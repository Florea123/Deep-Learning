import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Tuple

import timm
import torch
from prettytable import PrettyTable
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torchvision.transforms.v2.functional import hflip
from timed_decorator.simple_timed import timed
from tqdm import tqdm

try:
    import torch_tensorrt
    HAS_TORCH_TENSORRT = True
except ImportError:
    HAS_TORCH_TENSORRT = False

try:
    import openvino
    import openvino.torch
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False

print("HAS_TORCH_TENSORRT:", HAS_TORCH_TENSORRT)
print("HAS_OPENVINO:", HAS_OPENVINO)


class ClassificationModel(nn.Module):
    def __init__(self, backbone_name: str = "resnet34", num_classes: int = 10, cifar_stem: bool = False):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, pretrained=False)

        if cifar_stem:
            if hasattr(self.backbone, 'conv1'):
                in_c = self.backbone.conv1.in_channels
                out_c = self.backbone.conv1.out_channels
                self.backbone.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
            if hasattr(self.backbone, 'maxpool'):
                self.backbone.maxpool = nn.Identity()

        self.backbone.fc = nn.Linear(self.backbone.fc.weight.size(1), num_classes)

    def forward(self, x):
        return self.backbone(x)


class StandaloneOpenVINOModel(nn.Module):
    """OpenVINO model executed standalone without torch.compile."""
    def __init__(self, model_path: str, dtype: torch.dtype):
        super().__init__()
        core = openvino.Core()

        precision_map = {
            torch.float32: "f32",
            torch.float16: "f16",
            torch.bfloat16: "bf16",
        }
        hint = precision_map.get(dtype, "f32")

        config = {
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": hint,
            "EXECUTION_MODE_HINT": "PERFORMANCE",
            "NUM_STREAMS": "1",
            "CPU_DENORMALS_OPTIMIZATION": "YES",
        }

        cpu_device = torch.device("cpu")
        pt_model = create_model(model_path, cpu_device, "raw model", dtype=torch.float32)
        dummy = torch.randn(1, 3, 32, 32)
        ov_model = openvino.convert_model(pt_model, example_input=dummy)

        self.compiled = core.compile_model(ov_model, device_name="CPU", config=config)
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_np = x.detach().float().cpu().numpy()
        res = self.compiled([x_np])[0]
        return torch.from_numpy(res).to(device=x.device, dtype=x.dtype)


class StandaloneTensorRTModel(nn.Module):
    def __init__(self, model_path: str, dtype: torch.dtype):
        super().__init__()
        if not HAS_TORCH_TENSORRT:
            raise RuntimeError("torch_tensorrt is not installed")
        import torch_tensorrt

        device = torch.device("cuda")
        pt_model = create_model(model_path, device, "raw model", dtype=dtype)
        dummy = torch.randn(200, 3, 32, 32, device=device, dtype=dtype)
        scripted = torch.jit.trace(pt_model, dummy)

        enabled_precisions = {dtype} if dtype in [torch.float16, torch.bfloat16] else {torch.float32}
        self.trt_model = torch_tensorrt.compile(
            scripted,
            ir="ts",
            inputs=[torch_tensorrt.Input((200, 3, 32, 32), dtype=dtype)],
            enabled_precisions=enabled_precisions,
            workspace_size=1 << 30,
            truncate_long_and_double=True,
            optimization_level=5
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trt_model(x)


def create_model(model_path: str, device: torch.device, model_type: str, dtype: torch.dtype = torch.float32):
    if model_type == "OpenVINO (Standalone)":
        if device.type != "cpu":
            raise RuntimeError("OpenVINO is configured for CPU only")
        if not HAS_OPENVINO:
            raise RuntimeError("OpenVINO is not installed")
        return StandaloneOpenVINOModel(model_path, dtype)

    if model_type == "TensorRT (Standalone)":
        if device.type != "cuda":
            raise RuntimeError("TensorRT requires a CUDA device")
        if not HAS_TORCH_TENSORRT:
            raise RuntimeError("torch_tensorrt is not installed")
        return StandaloneTensorRTModel(model_path, dtype)

    model_data = torch.load(model_path, map_location=device, weights_only=True)

    model_name = model_data.get("model_name", "resnet34")
    num_classes = model_data.get("num_classes", 10)
    state_dict = model_data["model_state_dict"]

    cifar_stem = False
    if "backbone.conv1.weight" in state_dict and state_dict["backbone.conv1.weight"].shape[-1] == 3:
        cifar_stem = True

    model = ClassificationModel(model_name, num_classes, cifar_stem=cifar_stem)
    model.load_state_dict(state_dict)

    model = model.to(device=device, dtype=dtype)
    model.eval()

    if model_type == "raw model":
        return model
    if sys.version_info < (3, 14):
        if model_type == "scripted model":
            return torch.jit.script(model)
        if model_type == "traced model":
            return torch.jit.trace(model, torch.rand((5, 3, 32, 32), device=device, dtype=dtype))
        if model_type == "frozen model":
            return torch.jit.freeze(torch.jit.script(model))
        if model_type == "optimized for inference":
            return torch.jit.optimize_for_inference(torch.jit.script(model))

    print(f"Model {model_type} not supported")
    return None


@timed(stdout=False, return_time=True, use_seconds=True)
def tta_inference(model, batches: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], device: torch.device,
                  tta_type: str, dtype: torch.dtype = torch.float32) -> float:
    total = 0
    correct = 0

    for data, target in batches:
        data = data.to(device=device, dtype=dtype)

        predicted = model(data)
        if tta_type == "mirroring":
            predicted += model(hflip(data))
        elif tta_type == "translate":
            padding_size = 2
            image_size = 32
            padded = v2.functional.pad(data, [padding_size], fill=0.5)
            for i in [-2, 0, 2]:
                for j in [-2, 0, 2]:
                    if i == 0 and j == 0:
                        continue
                    x = padding_size + i
                    y = padding_size + j
                    predicted += model(padded[:, :, x:x + image_size, y:y + image_size])
        elif tta_type == "mirroring_and_translate":
            padding_size = 2
            image_size = 32
            padded = v2.functional.pad(data, [padding_size], fill=0.5)
            for i in [-2, 0, 2]:
                for j in [-2, 0, 2]:
                    if i == 0 and j == 0:
                        continue
                    x = padding_size + i
                    y = padding_size + j
                    aux = padded[:, :, x:x + image_size, y:y + image_size]
                    predicted += model(aux)
                    predicted += model(hflip(aux))

        correct += (predicted.cpu().argmax(dim=1) == target).sum().item()
        total += data.size(0)

    return round(correct / total, 4)


def inference(model, batches: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], device: torch.device, tta_type: str,
              dtype: torch.dtype, model_type: str) -> Tuple[float, float]:
    accuracy, elapsed = "N/A", "N/A"
    try:
        with torch.inference_mode():
            accuracy, elapsed = tta_inference(model, batches, device, tta_type, dtype)
    except Exception as e:
        print(f"Model type {model_type} failed on {dtype} on {device.type} because of {e}")

    return accuracy, elapsed


def prepare_data(data_path: str) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261), inplace=True)
    ])
    dataset = CIFAR10(root=data_path, train=False, transform=transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=200)
    return tuple([x for x in dataloader])


def warmup(model, device: torch.device, dtype: torch.dtype):
    sample = torch.rand((200, 3, 32, 32), device=device, dtype=dtype)
    try:
        with torch.inference_mode():
            model(sample)
    except Exception:
        pass

    if device.type == "cuda":
        torch.cuda.synchronize()


def do_speed_test(data: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
                  model_types: Tuple[str, ...],
                  dtypes: Tuple[torch.dtype, ...],
                  tta_types: Tuple[str, ...],
                  devices: Tuple[torch.device | None, ...],
                  model_path: str):
    tta_type = "none"
    with tqdm(total=len(devices) * len(dtypes) * len(model_types), desc="Speed experiments (No Autocast)") as tbar:
        for device, dtype in product(devices, dtypes):
            if device is None:
                tbar.update(len(model_types))
                continue
            speed_results = PrettyTable()
            speed_results.field_names = ["Device", "Dtype", "TTA Type", "Model Type", "Accuracy", "Elapsed"]

            for model_type in model_types:
                tbar.update()
                if "TensorRT" in model_type and (device.type != "cuda" or not HAS_TORCH_TENSORRT):
                    continue
                if "OpenVINO" in model_type and (device.type != "cpu" or not HAS_OPENVINO):
                    continue
                
                if device.type == "cpu" and dtype != torch.float32 and "OpenVINO" not in model_type:
                    continue

                model = create_model(model_path, device, model_type, dtype=dtype)
                if model is None:
                    continue
                warmup(model, device, dtype)
                accuracy, elapsed = inference(model, data, device, tta_type, dtype, model_type)
                speed_results.add_row([device, dtype, tta_type, model_type, accuracy, elapsed])

            if len(speed_results.rows) > 0:
                print(speed_results)


def do_tta_test(data: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
                model_types: Tuple[str, ...],
                dtypes: Tuple[torch.dtype, ...],
                tta_types: Tuple[str, ...],
                devices: Tuple[torch.device | None, ...],
                model_path: str):
    tta_results = PrettyTable()
    tta_results.field_names = ["Device", "Dtype", "TTA Type", "Model Type", "Accuracy", "Elapsed"]

    device = devices[0] if devices[0] is not None else devices[1]
    model_type = "raw model"

    for dtype, tta_type in tqdm(tuple(product(dtypes, tta_types)), desc="TTA experiments (No Autocast)"):
        if device is None:
            continue
        if device.type == "cpu" and dtype != torch.float32:
            continue
        model = create_model(model_path, device, model_type, dtype=dtype)
        accuracy, elapsed = inference(model, data, device, tta_type, dtype, model_type)
        tta_results.add_row([device, dtype, tta_type, model_type, accuracy, elapsed])

    print(tta_results)


def compare_autocast_vs_direct(data: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
                               devices: Tuple[torch.device | None, ...],
                               model_path: str):

    device = devices[0] if devices[0] is not None else devices[1]
    if device is None:
        return

    table = PrettyTable()
    table.field_names = [
        "Device", "Dtype", "Mode", "Accuracy", "Elapsed (s)", "Throughput (imgs/s)", "Degradation vs FP32"
    ]

    dtypes = [
        (torch.float32, "FP32"),
        (torch.float16, "FP16"),
        (torch.bfloat16, "BF16"),
    ]

    total_images = sum(d[0].size(0) for d in data)

    fp32_base_acc = None
    for dt, dt_name in dtypes:
        model_ac = create_model(model_path, device, "raw model", dtype=torch.float32)
        warmup(model_ac, device, dt)

        enable_ac = device.type == "cuda" and dt != torch.float32
        t0 = time.perf_counter()
        correct, total = 0, 0
        with torch.autocast(device_type=device.type, dtype=dt, enabled=enable_ac), torch.inference_mode():
            for x, y in data:
                x = x.to(device)
                pred = model_ac(x)
                correct += (pred.argmax(dim=1).cpu() == y).sum().item()
                total += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        el = time.perf_counter() - t0
        acc = round(correct / total, 4)

        if dt == torch.float32:
            fp32_base_acc = acc

        deg = f"{(acc - fp32_base_acc) * 100:+.2f}%" if fp32_base_acc else "0.00% (Baseline)"
        fps = f"{total_images / el:.1f}"
        table.add_row([device.type.upper(), dt_name, "With Autocast", acc, f"{el:.4f}", fps, deg])

    for dt, dt_name in dtypes:
        model_direct = create_model(model_path, device, "raw model", dtype=dt)
        warmup(model_direct, device, dt)

        t0 = time.perf_counter()
        correct, total = 0, 0
        with torch.inference_mode():
            for x, y in data:
                x = x.to(device=device, dtype=dt)
                pred = model_direct(x)
                correct += (pred.argmax(dim=1).cpu() == y).sum().item()
                total += x.size(0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        el = time.perf_counter() - t0
        acc = round(correct / total, 4)

        deg = f"{(acc - fp32_base_acc) * 100:+.2f}%"
        fps = f"{total_images / el:.1f}"
        table.add_row([device.type.upper(), dt_name, "DIRECT (No Autocast)", acc, f"{el:.4f}", fps, deg])

    print("\n" + str(table) + "\n")



def main(model_path: str):
    data_path = os.path.join(str(Path(__file__).parent.resolve()), "data")
    data = prepare_data(data_path)
    model_types = (
        "raw model",
        "scripted model",
        "traced model",
        "frozen model",
        "optimized for inference",
        "OpenVINO (Standalone)",
        "TensorRT (Standalone)",
    )
    dtypes = (
        torch.bfloat16,
        torch.half,
        torch.float32,
    )
    tta_types = (
        "none",
        "mirroring",
        "translate",
        "mirroring_and_translate",
    )
    devices = (
        torch.accelerator.current_accelerator() if torch.accelerator.is_available() else None,
        torch.device("cpu"),
    )

    do_speed_test(data, model_types, dtypes, tta_types, devices, model_path)
    do_tta_test(data, model_types, dtypes, tta_types, devices, model_path)
    compare_autocast_vs_direct(data, devices, model_path)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    if HAS_TORCH_TENSORRT:
        print("Torch-TensorRT is available.")
    else:
        print("Note: Torch-TensorRT is not installed, TensorRT speed experiments will be skipped.")

    if HAS_OPENVINO:
        print("OpenVINO is available for CPU inference.")

    torch.set_float32_matmul_precision('high')
    checkpoint_dir = os.path.join(str(Path(__file__).parent.resolve()), "checkpoints")
    resnet34_path = os.path.join(checkpoint_dir, "best_resnet34.pth")
    if os.path.exists(resnet34_path):
        model_path = resnet34_path
    else:
        model_path = os.path.join(checkpoint_dir, "best.pth")

    print(f"Testing inference optimization on model: {model_path}")
    main(model_path)
