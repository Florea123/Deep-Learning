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


def export_to_torchscript(model_path: str, pt_path: str) -> str:
    if os.path.exists(pt_path):
        print(f"[TorchScript] Using existing TorchScript model at: {pt_path}")
        return pt_path

    cpu_device = torch.device("cpu")
    model = create_model(model_path, cpu_device, "raw model")
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32, device=cpu_device)
    os.makedirs(os.path.dirname(pt_path), exist_ok=True)

    with torch.inference_mode():
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(pt_path)

    reloaded = torch.jit.load(pt_path, map_location=cpu_device)
    with torch.inference_mode():
        diff = torch.max(torch.abs(model(dummy_input) - reloaded(dummy_input))).item()

    file_size_mb = os.path.getsize(pt_path) / (1024 * 1024)
    print(f"[TorchScript] Export verified successfully! Max diff: {diff:.6e}, File size: {file_size_mb:.2f} MB\n")
    return pt_path


def create_model(model_path: str, device: torch.device, model_type: str):
    if model_type == "TorchScript C++ Model":
        pt_path = os.path.splitext(model_path)[0] + "_traced.pt"
        if not os.path.exists(pt_path):
            export_to_torchscript(model_path, pt_path)
        model = torch.jit.load(pt_path, map_location=device)
        model.eval()
        return model

    model_data = torch.load(model_path, map_location=device, weights_only=True)

    model_name = model_data.get("model_name", "resnet34")
    num_classes = model_data.get("num_classes", 10)
    state_dict = model_data["model_state_dict"]

    cifar_stem = False
    if "backbone.conv1.weight" in state_dict and state_dict["backbone.conv1.weight"].shape[-1] == 3:
        cifar_stem = True

    model = ClassificationModel(model_name, num_classes, cifar_stem=cifar_stem)
    model = model.to(device)
    model.load_state_dict(state_dict)
    model.eval()

    if model_type == "raw model":
        return model
    if sys.version_info < (3, 14):
        # These do not work anymore on py >= 3.14
        if model_type == "scripted model":
            return torch.jit.script(model)
        if model_type == "traced model":
            return torch.jit.trace(model, torch.rand((5, 3, 32, 32), device=device))
        if model_type == "frozen model":
            return torch.jit.freeze(torch.jit.script(model))
        if model_type == "optimized for inference":
            return torch.jit.optimize_for_inference(torch.jit.script(model))
    if model_type == "compiled model":
        return torch.compile(model, backend="inductor", dynamic=False)
    if model_type == "compiled reduce-overhead":
        return torch.compile(model, backend="inductor", mode="reduce-overhead", dynamic=False)
    if model_type == "compiled max-autotune":
        return torch.compile(model, backend="inductor", mode="max-autotune", dynamic=False)
    if model_type == "TensorRT":
        if device.type != "cuda":
            raise RuntimeError("Torch-TensorRT requires a CUDA device")
        if not HAS_TORCH_TENSORRT:
            raise RuntimeError("torch_tensorrt is not installed")
        return torch.compile(
            model,
            backend="torch_tensorrt",
            dynamic=False,
            options={"optimization_level": 5},
        )
    if model_type == "OpenVINO":
        if device.type != "cpu":
            raise RuntimeError("This OpenVINO experiment is configured for CPU only")
        if not HAS_OPENVINO:
            raise RuntimeError("OpenVINO is not installed")

        return torch.compile(
            model,
            backend="openvino",
            dynamic=False,
            options={
                "device": "CPU",
                "config": {"PERFORMANCE_HINT": "LATENCY"},
            },
        )
    print(f"Model {model_type} not supported")
    return None


@timed(stdout=False, return_time=True, use_seconds=True)
def tta_inference(model, batches: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], device: torch.device,
                  tta_type: str) -> float:
    total = 0
    correct = 0

    for data, target in batches:
        data = data.to(device)

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
    enable_autocast = device.type == "cuda" and dtype != torch.float32
    accuracy, elapsed = "N/A", "N/A"
    try:
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=enable_autocast), torch.inference_mode():
            accuracy, elapsed = tta_inference(model, batches, device, tta_type)
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
    enable_autocast = device.type == "cuda" and dtype != torch.float32
    sample = torch.rand((200, 3, 32, 32), device=device)

    try:
        with torch.inference_mode(), torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=enable_autocast,
        ):
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
    with tqdm(total=len(devices) * len(dtypes) * len(model_types), desc="Speed experiments") as tbar:
        for device, dtype in product(devices, dtypes):
            if device is None:
                tbar.update(len(model_types))
                continue
            speed_results = PrettyTable()
            speed_results.field_names = ["Device", "Dtype", "TTA Type", "Model Type", "Accuracy", "Elapsed"]

            for model_type in model_types:
                tbar.update()
                if model_type == "TensorRT" and (device.type != "cuda" or not HAS_TORCH_TENSORRT):
                    continue
                if model_type == "OpenVINO" and (device.type != "cpu" or not HAS_OPENVINO):
                    continue

                model = create_model(model_path, device, model_type)
                if model is None:
                    continue
                warmup(model, device, dtype)
                accuracy, elapsed = inference(model, data, device, tta_type, dtype, model_type)
                speed_results.add_row([device, dtype, tta_type, model_type, accuracy, elapsed])

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

    for dtype, tta_type in tqdm(tuple(product(dtypes, tta_types)), desc="TTA experiments"):
        if device is None:
            continue
        model = create_model(model_path, device, model_type)
        accuracy, elapsed = inference(model, data, device, tta_type, dtype, model_type)
        tta_results.add_row([device, dtype, tta_type, model_type, accuracy, elapsed])

    print(tta_results)


def check_libtorch_performance_gains(data: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
                                     devices: Tuple[torch.device | None, ...],
                                     model_path: str):

    pt_path = os.path.splitext(model_path)[0] + "_traced.pt"
    export_to_torchscript(model_path, pt_path)

    table = PrettyTable()
    table.field_names = [
        "Device", "Environment / Runtime", "Elapsed (s)", "Throughput (imgs/s)", "Speedup vs Python Raw", "Accuracy"
    ]

    total_images = sum(d[0].size(0) for d in data)

    for device in devices:
        if device is None:
            continue

        dev_name = device.type.upper()
        pt_model = create_model(model_path, device, "raw model")
        warmup(pt_model, device, torch.float32)
        pt_acc, pt_elapsed = inference(pt_model, data, device, "none", torch.float32, "raw model")

        ts_model = create_model(model_path, device, "TorchScript C++ Model")
        warmup(ts_model, device, torch.float32)
        ts_acc, ts_elapsed = inference(ts_model, data, device, "none", torch.float32, "TorchScript C++ Model")

        opt_model = create_model(model_path, device, "optimized for inference")
        opt_acc, opt_elapsed = "N/A", "N/A"
        if opt_model is not None:
            warmup(opt_model, device, torch.float32)
            opt_acc, opt_elapsed = inference(opt_model, data, device, "none", torch.float32, "optimized for inference")

        pt_fps = f"{total_images / pt_elapsed:.1f}" if isinstance(pt_elapsed, (int, float)) else "N/A"
        ts_fps = f"{total_images / ts_elapsed:.1f}" if isinstance(ts_elapsed, (int, float)) else "N/A"
        ts_speedup = f"{pt_elapsed / ts_elapsed:.2f}x" if isinstance(pt_elapsed, (int, float)) and isinstance(ts_elapsed, (int, float)) else "N/A"

        table.add_row([dev_name, "Python (Raw PyTorch)", pt_elapsed, pt_fps, "1.00x (Baseline)", pt_acc])
        table.add_row([dev_name, "LibTorch / TorchScript (C++)", ts_elapsed, ts_fps, ts_speedup, ts_acc])

        if isinstance(opt_elapsed, (int, float)):
            opt_fps = f"{total_images / opt_elapsed:.1f}"
            opt_speedup = f"{pt_elapsed / opt_elapsed:.2f}x"
            table.add_row([dev_name, "LibTorch (Optimized for Inf.)", opt_elapsed, opt_fps, opt_speedup, opt_acc])

    print("\n" + str(table) + "\n")


def main(model_path: str):
    data_path = os.path.join(str(Path(__file__).parent.resolve()), "data")
    data = prepare_data(data_path)

    pt_path = os.path.splitext(model_path)[0] + "_traced.pt"
    export_to_torchscript(model_path, pt_path)

    model_types = (
        "raw model",
        "scripted model",
        "traced model",
        "frozen model",
        "optimized for inference",
        "compiled model",
        "compiled reduce-overhead",
        "compiled max-autotune",
        "TorchScript C++ Model",
        "TensorRT",
        "OpenVINO",
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

    check_libtorch_performance_gains(data, devices, model_path)


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

    print(f"Testing LibTorch inference pipeline on model: {model_path}")
    main(model_path)
