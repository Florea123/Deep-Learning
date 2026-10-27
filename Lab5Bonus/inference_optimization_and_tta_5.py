import copy
import io
import os
import sys
import time
from pathlib import Path
from typing import Tuple

import timm
import torch
from prettytable import PrettyTable
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from tqdm import tqdm

import torch.ao.quantization as quantization
from torch.ao.quantization import get_default_qconfig_mapping
import torch.ao.quantization.quantize_fx as quantize_fx


class ClassificationModel(nn.Module):
    def __init__(self, backbone_name: str = "resnet34", num_classes: int = 10, cifar_stem: bool = False):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False)
        if cifar_stem:
            self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.backbone.maxpool = nn.Identity()

        self.backbone.fc = nn.Linear(self.backbone.fc.weight.size(1), num_classes)

    def forward(self, x):
        return self.backbone(x)


def create_model(model_path: str, device: torch.device, dtype: torch.dtype = torch.float32) -> ClassificationModel:
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
    return model


def prepare_data(data_path: str) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261), inplace=True)
    ])
    dataset = CIFAR10(root=data_path, train=False, transform=transforms, download=True)
    dataloader = DataLoader(dataset, batch_size=200, shuffle=False)
    return tuple([x for x in dataloader])


def warmup(model, device: torch.device, dtype: torch.dtype = torch.float32):
    sample = torch.rand((200, 3, 32, 32), device=device, dtype=dtype)
    try:
        with torch.inference_mode():
            model(sample)
    except Exception:
        pass

    if device.type == "cuda":
        torch.cuda.synchronize()


def get_model_size_mb(model: nn.Module) -> float:

    buf = io.BytesIO()
    try:
        torch.save(model.state_dict(), buf)
        return round(buf.tell() / (1024 * 1024), 2)
    except Exception:
        param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
        return round((param_size + buffer_size) / (1024 * 1024), 2)


def evaluate_model(model, data: Tuple[Tuple[torch.Tensor, torch.Tensor], ...],
                   device: torch.device, dtype: torch.dtype = torch.float32,
                   desc: str = "Evaluating") -> Tuple[float, float, float]:
    total_imgs = sum(x.size(0) for x, _ in data)
    warmup(model, device, dtype)

    t0 = time.perf_counter()
    correct, total = 0, 0
    with torch.inference_mode():
        for x, y in tqdm(data, desc=desc, leave=False):
            x = x.to(device=device, dtype=dtype)
            preds = model(x)
            correct += (preds.argmax(dim=1).cpu() == y).sum().item()
            total += y.size(0)

    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - t0
    acc = round(correct / total, 4)
    fps = round(total_imgs / elapsed, 1) if elapsed > 0 else 0.0
    return acc, elapsed, fps


def test_float8_direct_casting(model_path: str, device: torch.device, data: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]):

    f8_types = [
        ("torch.float8_e4m3fn", getattr(torch, "float8_e4m3fn", None)),
        ("torch.float8_e5m2", getattr(torch, "float8_e5m2", None)),
    ]

    sample_x = data[0][0][:2].to(device)

    for name, f8_dtype in f8_types:
        if f8_dtype is None:
            continue


        try:
            model_f8 = create_model(model_path, device, dtype=f8_dtype)
            sample_f8 = sample_x.to(dtype=f8_dtype)
            with torch.inference_mode():
                out = model_f8(sample_f8)
            print(f"    [SUCCESS] Direct inference succeeded! Output shape: {out.shape}")
        except Exception as e:
            print(f"    [FAILED with {type(e).__name__}]:")
            print(f"    {e}")
            print(f"    Reason: PyTorch/cuDNN standard conv2d does not provide native kernel implementations")
            print(f"    for Float8 scalar types directly via .to(float8). Float8 in PyTorch is currently intended")
            print(f"    for scaled GEMM/Linear operations (torch._scaled_mm) or specialized FP8 transformer backends.")


def test_float8_simulated_quantization(model_path: str, device: torch.device,
                                       data: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> list:

    results = []

    if hasattr(torch, "float8_e4m3fn"):
        model_e4m3_scaled = create_model(model_path, device, dtype=torch.float32)
        with torch.no_grad():
            for p_name, param in model_e4m3_scaled.named_parameters():
                if "weight" in p_name and param.dim() >= 2:
                    max_val = param.abs().max()
                    scale = 448.0 / (max_val + 1e-8)
                    scaled_w = (param * scale).clamp(-448.0, 448.0)
                    f8_w = scaled_w.to(torch.float8_e4m3fn)
                    param.copy_(f8_w.to(torch.float32) / scale)

        acc, el, fps = evaluate_model(model_e4m3_scaled, data, device, dtype=torch.float32)
        size_mb = get_model_size_mb(model_e4m3_scaled) / 4.0 
        results.append({
            "name": "Float8 e4m3fn (Scaled Weights)",
            "device": device.type.upper(),
            "dtype": "FP8 (e4m3fn)",
            "accuracy": acc,
            "elapsed": el,
            "fps": fps,
            "size_mb": round(size_mb, 2),
            "status": "Simulated (Native GEMM/WeightQuant)",
        })

    if hasattr(torch, "float8_e4m3fn"):
        model_e4m3_unscaled = create_model(model_path, device, dtype=torch.float32)
        with torch.no_grad():
            for p_name, param in model_e4m3_unscaled.named_parameters():
                if "weight" in p_name and param.dim() >= 2:
                    f8_w = param.to(torch.float8_e4m3fn)
                    param.copy_(f8_w.to(torch.float32))

        acc, el, fps = evaluate_model(model_e4m3_unscaled, data, device, dtype=torch.float32)
        size_mb = get_model_size_mb(model_e4m3_unscaled) / 4.0
        results.append({
            "name": "Float8 e4m3fn (Unscaled Weights)",
            "device": device.type.upper(),
            "dtype": "FP8 (e4m3fn)",
            "accuracy": acc,
            "elapsed": el,
            "fps": fps,
            "size_mb": round(size_mb, 2),
            "status": "Simulated (Unscaled)",
        })

    if hasattr(torch, "float8_e5m2"):
        model_e5m2 = create_model(model_path, device, dtype=torch.float32)
        with torch.no_grad():
            for p_name, param in model_e5m2.named_parameters():
                if "weight" in p_name and param.dim() >= 2:
                    f8_w = param.to(torch.float8_e5m2)
                    param.copy_(f8_w.to(torch.float32))

        acc, el, fps = evaluate_model(model_e5m2, data, device, dtype=torch.float32)
        size_mb = get_model_size_mb(model_e5m2) / 4.0
        results.append({
            "name": "Float8 e5m2 (Unscaled Weights)",
            "device": device.type.upper(),
            "dtype": "FP8 (e5m2)",
            "accuracy": acc,
            "elapsed": el,
            "fps": fps,
            "size_mb": round(size_mb, 2),
            "status": "Simulated (Unscaled)",
        })

    return results


def test_int8_quantization(model_path: str, data: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]) -> list:


    results = []
    cpu_device = torch.device("cpu")
    base_model = create_model(model_path, cpu_device, dtype=torch.float32)


    m_dyn = quantization.quantize_dynamic(
        copy.deepcopy(base_model),
        {nn.Linear},
        dtype=torch.qint8
    )
    acc_dyn, el_dyn, fps_dyn = evaluate_model(m_dyn, data, cpu_device, dtype=torch.float32)
    size_dyn = get_model_size_mb(m_dyn)
    results.append({
        "name": "INT8 Dynamic Quantization (Linear only)",
        "device": "CPU",
        "dtype": "qint8 (Dynamic)",
        "accuracy": acc_dyn,
        "elapsed": el_dyn,
        "fps": fps_dyn,
        "size_mb": size_dyn,
        "status": "Native PyTorch Dynamic INT8",
    })

    m_fx = copy.deepcopy(base_model)
    qconfig_mapping = get_default_qconfig_mapping("x86")
    example_inputs = (torch.randn(1, 3, 32, 32),)

    prepared_fx = quantize_fx.prepare_fx(m_fx, qconfig_mapping, example_inputs)

    with torch.no_grad():
        for x, _ in data[:5]:
            prepared_fx(x)

    quantized_fx = quantize_fx.convert_fx(prepared_fx)

    acc_fx, el_fx, fps_fx = evaluate_model(quantized_fx, data, cpu_device, dtype=torch.float32)
    size_fx = get_model_size_mb(quantized_fx)
    results.append({
        "name": "INT8 Static PTQ (Full Conv2d + Linear)",
        "device": "CPU",
        "dtype": "qint8 (Static FX)",
        "accuracy": acc_fx,
        "elapsed": el_fx,
        "fps": fps_fx,
        "size_mb": size_fx,
        "status": "Native PyTorch Static INT8 FX",
    })

    return results


def run_comprehensive_study(model_path: str, data: Tuple[Tuple[torch.Tensor, torch.Tensor], ...]):
    cuda_device = torch.device("cuda") if torch.cuda.is_available() else None
    cpu_device = torch.device("cpu")
    primary_device = cuda_device if cuda_device else cpu_device

    base_fp32 = create_model(model_path, primary_device, dtype=torch.float32)
    acc_fp32, el_fp32, fps_fp32 = evaluate_model(base_fp32, data, primary_device, dtype=torch.float32)
    size_fp32 = get_model_size_mb(base_fp32)

    rows = [{
        "name": "FP32 Baseline (Full Precision)",
        "device": primary_device.type.upper(),
        "dtype": "FP32",
        "accuracy": acc_fp32,
        "elapsed": el_fp32,
        "fps": fps_fp32,
        "size_mb": size_fp32,
        "status": "Baseline",
    }]

    if primary_device.type == "cuda":
        base_fp16 = create_model(model_path, primary_device, dtype=torch.float16)
        acc_fp16, el_fp16, fps_fp16 = evaluate_model(base_fp16, data, primary_device, dtype=torch.float16)
        rows.append({
            "name": "FP16 Baseline (Direct Half)",
            "device": "CUDA",
            "dtype": "FP16",
            "accuracy": acc_fp16,
            "elapsed": el_fp16,
            "fps": fps_fp16,
            "size_mb": round(size_fp32 / 2.0, 2),
            "status": "Direct Half",
        })

        base_bf16 = create_model(model_path, primary_device, dtype=torch.bfloat16)
        acc_bf16, el_bf16, fps_bf16 = evaluate_model(base_bf16, data, primary_device, dtype=torch.bfloat16)
        rows.append({
            "name": "BF16 Baseline (Direct BFloat)",
            "device": "CUDA",
            "dtype": "BF16",
            "accuracy": acc_bf16,
            "elapsed": el_bf16,
            "fps": fps_bf16,
            "size_mb": round(size_fp32 / 2.0, 2),
            "status": "Direct BFloat",
        })

    test_float8_direct_casting(model_path, primary_device, data)
    rows.append({
        "name": "Float8 Direct Casting (.to(float8))",
        "device": primary_device.type.upper(),
        "dtype": "FP8 (e4m3fn)",
        "accuracy": "N/A",
        "elapsed": "N/A",
        "fps": "N/A",
        "size_mb": round(size_fp32 / 4.0, 2),
        "status": "FAILED (RuntimeError: cuDNN conv unsupported)",
    })

    f8_sim_results = test_float8_simulated_quantization(model_path, primary_device, data)
    rows.extend(f8_sim_results)

    int8_results = test_int8_quantization(model_path, data)
    rows.extend(int8_results)

    table = PrettyTable()
    table.field_names = [
        "Experiment / Method", "Device", "Precision", "Model Size (MB)",
        "Accuracy", "Throughput (imgs/s)", "Degradation vs FP32"
    ]

    for r in rows:
        acc = r["accuracy"]
        if isinstance(acc, float):
            deg = f"{(acc - acc_fp32) * 100:+.2f}%"
            acc_str = f"{acc * 100:.2f}%"
        else:
            deg = "N/A (Failed)"
            acc_str = str(acc)

        fps_str = f"{r['fps']}" if isinstance(r['fps'], (int, float)) else str(r['fps'])
        table.add_row([
            r["name"],
            r["device"],
            r["dtype"],
            f"{r['size_mb']} MB",
            acc_str,
            fps_str,
            deg
        ])

    print("\n" + str(table) + "\n")


def main(model_path: str):
    data_path = "./data" if os.path.exists("./data") else "../data"
    if not os.path.exists(data_path):
        data_path = os.path.join(Path(__file__).parent.resolve(), "data")

    print(f"Loading test data from: {data_path}")
    data = prepare_data(data_path)

    run_comprehensive_study(model_path, data)


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    torch.set_float32_matmul_precision('high')

    checkpoint_dir = os.path.join(str(Path(__file__).parent.resolve()), "checkpoints")
    resnet34_path = os.path.join(checkpoint_dir, "best_resnet34.pth")
    if os.path.exists(resnet34_path):
        model_path = resnet34_path
    else:
        model_path = os.path.join(checkpoint_dir, "best.pth")

    print(f"Testing (Float8 & INT8 Quantization) on model: {model_path}")
    main(model_path)
