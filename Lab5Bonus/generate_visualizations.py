import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#eeeeee'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.7

OUTPUT_DIR = Path(__file__).parent / "visualizations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_notebook_vs_machine():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    categories = ["Base (No TTA)", "Mirroring", "Translate", "Mirror & Translate"]
    x = np.arange(len(categories))
    width = 0.35

    resnet18_acc = [86.28, 87.29, 87.35, 87.85]
    resnet34_acc = [96.40, 96.73, 96.51, 96.88]

    rects1 = ax1.bar(x - width/2, resnet18_acc, width, label='Notebook Baseline: ResNet-18', color='#7F8C8D', edgecolor='#333333')
    rects2 = ax1.bar(x + width/2, resnet34_acc, width, label='Machine Experiment: ResNet-34 (Ours)', color='#27AE60', edgecolor='#333333')

    ax1.set_title("A. Accuracy: Notebook (ResNet-18) vs. Machine (ResNet-34)", fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylabel("CIFAR-10 Test Accuracy (%)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax1.set_ylim(80.0, 100.0)
    ax1.grid(axis='y')
    ax1.legend(loc='lower right', frameon=True)

    for bar, acc in zip(rects1, resnet18_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6, f"{acc:.2f}%", ha='center', va='bottom', fontsize=8.5, color='#444444')
    for bar, acc in zip(rects2, resnet34_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.6, f"{acc:.2f}%\n(+10.1%)", ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#196F3D')


    models = ["ResNet-18\n(Notebook Baseline)", "ResNet-34 Raw\n(Machine Baseline)", "ResNet-34 JIT Opt.\n(Machine Optimized)", "ResNet-34 ONNX\n(Machine ORT)"]
    cuda_elapsed = [0.218, 3.652, 2.819, 3.545] 
    colors_b = ["#95A5A6", "#3498DB", "#2ECC71", "#F39C12"]

    bars_b = ax2.bar(models, cuda_elapsed, color=colors_b, width=0.55, edgecolor='#333333')
    ax2.set_title("B. CUDA Inference Latency for Full Evaluation Set", fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylabel("Elapsed Time (seconds - Lower is Better)", fontsize=11)
    ax2.set_ylim(0, 4.5)
    ax2.grid(axis='y')

    for bar, val in zip(bars_b, cuda_elapsed):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08, f"{val:.2f}s", ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    fig.suptitle("Data Analysis 1: Notebook Baseline (ResNet-18) vs. Machine Experiments (ResNet-34)", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig0_notebook_vs_machine_comparison.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_runtime_speedup():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    cuda_runtimes = [
        "PyTorch Raw\n(Baseline)",
        "PyTorch JIT\n(Optimized)",
        "ONNX Runtime\n(CUDA Provider)",
        "LibTorch C++\n(TorchScript)",
        "LibTorch C++\n(Optimized)"
    ]
    cuda_fps = [2738.1, 3550.0, 2817.0, 2710.4, 3515.8]
    cuda_speedups = [1.00, 1.30, 1.04, 0.99, 1.28]
    cuda_colors = ["#4A90E2", "#50E3C2", "#F5A623", "#9013FE", "#27AE60"]

    bars1 = ax1.bar(cuda_runtimes, cuda_fps, color=cuda_colors, width=0.55, edgecolor="#333333", linewidth=0.8)
    ax1.set_title("A. CUDA Inference Throughput (RTX 4050)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel("Throughput (images / second)", fontsize=11, fontweight='medium')
    ax1.set_ylim(0, 4200)
    ax1.grid(axis='y')

    for bar, fps, sp in zip(bars1, cuda_fps, cuda_speedups):
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, yval + 70, f"{fps:.0f} imgs/s\n({sp:.2f}x)",
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#222222')

    cpu_runtimes = [
        "PyTorch Raw\n(Baseline)",
        "PyTorch JIT\n(Optimized)",
        "OpenVINO\n(CPU Native)",
        "ONNX Runtime\n(CPU Provider)",
        "LibTorch C++\n(Optimized)"
    ]
    cpu_fps = [187.1, 274.5, 234.0, 208.6, 195.6]
    cpu_speedups = [1.00, 1.47, 1.25, 1.11, 1.05]
    cpu_colors = ["#4A90E2", "#50E3C2", "#E74C3C", "#F5A623", "#27AE60"]

    bars2 = ax2.bar(cpu_runtimes, cpu_fps, color=cpu_colors, width=0.55, edgecolor="#333333", linewidth=0.8)
    ax2.set_title("B. CPU Inference Throughput (x86_64 Host)", fontsize=13, fontweight='bold', pad=12)
    ax2.set_ylabel("Throughput (images / second)", fontsize=11, fontweight='medium')
    ax2.set_ylim(0, 340)
    ax2.grid(axis='y')

    for bar, fps, sp in zip(bars2, cpu_fps, cpu_speedups):
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2, yval + 6, f"{fps:.0f} imgs/s\n({sp:.2f}x)",
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#222222')

    fig.suptitle("Exercise 1, 2 & 3: Runtime & Framework Optimization Benchmarks", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig1_runtime_throughput_speedup.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_tta_analysis():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    tta_modes = ["None\n(Single Pass)", "Mirroring\n(Horizontal Flip)", "Translate\n(Shift 8 Patches)", "Mirroring &\nTranslate (16 Patches)"]
    accuracies = [96.41, 96.72, 96.51, 96.88]
    elapsed_times_fp16 = [1.94, 3.84, 17.23, 32.35]
    elapsed_times_fp32 = [3.65, 7.27, 32.61, 61.55]

    colors = ["#3498DB", "#2ECC71", "#E67E22", "#9B59B6"]

    bars = ax1.bar(tta_modes, accuracies, color=colors, width=0.5, edgecolor="#333333", linewidth=0.8)
    ax1.set_title("A. Accuracy Gain across TTA Strategies", fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel("CIFAR-10 Test Accuracy (%)", fontsize=11, fontweight='medium')
    ax1.set_ylim(96.0, 97.2)
    ax1.grid(axis='y')

    for bar, acc in zip(bars, accuracies):
        diff = acc - 96.41
        label = f"{acc:.2f}%\n" + (f"(+{diff:.2f}%)" if diff > 0 else "(Baseline)")
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03, label,
                 ha='center', va='bottom', fontsize=9.5, fontweight='bold')

    ax2.plot(elapsed_times_fp16, accuracies, 'o--', color='#2ECC71', linewidth=2.5, markersize=9, label='Direct FP16 (CUDA)')
    ax2.plot(elapsed_times_fp32, accuracies, 's--', color='#3498DB', linewidth=2, markersize=8, label='FP32 (CUDA)')

    for t_fp16, acc, mode in zip(elapsed_times_fp16, accuracies, ["None", "Mirroring", "Translate", "Mirror+Translate"]):
        offset_y = 0.04 if mode != "Translate" else -0.07
        ax2.annotate(f"{mode}\n({acc:.2f}%, {t_fp16:.1f}s)",
                     (t_fp16, acc),
                     textcoords="offset points",
                     xytext=(0, 10 if offset_y > 0 else -25),
                     ha='center', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='#F9F9F9', edgecolor='#CCCCCC', alpha=0.9))

    ax2.set_title("B. TTA Efficiency Frontier: Accuracy vs. Elapsed Time", fontsize=13, fontweight='bold', pad=12)
    ax2.set_xlabel("Elapsed Time for 10,000 Images (seconds - Lower is Better)", fontsize=11, fontweight='medium')
    ax2.set_ylabel("Accuracy (%) - Higher is Better", fontsize=11, fontweight='medium')
    ax2.set_ylim(96.3, 97.1)
    ax2.grid(True)
    ax2.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9)

    fig.suptitle("Test-Time Augmentation (TTA) Trade-Off: Accuracy Boost vs. Compute Overhead", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig2_tta_accuracy_latency_tradeoff.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_quantization_and_precision():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)

    methods = [
        "FP32 Baseline\n(Full Precision)",
        "FP16 Baseline\n(Direct Half)",
        "BF16 Baseline\n(Direct BFloat)",
        "Float8 e4m3fn\n(Scaled)",
        "Float8 e4m3fn\n(Unscaled)",
        "Float8 e5m2\n(Unscaled)",
        "INT8 Dynamic\n(Linear fc)",
        "INT8 Static PTQ\n(Conv+Linear)"
    ]
    model_sizes = [81.32, 40.66, 40.66, 20.33, 20.33, 20.33, 81.31, 20.52]
    accuracies = [96.40, 96.41, 96.32, 96.38, 96.48, 96.36, 96.38, 96.45]


    size_colors = ["#4A90E2", "#34495E", "#34495E", "#E67E22", "#D35400", "#E74C3C", "#95A5A6", "#27AE60"]
    bars_size = ax1.bar(methods, model_sizes, color=size_colors, width=0.55, edgecolor="#333333", linewidth=0.8)
    ax1.set_title("A. Memory Footprint Reduction across Formats", fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylabel("Model Size (Megabytes - MB)", fontsize=11, fontweight='medium')
    ax1.set_ylim(0, 100)
    ax1.grid(axis='y')

    for bar, size in zip(bars_size, model_sizes):
        compression = 81.32 / size
        label = f"{size:.1f} MB\n({compression:.1f}x)" if compression > 1.1 else f"{size:.1f} MB\n(1.0x)"
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2, label,
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

    acc_colors = ["#4A90E2", "#27AE60", "#F39C12", "#2980B9", "#1ABC9C", "#E74C3C", "#8E44AD", "#16A085"]
    bars_acc = ax2.bar(methods, accuracies, color=acc_colors, width=0.55, edgecolor="#333333", linewidth=0.8)
    ax2.set_title("B. Accuracy Retention (Zero Degradation in FP16 / INT8)", fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylabel("CIFAR-10 Accuracy (%)", fontsize=11, fontweight='medium')
    ax2.set_ylim(96.0, 96.8)
    ax2.axhline(96.40, color="#E74C3C", linestyle=":", linewidth=1.5, label="FP32 Baseline (96.40%)")
    ax2.grid(axis='y')
    ax2.legend(loc='upper right', frameon=True)

    for bar, acc in zip(bars_acc, accuracies):
        deg = acc - 96.40
        deg_str = f"+{deg:.2f}%" if deg >= 0 else f"{deg:.2f}%"
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{acc:.2f}%\n({deg_str})",
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig.suptitle("Exercise 4 & 5: Precision, Quantization & Model Compression Analysis", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig3_precision_and_quantization_comparison.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_autocast_vs_direct():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    dtypes = ["FP32", "FP16", "BF16"]
    x = np.arange(len(dtypes))
    width = 0.32

    throughput_autocast = [1202.8, 2222.1, 2229.5]
    throughput_direct = [1206.2, 2255.3, 2249.7]

    bars1 = ax.bar(x - width/2, throughput_autocast, width, label='With torch.autocast', color='#95A5A6', edgecolor='#333333')
    bars2 = ax.bar(x + width/2, throughput_direct, width, label='DIRECT Casting (No Autocast)', color='#27AE60', edgecolor='#333333')

    ax.set_title("Exercise 4: Autocast Overhead Elimination & Throughput Doubling in FP16/BF16", fontsize=13, fontweight='bold', pad=15)
    ax.set_ylabel("Inference Throughput (images / second)", fontsize=11, fontweight='medium')
    ax.set_xticks(x)
    ax.set_xticklabels(["FP32\n(Full Precision)", "FP16\n(Half Precision)", "BF16\n(BFloat16)"], fontsize=11, fontweight='bold')
    ax.set_ylim(0, 2700)
    ax.grid(axis='y')
    ax.legend(loc='upper left', fontsize=10.5, frameon=True)

    for bar, fps in zip(bars1, throughput_autocast):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40, f"{fps:.0f}",
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#555555')

    for bar, fps, base_fps in zip(bars2, throughput_direct, throughput_autocast):
        speedup_vs_fp32 = fps / 1206.2
        label = f"{fps:.0f} imgs/s\n({speedup_vs_fp32:.2f}x vs FP32)"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40, label,
                ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#1B5E20')

    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig4_autocast_vs_direct_throughput.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_executive_dashboard():
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=300)

    ax_tl = axes[0, 0]
    runtimes = ["PyTorch Raw", "ONNX Runtime", "LibTorch C++", "TorchScript Opt."]
    speeds = [2738, 2817, 2710, 3516]
    colors_tl = ["#3498DB", "#E67E22", "#9B59B6", "#2ECC71"]
    bars_tl = ax_tl.barh(runtimes, speeds, color=colors_tl, height=0.5, edgecolor='#333333')
    ax_tl.set_title("1. Execution Runtime Comparison (CUDA)", fontsize=12, fontweight='bold')
    ax_tl.set_xlabel("Throughput (imgs/s)", fontsize=10)
    ax_tl.set_xlim(0, 4200)
    ax_tl.grid(axis='x')
    for b in bars_tl:
        ax_tl.text(b.get_width() + 60, b.get_y() + b.get_height()/2, f"{b.get_width()} imgs/s", va='center', fontweight='bold', fontsize=9)


    ax_tr = axes[0, 1]
    q_names = ["FP32", "FP16", "Float8 Scaled", "INT8 Dynamic", "INT8 Static PTQ"]
    q_sizes = [81.3, 40.7, 20.3, 81.3, 20.5]
    q_accs = [96.40, 96.41, 96.38, 96.38, 96.45]
    ax_tr.scatter(q_sizes, q_accs, c=[1, 2, 3, 4, 5], cmap='viridis', s=250, edgecolors='#333333', linewidth=1.5, zorder=5)
    ax_tr.set_title("2. Model Compression vs. Accuracy Retention", fontsize=12, fontweight='bold')
    ax_tr.set_xlabel("Model Memory Footprint (MB - Lower is Better)", fontsize=10)
    ax_tr.set_ylabel("Accuracy (%) - Higher is Better", fontsize=10)
    ax_tr.set_ylim(96.2, 96.6)
    ax_tr.set_xlim(10, 90)
    ax_tr.grid(True)
    for name, s, a in zip(q_names, q_sizes, q_accs):
        ax_tr.annotate(f" {name}\n ({s:.1f}MB, {a:.2f}%)", (s, a), fontsize=8.5, fontweight='bold')

    ax_bl = axes[1, 0]
    tta_x = [1.94, 3.84, 17.23, 32.35]
    tta_y = [96.41, 96.72, 96.51, 96.88]
    tta_lbls = ["None", "Mirroring", "Translate", "Mirror+Translate"]
    ax_bl.plot(tta_x, tta_y, 'o-', color='#E74C3C', linewidth=2, markersize=8)
    ax_bl.set_title("3. Test-Time Augmentation (TTA) Accuracy Frontier", fontsize=12, fontweight='bold')
    ax_bl.set_xlabel("Evaluation Time (seconds)", fontsize=10)
    ax_bl.set_ylabel("Accuracy (%)", fontsize=10)
    ax_bl.set_ylim(96.3, 97.0)
    ax_bl.grid(True)
    for x, y, l in zip(tta_x, tta_y, tta_lbls):
        ax_bl.annotate(f"{l}: {y:.2f}%", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8.5, fontweight='bold')


    ax_br = axes[1, 1]
    precisions = ["FP32", "FP16 Direct", "BF16 Direct", "INT8 PTQ (CPU)"]
    throughputs_all = [1206, 2255, 2250, 386]
    colors_br = ["#34495E", "#27AE60", "#2ECC71", "#16A085"]
    bars_br = ax_br.bar(precisions, throughputs_all, color=colors_br, width=0.5, edgecolor='#333333')
    ax_br.set_title("4. Direct Precision Casting & Quantization Throughput", fontsize=12, fontweight='bold')
    ax_br.set_ylabel("Throughput (imgs/s)", fontsize=10)
    ax_br.set_ylim(0, 2700)
    ax_br.grid(axis='y')
    for b in bars_br:
        ax_br.text(b.get_x() + b.get_width()/2, b.get_height() + 40, f"{b.get_height()} imgs/s", ha='center', fontweight='bold', fontsize=9)

    fig.suptitle("Executive Dashboard: Deep Learning Model Inference Optimization (Lab 5)", fontsize=16, fontweight='bold', y=0.99)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig5_master_executive_dashboard.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


def plot_ex1_model_types():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    model_types = [
        "raw model",
        "scripted",
        "traced",
        "frozen",
        "optimized\nfor inf",
        "compiled\n(inductor)",
        "compiled\n(reduce-oh)",
        "compiled\n(max-auto)"
    ]
    x = np.arange(len(model_types))
    width = 0.26

    fp32_times = [3.64, 3.70, 3.64, 3.66, 2.82, 3.64, 3.65, 3.64]
    fp16_times = [2.03, 1.97, 1.98, 1.95, 1.78, 4.29, 1.93, 1.93]
    bf16_times = [1.94, 2.00, 1.97, 1.94, 1.78, 4.27, 4.87, 4.81]

    rects1 = ax1.bar(x - width, fp32_times, width, label='FP32', color='#34495E', edgecolor='#222222')
    rects2 = ax1.bar(x, fp16_times, width, label='FP16 (Half)', color='#27AE60', edgecolor='#222222')
    rects3 = ax1.bar(x + width, bf16_times, width, label='BF16 (BFloat)', color='#E67E22', edgecolor='#222222')

    ax1.set_title("A. CUDA Elapsed Time by Optimization Mode (10,000 images)", fontsize=12, fontweight='bold', pad=12)
    ax1.set_ylabel("Elapsed Time (seconds - Lower is Better)", fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_types, fontsize=9, fontweight='bold')
    ax1.set_ylim(0, 6.0)
    ax1.grid(axis='y')
    ax1.legend(loc='upper left', frameon=True)

    for bar in rects1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08, f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=7.5, color='#34495E')
    for bar in rects2:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08, f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=7.5, fontweight='bold', color='#1E8449')
    for bar in rects3:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08, f"{bar.get_height():.2f}", ha='center', va='bottom', fontsize=7.5, color='#B9770E')

    cpu_models = [
        "raw model",
        "scripted",
        "traced",
        "frozen",
        "optimized\nfor inf",
        "compiled",
        "reduce-oh",
        "max-auto",
        "ONNX\n(ORT)",
        "OpenVINO"
    ]
    cpu_times = [42.6, 42.2, 42.1, 40.2, 36.3, 43.1, 42.8, 42.8, 32.0, 42.8]
    cpu_colors = ["#4A90E2"] * 8 + ["#F5A623", "#E74C3C"]
    cpu_colors[4] = "#27AE60"

    bars_cpu = ax2.bar(cpu_models, cpu_times, color=cpu_colors, width=0.55, edgecolor='#333333')
    ax2.set_title("B. CPU Inference Latency Comparison (FP32)", fontsize=12, fontweight='bold', pad=12)
    ax2.set_ylabel("Elapsed Time (seconds - Lower is Better)", fontsize=11)
    ax2.set_ylim(0, 55)
    ax2.grid(axis='y')

    for bar, t in zip(bars_cpu, cpu_times):
        sp = 42.6 / t
        label = f"{t:.1f}s\n({sp:.2f}x)" if sp > 1.05 else f"{t:.1f}s"
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8, label,
                 ha='center', va='bottom', fontsize=8, fontweight='bold')

    fig.suptitle("Exercise 1 & 2: Exhaustive Evaluation of PyTorch JIT, Inductor Backends, and CPU Compilers", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    out_path = OUTPUT_DIR / "fig6_ex1_pytorch_compiler_and_jit_modes.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved: {out_path}")


if __name__ == "__main__":
    print(f"Generating comprehensive visualizations in: {OUTPUT_DIR}")
    plot_notebook_vs_machine()
    plot_runtime_speedup()
    plot_tta_analysis()
    plot_quantization_and_precision()
    plot_autocast_vs_direct()
    plot_executive_dashboard()
    plot_ex1_model_types()
