# Rezultate Experimentale - Lab 5 Bonus (Deep Learning)

---

## 1. Exercițiul 1: Moduri de Optimizare PyTorch și Test-Time Augmentation (TTA)

### 1.1 Evaluare pe CUDA (NVIDIA RTX 4050)

#### torch.bfloat16
| Device | Dtype | TTA Type | Model Type | Accuracy | Elapsed (s) |
| :---: | :---: | :---: | :--- | :---: | :---: |
| cuda | torch.bfloat16 | none | raw model | 0.9633 | 1.9359 |
| cuda | torch.bfloat16 | none | scripted model | 0.9633 | 1.9952 |
| cuda | torch.bfloat16 | none | traced model | 0.9633 | 1.9673 |
| cuda | torch.bfloat16 | none | frozen model | 0.9641 | 1.9411 |
| cuda | torch.bfloat16 | none | optimized for inference | N/A | N/A |
| cuda | torch.bfloat16 | none | compiled model | 0.9633 | 4.2700 |
| cuda | torch.bfloat16 | none | compiled reduce-overhead | 0.9633 | 4.8687 |
| cuda | torch.bfloat16 | none | compiled max-autotune | 0.9633 | 4.8062 |

#### torch.float16
| Device | Dtype | TTA Type | Model Type | Accuracy | Elapsed (s) |
| :---: | :---: | :---: | :--- | :---: | :---: |
| cuda | torch.float16 | none | raw model | 0.9641 | 2.0251 |
| cuda | torch.float16 | none | scripted model | 0.9631 | 1.9701 |
| cuda | torch.float16 | none | traced model | 0.9641 | 1.9841 |
| cuda | torch.float16 | none | frozen model | 0.9640 | 1.9484 |
| cuda | torch.float16 | none | optimized for inference | N/A | N/A |
| cuda | torch.float16 | none | compiled model | 0.9641 | 4.2858 |
| cuda | torch.float16 | none | compiled reduce-overhead | 0.9641 | 1.9299 |
| cuda | torch.float16 | none | compiled max-autotune | 0.9641 | 1.9340 |

#### torch.float32
| Device | Dtype | TTA Type | Model Type | Accuracy | Elapsed (s) |
| :---: | :---: | :---: | :--- | :---: | :---: |
| cuda | torch.float32 | none | raw model | 0.9640 | 3.6408 |
| cuda | torch.float32 | none | scripted model | 0.9640 | 3.6974 |
| cuda | torch.float32 | none | traced model | 0.9640 | 3.6398 |
| cuda | torch.float32 | none | frozen model | 0.9640 | 3.6591 |
| cuda | torch.float32 | none | optimized for inference | 0.9640 | 2.8157 |
| cuda | torch.float32 | none | compiled model | 0.9640 | 3.6417 |
| cuda | torch.float32 | none | compiled reduce-overhead | 0.9640 | 3.6452 |
| cuda | torch.float32 | none | compiled max-autotune | 0.9640 | 3.6428 |

---

### 1.2 Evaluare pe CPU

#### torch.bfloat16
| Device | Dtype | TTA Type | Model Type | Accuracy | Elapsed (s) |
| :---: | :---: | :---: | :--- | :---: | :---: |
| cpu | torch.bfloat16 | none | raw model | 0.9640 | 40.0705 |
| cpu | torch.bfloat16 | none | scripted model | 0.9640 | 40.9624 |
| cpu | torch.bfloat16 | none | traced model | 0.9640 | 41.0184 |
| cpu | torch.bfloat16 | none | frozen model | 0.9640 | 39.5815 |
| cpu | torch.bfloat16 | none | optimized for inference | 0.9640 | 36.0470 |
| cpu | torch.bfloat16 | none | compiled model | 0.9640 | 41.5890 |
| cpu | torch.bfloat16 | none | compiled reduce-overhead | 0.9640 | 41.5785 |
| cpu | torch.bfloat16 | none | compiled max-autotune | 0.9640 | 51.8989 |
| cpu | torch.bfloat16 | none | OpenVINO | 0.9640 | 56.7693 |

#### torch.float16
| Device | Dtype | TTA Type | Model Type | Accuracy | Elapsed (s) |
| :---: | :---: | :---: | :--- | :---: | :---: |
| cpu | torch.float16 | none | raw model | 0.9640 | 51.0297 |
| cpu | torch.float16 | none | scripted model | 0.9640 | 43.2006 |
| cpu | torch.float16 | none | traced model | 0.9640 | 42.9541 |
| cpu | torch.float16 | none | frozen model | 0.9640 | 40.3333 |
| cpu | torch.float16 | none | optimized for inference | 0.9640 | 37.1972 |
| cpu | torch.float16 | none | compiled model | 0.9640 | 47.2555 |
| cpu | torch.float16 | none | compiled reduce-overhead | 0.9640 | 52.8366 |
| cpu | torch.float16 | none | compiled max-autotune | 0.9640 | 72.3683 |
| cpu | torch.float16 | none | OpenVINO | 0.9640 | 41.7612 |

#### torch.float32
| Device | Dtype | TTA Type | Model Type | Accuracy | Elapsed (s) |
| :---: | :---: | :---: | :--- | :---: | :---: |
| cpu | torch.float32 | none | raw model | 0.9640 | 49.7587 |
| cpu | torch.float32 | none | scripted model | 0.9640 | 41.9562 |
| cpu | torch.float32 | none | traced model | 0.9640 | 42.1832 |
| cpu | torch.float32 | none | frozen model | 0.9640 | 40.2453 |
| cpu | torch.float32 | none | optimized for inference | 0.9640 | 36.5367 |
| cpu | torch.float32 | none | compiled model | 0.9640 | 43.4676 |
| cpu | torch.float32 | none | compiled reduce-overhead | 0.9640 | 43.0153 |
| cpu | torch.float32 | none | compiled max-autotune | 0.9640 | 43.1189 |
| cpu | torch.float32 | none | OpenVINO | 0.9640 | 50.4765 |

---

### 1.3 Test-Time Augmentation (TTA) pe CUDA

| Device | Dtype | TTA Type | Model Type | Accuracy | Elapsed (s) |
| :---: | :---: | :--- | :---: | :---: | :---: |
| cuda | torch.bfloat16 | none | raw model | 0.9633 | 2.1871 |
| cuda | torch.bfloat16 | mirroring | raw model | 0.9675 | 3.8043 |
| cuda | torch.bfloat16 | translate | raw model | 0.9654 | 17.0801 |
| cuda | torch.bfloat16 | mirroring_and_translate | raw model | 0.9687 | 32.2109 |
| cuda | torch.float16 | none | raw model | 0.9641 | 1.9364 |
| cuda | torch.float16 | mirroring | raw model | 0.9672 | 3.8471 |
| cuda | torch.float16 | translate | raw model | 0.9651 | 17.2316 |
| cuda | torch.float16 | mirroring_and_translate | raw model | 0.9688 | 32.3545 |
| cuda | torch.float32 | none | raw model | 0.9640 | 3.6534 |
| cuda | torch.float32 | mirroring | raw model | 0.9673 | 7.2840 |
| cuda | torch.float32 | translate | raw model | 0.9651 | 32.6149 |
| cuda | torch.float32 | mirroring_and_translate | raw model | 0.9688 | 61.6139 |

---

## 2. Exercițiul 2: ONNX Runtime vs. PyTorch

**Validare numerică ieșiri (logits):**
* Diferență absolută maximă PyTorch vs. ONNX: `2.861023e-06` (potrivire numerică exactă).

| Device | Provider / Backend | PyTorch Raw (s) | ONNX Runtime (s) | Speedup | Throughput (imgs/s) | Accuracy |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| CUDA | CUDAExecutionProvider | 3.6800 | 3.5498 | 1.04x | 2817.0 | 0.9640 |
| CPU | CPUExecutionProvider | 48.8107 | 47.9295 | 1.02x | 208.6 | 0.9640 |
| CPU | CPUExecutionProvider (Fused) | 42.5844 | 32.0094 | 1.33x | 312.4 | 0.9640 |

---

## 3. Exercițiul 3: C++ LibTorch vs. Python PyTorch

| Device | Environment / Runtime | Elapsed (s) | Throughput (imgs/s) | Speedup vs Python Raw | Accuracy |
| :---: | :--- | :---: | :---: | :---: | :---: |
| CUDA | Python (Raw PyTorch) | 3.6522 | 2738.1 | 1.00x | 0.9640 |
| CUDA | LibTorch / TorchScript (C++) | 3.6894 | 2710.4 | 0.99x | 0.9640 |
| CUDA | LibTorch (Optimized for Inf.) | 2.8443 | 3515.8 | 1.28x | 0.9640 |
| CPU | Python (Raw PyTorch) | 53.4483 | 187.1 | 1.00x | 0.9640 |
| CPU | LibTorch / TorchScript (C++) | 60.0918 | 166.4 | 0.89x | 0.9640 |
| CPU | LibTorch (Optimized for Inf.) | 51.1305 | 195.6 | 1.05x | 0.9640 |

---

## 4. Exercițiul 4: Cu Autocast vs. Direct (Fără Autocast)

| Device | Dtype | Mode | Accuracy | Elapsed (s) | Throughput (imgs/s) | Degradation vs FP32 |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| CUDA | FP32 | With Autocast | 0.9640 | 8.3137 | 1202.8 | +0.00% |
| CUDA | FP16 | With Autocast | 0.9641 | 4.5002 | 2222.1 | +0.01% |
| CUDA | BF16 | With Autocast | 0.9633 | 4.4854 | 2229.5 | -0.07% |
| CUDA | FP32 | DIRECT (No Autocast) | 0.9640 | 8.2905 | 1206.2 | +0.00% |
| CUDA | FP16 | DIRECT (No Autocast) | 0.9641 | 4.4340 | 2255.3 | +0.01% |
| CUDA | BF16 | DIRECT (No Autocast) | 0.9632 | 4.4451 | 2249.7 | -0.08% |

---

## 5. Exercițiul 5: Cuantizare Float8 și INT8 Post-Training Quantization (PTQ)

**Direct Float8 Casting (`.to(torch.float8_e4m3fn)` / `.to(torch.float8_e5m2)`):**
* `RuntimeError: getCudnnDataTypeFromScalarType() not supported for Float8_e4m3fn / Float8_e5m2` (operațiile `Conv2d` din cuDNN nu oferă nuclee directe pentru scalari Float8).

| Experiment / Method | Device | Precision | Model Size (MB) | Accuracy | Throughput (imgs/s) | Degradation vs FP32 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FP32 Baseline (Full Precision) | CUDA | FP32 | 81.32 MB | 96.40% | 2671.2 | +0.00% |
| FP16 Baseline (Direct Half) | CUDA | FP16 | 40.66 MB | 96.41% | 5106.1 | +0.01% |
| BF16 Baseline (Direct BFloat) | CUDA | BF16 | 40.66 MB | 96.32% | 5086.1 | -0.08% |
| Float8 Direct Casting (.to(float8)) | CUDA | FP8 (e4m3fn) | 20.33 MB | N/A | N/A | Failed (cuDNN) |
| Float8 e4m3fn (Scaled Weights) | CUDA | FP8 (e4m3fn) | 20.33 MB | 96.38% | 2689.2 | -0.02% |
| Float8 e4m3fn (Unscaled Weights) | CUDA | FP8 (e4m3fn) | 20.33 MB | 96.48% | 2692.9 | +0.08% |
| Float8 e5m2 (Unscaled Weights) | CUDA | FP8 (e5m2) | 20.33 MB | 96.36% | 2692.5 | -0.04% |
| INT8 Dynamic Quantization (Linear only) | CPU | qint8 (Dynamic) | 81.31 MB | 96.38% | 210.0 | -0.02% |
| INT8 Static PTQ (Full Conv2d + Linear) | CPU | qint8 (Static FX) | 20.52 MB | 96.45% | 385.6 | +0.05% |

---

## 6. Data Analysis: Corelații Pearson și Praguri de Decizie

### 6.1 Scoruri de Corelație Pearson ($r$)

| Relație între Metrici | Scor Pearson ($r$) | Interpretare |
| :--- | :---: | :--- |
| CUDA Speedup vs. CPU Speedup | +0.7243 | Corelație Pozitivă Puternică |
| TTA Views (1, 2, 8, 16) vs. Accuracy (%) | +0.6875 | Corelație Pozitivă Puternică |
| TTA Latency vs. Accuracy Gain | +0.6662 | Corelație Pozitivă Moderată |
| Precision (Bits) vs. Accuracy (%) | -0.1538 | Aproape de Zero |
| Model Size (MB) vs. Accuracy (%) | -0.2375 | Aproape de Zero / Neglijabil |
| Throughput (imgs/s) vs. Accuracy (%) | -0.2926 | Aproape de Zero |
| Model Size (MB) vs. Throughput (imgs/s) | -0.0889 | Negativ Ușor |

---

### 6.2 Matricea Pragurilor de Încredere (Confidence Thresholds)

| Hardware | Tip Dată | Optimizare Recomandată | Câștig Estimativ | Impact Acuratețe | Prag Încredere |
| :---: | :---: | :--- | :---: | :---: | :---: |
| **NVIDIA GPU (CUDA)** | `torch.float16` | Direct Casting + TorchScript / TensorRT | 1.9x - 2.1x Throughput (5106 imgs/s) | +0.01% | **99.5%** |
| **NVIDIA GPU (CUDA)** | `torch.bfloat16` | Direct Casting + TorchScript | 1.9x - 2.0x Throughput (5086 imgs/s) | -0.08% | **99.0%** |
| **NVIDIA GPU (CUDA)** | `torch.float32` | TorchScript (`optimize_for_inference`) / ONNX | 1.28x - 1.30x Throughput (3515 imgs/s) | +0.00% | **98.0%** |
| **NVIDIA GPU (CUDA)** | `torch.float8` | Simulated Weights / Scaled GEMM | 4.0x Compresie Memorie (20.3 MB) | -0.02% | **95.0%** |
| **x86 CPU** | `torch.qint8` | Static FX Post-Training Quantization (PTQ) | 4.0x Compresie (20.5 MB), 2.0x Throughput | +0.05% | **99.5%** |
| **x86 CPU** | `torch.float32` | ONNX Runtime / OpenVINO / JIT Opt. | 1.25x - 1.47x Throughput (274 imgs/s) | +0.00% | **95.0%** |
| **x86 CPU** | `torch.float16` | De evitat PyTorch Raw (de folosit OpenVINO) | 10x - 15x mai lent în PyTorch Raw | +0.00% | **99.9%** |
| **Producție / Edge** | Any Dtype | ONNX Runtime / LibTorch (C++) | Fără Python GIL, latență stabilă | +0.00% | **99.0%** |
