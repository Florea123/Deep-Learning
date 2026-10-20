* **GPU:** NVIDIA GeForce RTX 4050 Laptop GPU (Arhitectură Ada Lovelace, 6 GB VRAM)
* **OS:** Windows 11 (PowerShell)
* **Python:** 3.11.9
* **PyTorch & CUDA:** PyTorch 2.6.0 + cu124, cuDNN activat (`benchmark=True`)
* **Durată antrenament:** 100 de epoci per experiment

## Tabel cu Rezultatele Tuturor Experimentelor

| # | Pipeline / Script | Arhitectură | Batch Size | Precizie (Dtype) | Timp Total (100 epoci) | Viteză / Epocă | Train Acc | Train Loss | Val Acc | Val Loss | **Best Val Acc** |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | [Pipeline.py](Pipeline.py) (Baseline) | ResNet18 | 32 | FP32 | 40:05 | 24.06 s/it | 0.784 | 0.947 | - | - | **87.5%** |
| **2** | [Pipeline_Batched.py](Pipeline_Batched.py) | ResNet18 | 32 | FP32 | 44:12 | 26.53 s/it | 0.783 | 0.955 | - | - | **87.3%** |
| **3** | [Pipeline_Automatic.py](Pipeline_Automatic.py) | ResNet18 | 32 | AMP (FP16/FP32) | 27:38 | 16.59 s/it | 0.776 | 0.982 | 0.861 | 0.569 | **87.7%** |
| **4** | [EfficientPipeline.py](EfficientPipeline.py) | ResNet18 | 128 | FP32 | 22:45 | 13.66 s/it | 0.803 | 0.882 | 0.886 | 0.409 | **88.7%** |
| **5** | [EfficientPipeline_Batched.py](EfficientPipeline_Batched.py) | ResNet18 | 128 | FP32 | 08:16 | 4.97 s/it | 0.802 | 0.890 | - | - | **89.0%** |
| **6** | [EfficientPipeline_Automatic.py](EfficientPipeline_Automatic.py) | ResNet18 | 128 | AMP (FP16/FP32) | **08:03** | **4.84 s/it** | 0.807 | 0.870 | 0.886 | **0.385** | **88.6%** |
| **7** | [EfficientPipeline_16float.py](EfficientPipeline_16float.py) | ResNet18 | 128 | Direct Float16 | 11:16 | 6.76 s/it | 0.807 | 0.858 | 0.887 | 0.410 | **88.7%** |
| **8** | [EfficientPipeline_8float.py](EfficientPipeline_8float.py) | ResNet18 | 128 | Direct Float8 | *Crash* | *Eroare imediată* | - | - | - | - | **N/A (cuDNN incompatibil)** |
| **9** | **[EfficientPipeline_5.py](EfficientPipeline_5.py)** | **ResNet34 (CIFAR Stem)** | **128** | **AMP (FP16/FP32)** | **1:05:27** | **39.27 s/it** | **0.920** | **0.918** | **0.964** | - | **97.0%**  |

---

