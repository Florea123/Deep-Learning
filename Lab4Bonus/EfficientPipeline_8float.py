import os
import sys
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.transforms import v2
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, Dataset
import timm
from tqdm import tqdm


class CachedDataset(Dataset):
    def __init__(self, dataset: Dataset):
        self.dataset = tuple([x for x in dataset])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        return self.dataset[i]


class ClassificationModel(nn.Module):
    def __init__(self, backbone_name: str = 'resnet18', num_classes: int = 10):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(backbone_name, pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.weight.size(1), num_classes)

    def forward(self, x):
        return self.backbone(x)


def get_dataset(data_path: str, is_train: bool):
    initial_transforms = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ]
    if is_train:
        initial_transforms.append(v2.Pad(padding=4, fill=0.5))
    else:
        initial_transforms.append(
            v2.Normalize(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261), inplace=True)
        )

    cifar10 = CIFAR10(root=data_path, train=is_train, transform=v2.Compose(initial_transforms), download=True)
    return CachedDataset(cifar10)


def get_cutmix_or_mixup(num_classes: int = 10):
    return v2.RandomChoice([
        v2.CutMix(num_classes=num_classes),
        v2.MixUp(num_classes=num_classes),
        v2.Identity(),
    ])


class EfficientTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: nn.Module,
        batch_size: int = 128,
        val_batch_size: int = 500,
        epochs: int = 100,
        save_path: str = "best_efficient_pipeline_8float.pth",
        data_path: str = "./data",
        dtype: torch.dtype = torch.float8_e4m3fn
    ):
        self.device = torch.accelerator.current_accelerator()
        self.dtype = dtype
        print(f"Using device {self.device} (Direct float8: {self.dtype})")

        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')

        model = model.to(self.device).to(self.dtype)
        if sys.version_info < (3, 14):
            self.model = torch.jit.script(model)
        else:
            self.model = model

        self.optimizer = optimizer
        self.criterion = criterion.to(self.device)
        self.epochs = epochs
        self.save_path = save_path

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)

        train_dataset = get_dataset(data_path, is_train=True)
        val_dataset = get_dataset(data_path, is_train=False)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            pin_memory=True
        )

        self.batch_transforms = v2.Compose([
            v2.RandomCrop(size=32),
            v2.RandomHorizontalFlip(),
            v2.RandomErasing(scale=(0.01, 0.15), value=0.5, inplace=True),
            v2.Normalize(mean=(0.491, 0.482, 0.446), std=(0.247, 0.243, 0.261), inplace=True),
        ])

        self.cutmix_mixup = get_cutmix_or_mixup(10)
        self.best_val_acc = 0.0
        self.batch_size = batch_size

    def step(self, data: torch.Tensor, target: torch.Tensor):
        predicted = self.model(data)
        loss = self.criterion(predicted, target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return predicted, loss

    def train_epoch(self):
        self.model.train()
        total = 0
        correct = 0
        total_loss = 0.0

        for data, target in tqdm(self.train_loader, desc="Training", leave=False):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            data = self.batch_transforms(data)
            data, target = self.cutmix_mixup(data, target)

            data = data.to(dtype=self.dtype)

            predicted, loss = self.step(data, target)

            if target.ndim > 1:
                target = target.argmax(1)

            correct += predicted.argmax(1).eq(target).sum().item()
            total += data.size(0)
            total_loss += float(loss.item())

        return correct / total, total_loss / len(self.train_loader)

    @torch.inference_mode()
    def validate(self):
        self.model.eval()
        total = 0
        correct = 0
        total_loss = 0.0

        for data, target in tqdm(self.val_loader, desc="Validation", leave=False):
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)
            data = data.to(dtype=self.dtype)
            predicted = self.model(data)
            loss = self.criterion(predicted, target)

            correct += predicted.argmax(1).eq(target).sum().item()
            total += data.size(0)
            total_loss += float(loss.item())

        return correct / total, total_loss / len(self.val_loader)

    def run(self, epochs: int | None = None):
        if epochs is None:
            epochs = self.epochs

        with tqdm(range(epochs), desc="Training", ncols=140) as pbar:
            for _ in pbar:
                tr_acc, tr_loss = self.train_epoch()
                va_acc, va_loss = self.validate()
                self.scheduler.step()

                if va_acc > self.best_val_acc:
                    self.best_val_acc = va_acc
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'model_name': self.model.backbone_name,
                        'num_classes': int(self.model.backbone.fc.weight.shape[0]),
                    }, self.save_path)

                pbar.set_postfix(train_acc=tr_acc, train_loss=round(tr_loss, 3), val_acc=va_acc,
                                 val_loss=round(va_loss, 3), best_val_acc=self.best_val_acc)


def main():
    dtype = torch.float8_e4m3fn
    model = ClassificationModel(backbone_name='resnet18', num_classes=10).to(dtype=dtype)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        nesterov=True,
        weight_decay=5e-4,
        fused=True
    )

    criterion = nn.CrossEntropyLoss()

    trainer = EfficientTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        batch_size=128,
        val_batch_size=500,
        epochs=100,
        save_path="best_efficient_pipeline_8float.pth",
        data_path="./data",
        dtype=dtype
    )
    trainer.run()


if __name__ == "__main__":
    main()
