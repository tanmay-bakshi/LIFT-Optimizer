#!/usr/bin/env python
"""Finetune ResNet‑50 on CIFAR‑10 using PyTorch.

This script is self‑contained and uses hard‑coded constants (defined below)
instead of command‑line flags. It will download the CIFAR‑10 dataset and
pretrained ImageNet weights for ResNet‑50 on first run, then fine‑tune the
entire network end‑to‑end. Validation accuracy is reported after each epoch and
the best model checkpoint is saved.

Edit the constants in the *Hyper‑parameters* section to change training
behaviour.

Author: ChatGPT
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from lift_optimizer import LIFTSparseAdamW

# ----------------------------- Hyper‑parameters ----------------------------- #
# Dataset and checkpoints
DATA_DIR: str = "./data"
CHECKPOINT_DIR: str = "./checkpoints"

# Training parameters
NUM_EPOCHS: int = 5
BATCH_SIZE: int = 128
NUM_WORKERS: int = 4  # 0 for Windows if you hit issues
BASE_LR: float = 0.01
MOMENTUM: float = 0.9
WEIGHT_DECAY: float = 5e-4
LR_DECAY_EPOCHS: Tuple[int, ...] = (3,)  # Decay LR by 0.1 at these epochs

# Runtime
DEVICE: torch.device = torch.device("mps")
SEED: int = 42

# --------------------------- Reproducibility utils -------------------------- #
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True  # Enable Winograd, etc.

# ---------------------------- Dataset & Dataloaders ------------------------- #
_mean = (0.4914, 0.4822, 0.4465)
_std = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(_mean, _std),
])


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders.

    :returns: Training dataloader, validation dataloader.
    """
    train_set = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=train_transform)
    val_set = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=val_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return train_loader, val_loader


# -------------------- Model, Loss, Optimizer, and Scheduler ----------------- #

def build_model() -> nn.Module:
    """Load pretrained ResNet‑50 and replace the classification head.

    :returns: Model ready for training.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    num_features: int = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    return model.to(DEVICE)


def build_optimizer(model: nn.Module) -> optim.Optimizer:
    """Create SGD optimizer.

    :param model: The model whose parameters will be optimized.
    :returns: SGD optimizer over all model parameters.
    """
    # return optim.SGD(model.parameters(), lr=BASE_LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    return LIFTSparseAdamW(
        model.named_parameters(),
        num_principal=4,
        filter_rank=8,
        update_interval=200,
    )


def adjust_learning_rate(optimizer: optim.Optimizer, epoch: int) -> None:
    """Decay learning rate by 0.1 at predefined epochs.

    :param optimizer: Optimizer whose LR will be updated.
    :param epoch: Current epoch index (1‑based).
    """
    if epoch in LR_DECAY_EPOCHS:
        for param_group in optimizer.param_groups:
            new_lr = param_group["lr"] * 0.1
            param_group["lr"] = new_lr
        print(f"Learning rate decayed to {new_lr:.6f}")


# ----------------------------- Train & Validate ----------------------------- #

def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, epoch: int) -> None:
    """Train the model for one epoch.

    :param model: Model to train.
    :param loader: DataLoader for training data.
    :param criterion: Loss function.
    :param optimizer: Optimizer updating the model.
    :param epoch: Epoch index (1‑based).
    """
    model.train()
    running_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    for step, (images, targets) in enumerate(loader, start=1):
        images, targets = images.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)

        # Forward
        outputs = model(images)
        loss = criterion(outputs, targets)

        # Backward
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(targets).sum().item()
        total_samples += targets.size(0)

        if step % 100 == 0 or step == len(loader):
            acc = total_correct / total_samples * 100
            avg_loss = running_loss / total_samples
            print(f"Epoch [{epoch}/{NUM_EPOCHS}] Step [{step}/{len(loader)}] Loss: {avg_loss:.4f} Acc: {acc:.2f}%")


@torch.no_grad()

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
    """Evaluate the model on validation data.

    :param model: Trained model.
    :param loader: DataLoader for validation data.
    :param criterion: Loss function.
    :returns: Tuple of (average loss, accuracy percentage).
    """
    model.eval()
    total_loss: float = 0.0
    total_correct: int = 0
    total_samples: int = 0

    for images, targets in loader:
        images, targets = images.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        _, preds = outputs.max(1)
        total_correct += preds.eq(targets).sum().item()
        total_samples += targets.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples * 100
    return avg_loss, accuracy


# ---------------------------------- Main ----------------------------------- #

def main() -> None:
    """Entry point for training loop."""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_loader, val_loader = build_dataloaders()

    model = build_model()
    criterion: nn.Module = nn.CrossEntropyLoss()
    optimizer = build_optimizer(model)

    best_acc: float = 0.0

    for epoch in range(1, NUM_EPOCHS + 1):
        adjust_learning_rate(optimizer, epoch)
        train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(f"Validation — Epoch {epoch}/{NUM_EPOCHS}: Loss={val_loss:.4f} | Acc={val_acc:.2f}%")

        # Save best model
        if val_acc > best_acc + 1e-3:
            best_acc = val_acc
            checkpoint_path = Path(CHECKPOINT_DIR) / "best_resnet50_cifar10.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": best_acc,
            }, checkpoint_path)
            print(f"New best model saved to {checkpoint_path}")

    print(f"Training complete. Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    main()
