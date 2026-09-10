"""
Fine-tuning script for the document classifier.

Expected data layout (ImageFolder convention):

    dataset/
        train/
            AADHAAR/*.jpg
            PAN/*.jpg
            UNKNOWN/*.jpg
        val/
            AADHAAR/*.jpg
            PAN/*.jpg
            UNKNOWN/*.jpg

Run:
    python -m src.train_classifier --data_dir dataset --epochs 15 --backbone resnet18
"""
from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .classifier import build_model

# Augmentations chosen to mimic real-world capture conditions called out in
# the brief: blur, skew, lighting variation, noise.
TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=15),           # skew
        transforms.RandomPerspective(distortion_scale=0.2, p=0.3),  # perspective skew
        transforms.ColorJitter(brightness=0.3, contrast=0.3),        # lighting
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),    # blur
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.1),                  # occlusion / noise proxy
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

VAL_TRANSFORM = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def train(data_dir: str, epochs: int, backbone: str, out_path: str, lr: float = 1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = datasets.ImageFolder(f"{data_dir}/train", transform=TRAIN_TRANSFORM)
    val_ds = datasets.ImageFolder(f"{data_dir}/val", transform=VAL_TRANSFORM)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=4)

    model = build_model(backbone, num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_acc = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_ds)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = correct / total if total else 0.0

        print(f"Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    torch.save(best_state, out_path)
    print(f"Best val_acc={best_acc:.4f}. Saved weights to {out_path}")
    print(f"Class index mapping: {train_ds.class_to_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "vit_b_16"])
    parser.add_argument("--out_path", type=str, default="models/classifier.pt")
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    train(args.data_dir, args.epochs, args.backbone, args.out_path, args.lr)
