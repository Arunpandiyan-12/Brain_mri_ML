"""
Brain Tumor Model Training Script
===================================
Dataset  : Kaggle Brain Tumor 4-Class + BR35H binary (with domain adaptation)
Strategy : Curriculum learning + progressive unfreezing + mixup augmentation

Run:
  python train.py --data_dir /path/to/data --epochs 50 --batch_size 32

Expected data directory structure:
  data/
    train/
      glioma/       *.jpg
      meningioma/   *.jpg
      pituitary/    *.jpg
      no_tumor/     *.jpg
    val/
      <same structure>
    test/
      <same structure>
"""
import os
import argparse
import logging
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import cv2

from ml.model import BrainTumorModel, apply_clahe, CLASSES, build_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Dataset with CLAHE preprocessing ─────────────────────────────────────────

class BrainMRIDataset(Dataset):
    def __init__(self, root: str, transform=None, apply_clahe_flag=True):
        self.dataset = ImageFolder(root)
        self.transform = transform
        self.apply_clahe_flag = apply_clahe_flag
        self.classes = self.dataset.classes

    def __len__(self): return len(self.dataset)

    def __getitem__(self, idx):
        path, label = self.dataset.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.apply_clahe_flag:
            img = apply_clahe(img)
        if self.transform:
            img = self.transform(img)
        # Synthetic clinical features (zeros for training; real data in deployment)
        clinical = torch.zeros(4, dtype=torch.float32)
        return img, clinical, torch.tensor(label, dtype=torch.long)


# ── Loss: Focal + Label Smoothing for class imbalance ────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss: addresses severe class imbalance.
    FL(p_t) = -alpha_t * (1-p_t)^gamma * log(p_t)
    """
    def __init__(self, gamma=2.0, alpha=None, label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        num_classes = logits.size(1)
        # Label smoothing
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_probs = nn.functional.log_softmax(logits, dim=1)
        probs     = log_probs.exp()
        focal_w   = (1 - probs) ** self.gamma
        loss      = -(focal_w * smooth_targets * log_probs).sum(dim=1)
        return loss.mean()


class JointLoss(nn.Module):
    """
    Joint loss for classification + urgency regression.
    L = alpha * L_focal + beta * L_urgency_mse
    """
    def __init__(self, alpha=0.8, beta=0.2, gamma=2.0):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, label_smoothing=0.1)
        self.mse   = nn.MSELoss()
        self.alpha = alpha
        self.beta  = beta

    def forward(self, logits, cls_targets, urgency_pred, urgency_targets):
        l_cls = self.focal(logits, cls_targets)
        l_urg = self.mse(urgency_pred, urgency_targets)
        return self.alpha * l_cls + self.beta * l_urg, l_cls, l_urg


# ── Weighted sampler for class imbalance ─────────────────────────────────────

def get_weighted_sampler(dataset):
    labels = [s[1] for s in dataset.dataset.samples]
    counts = Counter(labels)
    total  = len(labels)
    weights = [total / (len(counts) * counts[l]) for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = cls_loss = urg_loss = correct = total = 0

    for batch_idx, (images, clinical, labels) in enumerate(loader):
        images   = images.to(device)
        clinical = clinical.to(device)
        labels   = labels.to(device)

        # Synthetic urgency targets (0=no_tumor=low, 1=glioma=high)
        urgency_targets = torch.where(
            labels == 0, torch.ones_like(labels, dtype=torch.float32) * 0.8,
            torch.where(labels == 3, torch.zeros_like(labels, dtype=torch.float32) + 0.1,
                        torch.ones_like(labels, dtype=torch.float32) * 0.5)
        ).to(device)

        optimizer.zero_grad()
        out = model(images, clinical, calibrate=False)
        loss, l_cls, l_urg = criterion(
            out["logits"], labels, out["urgency_score"], urgency_targets
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        cls_loss   += l_cls.item()
        urg_loss   += l_urg.item()
        preds       = out["logits"].argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

        if batch_idx % 20 == 0:
            logger.info(
                f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"Loss={loss.item():.4f} Cls={l_cls.item():.4f} Urg={l_urg.item():.4f} "
                f"Acc={correct/total:.3f}"
            )

    return total_loss / len(loader), correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    all_preds, all_labels = [], []

    for images, clinical, labels in loader:
        images   = images.to(device)
        clinical = clinical.to(device)
        labels   = labels.to(device)
        urg_tgts = torch.zeros(labels.size(0)).to(device)

        out  = model(images, clinical, calibrate=True)
        loss, l_cls, _ = criterion(out["logits"], labels, out["urgency_score"], urg_tgts)

        total_loss += l_cls.item()
        preds  = out["probs"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Per-class accuracy
    from sklearn.metrics import classification_report
    report = classification_report(all_labels, all_preds, target_names=CLASSES, zero_division=0)
    return total_loss / len(loader), correct / total, report


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # Datasets
    train_tf = build_transforms(train=True)
    val_tf   = build_transforms(train=False)

    train_ds = BrainMRIDataset(f"{args.data_dir}/train", transform=train_tf)
    val_ds   = BrainMRIDataset(f"{args.data_dir}/val",   transform=val_tf)

    sampler = get_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model = BrainTumorModel(num_classes=4, pretrained=False).to(device)

    # Curriculum: Phase 1 — train heads only
    for name, p in model.named_parameters():
        if "backbone" in name:
            p.requires_grad = False

    criterion = JointLoss(alpha=0.8, beta=0.2, gamma=2.0)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    import json

    best_val_acc = 0.0
    history = []
    Path("ml/weights").mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):

        if epoch == 5:
            logger.info("Phase 2: Unfreezing backbone for fine-tuning")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - 5
            )

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        val_loss, val_acc, report = eval_epoch(
            model, val_loader, criterion, device
        )

        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_acc": round(float(train_acc), 4),
            "val_acc": round(float(val_acc), 4),
            "train_loss": round(float(train_loss), 4),
            "val_loss": round(float(val_loss), 4),
        })

        with open("ml/weights/training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        logger.info(
            f"\n{'='*60}\n"
            f"Epoch {epoch}/{args.epochs}\n"
            f"Train Loss={train_loss:.4f}  Acc={train_acc:.4f}\n"
            f"Val   Loss={val_loss:.4f}  Acc={val_acc:.4f}\n"
            f"{'='*60}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "ml/weights/brain_tumor_v1.pth")
            logger.info(f"Saved best model: val_acc={val_acc:.4f}")

        if epoch % 10 == 0:
            logger.info(f"\nClassification Report:\n{report}")

    logger.info(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
    logger.info("Weights saved to ml/weights/brain_tumor_v1.pth")
    logger.info("History saved to ml/weights/training_history.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",    default="data",    help="Dataset root directory")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    main(parser.parse_args())
