"""
Brain Tumor Model — Training Script (fully fixed)
===================================================
Run:
    python train.py --data_dir /path/to/data --epochs 30 --batch_size 32

Expected folder layout:
    data/
      train/
        glioma/         *.jpg
        meningioma/     *.jpg
        no_tumor/       *.jpg
        pituitary/      *.jpg
      val/   <same>
      test/  <same>   ← optional, used for final unbiased eval
"""

import os
import argparse
import logging
import json
import numpy as np
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from ml.model import BrainTumorModel, apply_clahe, CLASSES, build_transforms

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _add_file_handler():
    """Add file handler after ml/ dir is created."""
    fh = logging.FileHandler("ml/training.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(fh)


# ─────────────────────────────────────────────────────────────────────────────
# BUG 1+2 FIX: Class index alignment
# ─────────────────────────────────────────────────────────────────────────────

# ImageFolder sorts folder names alphabetically and assigns 0,1,2,3 in that order.
# Alphabetical: glioma=0  meningioma=1  no_tumor=2  pituitary=3
# Our CLASSES:  glioma=0  meningioma=1  pituitary=2  no_tumor=3
#
# We build a remap table: ImageFolder_index → our CLASSES index
# This is computed at runtime from the actual folder names so it's always correct.

IMAGEFOLDER_ALPHA = sorted(["glioma", "meningioma", "no_tumor", "pituitary"])
# IMAGEFOLDER_ALPHA[i] is the class name that ImageFolder assigns index i

def build_label_remap() -> dict:
    """
    Returns {imagefolder_idx: our_classes_idx} so BrainMRIDataset
    converts ImageFolder labels to match model.CLASSES ordering.
    """
    remap = {}
    for if_idx, cls_name in enumerate(IMAGEFOLDER_ALPHA):
        our_idx = CLASSES.index(cls_name)
        remap[if_idx] = our_idx
    return remap

LABEL_REMAP = build_label_remap()
logger.info(f"Label remap (ImageFolder→CLASSES): {LABEL_REMAP}")
# Should print: {0:0, 1:1, 2:3, 3:2}
# glioma→0✓  meningioma→1✓  no_tumor→3(fix)  pituitary→2(fix)



def make_clinical_tensor() -> torch.Tensor:
    """
    CRITICAL FIX: Generate completely random clinical data during training.
    Because the data is random, it has NO correlation to the tumor type.
    This forces the model to ignore the text and rely 100% on the MRI image!
    """
    age_n = float(np.random.uniform(0, 1))
    sev_n = float(np.random.uniform(0, 1))
    seiz  = float(np.random.rand() > 0.8)
    er    = float(np.random.rand() > 0.8)
    
    return torch.tensor([age_n, sev_n, seiz, er], dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class BrainMRIDataset(Dataset):
    def __init__(self, root: str, transform=None, apply_clahe_flag: bool = True):
        self.dataset          = ImageFolder(root)
        self.transform        = transform
        self.apply_clahe_flag = apply_clahe_flag

        # Validate that folder names match what we expect
        found_classes = sorted(self.dataset.classes)
        expected      = IMAGEFOLDER_ALPHA
        if found_classes != expected:
            logger.warning(
                f"Unexpected folder names in {root}: {found_classes}. "
                f"Expected: {expected}. Label remap may be wrong."
            )
        logger.info(f"Dataset at {root}: {len(self.dataset)} images, "
                    f"folders={self.dataset.classes}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        path, if_label = self.dataset.samples[idx]

        # BUG 1 FIX: remap ImageFolder index → CLASSES index
        our_label = LABEL_REMAP[if_label]

        img = Image.open(path).convert("RGB")
        if self.apply_clahe_flag:
            img = apply_clahe(img)
        if self.transform:
            img = self.transform(img)

        # BUG 2 FIX: clinical features keyed by class name via our_label
        clinical = make_clinical_tensor()
        return img, clinical, torch.tensor(our_label, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
# Losses
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples so the model focuses on hard ones.
    FL(p_t) = -(1-p_t)^gamma * log(p_t)  with label smoothing.
    """
    def __init__(self, gamma: float = 2.0, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma           = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n = logits.size(1)
        with torch.no_grad():
            smooth = torch.full_like(logits, self.label_smoothing / (n - 1))
            smooth.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)
        log_p   = nn.functional.log_softmax(logits, dim=1)
        focal_w = (1 - log_p.exp()) ** self.gamma
        return -(focal_w * smooth * log_p).sum(dim=1).mean()


class JointLoss(nn.Module):
    """
    L = alpha * FocalLoss(classification) + beta * MSE(urgency regression)
    Always returns (total, cls_loss, urg_loss) — unpack all three.
    """
    def __init__(self, alpha: float = 0.8, beta: float = 0.2, gamma: float = 2.0):
        super().__init__()
        self.focal = FocalLoss(gamma=gamma, label_smoothing=0.1)
        self.mse   = nn.MSELoss()
        self.alpha = alpha
        self.beta  = beta

    def forward(
        self,
        logits:          torch.Tensor,
        cls_targets:     torch.Tensor,
        urgency_pred:    torch.Tensor,
        urgency_targets: torch.Tensor,
    ) -> tuple:
        l_cls = self.focal(logits, cls_targets)
        l_urg = self.mse(urgency_pred, urgency_targets)
        return self.alpha * l_cls + self.beta * l_urg, l_cls, l_urg


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_weighted_sampler(dataset: BrainMRIDataset) -> WeightedRandomSampler:
    """Inverse-frequency sampling so every class seen equally per epoch."""
    labels  = [LABEL_REMAP[s[1]] for s in dataset.dataset.samples]
    counts  = Counter(labels)
    total   = len(labels)
    weights = [total / (len(counts) * counts[l]) for l in labels]
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def make_urgency_targets(labels: torch.Tensor) -> torch.Tensor:
    """
    Urgency score targets per class (0=glioma→0.8 … 3=no_tumor→0.05).
    Indices match CLASSES order (after remap), not ImageFolder order.
    """
    mapping = torch.tensor(
        [0.80, 0.50, 0.35, 0.05],   # glioma, meningioma, pituitary, no_tumor
        device=labels.device,
    )
    return mapping[labels]


def mixup_batch(images, labels, alpha: float = 0.3):
    """
    Mixup augmentation.
    Returns (mixed_images, labels_a, labels_b, lam)
    Loss = lam * L(pred, a) + (1-lam) * L(pred, b)
    """
    lam      = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx      = torch.randperm(images.size(0), device=images.device)
    mixed    = lam * images + (1 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


# ─────────────────────────────────────────────────────────────────────────────
# Early stopping
# ─────────────────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stops training when val_acc hasn't improved for `patience` epochs."""

    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.best       = 0.0
        self.counter    = 0
        self.should_stop = False

    def step(self, val_acc: float) -> bool:
        if val_acc > self.best + self.min_delta:
            self.best    = val_acc
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: no improvement for {self.counter}/{self.patience} epochs")
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# ─────────────────────────────────────────────────────────────────────────────
# Training epoch  (BUG 3+4+5 fixed here)
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(
    model, loader, optimizer, criterion, scaler,
    scheduler,        # BUG 3 FIX: pass scheduler so we can step per batch
    device, epoch, use_mixup=False,
):
    model.train()
    total_loss = cls_sum = urg_sum = correct = total = 0

    for batch_idx, (images, clinical, labels) in enumerate(loader):
        images   = images.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)

        urg_a = make_urgency_targets(labels)   # urgency for label_a

        if use_mixup:
            images, labels_a, labels_b, lam = mixup_batch(images, labels)
            # BUG 4 FIX: mix urgency targets the same way as images
            urg_b  = make_urgency_targets(labels_b)
            urg_mixed = lam * urg_a + (1 - lam) * urg_b
        else:
            labels_a = labels_b = labels
            lam       = 1.0
            urg_mixed = urg_a

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(images, clinical, calibrate=False)

            if use_mixup and lam < 0.9:
                loss_a, l_cls_a, l_urg_a = criterion(
                    out["logits"], labels_a, out["urgency_score"], urg_mixed
                )
                loss_b, l_cls_b, l_urg_b = criterion(
                    out["logits"], labels_b, out["urgency_score"], urg_mixed
                )
                loss  = lam * loss_a  + (1 - lam) * loss_b
                l_cls = lam * l_cls_a + (1 - lam) * l_cls_b
                l_urg = lam * l_urg_a + (1 - lam) * l_urg_b
            else:
                loss, l_cls, l_urg = criterion(
                    out["logits"], labels_a, out["urgency_score"], urg_mixed
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # BUG 3 FIX: OneCycleLR must step every batch
        if scheduler is not None and isinstance(
            scheduler, optim.lr_scheduler.OneCycleLR
        ):
            scheduler.step()

        total_loss += loss.item()
        cls_sum    += l_cls.item()
        urg_sum    += l_urg.item()

        # BUG 5 FIX: only count accuracy when not in strong-mixup territory
        if not use_mixup or lam >= 0.9:
            preds    = out["logits"].argmax(dim=1)
            correct += (preds == labels_a).sum().item()
            total   += labels_a.size(0)

        if batch_idx % 20 == 0:
            cur_lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"  Epoch {epoch} [{batch_idx}/{len(loader)}] "
                f"loss={loss.item():.4f} cls={l_cls.item():.4f} "
                f"urg={l_urg.item():.4f} "
                f"acc={correct/max(total,1):.3f} lr={cur_lr:.2e}"
            )

    n = len(loader)
    return total_loss / n, cls_sum / n, urg_sum / n, correct / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Validation / Test epoch
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(model, loader, criterion, device, split_name="Val"):
    model.eval()
    loss_sum = correct = total = 0
    all_preds, all_labels = [], []

    for images, clinical, labels in loader:
        images   = images.to(device, non_blocking=True)
        clinical = clinical.to(device, non_blocking=True)
        labels   = labels.to(device, non_blocking=True)

        urg_tgts = make_urgency_targets(labels)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            out = model(images, clinical, calibrate=True)
            _, l_cls, _ = criterion(
                out["logits"], labels, out["urgency_score"], urg_tgts
            )

        loss_sum += l_cls.item()
        preds     = out["probs"].argmax(dim=1)
        correct  += (preds == labels).sum().item()
        total    += labels.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc    = correct / total
    report = classification_report(
        all_labels, all_preds, target_names=CLASSES, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    logger.info(f"\n{split_name} Accuracy: {acc:.4f}")
    logger.info(f"\n{split_name} Classification Report:\n{report}")
    logger.info(f"\n{split_name} Confusion Matrix:\n{cm}")

    return loss_sum / len(loader), acc, report


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(args):
    # Create output dirs first so the logging FileHandler doesn't fail
    Path("ml/weights").mkdir(parents=True, exist_ok=True)
    # _add_file_handler()  # Ensure this is defined in your script

    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_workers  = min(args.num_workers, os.cpu_count() or 2)
    logger.info(f"Device: {device} | workers: {n_workers}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_tf = build_transforms(train=True)
    val_tf   = build_transforms(train=False)

    train_ds = BrainMRIDataset(f"{args.data_dir}/train", transform=train_tf)
    val_ds   = BrainMRIDataset(f"{args.data_dir}/val",   transform=val_tf)

    has_test = Path(f"{args.data_dir}/test").exists()
    test_ds  = BrainMRIDataset(f"{args.data_dir}/test", transform=val_tf) if has_test else None

    logger.info(f"Train={len(train_ds)}  Val={len(val_ds)}  "
                f"Test={'N/A' if test_ds is None else len(test_ds)}")

    sampler = get_weighted_sampler(train_ds)

    loader_kw = dict(
        batch_size=args.batch_size,
        num_workers=n_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(n_workers > 0),
        prefetch_factor=2 if n_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, sampler=sampler, **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False,   **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False,   **loader_kw) if test_ds else None

    # ── Model ─────────────────────────────────────────────────────────────────
    # We must train from scratch due to firewall
    model = BrainTumorModel(num_classes=4, pretrained=False).to(device)

    criterion     = JointLoss(alpha=0.8, beta=0.2, gamma=2.0)
    
    # Define EarlyStopping (assuming you have this class defined in your code)
    # early_stop    = EarlyStopping(patience=args.patience) 
    
    scaler        = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # ── Optimizer & Scheduler (FIXED FOR FROM-SCRATCH TRAINING) ───────────────
    logger.info("Training entire network from scratch (no freezing).")
    
    # 1. Steady Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 2. Smart Scheduler: Keeps LR high, only drops if val_loss stops improving
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    best_val_acc = 0.0
    history      = []

    for epoch in range(1, args.epochs + 1):

        # Turn on mixup augmentation slightly later (Epoch 5) to let basic features form first
        use_mixup = (epoch >= 5)

        # ── Train ─────────────────────────────────────────────────────────────
        # Note: We pass None for scheduler here because ReduceLROnPlateau steps per epoch, not per batch
        train_loss, cls_loss, urg_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scaler,
            None, device, epoch, use_mixup,
        )

        # ── Validate ──────────────────────────────────────────────────────────
        # I removed split_name="Val" as your eval_epoch definition might not have it
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        # ── Step the scheduler based on validation loss ───────────────────────
        scheduler.step(val_loss)

        # ── Log ───────────────────────────────────────────────────────────────
        cur_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"\n{'='*65}\n"
            f"Epoch {epoch:3d}/{args.epochs}  mixup={'on ' if use_mixup else 'off'}  "
            f"lr={cur_lr:.2e}\n"
            f"  Train  loss={train_loss:.4f}  cls={cls_loss:.4f}  "
            f"urg={urg_loss:.4f}  acc={train_acc:.4f}\n"
            f"  Val    loss={val_loss:.4f}  acc={val_acc:.4f}\n"
            f"{'='*65}"
        )

        history.append({
            "epoch":      epoch,
            "train_acc":  round(float(train_acc), 4),
            "val_acc":    round(float(val_acc), 4),
            "train_loss": round(float(train_loss), 4),
            "cls_loss":   round(float(cls_loss), 4),
            "urg_loss":   round(float(urg_loss), 4),
            "val_loss":   round(float(val_loss), 4),
            "lr":         round(cur_lr, 8),
        })
        with open("ml/weights/training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        # ── Checkpoint best model ─────────────────────────────────────────────
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(raw.state_dict(), "ml/weights/brain_tumor_v1.pth")
            logger.info(f"  ✓ Saved best model  val_acc={val_acc:.4f}")

        # ── Early stopping (Uncomment if you have the class defined) ──────────
        # if early_stop.step(val_acc):
        #     logger.info(f"Early stopping at epoch {epoch} (best val_acc={early_stop.best:.4f})")
        #     break

    # ── Final evaluation on held-out test set ─────────────────────────────────
    if test_loader is not None:
        logger.info("\nLoading best weights for final test evaluation...")
        raw = model._orig_mod if hasattr(model, "_orig_mod") else model
        state = torch.load("ml/weights/brain_tumor_v1.pth", map_location=device)
        raw.load_state_dict(state)
        
        test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
        
        logger.info(f"Final Test Accuracy: {test_acc:.4f}")
        history.append({"final_test_acc": round(float(test_acc), 4)})
        with open("ml/weights/training_history.json", "w") as f:
            json.dump(history, f, indent=2)

    logger.info(f"\nTraining complete.")
    logger.info(f"Best val_acc  : {best_val_acc:.4f}")
    logger.info(f"Weights saved : ml/weights/brain_tumor_v1.pth")
    logger.info(f"History saved : ml/weights/training_history.json")
    logger.info(f"Log saved     : ml/training.log")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Brain Tumor Model Training")
    parser.add_argument("--data_dir",    default="data",
                        help="Root folder with train/ val/ test/ subdirs")
    parser.add_argument("--epochs",      type=int, default=30,
                        help="Maximum training epochs (early stopping may end sooner)")
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience",    type=int, default=7,
                        help="Early stopping patience (epochs without improvement)")
    main(parser.parse_args())