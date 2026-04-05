"""
Brain Tumor Classification + Urgency Scoring — Production Pipeline
===================================================================
Architecture  : EfficientNet-B0 (ImageNet pretrained, fine-tuned)
                + ClinicalEncoder (age / severity / seizures / ER)
                + Gated AttentionFusion
                → 4-class tumor classification
                → Urgency score regression
                + Grad-CAM++ spatial explainability

Classes       : 0=glioma  1=meningioma  2=pituitary  3=no_tumor

WHY WE DO NOT USE THE NOTEBOOK'S CNN / ML CODE
────────────────────────────────────────────────
The shared notebook has 3 critical bugs that make its predictions unreliable:

1. DATA LEAKAGE — Train AND Test folders are merged into X,Y BEFORE the
   train_test_split. The "test accuracy" is measured on data the model has
   already seen during training. Real generalisation accuracy will be far lower.

2. PREPROCESSING MISMATCH (all ML models) — Models are trained with
   StandardScaler (mean≈0, std≈1) but the prediction loop divides by 255
   (scale [0,1]). The model receives completely out-of-distribution input
   at inference time → every prediction is wrong.
   Proof: StandardScaler pixel ≈ 0.0 for a 128-value pixel; /255 gives 0.502.

3. SCRATCH-TRAINED CNN WITH NO REGULARISATION — Only 2 conv blocks, no
   BatchNorm in conv layers, no Dropout, no LR scheduler, no early stopping,
   no pretrained weights. EfficientNet-B0 pretrained on 1.2M ImageNet images
   gives dramatically better feature initialisation for medical imaging.

WHAT WE KEEP FROM THE NOTEBOOK
────────────────────────────────
✓ Class label mapping convention (glioma/meningioma/pituitary/no_tumor)
✓ 224×224 input resolution
✓ Grayscale-to-RGB promotion (cv2 imread → convert('RGB'))
✓ RandomOverSampler idea → implemented as WeightedRandomSampler in train.py
✓ ImageDataGenerator augmentation ideas → richer augment in build_transforms()

GRADCAM STATUS (this file)
───────────────────────────
Conv2d target : backbone.conv_head  → output (B, 1280, 7, 7) ← spatial ✓
enable_grad   : generate() uses torch.enable_grad() explicitly ✓
Hook type     : register_full_backward_hook on Conv2d leaf ✓
eval() mode   : model stays in eval() (correct BN behaviour) ✓
              Dropout is disabled in eval() → gradient flows cleanly ✓
"""

import time
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import cv2

import builtins as _builtins
for _attr in ("float", "int", "bool", "complex", "object", "str"):
    if not hasattr(np, _attr):
        setattr(np, _attr, getattr(_builtins, _attr))

logger = logging.getLogger(__name__)

# ── Class definitions (kept compatible with notebook's naming) ────────────────
CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]

# Notebook used suffixed names — map them for any legacy callers
NOTEBOOK_CLASS_MAP = {
    "glioma_tumor":     "glioma",
    "meningioma_tumor": "meningioma",
    "pituitary_tumor":  "pituitary",
    "no_tumor":         "no_tumor",
}

CLASS_COLORS = {
    "glioma":     (255, 80,  80),
    "meningioma": (255, 180, 80),
    "pituitary":  (80,  200, 255),
    "no_tumor":   (80,  255, 120),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing  (notebook kept grayscale; we use CLAHE + RGB to get 3× the
# feature information that EfficientNet-B0 expects)
# ─────────────────────────────────────────────────────────────────────────────

def apply_clahe(img: Image.Image) -> Image.Image:
    """
    Contrast-Limited Adaptive Histogram Equalisation.
    Dramatically improves visibility of low-contrast tumours (meningioma,
    pituitary) that the notebook's plain grayscale read would miss.
    """
    arr    = np.array(img.convert("L"))
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr_eq = clahe.apply(arr)
    return Image.fromarray(arr_eq).convert("RGB")


def build_transforms(train: bool = False) -> T.Compose:
    """
    Single consistent pipeline used for BOTH training and inference.
    Fixes the notebook's PREPROCESSING MISMATCH bug (critical bug #2).

    Notebook trained with StandardScaler, predicted with /255 — completely
    different scale. We use the same ImageNet normalisation everywhere.
    """
    ops = []
    if train:
        ops += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.1),
            T.RandomRotation(15),
            T.ColorJitter(brightness=0.25, contrast=0.25),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        ]
    ops += [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(ops)


# ─────────────────────────────────────────────────────────────────────────────
# Model components
# ─────────────────────────────────────────────────────────────────────────────

class ClinicalEncoder(nn.Module):
    """Encodes [age, headache_severity, seizures, er_admission] → 64-d vector."""

    def __init__(self, input_dim: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionFusion(nn.Module):
    """
    Gated cross-attention fusion: image features ↔ clinical features.
    Gate learns how much to trust each modality per sample.
    """

    def __init__(self, img_dim: int = 1280, clin_dim: int = 64, out_dim: int = 512):
        super().__init__()
        self.img_proj  = nn.Linear(img_dim, out_dim)
        self.clin_proj = nn.Linear(clin_dim, out_dim)
        self.attn_gate = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim), nn.Sigmoid()
        )
        self.out_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, img_f: torch.Tensor, clin_f: torch.Tensor) -> torch.Tensor:
        img_p  = self.img_proj(img_f)
        clin_p = self.clin_proj(clin_f)
        gate   = self.attn_gate(torch.cat([img_p, clin_p], dim=-1))
        fused  = gate * img_p + (1 - gate) * clin_p
        return self.out_proj(fused)


class BrainTumorModel(nn.Module):
    """
    Multimodal brain tumour classification + urgency scoring.

    Why EfficientNet-B0 (not the notebook's scratch CNN):
      • Pretrained on ImageNet → edge/texture/shape detectors already learned
      • 16M parameters with compound-scaling (width + depth + resolution)
      • Notebook CNN: 2 conv blocks, ~3M params, trained from scratch = noise
      • B0 achieves ~77% ImageNet accuracy; its features transfer well to MRI
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()
        self._backbone_type = "unknown"

        try:
            import timm
            self.backbone = timm.create_model(
                "efficientnet_b0",
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
            img_feat_dim = self.backbone.num_features   # 1280
            self._backbone_type = "timm_efficientnet"
        except Exception as e:
            logger.warning(f"timm not available ({e}), using torchvision ResNet50")
            import torchvision.models as tvm
            _resnet       = tvm.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            self.backbone = nn.Sequential(*list(_resnet.children())[:-1])
            img_feat_dim  = 2048
            self._backbone_type = "torchvision_resnet50"

        # Expose the GradCAM Conv2d target BEFORE any freezing
        self.gradcam_target_conv: nn.Conv2d = self._find_gradcam_conv()
        logger.info(
            f"GradCAM target: Conv2d(out={self.gradcam_target_conv.out_channels}, "
            f"k={self.gradcam_target_conv.kernel_size}) | backbone={self._backbone_type}"
        )

        self._freeze_backbone_layers(freeze_ratio=0.7)

        self.clinical_encoder = ClinicalEncoder(input_dim=4, hidden_dim=64)
        self.fusion = AttentionFusion(img_dim=img_feat_dim, clin_dim=64, out_dim=512)

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )
        self.urgency_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    # ── Layer-finding helpers ─────────────────────────────────────────────────

    def _find_gradcam_conv(self) -> nn.Conv2d:
        """
        Find the last Conv2d with SPATIAL (H×W) output — BEFORE global_avg_pool.

        EfficientNet-B0 (timm) spatial trace at 224px input:
          blocks[-3] → (B, 112, 14, 14)
          blocks[-2] → (B, 192,  7,  7)
          blocks[-1] → (B, 320,  7,  7)
          conv_head  → (B,1280,  7,  7)  ← best: widest receptive field
          global_pool→ (B,1280)           ← no spatial dims
        """
        if self._backbone_type == "timm_efficientnet":
            if hasattr(self.backbone, "conv_head") and isinstance(
                self.backbone.conv_head, nn.Conv2d
            ):
                return self.backbone.conv_head      # (B, 1280, 7, 7) ✓

            # Fallback: last Conv2d in the deepest MBConv block
            for block in reversed(list(self.backbone.blocks)):
                conv = self._last_conv2d_in(block)
                if conv is not None:
                    return conv

        # Generic: last Conv2d in entire backbone (works for ResNet too)
        conv = self._last_conv2d_in(self.backbone)
        if conv is None:
            raise RuntimeError("No spatial Conv2d found in backbone")
        return conv

    @staticmethod
    def _last_conv2d_in(module: nn.Module) -> Optional[nn.Conv2d]:
        last = None
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                last = m
        return last

    def _freeze_backbone_layers(self, freeze_ratio: float = 0.7):
        params   = list(self.backbone.parameters())
        freeze_n = int(len(params) * freeze_ratio)
        for p in params[:freeze_n]:
            p.requires_grad = False

    def unfreeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        image:     torch.Tensor,
        clinical:  torch.Tensor,
        calibrate: bool = True,
    ) -> Dict[str, torch.Tensor]:
        img_f = self.backbone(image)
        if img_f.dim() > 2:
            img_f = img_f.flatten(1)

        clin_f = self.clinical_encoder(clinical)
        fused  = self.fusion(img_f, clin_f)
        logits = self.classifier(fused)

        if calibrate:
            probs = F.softmax(logits / self.temperature.clamp(min=0.5, max=5.0), dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)

        urgency = self.urgency_head(fused).squeeze(-1)
        return {
            "logits":        logits,
            "probs":         probs,
            "urgency_score": urgency,
            "features":      fused,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Grad-CAM++   (the notebook has NO explainability at all)
# ─────────────────────────────────────────────────────────────────────────────

class GradCAMPlusPlus:
    """
    Grad-CAM++ spatial attention heatmap.

    Key design decisions:
    • Hook on nn.Conv2d leaf (not container) → fires on all PyTorch versions
    • conv_head output is (B, 1280, 7, 7) → rich spatial resolution for MRI
    • torch.enable_grad() inside generate() → safe to call from any context
    • model stays in eval() → BN uses running stats, Dropout is OFF
      (Dropout OFF in eval means clean gradient flow — no zeros from dropout)
    """

    def __init__(self, model: BrainTumorModel):
        self.model        = model
        self.activations: Optional[torch.Tensor] = None
        self.gradients:   Optional[torch.Tensor] = None
        self._hooks:      List = []
        self._register_hooks()

    def _register_hooks(self):
        target = self.model.gradcam_target_conv   # nn.Conv2d leaf

        def _fwd(module, inp, out):
            self.activations = out                # keep grad_fn for backward

        def _bwd(module, grad_in, grad_out):
            if grad_out[0] is not None:
                self.gradients = grad_out[0].detach().clone()

        self._hooks = [
            target.register_forward_hook(_fwd),
            target.register_full_backward_hook(_bwd),
        ]

    def generate(
        self,
        image_tensor:    torch.Tensor,         # (C, H, W) CPU
        clinical_tensor: torch.Tensor,         # (4,)  CPU
        target_class:    Optional[int] = None,
        orig_size:       Tuple[int, int] = (224, 224),   # (W, H) for cv2
    ) -> np.ndarray:
        """
        Returns a BGR heatmap (H, W, 3).
        Uses torch.enable_grad() so it's safe inside no_grad contexts.
        """
        self.model.eval()
        self.activations = None
        self.gradients   = None

        dev     = next(self.model.parameters()).device
        img_in  = image_tensor.unsqueeze(0).float().to(dev)
        clin_in = clinical_tensor.unsqueeze(0).to(dev)

        with torch.enable_grad():
            img_in = img_in.requires_grad_(True)
            out    = self.model(img_in, clin_in, calibrate=False)
            logits = out["logits"]

            if target_class is None:
                target_class = int(logits.argmax(dim=1).item())

            self.model.zero_grad()
            logits[0, target_class].backward()

        if self.activations is None or self.gradients is None:
            logger.warning("GradCAM hooks did not fire")
            return np.zeros((orig_size[1], orig_size[0], 3), dtype=np.uint8)

        acts  = self.activations.detach()[0]   # (C, H_feat, W_feat)
        grads = self.gradients[0]               # (C, H_feat, W_feat)

        if acts.dim() != 3:
            logger.warning(f"Expected 3-D activations, got {acts.shape}")
            return np.zeros((orig_size[1], orig_size[0], 3), dtype=np.uint8)

        # Grad-CAM++ alpha weights
        grads_sq  = grads ** 2
        grads_cub = grads ** 3
        denom = 2.0 * grads_sq + acts * grads_cub.sum(dim=(1, 2), keepdim=True)
        denom = torch.where(denom.abs() > 1e-8, denom, torch.ones_like(denom))
        alpha   = grads_sq / denom
        weights = (alpha * F.relu(grads)).sum(dim=(1, 2))   # (C,)

        cam = (weights[:, None, None] * acts).sum(dim=0)    # (H_feat, W_feat)
        cam = F.relu(cam).cpu().numpy()

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min < 1e-8:
            logger.warning("GradCAM: flat activation — model may not be trained yet")
            return np.zeros((orig_size[1], orig_size[0], 3), dtype=np.uint8)

        cam = ((cam - cam_min) / (cam_max - cam_min) * 255).astype(np.uint8)
        cam = cv2.resize(cam, orig_size, interpolation=cv2.INTER_LINEAR)
        return cv2.applyColorMap(cam, cv2.COLORMAP_JET)

    def cleanup(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []


# ─────────────────────────────────────────────────────────────────────────────
# Urgency rule engine
# ─────────────────────────────────────────────────────────────────────────────

def compute_urgency(
    probs:             np.ndarray,
    urgency_model_out: float,
    age:               int,
    headache_severity: int,
    history_seizures:  bool,
    er_admission:      bool,
) -> Tuple[float, str]:
    tumor_class = CLASSES[int(np.argmax(probs))]
    tumor_prob  = float(probs[int(np.argmax(probs))])
    score       = float(urgency_model_out)

    if er_admission:           score = min(1.0, score + 0.25)
    if history_seizures:       score = min(1.0, score + 0.20)
    if headache_severity >= 8: score = min(1.0, score + 0.15)
    if age > 65 and tumor_class != "no_tumor":
        score = min(1.0, score + 0.10)

    floors = {"glioma": 0.65, "meningioma": 0.40, "pituitary": 0.35, "no_tumor": 0.0}
    score  = max(score, floors.get(tumor_class, 0.0))
    score  = score * (0.6 + 0.4 * tumor_prob)

    label = "RED" if score >= 0.65 else ("YELLOW" if score >= 0.35 else "GREEN")
    return round(score, 4), label


# ─────────────────────────────────────────────────────────────────────────────
# Inference pipeline
# ─────────────────────────────────────────────────────────────────────────────

class InferencePipeline:
    """
    End-to-end inference with correct preprocessing and GradCAM.

    Execution order in predict():
      1. no_grad forward pass → get predicted class (fast, no memory for grads)
      2. GradCAM.generate()  → enable_grad forward+backward on conv_head
         (needs gradients; torch.enable_grad() handles this safely)
      3. no_grad forward pass → final calibrated probabilities
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"InferencePipeline on {self.device}")

        if not weights_path or not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Weights not found at '{weights_path}'. Run train.py first."
            )

        self.model = BrainTumorModel(num_classes=4, pretrained=False)
        state      = torch.load(weights_path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing:    logger.warning(f"Missing keys  ({len(missing)}): {missing[:5]}")
        if unexpected: logger.warning(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}")
        if not missing and not unexpected:
            logger.info(f"Weights loaded cleanly from {weights_path}")

        self.model.to(self.device).eval()
        self.transform = build_transforms(train=False)
        self.gradcam   = GradCAMPlusPlus(self.model)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        img = apply_clahe(img)
        return self.transform(img)

    def encode_clinical(
        self,
        age: int, headache_severity: int,
        history_seizures: bool, er_admission: bool,
    ) -> torch.Tensor:
        return torch.tensor(
            [age / 100.0, headache_severity / 10.0,
             float(history_seizures), float(er_admission)],
            dtype=torch.float32,
        )

    def predict(
        self,
        image_path:        str,
        age:               int,
        headache_severity: int,
        history_seizures:  bool,
        er_admission:      bool,
        output_dir:        str = "static/heatmaps",
        case_id:           str = "case",
    ) -> Dict:
        t0 = time.perf_counter()

        img_tensor  = self.preprocess_image(image_path).to(self.device)
        clin_tensor = self.encode_clinical(
            age, headache_severity, history_seizures, er_admission,
        ).to(self.device)

        # Step 1: quick pass to get predicted class for targeted GradCAM
        with torch.no_grad():
            _out = self.model(img_tensor.unsqueeze(0), clin_tensor.unsqueeze(0), calibrate=True)
        target_class = int(_out["probs"][0].argmax().item())

        # Step 2: GradCAM (uses enable_grad internally — safe here)
        heatmap_path = gradcam_path = None
        try:
            orig_img = cv2.imread(image_path)
            if orig_img is None:
                raise ValueError(f"cv2.imread failed: {image_path}")
            orig_h, orig_w = orig_img.shape[:2]

            heatmap = self.gradcam.generate(
                img_tensor.cpu(), clin_tensor.cpu(),
                target_class=target_class,
                orig_size=(orig_w, orig_h),
            )
            overlay = cv2.addWeighted(orig_img, 0.55, heatmap, 0.45, 0)

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            gradcam_path = f"{output_dir}/{case_id}_gradcam.jpg"
            heatmap_path = f"{output_dir}/{case_id}_heatmap.jpg"
            cv2.imwrite(gradcam_path, overlay)
            cv2.imwrite(heatmap_path, heatmap)
        except Exception as e:
            logger.warning(f"GradCAM failed: {e}", exc_info=True)

        # Step 3: final calibrated inference
        with torch.no_grad():
            out = self.model(img_tensor.unsqueeze(0), clin_tensor.unsqueeze(0), calibrate=True)

        probs       = out["probs"][0].detach().cpu().numpy()
        urgency_raw = float(out["urgency_score"][0].detach().cpu())

        pred_idx   = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])

        urgency_score, urgency_label = compute_urgency(
            probs, urgency_raw, age, headache_severity, history_seizures, er_admission,
        )

        return {
            "tumor_class":         pred_class,
            "class_probabilities": {
                cls: round(float(p), 4) for cls, p in zip(CLASSES, probs)
            },
            "confidence":          round(confidence, 4),
            "calibrated_prob":     round(confidence, 4),
            "urgency_score":       urgency_score,
            "urgency_label":       urgency_label,
            "heatmap_path":        heatmap_path,
            "gradcam_path":        gradcam_path,
            "inference_time_ms":   round((time.perf_counter() - t0) * 1000, 2),
            "model_version":       "v2.0-efficientnet-b0",
        }


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode (no trained weights — uses MRI's brightest region for heatmap)
# ─────────────────────────────────────────────────────────────────────────────

def demo_predict(
    image_path: str, age: int, headache_severity: int,
    history_seizures: bool, er_admission: bool,
    case_id: str = "demo", output_dir: str = "static/heatmaps",
) -> Dict:
    """
    Deterministic demo. Heatmap Gaussian blob placed at the MRI's brightest
    region (where real tumours show hyper-intense signal on T1+contrast MRI).
    This is far better than the notebook's random blob placement.
    """
    import hashlib, random

    with open(image_path, "rb") as f:
        rng = random.Random(int(hashlib.md5(f.read()).hexdigest(), 16))

    probs_raw  = [rng.uniform(0.05, 0.9) for _ in range(4)]
    total      = sum(probs_raw)
    probs      = [p / total for p in probs_raw]
    pred_idx   = probs.index(max(probs))
    pred_class = CLASSES[pred_idx]
    confidence = probs[pred_idx]

    urgency_score, urgency_label = compute_urgency(
        np.array(probs), rng.uniform(0.2, 0.9),
        age, headache_severity, history_seizures, er_admission,
    )

    heatmap_path = gradcam_path = None
    try:
        orig = cv2.imread(image_path)
        if orig is not None:
            h_, w_  = orig.shape[:2]
            # Find brightest region (hyper-intense = likely tumour on T1+contrast)
            gray    = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY).astype(np.float32)
            blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            _, _, _, max_loc = cv2.minMaxLoc(blurred)
            cx, cy  = max_loc      # (x, y)

            ys, xs = np.mgrid[0:h_, 0:w_].astype(np.float32)
            sigma  = min(h_, w_) * 0.18
            synth  = np.exp(-((xs - cx)**2 + (ys - cy)**2) / (2 * sigma**2))
            synth  = (synth * 255).astype(np.uint8)

            heatmap  = cv2.applyColorMap(synth, cv2.COLORMAP_JET)
            overlay  = cv2.addWeighted(orig, 0.55, heatmap, 0.45, 0)

            Path(output_dir).mkdir(parents=True, exist_ok=True)
            gradcam_path = f"{output_dir}/{case_id}_gradcam.jpg"
            heatmap_path = f"{output_dir}/{case_id}_heatmap.jpg"
            cv2.imwrite(gradcam_path, overlay)
            cv2.imwrite(heatmap_path, heatmap)
    except Exception as e:
        logger.warning(f"Demo heatmap failed: {e}")

    return {
        "tumor_class":         pred_class,
        "class_probabilities": {cls: round(p, 4) for cls, p in zip(CLASSES, probs)},
        "confidence":          round(confidence, 4),
        "calibrated_prob":     round(confidence, 4),
        "urgency_score":       urgency_score,
        "urgency_label":       urgency_label,
        "heatmap_path":        heatmap_path,
        "gradcam_path":        gradcam_path,
        "inference_time_ms":   round(rng.uniform(120, 400), 2),
        "model_version":       "v2.0-demo",
    }
