"""
Brain Tumor Classification + Urgency Scoring ML Pipeline
=========================================================
Architecture : EfficientNet-B3 backbone (ImageNet pretrained)
               + Multimodal Fusion head (image features + clinical metadata)
               → 4-class tumor classification
               → Urgency score regression

Classes      : 0=glioma  1=meningioma  2=pituitary  3=no_tumor

Explainability: Grad-CAM++ on last conv block
Calibration   : Temperature scaling post-hoc
"""

import time
import logging
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import cv2

if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "bool"):
    np.bool = bool
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "str"):
    np.str = str

logger = logging.getLogger(__name__)

CLASSES = ["glioma", "meningioma", "pituitary", "no_tumor"]
CLASS_COLORS = {
    "glioma":     (255, 80,  80),
    "meningioma": (255, 180, 80),
    "pituitary":  (80,  200, 255),
    "no_tumor":   (80,  255, 120),
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def build_transforms(train: bool = False) -> T.Compose:
    ops = []
    if train:
        ops += [
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2, contrast=0.3),
            T.RandomGrayscale(p=0.1),
        ]
    ops += [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    return T.Compose(ops)


def apply_clahe(img: Image.Image) -> Image.Image:
    arr    = np.array(img.convert("L"))
    clahe  = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    arr_eq = clahe.apply(arr)
    return Image.fromarray(arr_eq).convert("RGB")


class ClinicalEncoder(nn.Module):
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
    def __init__(self, img_dim: int = 1536, clin_dim: int = 64, out_dim: int = 512):
        super().__init__()
        self.img_proj  = nn.Linear(img_dim, out_dim)
        self.clin_proj = nn.Linear(clin_dim, out_dim)
        self.attn_gate = nn.Sequential(nn.Linear(out_dim * 2, out_dim), nn.Sigmoid())
        self.out_proj  = nn.Sequential(
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
    Multimodal Brain Tumor Classification + Urgency Scoring Model.

    Backbone    : EfficientNet-B3 (pretrained, fine-tuned last 2 blocks)
    Fusion      : Gated Cross-Attention (image ↔ clinical)
    Head 1      : 4-class tumor classification  (softmax)
    Head 2      : Urgency score regression      (sigmoid → [0,1])
    Calibration : Learnable temperature T (post-hoc scaling)
    """

    def __init__(self, num_classes: int = 4, pretrained: bool = True):
        super().__init__()

        try:
            import timm
            self.backbone = timm.create_model(
                "efficientnet_b3",
                pretrained=pretrained,
                num_classes=0,
                global_pool="avg",
            )
            img_feat_dim = self.backbone.num_features
        except Exception:
            import torchvision.models as tvm
            _resnet       = tvm.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
            self.backbone = nn.Sequential(*list(_resnet.children())[:-1])
            img_feat_dim  = 2048

        self._freeze_backbone_layers(freeze_ratio=0.7)

        self.clinical_encoder = ClinicalEncoder(input_dim=4, hidden_dim=64)

        self.fusion = AttentionFusion(
            img_dim=img_feat_dim,
            clin_dim=64,
            out_dim=512,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self.urgency_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def _freeze_backbone_layers(self, freeze_ratio: float = 0.7):
        params   = list(self.backbone.parameters())
        freeze_n = int(len(params) * freeze_ratio)
        for p in params[:freeze_n]:
            p.requires_grad = False

    def forward(
        self,
        image:    torch.Tensor,
        clinical: torch.Tensor,
        calibrate: bool = True,
    ) -> Dict[str, torch.Tensor]:
        img_f = self.backbone(image)
        if img_f.dim() > 2:
            img_f = img_f.flatten(1)

        clin_f = self.clinical_encoder(clinical)
        fused  = self.fusion(img_f, clin_f)
        logits = self.classifier(fused)

        if calibrate:
            probs = F.softmax(logits / self.temperature.clamp(min=0.1), dim=-1)
        else:
            probs = F.softmax(logits, dim=-1)

        urgency = self.urgency_head(fused).squeeze(-1)

        return {
            "logits":        logits,
            "probs":         probs,
            "urgency_score": urgency,
            "features":      fused,
        }


class GradCAMPlusPlus:
    """
    Grad-CAM++ on the last convolutional block of EfficientNet-B3.
    Reference: Chattopadhyay et al., 2018
    """

    def __init__(self, model: BrainTumorModel):
        self.model       = model
        self.gradients:  Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hook_handles = []
        self._register_hooks()

    def _register_hooks(self):
        try:
            target_layer = self.model.backbone.blocks[-1]
        except (AttributeError, IndexError):
            children     = list(self.model.backbone.children())
            target_layer = children[-3] if len(children) >= 3 else children[-1]

        def _save_activation(module, input, output):
            self.activations = output.detach()

        def _save_gradient(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        h1 = target_layer.register_forward_hook(_save_activation)
        h2 = target_layer.register_backward_hook(_save_gradient)
        self._hook_handles = [h1, h2]

    def generate(
        self,
        image_tensor:    torch.Tensor,
        clinical_tensor: torch.Tensor,
        target_class:    Optional[int] = None,
    ) -> np.ndarray:
        self.model.eval()
        image_tensor    = image_tensor.unsqueeze(0).requires_grad_(True)
        clinical_tensor = clinical_tensor.unsqueeze(0)

        output = self.model(image_tensor, clinical_tensor)
        logits = output["logits"]

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, target_class].backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224, 3), dtype=np.uint8)

        grads     = self.gradients[0]
        acts      = self.activations[0]
        grads_sq  = grads ** 2
        grads_cub = grads ** 3
        denom     = 2 * grads_sq + acts * grads_cub.sum(dim=(1, 2), keepdim=True)
        denom     = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha     = grads_sq / denom
        weights   = (alpha * F.relu(grads)).sum(dim=(1, 2))

        cam = (weights[:, None, None] * acts).sum(dim=0)
        cam = F.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (224, 224))
        return cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)

    def cleanup(self):
        for h in self._hook_handles:
            h.remove()


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

    if er_admission:
        score = min(1.0, score + 0.25)
    if history_seizures:
        score = min(1.0, score + 0.20)
    if headache_severity >= 8:
        score = min(1.0, score + 0.15)
    if age > 65 and tumor_class != "no_tumor":
        score = min(1.0, score + 0.10)

    class_floors = {"glioma": 0.65, "meningioma": 0.40, "pituitary": 0.35, "no_tumor": 0.0}
    score = max(score, class_floors.get(tumor_class, 0.0))
    score = score * (0.6 + 0.4 * tumor_prob)

    if score >= 0.65:
        label = "RED"
    elif score >= 0.35:
        label = "YELLOW"
    else:
        label = "GREEN"

    return round(score, 4), label


class InferencePipeline:
    """
    Full end-to-end inference pipeline.

    Steps:
      1. Load and preprocess MRI image (CLAHE + ImageNet normalization)
      2. Encode clinical metadata into feature vector
      3. Run multimodal EfficientNet-B3 model
      4. Generate Grad-CAM++ heatmap overlay
      5. Compute urgency score via clinical rule engine
      6. Return structured result dict

    Raises FileNotFoundError if weights are missing so scan.py can
    cleanly fall back to demo mode rather than running with random weights.
    """

    def __init__(self, weights_path: Optional[str] = None, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"InferencePipeline initializing on device: {self.device}")

        if not weights_path or not Path(weights_path).exists():
            raise FileNotFoundError(
                f"Model weights not found at '{weights_path}'. "
                "Run train.py first to generate ml/weights/brain_tumor_v1.pth"
            )

        self.model = BrainTumorModel(num_classes=4, pretrained=False)

        state = torch.load(weights_path, map_location=self.device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(state, strict=False)

        if missing:
            logger.warning(f"Missing keys in state dict ({len(missing)}): {missing[:5]}")
        if unexpected:
            logger.warning(f"Unexpected keys in state dict ({len(unexpected)}): {unexpected[:5]}")
        if not missing and not unexpected:
            logger.info(f"✅ Weights loaded cleanly from {weights_path}")
        else:
            logger.info(f"✅ Weights loaded from {weights_path} (with some key mismatches — see above)")

        self.model.to(self.device)
        self.model.eval()
        self.transform = build_transforms(train=False)
        self.gradcam   = GradCAMPlusPlus(self.model)

    def preprocess_image(self, image_path: str) -> torch.Tensor:
        img = Image.open(image_path).convert("RGB")
        img = apply_clahe(img)
        return self.transform(img)

    def encode_clinical(
        self,
        age:               int,
        headache_severity: int,
        history_seizures:  bool,
        er_admission:      bool,
    ) -> torch.Tensor:
        feats = [
            age / 100.0,
            headache_severity / 10.0,
            float(history_seizures),
            float(er_admission),
        ]
        return torch.tensor(feats, dtype=torch.float32)

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
            age, headache_severity, history_seizures, er_admission
        ).to(self.device)

        self.model.eval()
        out = self.model(
            img_tensor.unsqueeze(0),
            clin_tensor.unsqueeze(0),
            calibrate=True,
        )

        heatmap_path = None
        gradcam_path = None
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            orig_img = cv2.imread(image_path)
            if orig_img is not None:
                orig_img = cv2.resize(orig_img, (224, 224))
                heatmap  = self.gradcam.generate(
                    img_tensor.cpu(), clin_tensor.cpu(),
                    target_class=int(out["probs"][0].argmax().item())
                )
                overlay      = cv2.addWeighted(orig_img, 0.5, heatmap, 0.5, 0)
                gradcam_path = f"{output_dir}/{case_id}_gradcam.jpg"
                heatmap_path = f"{output_dir}/{case_id}_heatmap.jpg"
                cv2.imwrite(gradcam_path, overlay)
                cv2.imwrite(heatmap_path, heatmap)
        except Exception as e:
            logger.warning(f"GradCAM generation failed: {e}")

        with torch.no_grad():
            probs       = out["probs"][0].detach().cpu().numpy()
            urgency_raw = out["urgency_score"][0].detach().cpu().item()

        pred_class_idx = int(np.argmax(probs))
        pred_class     = CLASSES[pred_class_idx]
        confidence     = float(probs[pred_class_idx])

        urgency_score, urgency_label = compute_urgency(
            probs, urgency_raw, age, headache_severity,
            history_seizures, er_admission,
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
            "model_version":       "v1.0-efficientnet-b3",
        }


def demo_predict(
    image_path:        str,
    age:               int,
    headache_severity: int,
    history_seizures:  bool,
    er_admission:      bool,
    case_id:           str = "demo",
    output_dir:        str = "static/heatmaps",
) -> Dict:
    """
    Synthetic prediction for demo/testing when no trained weights exist.
    Uses image MD5 hash for deterministic results per image.
    """
    import hashlib
    import random

    with open(image_path, "rb") as f:
        h = int(hashlib.md5(f.read()).hexdigest(), 16)
    rng = random.Random(h)

    probs_raw = [rng.uniform(0.05, 0.9) for _ in range(4)]
    total     = sum(probs_raw)
    probs     = [p / total for p in probs_raw]

    pred_class_idx = probs.index(max(probs))
    pred_class     = CLASSES[pred_class_idx]
    confidence     = probs[pred_class_idx]

    urgency_score, urgency_label = compute_urgency(
        np.array(probs), rng.uniform(0.2, 0.9),
        age, headache_severity, history_seizures, er_admission,
    )

    heatmap_path = None
    gradcam_path = None
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        orig = cv2.imread(image_path)
        if orig is not None:
            orig  = cv2.resize(orig, (224, 224))
            h_, w_ = orig.shape[:2]
            cx    = rng.randint(w_ // 4, 3 * w_ // 4)
            cy    = rng.randint(h_ // 4, 3 * h_ // 4)
            ys, xs = np.mgrid[0:h_, 0:w_].astype(np.float32)
            sigma  = 40.0
            synthetic    = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))
            synthetic    = (synthetic * 255).astype(np.uint8)
            heatmap      = cv2.applyColorMap(synthetic, cv2.COLORMAP_JET)
            overlay      = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)
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
        "model_version":       "v1.0-demo",
    }