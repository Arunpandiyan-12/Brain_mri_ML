# 🧠 NeuroTriage AI — Brain MRI Detection & Triage System
### Final Year ML Project · Full-Stack Intelligent Medical AI System

---

## 🏗 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React)                        │
│  Login → Upload MRI → Patient Form → Analyze → Results/Queue   │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP (JWT Bearer)
┌────────────────────────────▼────────────────────────────────────┐
│                       FASTAPI BACKEND                           │
│                                                                 │
│  /auth  →  JWT auth (register/login)                           │
│  /cases →  CRUD for patient cases                              │
│  /scan  →  Upload MRI + trigger inference + serve heatmaps     │
│  /queue →  Priority queue + stats                              │
└────────────────────────────┬────────────────────────────────────┘
              ┌──────────────┴───────────────┐
              ▼                              ▼
┌─────────────────────┐          ┌─────────────────────────┐
│    ML Pipeline      │          │     SQLite Database      │
│                     │          │                         │
│  EfficientNet-B3    │          │  users                  │
│  + ClinicalEncoder  │          │  scan_cases             │
│  + AttentionFusion  │          │  scan_results           │
│  + Grad-CAM++       │          └─────────────────────────┘
│  + UrgencyScorer    │
│  → 4-class probs    │
│  → urgency RED/Y/G  │
│  → heatmap overlay  │
└─────────────────────┘
```

---

## 🚀 Quick Start (Local Development)

### 1. Backend Setup

```bash
cd backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs
Default login: `admin` / `admin123`

### 2. Frontend Setup

```bash
cd frontend
npm install
npm start
```

App: http://localhost:3000

### 3. Docker (Production)

```bash
docker-compose up --build
```

- Frontend: http://localhost:3000
- Backend API: http://localhost:8000/docs

---

## 🧬 ML Model Architecture

### Backbone: EfficientNet-B3
- ImageNet pretrained → fine-tuned (last 30% of layers)
- Global average pooling → 1536-dim feature vector
- Chosen for best accuracy/compute tradeoff on medical imaging

### Clinical Encoder
```
age (norm) + headache_severity + seizure_hx + er_admission
    → Linear(4→32) → LayerNorm → GELU → Linear(32→64)
```

### Gated Attention Fusion
```
img_feat [1536] + clin_feat [64]
    → Cross-attention gate (sigmoid weighting)
    → Fused representation [512]
```

### Dual-Head Output
1. **Classifier**: 4-class softmax (glioma / meningioma / pituitary / no_tumor)
2. **Urgency head**: Sigmoid regression → 0–1 score

### Calibration: Temperature Scaling
```
calibrated_logits = logits / T    (T is learned parameter)
```

---

## 🎓 Training Your Own Model

### 1. Prepare Dataset
```
data/
  train/
    glioma/        (from Kaggle Brain Tumor Dataset)
    meningioma/
    pituitary/
    no_tumor/      (from BR35H dataset — reclassified)
  val/
    <same structure>
```

Download datasets:
- Kaggle Brain Tumor: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
- BR35H: https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection

### 2. Run Training
```bash
cd backend
python train.py --data_dir ./data --epochs 50 --batch_size 32
```

Weights saved to: `ml/weights/brain_tumor_v1.pth`

### 3. Training Curriculum
- **Phase 1** (epochs 1-4): Freeze backbone, train heads only (LR=1e-3)
- **Phase 2** (epochs 5-50): Unfreeze all, fine-tune full model (LR=1e-4)
- **Loss**: Focal Loss (γ=2) + Label Smoothing (0.1) + MSE urgency
- **Sampling**: Weighted random sampler (balances class imbalance)
- **Scheduler**: Cosine annealing

---

## 🔬 Explainability: Grad-CAM++

Grad-CAM++ produces **class-discriminative localization maps** — a heatmap overlay showing which brain regions most influenced the classification.

**How it works:**
1. Forward pass → record activations at last conv block
2. Backward pass on target class score → get gradients
3. Compute pixel-wise importance: `α = grad²/ (2·grad² + A·Σgrad³)`
4. Weighted sum of activation maps
5. ReLU + resize + jet colormap overlay

**Better than standard Grad-CAM:**
- Weighted second-order gradients (better localization)
- More precise region highlighting
- Works well on EfficientNet's depthwise convolutions

---

## 🚦 Urgency Scoring Algorithm

```
base_score = model_urgency_output (0-1)

# Clinical rule engine boosts:
+0.25  if ER admission
+0.20  if history of seizures
+0.15  if headache_severity ≥ 8
+0.10  if age > 65 AND tumor detected

# Class floor thresholds (evidence-based):
glioma     → min 0.65 (high-grade risk)
meningioma → min 0.40
pituitary  → min 0.35
no_tumor   → min 0.00

# Final label:
score ≥ 0.65 → RED    (Immediate review)
score ≥ 0.35 → YELLOW (Priority review)
score < 0.35 → GREEN  (Routine review)
```

---

## 📊 Evaluation Metrics (Clinical Deployment)

| Metric | Why It Matters |
|--------|---------------|
| Sensitivity (Recall) | Must not miss tumors (patient safety) |
| Specificity | Avoid unnecessary procedures |
| AUC-ROC | Overall discriminative power |
| Expected Calibration Error (ECE) | Probability reliability |
| Cohen's Kappa | Inter-rater agreement proxy |
| F1 (macro) | Balanced across imbalanced classes |
| Inference latency (p95) | Clinical workflow SLA |

**Target for deployment:**
- Sensitivity (glioma) ≥ 95%
- Specificity ≥ 90%
- ECE ≤ 0.05 (well calibrated)
- Latency ≤ 500ms p95

---

## 🗄 API Reference

### Auth
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/auth/register` | Create account |
| POST | `/api/v1/auth/login`    | Get JWT token |
| GET  | `/api/v1/auth/me`       | Current user |

### Cases
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST   | `/api/v1/cases`          | Create patient case |
| GET    | `/api/v1/cases`          | List all cases |
| GET    | `/api/v1/cases/{id}`     | Get single case |
| DELETE | `/api/v1/cases/{id}`     | Delete case |

### Scan / Inference
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/scan/upload/{id}`  | Upload MRI image |
| POST | `/api/v1/scan/analyze/{id}` | Trigger AI analysis |
| GET  | `/api/v1/scan/result/{id}`  | Get result |
| GET  | `/api/v1/scan/heatmap/{id}` | Get Grad-CAM image |
| GET  | `/api/v1/scan/status/{id}`  | Poll status |

### Queue
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET  | `/api/v1/queue`         | Get ordered queue |
| POST | `/api/v1/queue/reorder` | Manual reorder |
| GET  | `/api/v1/queue/stats`   | Dashboard stats |

---

## 🎨 UI Color Palette

```
Background:   #030712  (near-black navy)
Surface:      #0d1f38  (card background)
Border:       rgba(0,229,255,0.1)
Cyan accent:  #00e5ff  (primary actions)
Red critical: #ff4444  (urgency RED)
Yellow warn:  #ffaa00  (urgency YELLOW)
Green safe:   #00e676  (urgency GREEN)
Text primary: #e2e8f0
Text muted:   #64748b
Font display: Exo 2
Font body:    DM Sans
Font mono:    JetBrains Mono
```

---

## 🔐 Security Notes
- JWT tokens expire in 8 hours
- bcrypt password hashing
- CORS restricted in production
- All endpoints require authentication except /auth/*
- File upload validates extension and size

---

## 📈 Beyond Grad-CAM: Future Improvements

1. **SHAP DeepExplainer** — game-theoretic feature attribution
2. **ScoreCAM** — perturbation-based, no gradient required
3. **Concept Activation Vectors (TCAV)** — human-interpretable concepts
4. **Uncertainty quantification** — Monte Carlo Dropout (epistemic) + temperature calibration (aleatoric)
5. **Ensemble disagreement** — 5-model ensemble with disagreement score
6. **LIME** — local interpretable model-agnostic explanations

---

*Built for Final Year ML Project · Department of Computer Science*
