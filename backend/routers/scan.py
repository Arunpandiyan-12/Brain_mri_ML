"""
MRI Scan inference routes.

POST /scan/upload/{case_id}        — upload MRI image
POST /scan/analyze/{case_id}       — run ML inference
GET  /scan/result/{case_id}        — get result
GET  /scan/heatmap/{case_id}       — serve heatmap image
GET  /scan/status/{case_id}        — get processing status
GET  /scan/training-history        — get epoch-wise training accuracy history
"""
import json
import logging
import shutil
import traceback
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from models.database import get_db, ScanCase, ScanResult
from models.schemas import ResultOut
from utils.auth import get_current_user

logger = logging.getLogger(__name__)

UPLOAD_DIR           = Path("static/uploads")
HEATMAP_DIR          = Path("static/heatmaps")
TRAINING_HISTORY_PATH = Path("ml/weights/training_history.json")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        try:
            from ml.model import InferencePipeline
            _pipeline = InferencePipeline(weights_path="ml/weights/brain_tumor_v1.pth")
            logger.info("✅ Real inference pipeline loaded successfully")
        except Exception as e:
            logger.error(f"❌ Pipeline init failed — falling back to demo mode\n{traceback.format_exc()}")
            _pipeline = "demo"
    return _pipeline


router = APIRouter(prefix="/scan", tags=["scan"])


@router.post(
    "/upload/{case_id}",
    summary="Upload MRI image for a case",
    description="Accepts JPG, PNG, BMP, TIFF, or DICOM files. Saves the image and marks the case as uploaded.",
)
async def upload_mri(
    case_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_user),
):
    allowed = {".jpg", ".jpeg", ".png", ".dcm", ".bmp", ".tiff"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"File type '{ext}' not supported. Allowed: {sorted(allowed)}")

    q = await db.execute(select(ScanCase).where(ScanCase.case_id == case_id))
    case = q.scalar_one_or_none()
    if not case:
        raise HTTPException(404, "Case not found")

    dest = UPLOAD_DIR / f"{case_id}{ext}"
    with dest.open("wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    case.image_path = str(dest)
    case.status = "uploaded"
    await db.commit()

    return {"message": "Image uploaded successfully", "path": str(dest), "case_id": case_id}


@router.post(
    "/analyze/{case_id}",
    summary="Run ML inference on uploaded MRI",
    description="Triggers background inference using the trained EfficientNet-B3 model. Poll /scan/status/{case_id} to track progress.",
)
async def analyze_mri(
    case_id: str,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_user),
):
    q = await db.execute(
        select(ScanCase)
        .where(ScanCase.case_id == case_id)
        .options(selectinload(ScanCase.result))
    )
    case = q.scalar_one_or_none()
    if not case:
        raise HTTPException(404, "Case not found")
    if not case.image_path or not Path(case.image_path).exists():
        raise HTTPException(400, "No MRI image found for this case. Upload an image first.")

    case.status = "processing"
    await db.commit()

    background_tasks.add_task(_run_inference, case_id, db)

    return {"message": "Analysis started", "case_id": case_id, "status": "processing"}


@router.get(
    "/result/{case_id}",
    response_model=ResultOut,
    summary="Get inference result for a case",
    description="Returns tumor class, confidence, urgency label, class probabilities, and Grad-CAM paths.",
)
async def get_result(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_user),
):
    q = await db.execute(
        select(ScanCase)
        .where(ScanCase.case_id == case_id)
        .options(selectinload(ScanCase.result))
    )
    case = q.scalar_one_or_none()
    if not case:
        raise HTTPException(404, "Case not found")
    if not case.result:
        raise HTTPException(404, f"No result available yet. Current status: {case.status}")
    return case.result


@router.get(
    "/heatmap/{case_id}",
    summary="Serve Grad-CAM heatmap image",
    description="Returns the heatmap JPEG for a case. Use ?type=gradcam for overlay or ?type=heatmap for raw heatmap.",
)
async def get_heatmap(
    case_id: str,
    type: str = "gradcam",
    token: str = None,
    db: AsyncSession = Depends(get_db),
):
    q = await db.execute(
        select(ScanCase)
        .where(ScanCase.case_id == case_id)
        .options(selectinload(ScanCase.result))
    )
    case = q.scalar_one_or_none()
    if not case or not case.result:
        raise HTTPException(404, "No result found for this case")

    path = case.result.gradcam_path if type == "gradcam" else case.result.heatmap_path
    if not path or not Path(path).exists():
        raise HTTPException(404, "Heatmap file not found. Run analysis first.")

    return FileResponse(path, media_type="image/jpeg")


@router.get(
    "/status/{case_id}",
    summary="Get processing status of a case",
    description="Returns current status: pending | uploaded | processing | done | error",
)
async def get_status(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_user),
):
    q = await db.execute(select(ScanCase).where(ScanCase.case_id == case_id))
    case = q.scalar_one_or_none()
    if not case:
        raise HTTPException(404, "Case not found")
    return {"case_id": case_id, "status": case.status}


@router.get(
    "/training-history",
    summary="Get model training accuracy history",
    description="Returns epoch-wise train and validation accuracy logged during model training. Used to render the accuracy comparison chart in the Results page.",
)
async def get_training_history(_=Depends(get_current_user)):
    if not TRAINING_HISTORY_PATH.exists():
        raise HTTPException(
            404,
            "Training history not found. Run train.py first to generate ml/weights/training_history.json"
        )

    try:
        with TRAINING_HISTORY_PATH.open("r") as f:
            history = json.load(f)
    except json.JSONDecodeError:
        raise HTTPException(500, "Training history file is corrupted or malformed.")

    return JSONResponse(content={
        "status":  "success",
        "epochs":  len(history),
        "history": history,
    })


async def _run_inference(case_id: str, db: AsyncSession):
    from models.database import AsyncSessionLocal, ScanCase, ScanResult

    async with AsyncSessionLocal() as session:
        try:
            q = await session.execute(
                select(ScanCase)
                .where(ScanCase.case_id == case_id)
                .options(selectinload(ScanCase.result))
            )
            case = q.scalar_one_or_none()
            if not case:
                return

            pipeline = get_pipeline()

            if pipeline == "demo":
                from ml.model import demo_predict
                result_data = demo_predict(
                    image_path=case.image_path,
                    age=case.age,
                    headache_severity=case.headache_severity,
                    history_seizures=case.history_seizures,
                    er_admission=case.er_admission,
                    case_id=case_id,
                    output_dir=str(HEATMAP_DIR),
                )
            else:
                result_data = pipeline.predict(
                    image_path=case.image_path,
                    age=case.age,
                    headache_severity=case.headache_severity,
                    history_seizures=case.history_seizures,
                    er_admission=case.er_admission,
                    output_dir=str(HEATMAP_DIR),
                    case_id=case_id,
                )

            if case.result:
                result = case.result
            else:
                result = ScanResult(case_id_fk=case.id)
                session.add(result)

            result.tumor_class         = result_data["tumor_class"]
            result.class_probabilities = result_data["class_probabilities"]
            result.confidence          = result_data["confidence"]
            result.calibrated_prob     = result_data["calibrated_prob"]
            result.urgency_score       = result_data["urgency_score"]
            result.urgency_label       = result_data["urgency_label"]
            result.heatmap_path        = result_data["heatmap_path"]
            result.gradcam_path        = result_data["gradcam_path"]
            result.inference_time_ms   = result_data["inference_time_ms"]
            result.model_version       = result_data["model_version"]

            case.status = "done"

            await _auto_reorder_queue(session)
            await session.commit()

            logger.info(f"Inference complete for {case_id}: {result_data['tumor_class']} ({result_data['urgency_label']})")

        except Exception as e:
            logger.error(f"Inference failed for {case_id}: {e}\n{traceback.format_exc()}")
            q = await session.execute(select(ScanCase).where(ScanCase.case_id == case_id))
            case = q.scalar_one_or_none()
            if case:
                case.status = "error"
                await session.commit()


async def _auto_reorder_queue(session: AsyncSession):
    q = await session.execute(
        select(ScanCase)
        .where(ScanCase.status.in_(["done", "uploaded", "pending", "processing"]))
        .options(selectinload(ScanCase.result))
    )
    cases = q.scalars().all()

    urgency_order = {"RED": 0, "YELLOW": 1, "GREEN": 2, None: 3}

    def sort_key(c):
        label = c.result.urgency_label if c.result else None
        score = -(c.result.urgency_score if c.result else 0)
        return (urgency_order.get(label, 3), score, c.created_at)

    cases.sort(key=sort_key)
    for i, c in enumerate(cases, start=1):
        c.queue_position = i