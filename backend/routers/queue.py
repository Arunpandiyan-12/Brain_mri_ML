"""
Queue management routes.
GET  /queue          — get ordered queue
POST /queue/reorder  — manual reorder
GET  /queue/stats    — dashboard statistics
"""
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from sqlalchemy.orm import selectinload

from models.database import get_db, ScanCase, ScanResult
from models.schemas import QueueItem, QueueReorder
from utils.auth import get_current_user

router = APIRouter(prefix="/queue", tags=["queue"])

# Urgency priority — lower number = treated first
URGENCY_PRIORITY = {"RED": 0, "YELLOW": 1, "GREEN": 2, None: 3}


@router.get("", response_model=list[QueueItem])
async def get_queue(
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_user),
):
    q = await db.execute(
        select(ScanCase)
        .options(selectinload(ScanCase.result))
        .order_by(ScanCase.queue_position)
    )
    cases = q.scalars().all()

    # ── Auto-sort: severity first, then age (elderly first) ──────────
    # Cases with a result get sorted by urgency → age desc.
    # Cases without a result (still processing) go to the end.
    def sort_key(c):
        urgency = c.result.urgency_label if c.result else None
        age     = c.age or 0
        return (URGENCY_PRIORITY[urgency], -(age))

    sorted_cases = sorted(cases, key=sort_key)

    # Write the new positions back so drag-and-drop reorder stays consistent
    for position, c in enumerate(sorted_cases, start=1):
        if c.queue_position != position:
            c.queue_position = position
    await db.commit()

    items = []
    for c in sorted_cases:
        items.append(QueueItem(
            case_id=c.case_id,
            patient_name=c.patient_name,
            age=c.age,
            gender=c.gender,
            headache_severity=c.headache_severity,
            history_seizures=getattr(c, "history_seizures", False),
            urgency_label=c.result.urgency_label if c.result else None,
            urgency_score=c.result.urgency_score if c.result else None,
            tumor_class=c.result.tumor_class if c.result else None,
            status=c.status,
            queue_position=c.queue_position,
            created_at=c.created_at,
        ))
    return items


@router.post("/reorder", status_code=200)
async def reorder_queue(
    payload: QueueReorder,
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_user),
):
    """
    Manually reorder the queue by providing ordered list of case_ids.
    Drag-and-drop positions override the auto-sort until next new scan arrives.
    """
    for position, case_id in enumerate(payload.ordered_case_ids, start=1):
        q = await db.execute(select(ScanCase).where(ScanCase.case_id == case_id))
        case = q.scalar_one_or_none()
        if case:
            case.queue_position = position

    await db.commit()
    return {"message": "Queue reordered", "count": len(payload.ordered_case_ids)}


@router.get("/stats")
async def get_stats(
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_user),
):
    """Dashboard statistics."""
    total_q = await db.execute(select(func.count(ScanCase.id)))
    total   = total_q.scalar()

    status_q = await db.execute(
        select(ScanCase.status, func.count(ScanCase.id))
        .group_by(ScanCase.status)
    )
    status_counts = dict(status_q.all())

    urgency_q = await db.execute(
        select(ScanResult.urgency_label, func.count(ScanResult.id))
        .group_by(ScanResult.urgency_label)
    )
    urgency_counts = dict(urgency_q.all())

    class_q = await db.execute(
        select(ScanResult.tumor_class, func.count(ScanResult.id))
        .group_by(ScanResult.tumor_class)
    )
    class_counts = dict(class_q.all())

    avg_conf_q = await db.execute(select(func.avg(ScanResult.confidence)))
    avg_conf   = avg_conf_q.scalar()

    avg_time_q = await db.execute(select(func.avg(ScanResult.inference_time_ms)))
    avg_time   = avg_time_q.scalar()

    return {
        "total_cases":      total,
        "status_counts":    status_counts,
        "urgency_counts":   urgency_counts,
        "class_counts":     class_counts,
        "avg_confidence":   round(avg_conf or 0, 4),
        "avg_inference_ms": round(avg_time or 0, 2),
    }