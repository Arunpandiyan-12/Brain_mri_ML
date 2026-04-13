"""
Scan Case CRUD routes.
"""
import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.exc import IntegrityError  # ✅ FIXED

from models.database import get_db, ScanCase, ScanResult
from models.schemas import CaseCreate, CaseOut, CaseUpdate
from utils.auth import get_current_user

router = APIRouter(prefix="/cases", tags=["cases"])


@router.post("", response_model=CaseOut, status_code=201)
async def create_case(
    payload: CaseCreate,
    db: AsyncSession = Depends(get_db),
    current_user=Depends(get_current_user),
):
    # queue position
    q = await db.execute(select(ScanCase).order_by(ScanCase.queue_position.desc()))
    last = q.scalars().first()
    next_pos = (last.queue_position + 1) if last else 1

    # ✅ RETRY LOGIC (IMPORTANT)
    for _ in range(5):
        case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"

        case = ScanCase(
            **payload.model_dump(exclude={"case_id"}),
            case_id=case_id,
            queue_position=next_pos,
            created_by=current_user.get("id"),
        )

        db.add(case)

        try:
            await db.commit()
            await db.refresh(case)
            return await _load_case(db, case.id)

        except IntegrityError:
            await db.rollback()

    raise HTTPException(500, "Failed to generate unique case_id")


@router.get("", response_model=list[CaseOut])
async def list_cases(
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_user),
):
    q = await db.execute(
        select(ScanCase)
        .options(selectinload(ScanCase.result))
        .order_by(ScanCase.queue_position)
    )
    return q.scalars().all()


@router.get("/{case_id}", response_model=CaseOut)
async def get_case(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_user),
):
    case = await _load_case_by_str_id(db, case_id)
    if not case:
        raise HTTPException(404, "Case not found")
    return case


@router.put("/{case_id}", response_model=CaseOut)
async def update_case(
    case_id: str,
    payload: CaseUpdate,
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_user),
):
    case = await _load_case_by_str_id(db, case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    for k, v in payload.model_dump(exclude_none=True).items():
        setattr(case, k, v)

    await db.commit()
    return await _load_case(db, case.id)


@router.delete("/{case_id}", status_code=204)
async def delete_case(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    _=Depends(get_current_user),
):
    case = await _load_case_by_str_id(db, case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    await db.delete(case)
    await db.commit()


# ── Helpers ─────────────────────────────────────────

async def _load_case(db: AsyncSession, case_pk: int) -> ScanCase:
    q = await db.execute(
        select(ScanCase)
        .where(ScanCase.id == case_pk)
        .options(selectinload(ScanCase.result))
    )
    return q.scalar_one_or_none()


async def _load_case_by_str_id(db: AsyncSession, case_id: str) -> ScanCase:
    q = await db.execute(
        select(ScanCase)
        .where(ScanCase.case_id == case_id)
        .options(selectinload(ScanCase.result))
    )
    return q.scalar_one_or_none()