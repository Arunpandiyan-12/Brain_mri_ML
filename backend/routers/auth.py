"""
Authentication routes: register, login, me
Supports login with either username or email.
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_

from models.database import get_db, User
from models.schemas import UserCreate, UserLogin, Token
from utils.auth import hash_password, verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=Token)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    q = await db.execute(
        select(User).where(
            or_(User.username == payload.username, User.email == payload.email)
        )
    )
    if q.scalar_one_or_none():
        raise HTTPException(400, "Username or email already exists")

    user = User(
        username=payload.username,
        email=payload.email,
        hashed_pw=hash_password(payload.password),
        role=payload.role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    token = create_access_token({"sub": user.username, "role": user.role, "id": user.id})
    return Token(
        access_token=token,
        user={"id": user.id, "username": user.username, "role": user.role, "email": user.email},
    )


@router.post("/login", response_model=Token)
async def login(payload: UserLogin, db: AsyncSession = Depends(get_db)):
    q = await db.execute(
        select(User).where(
            or_(User.username == payload.username, User.email == payload.username)
        )
    )
    user = q.scalar_one_or_none()

    if not user or not verify_password(payload.password, user.hashed_pw):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token({"sub": user.username, "role": user.role, "id": user.id})
    return Token(
        access_token=token,
        user={"id": user.id, "username": user.username, "role": user.role, "email": user.email},
    )


@router.get("/me")
async def me(current_user=Depends(get_current_user)):
    return current_user