"""
Database configuration using SQLAlchemy + SQLite (async).
Tables: users, scan_cases, scan_results
"""
from sqlalchemy import (
    Column, Integer, String, Float, Boolean,
    DateTime, Text, JSON, ForeignKey
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./brain_triage.db")

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id         = Column(Integer, primary_key=True, index=True)
    username   = Column(String(64), unique=True, nullable=False, index=True)
    email      = Column(String(128), unique=True, nullable=False)
    hashed_pw  = Column(String(256), nullable=False)
    role       = Column(String(32), default="radiologist")   # radiologist | admin
    created_at = Column(DateTime, default=datetime.utcnow)

    cases = relationship("ScanCase", back_populates="created_by_user")


class ScanCase(Base):
    __tablename__ = "scan_cases"

    id             = Column(Integer, primary_key=True, index=True)
    case_id        = Column(String(64), unique=True, nullable=False, index=True)
    patient_name   = Column(String(128), nullable=False)
    age            = Column(Integer, nullable=False)
    gender         = Column(String(16))
    headache_severity = Column(Integer, default=0)   # 0-10 VAS scale
    history_seizures  = Column(Boolean, default=False)
    er_admission      = Column(Boolean, default=False)
    image_path        = Column(String(512))
    status         = Column(String(32), default="pending")  # pending|processing|done|error
    queue_position = Column(Integer, default=9999)
    created_at     = Column(DateTime, default=datetime.utcnow)
    updated_at     = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_by     = Column(Integer, ForeignKey("users.id"), nullable=True)

    created_by_user = relationship("User", back_populates="cases")
    result          = relationship("ScanResult", back_populates="case", uselist=False)


class ScanResult(Base):
    __tablename__ = "scan_results"

    id               = Column(Integer, primary_key=True, index=True)
    case_id_fk       = Column(Integer, ForeignKey("scan_cases.id"), unique=True)
    tumor_class      = Column(String(64))         # glioma | meningioma | pituitary | no_tumor
    class_probabilities = Column(JSON)            # {class: prob}
    confidence       = Column(Float)
    urgency_score    = Column(Float)              # 0-1 continuous
    urgency_label    = Column(String(16))         # RED | YELLOW | GREEN
    heatmap_path     = Column(String(512))
    gradcam_path     = Column(String(512))
    calibrated_prob  = Column(Float)              # temperature-scaled
    model_version    = Column(String(32), default="v1.0")
    inference_time_ms = Column(Float)
    created_at       = Column(DateTime, default=datetime.utcnow)

    case = relationship("ScanCase", back_populates="result")


async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
