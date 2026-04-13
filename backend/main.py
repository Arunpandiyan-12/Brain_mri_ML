"""
Brain MRI Triage System — FastAPI Backend
==========================================
Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from models.database import init_db
from routers import auth, cases, scan, queue, report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize DB and create demo admin user on startup."""
    logger.info("🚀 Starting Brain MRI Triage System...")
    await init_db()

    # Create default admin if not exists
    from models.database import AsyncSessionLocal, User
    from sqlalchemy import select
    from utils.auth import hash_password

    async with AsyncSessionLocal() as db:
        q = await db.execute(select(User).where(User.username == "admin"))
        if not q.scalar_one_or_none():
            admin = User(
                username="admin",
                email="admin@hospital.ai",
                hashed_pw=hash_password("admin123"),
                role="admin",
            )
            db.add(admin)
            await db.commit()
            logger.info("✅ Default admin created: admin / admin123")

    # Ensure static dirs
    Path("static/uploads").mkdir(parents=True, exist_ok=True)
    Path("static/heatmaps").mkdir(parents=True, exist_ok=True)
    Path("ml/weights").mkdir(parents=True, exist_ok=True)

    logger.info("✅ Database initialized")
    yield
    logger.info("👋 Shutting down...")


app = FastAPI(
    title="Brain MRI Triage System",
    description="""
## 🧠 Intelligent Brain MRI Triage API

Multimodal deep learning system for:
- 4-class brain tumor classification (glioma, meningioma, pituitary, no_tumor)
- Grad-CAM++ explainability heatmaps
- Clinical metadata fusion (age, seizures, ER admission)
- Automated urgency scoring (RED/YELLOW/GREEN)
- Priority queue management for radiologist workflow

**Default credentials:** `admin` / `admin123`
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*","https://brain-mri-ml.vercel.app"],   # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files ──────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router,   prefix="/api/v1")
app.include_router(cases.router,  prefix="/api/v1")
app.include_router(scan.router,   prefix="/api/v1")
app.include_router(queue.router,  prefix="/api/v1")
app.include_router(report.router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "system": "Brain MRI Triage System",
        "version": "1.0.0",
        "docs": "/docs",
        "status": "operational",
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}
