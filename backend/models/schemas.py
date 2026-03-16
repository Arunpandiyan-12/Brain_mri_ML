"""
Pydantic request/response schemas.
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, List
from datetime import datetime


# ── Auth ──────────────────────────────────────────────────────────────────────
class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: str = "radiologist"

class UserLogin(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Dict


# ── Case ──────────────────────────────────────────────────────────────────────
class CaseCreate(BaseModel):
    case_id:           str
    patient_name:      str
    age:               int = Field(..., ge=0, le=120)
    gender:            Optional[str] = None
    headache_severity: int  = Field(0, ge=0, le=10)
    history_seizures:  bool = False
    er_admission:      bool = False

class CaseUpdate(BaseModel):
    status:         Optional[str] = None
    queue_position: Optional[int] = None

class CaseOut(BaseModel):
    id:               int
    case_id:          str
    patient_name:     str
    age:              int
    gender:           Optional[str]
    headache_severity: int
    history_seizures: bool
    er_admission:     bool
    status:           str
    queue_position:   int
    image_path:       Optional[str]
    created_at:       datetime
    result:           Optional["ResultOut"] = None

    class Config:
        from_attributes = True


# ── Result ────────────────────────────────────────────────────────────────────
class ResultOut(BaseModel):
    tumor_class:         str
    class_probabilities: Dict[str, float]
    confidence:          float
    urgency_score:       float
    urgency_label:       str
    calibrated_prob:     float
    heatmap_path:        Optional[str]
    inference_time_ms:   float
    model_version:       str
    created_at:          datetime

    class Config:
        from_attributes = True

CaseOut.model_rebuild()


# ── Queue ─────────────────────────────────────────────────────────────────────
class QueueItem(BaseModel):
    case_id:        str
    patient_name:   str
    age:            int
    gender:         str | None
    headache_severity: int | None
    urgency_label:  Optional[str]
    urgency_score:  Optional[float]
    tumor_class:    Optional[str]
    status:         str
    queue_position: int
    created_at:     datetime

class QueueReorder(BaseModel):
    ordered_case_ids: List[str]
