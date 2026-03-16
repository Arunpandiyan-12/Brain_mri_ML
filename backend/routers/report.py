"""
PDF Report Generator — Brain Tumor Detection Center
GET /api/v1/report/{case_id}  →  downloadable PDF
"""

import io
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from models.database import get_db, ScanCase, ScanResult
from utils.auth import get_current_user

router = APIRouter(prefix="/report", tags=["report"])

TUMOR_LABELS = {
    "glioma":     "Glioma",
    "meningioma": "Meningioma",
    "pituitary":  "Pituitary Tumor",
    "no_tumor":   "No Tumor Detected",
}

SEVERITY_MAP = {
    "RED":    "SEVERE",
    "YELLOW": "MODERATE",
    "GREEN":  "LOW RISK",
}

SEVERITY_COLOR = {
    "RED":    (0.85, 0.15, 0.15),
    "YELLOW": (0.90, 0.55, 0.00),
    "GREEN":  (0.10, 0.65, 0.30),
}

FINDINGS_TEXT = {
    "glioma": (
        "Glioma is a type of tumor that occurs in the brain and spinal cord, "
        "originating from glial cells. Gliomas can be low-grade (slow-growing) or "
        "high-grade (fast-growing and aggressive). Immediate neurosurgical consultation "
        "is strongly recommended. Further imaging with contrast MRI and MR spectroscopy "
        "is advised for grading."
    ),
    "meningioma": (
        "Meningioma is a tumor that arises from the meninges — the membranes that surround "
        "the brain and spinal cord. Most meningiomas are noncancerous (benign), though rarely "
        "they can be cancerous. Clinical follow-up advised. Neurosurgical evaluation "
        "recommended for symptomatic lesions."
    ),
    "pituitary": (
        "A pituitary region lesion has been identified. Pituitary adenomas are typically "
        "benign tumors of the pituitary gland. Endocrinological workup and hormonal panel "
        "are recommended. Ophthalmology review for visual field assessment is advised. "
        "MRI with dedicated pituitary protocol suggested."
    ),
    "no_tumor": (
        "No tumor detected by AI analysis. The scan does not show evidence of an intracranial "
        "neoplasm at this time. Clinical correlation with patient symptoms is essential. "
        "Routine follow-up as clinically indicated. If symptoms persist, repeat imaging "
        "may be warranted."
    ),
}

IMPRESSION_TEXT = {
    "glioma":     "Model suggests Glioma with {conf:.1f}% probability. Immediate review recommended.",
    "meningioma": "Model suggests Meningioma with {conf:.1f}% probability.",
    "pituitary":  "Model suggests Pituitary Tumor with {conf:.1f}% probability.",
    "no_tumor":   "Model suggests No Tumor Detected with {conf:.1f}% probability.",
}


@router.get(
    "/{case_id}",
    summary="Download PDF report for a case",
    description="Generates and streams a formatted PDF diagnostic report including patient info, prediction results, class probabilities, findings, impression, and scan images.",
)
async def generate_report(
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
        raise HTTPException(404, "No analysis result found. Run analysis first.")

    try:
        pdf_bytes = _build_pdf(case, case.result)
    except ImportError:
        raise HTTPException(500, "reportlab not installed. Run: pip install reportlab")

    filename = f"MRI_Report_{case.case_id}_{case.patient_name.replace(' ', '_')}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def _build_pdf(case: ScanCase, result: ScanResult) -> bytes:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, Image as RLImage,
    )

    buf  = io.BytesIO()
    W, H = A4
    LM = RM = 1.8 * cm
    TM = BM = 1.5 * cm
    CW = W - LM - RM

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        topMargin=TM, bottomMargin=BM,
        leftMargin=LM, rightMargin=RM,
    )

    BLUE      = colors.HexColor("#1565C0")
    DARK      = colors.HexColor("#0D1B2A")
    GRAY      = colors.HexColor("#546E7A")
    WHITE     = colors.white
    DIVIDER   = colors.HexColor("#BBDEFB")
    BG_INFO   = colors.HexColor("#F5F9FF")

    sev       = SEVERITY_MAP.get(result.urgency_label, "LOW RISK")
    sev_rgb   = SEVERITY_COLOR.get(result.urgency_label, (0.1, 0.65, 0.3))
    SEV_COLOR = colors.Color(*sev_rgb)

    def ps(name, **kw):
        return ParagraphStyle(name, **kw)

    s_header_title = ps("ht", fontName="Helvetica-Bold", fontSize=15,
                        textColor=WHITE, alignment=TA_CENTER)
    s_header_sub   = ps("hs", fontName="Helvetica", fontSize=8,
                        textColor=colors.HexColor("#BBDEFB"), alignment=TA_CENTER)
    s_label        = ps("lbl", fontName="Helvetica-Bold", fontSize=8, textColor=DARK)
    s_value        = ps("val", fontName="Helvetica",      fontSize=8, textColor=DARK)
    s_norm8        = ps("n8",  fontName="Helvetica",      fontSize=8, textColor=DARK, leading=12)
    s_norm7        = ps("n7",  fontName="Helvetica",      fontSize=7, textColor=GRAY, alignment=TA_CENTER)
    s_sec          = ps("sec", fontName="Helvetica-Bold", fontSize=9, textColor=DARK,
                        spaceBefore=4, spaceAfter=3)
    s_banner       = ps("ban", fontName="Helvetica-Bold", fontSize=10,
                        textColor=WHITE, alignment=TA_CENTER)
    s_sig_bold     = ps("sigb", fontName="Helvetica-Bold", fontSize=7,
                        textColor=DARK, alignment=TA_CENTER)
    s_sig          = ps("sig",  fontName="Helvetica",      fontSize=7,
                        textColor=GRAY, alignment=TA_CENTER)

    story = []

    # ── Header banner ─────────────────────────────────────────────────────────
    header_tbl = Table(
        [
            [Paragraph("BRAIN TUMOR DETECTION CENTER", s_header_title)],
            [Paragraph("MRI Brain Scan Analysis Report", s_header_sub)],
            [Paragraph("AI-Assisted Diagnostic Report",  s_header_sub)],
        ],
        colWidths=[CW],
        rowHeights=[22, 13, 13],
    )
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("LEFTPADDING",   (0, 0), (-1, -1), 6),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 6))

    # ── Patient info table ────────────────────────────────────────────────────
    now_str = datetime.utcnow().strftime("%d %b %Y %I:%M %p")
    rep_str = result.created_at.strftime("%d %b %Y %I:%M %p") if result.created_at else now_str
    seizure = "Yes" if case.history_seizures else "No"

    info_data = [
        [
            Paragraph("Patient Name:", s_label),
            Paragraph(case.patient_name, s_value),
            Paragraph("Registered on:", s_label),
            Paragraph(now_str, s_value),
        ],
        [
            Paragraph("Age / Sex:", s_label),
            Paragraph(f"{case.age} / {case.gender or 'N/A'}", s_value),
            Paragraph("Reported on:", s_label),
            Paragraph(rep_str, s_value),
        ],
        [
            Paragraph("Case ID:", s_label),
            Paragraph(str(case.case_id), s_value),
            Paragraph("Seizure History:", s_label),
            Paragraph(seizure, s_value),
        ],
    ]
    info_tbl = Table(
        info_data,
        colWidths=[CW * 0.18, CW * 0.30, CW * 0.20, CW * 0.32],
        rowHeights=16,
    )
    info_tbl.setStyle(TableStyle([
        ("BOX",           (0, 0), (-1, -1), 1,   BLUE),
        ("INNERGRID",     (0, 0), (-1, -1), 0.3, DIVIDER),
        ("BACKGROUND",    (0, 0), (-1, -1), BG_INFO),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING",   (0, 0), (-1, -1), 5),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(info_tbl)
    story.append(Spacer(1, 6))

    # ── Section banner ────────────────────────────────────────────────────────
    banner_tbl = Table(
        [[Paragraph("MRI BRAIN - AI ANALYSIS REPORT", s_banner)]],
        colWidths=[CW],
        rowHeights=[18],
    )
    banner_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), BLUE),
        ("TOPPADDING",    (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(banner_tbl)
    story.append(Spacer(1, 8))

    # ── Prediction results ────────────────────────────────────────────────────
    story.append(Paragraph("Prediction Results", s_sec))

    tumor_present = "No" if result.tumor_class == "no_tumor" else "Yes"
    tumor_label   = TUMOR_LABELS.get(result.tumor_class, result.tumor_class)
    conf_pct      = result.confidence * 100
    sev_hex       = "#{:02x}{:02x}{:02x}".format(
        int(sev_rgb[0] * 255), int(sev_rgb[1] * 255), int(sev_rgb[2] * 255)
    )

    pred_data = [
        [Paragraph(f"Tumor Present:  {tumor_present}", s_norm8),  ""],
        [Paragraph(f"Predicted Class:  {tumor_label}", s_norm8),  ""],
        [Paragraph(f"Confidence:  {conf_pct:.1f}%",    s_norm8),  ""],
        [
            Paragraph("Severity:", s_norm8),
            Paragraph(f'<font color="{sev_hex}"><b>{sev}</b></font>', s_norm8),
        ],
    ]
    pred_tbl = Table(pred_data, colWidths=[CW * 0.35, CW * 0.65], rowHeights=14)
    pred_tbl.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(pred_tbl)
    story.append(Spacer(1, 6))

    # ── Class probabilities ───────────────────────────────────────────────────
    story.append(Paragraph("Class Probabilities:", s_sec))
    probs = result.class_probabilities or {}
    prob_rows = []
    for cls in ["glioma", "no_tumor", "pituitary", "meningioma"]:
        p    = probs.get(cls, 0.0) * 100
        lbl  = TUMOR_LABELS.get(cls, cls)
        bold = cls == result.tumor_class
        style = ps(
            f"cp_{cls}",
            fontName="Helvetica-Bold" if bold else "Helvetica",
            fontSize=8,
            textColor=DARK,
            leading=13,
        )
        prob_rows.append([Paragraph(f"{lbl}:", style), Paragraph(f"{p:.1f}%", style)])

    prob_tbl = Table(prob_rows, colWidths=[CW * 0.35, CW * 0.65], rowHeights=13)
    prob_tbl.setStyle(TableStyle([
        ("TOPPADDING",    (0, 0), (-1, -1), 1),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ("LEFTPADDING",   (0, 0), (-1, -1), 20),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
    ]))
    story.append(prob_tbl)
    story.append(Spacer(1, 8))

    # ── Findings ──────────────────────────────────────────────────────────────
    story.append(Paragraph("Findings", s_sec))
    story.append(Paragraph(FINDINGS_TEXT.get(result.tumor_class, ""), s_norm8))
    story.append(Spacer(1, 8))

    # ── Impression ────────────────────────────────────────────────────────────
    story.append(Paragraph("Impression", s_sec))
    imp = IMPRESSION_TEXT.get(result.tumor_class, "").format(conf=conf_pct)
    story.append(Paragraph(imp, s_norm8))
    story.append(Spacer(1, 8))

    # ── Scan images ───────────────────────────────────────────────────────────
    story.append(Paragraph("Scan Images", s_sec))

    img_w = (CW - 16) / 2
    img_h = img_w * 0.85

    def load_img(path, w, h):
        if path and Path(path).exists():
            try:
                ri = RLImage(path, width=w, height=h)
                ri.hAlign = "CENTER"
                return ri
            except Exception:
                pass
        return Paragraph(
            "[Image unavailable]",
            ps("na", fontName="Helvetica", fontSize=7, textColor=GRAY, alignment=TA_CENTER),
        )

    orig_img = load_img(case.image_path, img_w, img_h)
    gcam_img = load_img(result.gradcam_path or result.heatmap_path, img_w, img_h)

    img_tbl = Table(
        [
            [orig_img, gcam_img],
            [Paragraph("Original MRI", s_norm7), Paragraph("Grad-CAM Heatmap", s_norm7)],
        ],
        colWidths=[img_w + 8, img_w + 8],
        rowHeights=[img_h, 14],
    )
    img_tbl.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING",    (0, 0), (-1, -1), 2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("BOX",           (0, 0), (0, 0),   0.5, DIVIDER),
        ("BOX",           (1, 0), (1, 0),   0.5, DIVIDER),
    ]))
    story.append(img_tbl)
    story.append(Spacer(1, 14))

    # ── Divider ───────────────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=DIVIDER, spaceAfter=10))

    # ── Signature footer ──────────────────────────────────────────────────────
    col_w = CW / 3

    def sig_cell(title, subtitle):
        t = Table(
            [
                [Paragraph(title,    s_sig_bold)],
                [Paragraph(subtitle, s_sig)],
            ],
            colWidths=[col_w - 8],
        )
        t.setStyle(TableStyle([
            ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
            ("TOPPADDING",    (0, 0), (-1, -1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1),
        ]))
        return t

    sig_tbl = Table(
        [[
            sig_cell("Radiologic Technologist", "(MSc, PGDM)"),
            sig_cell("Reporting Radiologist",   "(MD, Radiologist)"),
            sig_cell("Consulting Doctor",        "(MD, Radiologist)"),
        ]],
        colWidths=[col_w, col_w, col_w],
        rowHeights=28,
    )
    sig_tbl.setStyle(TableStyle([
        ("ALIGN",         (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",        (0, 0), (-1, -1), "TOP"),
        ("LINEABOVE",     (0, 0), (-1, 0),  0.5, DIVIDER),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(sig_tbl)

    doc.build(story)
    return buf.getvalue()