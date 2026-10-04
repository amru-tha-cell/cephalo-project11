"""X-ray upload -> creates a Case row."""
import cv2
import numpy as np
from fastapi import APIRouter, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.database import get_db
from app.models.db_models import Case, CaseEvent
from app.schemas.schemas import CaseCreateOut
from app.utils.image_utils import save_upload_bytes, load_grayscale

router = APIRouter()


@router.post("/", response_model=CaseCreateOut)
async def upload_xray(
    file: UploadFile = File(...),
    patient_ref: str = Form(None),
    pixel_spacing_mm: float = Form(None),
    db: Session = Depends(get_db),
):
    file_bytes = await file.read()
    path = save_upload_bytes(file_bytes, file.filename)
    gray = load_grayscale(path)
    h, w = gray.shape[:2]

    case = Case(
        patient_ref=patient_ref,
        image_path=path,
        image_width=w,
        image_height=h,
        pixel_spacing_mm=pixel_spacing_mm or settings.DEFAULT_PIXEL_SPACING_MM,
        status="uploaded",
    )
    db.add(case)
    db.commit()
    db.refresh(case)

    db.add(CaseEvent(case_id=case.id, event_type="upload",
                      details_json={"filename": file.filename, "width": w, "height": h}))
    db.commit()
    return case
