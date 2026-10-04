"""Cephalometric measurements: fetch + trigger recalculation (Feature #2)."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.db_models import Case, Measurement
from app.services.correction_service import recalculate_measurements

router = APIRouter()


@router.get("/{case_id}")
def get_measurements(case_id: int, source: str = "predicted", db: Session = Depends(get_db)):
    case = db.query(Case).get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    rows = db.query(Measurement).filter(Measurement.case_id == case_id, Measurement.source == source).all()
    return [
        {"name": m.name, "value": m.value, "unit": m.unit,
         "normal_low": m.normal_low, "normal_high": m.normal_high, "interpretation": m.interpretation}
        for m in rows
    ]


@router.post("/{case_id}/recalculate")
def recalc(case_id: int, db: Session = Depends(get_db)):
    case = db.query(Case).get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return recalculate_measurements(db, case)
