"""Automated Analysis Report generation."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.db_models import Case, CaseEvent
from app.services.report_service import build_case_report

router = APIRouter()


@router.get("/{case_id}")
def get_report(case_id: int, db: Session = Depends(get_db)):
    case = db.query(Case).get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    report = build_case_report(db, case)
    db.add(CaseEvent(case_id=case_id, event_type="report", details_json={"status": case.status}))
    db.commit()
    return report
