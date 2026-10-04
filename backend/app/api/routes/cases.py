"""Case CRUD + history."""
from typing import List
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.db_models import Case, CaseEvent
from app.schemas.schemas import CaseCreateOut

router = APIRouter()


@router.get("/", response_model=List[CaseCreateOut])
def list_cases(db: Session = Depends(get_db)):
    return db.query(Case).order_by(Case.created_at.desc()).all()


@router.get("/{case_id}", response_model=CaseCreateOut)
def get_case(case_id: int, db: Session = Depends(get_db)):
    case = db.query(Case).get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return case


@router.get("/{case_id}/history")
def get_case_history(case_id: int, db: Session = Depends(get_db)):
    events = db.query(CaseEvent).filter(CaseEvent.case_id == case_id).order_by(CaseEvent.timestamp).all()
    return [
        {"event_type": e.event_type, "details": e.details_json, "timestamp": e.timestamp}
        for e in events
    ]


@router.delete("/{case_id}")
def delete_case(case_id: int, db: Session = Depends(get_db)):
    case = db.query(Case).get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    db.delete(case)
    db.commit()
    return {"deleted": case_id}
