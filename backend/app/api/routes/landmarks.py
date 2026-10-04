"""Get predicted/ground-truth/corrected landmarks; apply clinician drag-corrections (Feature #1)."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.db_models import Case, PredictedLandmark, GroundTruthLandmark, CorrectedLandmark
from app.schemas.schemas import CorrectedLandmarkIn
from app.services.correction_service import apply_correction

router = APIRouter()


@router.get("/{case_id}")
def get_landmarks(case_id: int, db: Session = Depends(get_db)):
    case = db.query(Case).get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    predicted = db.query(PredictedLandmark).filter(PredictedLandmark.case_id == case_id).all()
    corrected = db.query(CorrectedLandmark).filter(CorrectedLandmark.case_id == case_id).all()
    ground_truth = db.query(GroundTruthLandmark).filter(GroundTruthLandmark.case_id == case_id).all()
    return {
        "predicted": [
            {"landmark_index": p.landmark_index, "name": p.name, "x": p.x, "y": p.y,
             "confidence": p.confidence, "uncertainty_std_mm": p.uncertainty_std_mm,
             "reliability": p.reliability}
            for p in predicted
        ],
        "corrected": [
            {"landmark_index": c.landmark_index, "name": c.name, "x": c.x, "y": c.y,
             "corrected_by": c.corrected_by, "corrected_at": c.corrected_at}
            for c in corrected
        ],
        "ground_truth": [
            {"landmark_index": g.landmark_index, "name": g.name, "x": g.x, "y": g.y}
            for g in ground_truth
        ],
    }


@router.put("/{case_id}/{landmark_index}")
def correct_landmark(case_id: int, landmark_index: int, payload: CorrectedLandmarkIn,
                      db: Session = Depends(get_db)):
    case = db.query(Case).get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    if not (0 <= landmark_index < 19):
        raise HTTPException(status_code=400, detail="landmark_index must be 0-18")
    return apply_correction(db, case, landmark_index, payload.x, payload.y, payload.corrected_by)
