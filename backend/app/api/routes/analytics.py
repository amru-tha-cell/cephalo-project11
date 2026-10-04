"""
Real-time Error/Performance Analytics (Feature #4), Landmark-wise Error
Analysis (Feature #5), and Before-vs-After comparison (Feature #6).
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.db_models import Case, PredictedLandmark, GroundTruthLandmark, EvaluationResult
from app.ml.metrics import compute_landmark_errors_mm, compute_mre, compute_sdr, landmark_wise_error
from app.services.correction_service import before_vs_after

router = APIRouter()


@router.get("/mre-sdr/{case_id}")
def mre_sdr(case_id: int, db: Session = Depends(get_db)):
    """Requires ground-truth landmarks to exist for this case (e.g. ISBI test-set case)."""
    case = db.query(Case).get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")

    gt = {g.landmark_index: (g.x, g.y) for g in
          db.query(GroundTruthLandmark).filter(GroundTruthLandmark.case_id == case_id).all()}
    pred = {p.landmark_index: (p.x, p.y) for p in
            db.query(PredictedLandmark).filter(PredictedLandmark.case_id == case_id).all()}
    if not gt:
        raise HTTPException(status_code=400, detail="No ground-truth landmarks for this case")

    errors_mm = compute_landmark_errors_mm(pred, gt, case.pixel_spacing_mm)
    mre = compute_mre(errors_mm)
    sdr = compute_sdr(errors_mm)
    per_landmark = landmark_wise_error(errors_mm)

    db.add(EvaluationResult(case_id=case_id, source="predicted", mre_mm=mre,
                             sdr_json=sdr, per_landmark_error_json=per_landmark))
    db.commit()

    return {"case_id": case_id, "mre_mm": round(mre, 3), "sdr": sdr, "per_landmark_error_mm": per_landmark}


@router.get("/landmark-error/{case_id}")
def landmark_error(case_id: int, db: Session = Depends(get_db)):
    """Landmark-wise error breakdown, worst-first, to surface 'difficult' landmarks."""
    latest = (
        db.query(EvaluationResult)
        .filter(EvaluationResult.case_id == case_id)
        .order_by(EvaluationResult.created_at.desc())
        .first()
    )
    if not latest:
        raise HTTPException(status_code=404, detail="No evaluation found; call /mre-sdr first")
    return latest.per_landmark_error_json


@router.get("/uncertainty/{case_id}")
def uncertainty(case_id: int, db: Session = Depends(get_db)):
    """Per-landmark confidence + MC-Dropout uncertainty, for charting."""
    rows = db.query(PredictedLandmark).filter(PredictedLandmark.case_id == case_id).all()
    if not rows:
        raise HTTPException(status_code=404, detail="No predictions found; run inference first")
    return [
        {"name": r.name, "confidence": r.confidence, "uncertainty_std_mm": r.uncertainty_std_mm,
         "reliability": r.reliability}
        for r in rows
    ]


@router.get("/before-after/{case_id}")
def before_after(case_id: int, db: Session = Depends(get_db)):
    case = db.query(Case).get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    return before_vs_after(db, case)
