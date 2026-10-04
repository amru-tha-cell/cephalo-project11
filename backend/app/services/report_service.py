"""
Builds the Automated Analysis Report: combines case info, current landmarks
(predicted + any corrections), measurements, reliability summary, and
MRE/SDR evaluation (when ground truth is available) into one JSON payload
the frontend can render as a clinical report.
"""
from sqlalchemy.orm import Session

from app.core.config import settings
from app.ml.reliability import reliability_summary
from app.models.db_models import (
    Case, PredictedLandmark, CorrectedLandmark, Measurement, EvaluationResult
)


def build_case_report(db: Session, case: Case) -> dict:
    predictions = db.query(PredictedLandmark).filter(PredictedLandmark.case_id == case.id).all()
    corrections = db.query(CorrectedLandmark).filter(CorrectedLandmark.case_id == case.id).all()
    corrected_idx = {c.landmark_index for c in corrections}

    predicted_measurements = [
        {"name": m.name, "value": m.value, "unit": m.unit, "interpretation": m.interpretation}
        for m in db.query(Measurement).filter(Measurement.case_id == case.id, Measurement.source == "predicted")
    ]
    corrected_measurements = [
        {"name": m.name, "value": m.value, "unit": m.unit, "interpretation": m.interpretation}
        for m in db.query(Measurement).filter(Measurement.case_id == case.id, Measurement.source == "corrected")
    ]

    latest_eval = (
        db.query(EvaluationResult)
        .filter(EvaluationResult.case_id == case.id)
        .order_by(EvaluationResult.created_at.desc())
        .first()
    )

    mc_like = [{"reliability": p.reliability} for p in predictions]

    return {
        "case_id": case.id,
        "patient_ref": case.patient_ref,
        "status": case.status,
        "image_size": {"width": case.image_width, "height": case.image_height},
        "pixel_spacing_mm": case.pixel_spacing_mm,
        "num_landmarks_detected": len(predictions),
        "num_landmarks_corrected": len(corrected_idx),
        "reliability_summary": reliability_summary(mc_like) if predictions else None,
        "predicted_measurements": predicted_measurements,
        "corrected_measurements": corrected_measurements or None,
        "evaluation": {
            "mre_mm": latest_eval.mre_mm,
            "sdr": latest_eval.sdr_json,
            "per_landmark_error_mm": latest_eval.per_landmark_error_json,
        } if latest_eval else None,
        "generated_from": "Swin Transformer Encoder + Attention U-Net Decoder "
                           "(19 heatmaps + 2D offset regression, MC-Dropout uncertainty)",
    }
