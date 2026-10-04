"""
Handles clinician drag-corrections of a landmark and the resulting automatic
recalculation of cephalometric measurements (Feature #1 + #2), plus the
Before-vs-After comparison (Feature #6).
"""
from sqlalchemy.orm import Session

from app.core.config import settings
from app.ml.cephalometrics import compute_all_measurements
from app.ml.metrics import euclidean_px
from app.models.db_models import (
    Case, PredictedLandmark, CorrectedLandmark, Measurement, CaseEvent
)


def _latest_landmark_set(db: Session, case_id: int) -> dict:
    """Effective landmark set = predicted, overridden by any corrections. Returns {idx: (x, y)}."""
    predicted = {p.landmark_index: (p.x, p.y) for p in
                 db.query(PredictedLandmark).filter(PredictedLandmark.case_id == case_id).all()}
    corrected = {c.landmark_index: (c.x, c.y) for c in
                 db.query(CorrectedLandmark).filter(CorrectedLandmark.case_id == case_id).all()}
    predicted.update(corrected)
    return predicted


def apply_correction(db: Session, case: Case, landmark_index: int, x: float, y: float,
                      corrected_by: str = "clinician") -> dict:
    predicted = db.query(PredictedLandmark).filter(
        PredictedLandmark.case_id == case.id, PredictedLandmark.landmark_index == landmark_index
    ).first()
    original_x = predicted.x if predicted else None
    original_y = predicted.y if predicted else None
    name = settings.LANDMARK_NAMES[landmark_index]

    existing = db.query(CorrectedLandmark).filter(
        CorrectedLandmark.case_id == case.id, CorrectedLandmark.landmark_index == landmark_index
    ).first()
    if existing:
        existing.x, existing.y = x, y
        existing.corrected_by = corrected_by
    else:
        existing = CorrectedLandmark(
            case_id=case.id, landmark_index=landmark_index, name=name,
            x=x, y=y, original_x=original_x, original_y=original_y, corrected_by=corrected_by,
        )
        db.add(existing)

    case.status = "corrected"
    db.add(CaseEvent(case_id=case.id, event_type="correction",
                      details_json={"landmark_index": landmark_index, "x": x, "y": y}))
    db.commit()

    measurements = recalculate_measurements(db, case)
    return {"landmark": {"landmark_index": landmark_index, "name": name, "x": x, "y": y},
            "measurements": measurements}


def recalculate_measurements(db: Session, case: Case) -> list:
    """Automatic Measurement Recalculation (Feature #2): recompute cephalometric
    measurements using the CURRENT effective landmark set (predicted + corrections)."""
    landmark_points = _latest_landmark_set(db, case.id)
    db.query(Measurement).filter(Measurement.case_id == case.id, Measurement.source == "corrected").delete()
    measurements = compute_all_measurements(landmark_points)
    for m in measurements:
        db.add(Measurement(case_id=case.id, source="corrected", **m))
    db.add(CaseEvent(case_id=case.id, event_type="recalculation",
                      details_json={"num_measurements": len(measurements)}))
    db.commit()
    return measurements


def before_vs_after(db: Session, case: Case, pixel_spacing_mm: float = None) -> dict:
    """Feature #6: compares AI predictions vs clinician-corrected landmarks, and the
    resulting effect on cephalometric measurements."""
    pixel_spacing_mm = pixel_spacing_mm or case.pixel_spacing_mm

    predicted = {p.landmark_index: p for p in
                 db.query(PredictedLandmark).filter(PredictedLandmark.case_id == case.id).all()}
    corrected = {c.landmark_index: c for c in
                 db.query(CorrectedLandmark).filter(CorrectedLandmark.case_id == case.id).all()}

    landmark_shift_mm = {}
    for idx, c in corrected.items():
        if idx in predicted:
            p = predicted[idx]
            landmark_shift_mm[settings.LANDMARK_NAMES[idx]] = round(
                euclidean_px((p.x, p.y), (c.x, c.y)) * pixel_spacing_mm, 3
            )

    predicted_measurements = {m.name: m.value for m in db.query(Measurement).filter(
        Measurement.case_id == case.id, Measurement.source == "predicted").all()}
    corrected_measurements = {m.name: m.value for m in db.query(Measurement).filter(
        Measurement.case_id == case.id, Measurement.source == "corrected").all()}

    deltas = {
        name: round(corrected_measurements[name] - predicted_measurements[name], 3)
        for name in predicted_measurements if name in corrected_measurements
    }

    return {
        "predicted_landmarks": [p.__dict__ for p in predicted.values()],
        "corrected_landmarks": [c.__dict__ for c in corrected.values()],
        "predicted_measurements": predicted_measurements,
        "corrected_measurements": corrected_measurements,
        "measurement_deltas": deltas,
        "landmark_shift_mm": landmark_shift_mm,
    }
