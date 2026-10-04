"""
Orchestrates the full model pipeline for a case:
Preprocess -> Model forward (Swin encoder -> Attention U-Net decoder) ->
Coordinate extraction & sub-pixel refinement -> MC-Dropout uncertainty ->
Reliability classification -> persist to DB -> compute clinical measurements.
"""
import torch
import numpy as np
from sqlalchemy.orm import Session

from app.core.config import settings
from app.ml.model import build_model
from app.ml.preprocessing import preprocess_image
from app.ml.mc_dropout import mc_dropout_predict
from app.ml.reliability import annotate_reliability
from app.ml.cephalometrics import compute_all_measurements
from app.models.db_models import Case, PredictedLandmark, Measurement, CaseEvent
from app.utils.image_utils import load_grayscale, draw_landmarks_overlay, save_processed_image

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model_cache = {}


def get_model():
    """Lazily build & cache the model (singleton) so weights are loaded once per process."""
    if "model" not in _model_cache:
        _model_cache["model"] = build_model(device=_device)
    return _model_cache["model"]


def run_inference_for_case(db: Session, case: Case) -> dict:
    model = get_model()

    raw_gray = load_grayscale(case.image_path)
    tensor_img, scale_x, scale_y = preprocess_image(raw_gray)
    input_tensor = torch.from_numpy(tensor_img).float().unsqueeze(0).unsqueeze(0).to(_device)

    mc_results, _ = mc_dropout_predict(
        model, input_tensor, scale_x, scale_y,
        pixel_spacing_mm=case.pixel_spacing_mm,
    )
    mc_results = annotate_reliability(mc_results)

    # Clear previous predictions for this case, then persist fresh ones
    db.query(PredictedLandmark).filter(PredictedLandmark.case_id == case.id).delete()
    landmark_points = {}
    saved = []
    for r in mc_results:
        name = settings.LANDMARK_NAMES[r["landmark_index"]]
        row = PredictedLandmark(
            case_id=case.id, landmark_index=r["landmark_index"], name=name,
            x=r["x"], y=r["y"], confidence=r["confidence"],
            uncertainty_std_px=r["uncertainty_std_px"], uncertainty_std_mm=r["uncertainty_std_mm"],
            reliability=r["reliability"],
        )
        db.add(row)
        saved.append(row)
        landmark_points[r["landmark_index"]] = (r["x"], r["y"])

    # Cephalometric measurements from the freshly predicted landmarks
    db.query(Measurement).filter(Measurement.case_id == case.id, Measurement.source == "predicted").delete()
    measurements = compute_all_measurements(landmark_points)
    for m in measurements:
        db.add(Measurement(case_id=case.id, source="predicted", **m))

    # Save a landmark overlay preview image for the dashboard
    overlay = draw_landmarks_overlay(raw_gray, mc_results)
    overlay_path = save_processed_image(overlay, "landmarks_overlay")

    case.status = "inferred"
    db.add(CaseEvent(case_id=case.id, event_type="inference",
                      details_json={"num_landmarks": len(mc_results), "device": _device}))
    db.commit()

    return {
        "landmarks": mc_results,
        "measurements": measurements,
        "heatmap_overlay_path": overlay_path,
    }
