"""
Evaluation metrics: Mean Radial Error (MRE) and Successful Detection Rate (SDR),
plus landmark-wise error breakdown -- used to score predictions against
ground truth (ISBI test set) and, separately, to compare AI-vs-clinician-corrected
landmarks (Before-vs-After analysis).
"""
import math
from app.core.config import settings


def euclidean_px(p1, p2) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def compute_landmark_errors_mm(pred: dict, gt: dict, pixel_spacing_mm: float) -> dict:
    """pred/gt: {landmark_index: (x, y)} in original pixel space.
    Returns {landmark_index: error_mm}."""
    errors = {}
    for idx, gt_xy in gt.items():
        if idx not in pred:
            continue
        errors[idx] = euclidean_px(pred[idx], gt_xy) * pixel_spacing_mm
    return errors


def compute_mre(errors_mm: dict) -> float:
    """Mean Radial Error across all landmarks, in mm."""
    if not errors_mm:
        return 0.0
    return sum(errors_mm.values()) / len(errors_mm)


def compute_sdr(errors_mm: dict, thresholds_mm=None) -> dict:
    """Successful Detection Rate: fraction of landmarks with error <= threshold,
    for each threshold in mm. Returns {'2.0mm': 0.81, '2.5mm': 0.88, ...}."""
    thresholds_mm = thresholds_mm or settings.SDR_THRESHOLDS_MM
    n = len(errors_mm)
    if n == 0:
        return {f"{t}mm": 0.0 for t in thresholds_mm}
    sdr = {}
    for t in thresholds_mm:
        successes = sum(1 for e in errors_mm.values() if e <= t)
        sdr[f"{t}mm"] = round(successes / n, 4)
    return sdr


def landmark_wise_error(errors_mm: dict, landmark_names=None) -> dict:
    """Maps landmark_index -> {'name':..., 'error_mm':...}, sorted worst-first
    to surface the most 'difficult' landmarks for the dashboard."""
    landmark_names = landmark_names or settings.LANDMARK_NAMES
    named = {
        landmark_names[idx] if idx < len(landmark_names) else str(idx): round(err, 3)
        for idx, err in errors_mm.items()
    }
    return dict(sorted(named.items(), key=lambda kv: kv[1], reverse=True))
