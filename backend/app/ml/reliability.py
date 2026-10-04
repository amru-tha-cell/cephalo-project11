"""
Reliability Classification.

Buckets each landmark's MC-Dropout uncertainty (std, in mm) into
High / Moderate / Low reliability, using clinically-motivated thresholds
(configurable in app/core/config.py).
"""
from app.core.config import settings


def classify_reliability(std_mm: float) -> str:
    if std_mm < settings.RELIABILITY_HIGH_STD_MM:
        return "High"
    elif std_mm < settings.RELIABILITY_MODERATE_STD_MM:
        return "Moderate"
    else:
        return "Low"


def annotate_reliability(mc_results: list) -> list:
    """Takes the list of dicts from mc_dropout_predict and adds a 'reliability' key."""
    for r in mc_results:
        r["reliability"] = classify_reliability(r["uncertainty_std_mm"])
    return mc_results


def reliability_summary(mc_results: list) -> dict:
    """Aggregate counts, useful for a dashboard summary widget."""
    counts = {"High": 0, "Moderate": 0, "Low": 0}
    for r in mc_results:
        counts[r["reliability"]] += 1
    total = max(len(mc_results), 1)
    return {
        "counts": counts,
        "percentages": {k: round(v / total * 100, 1) for k, v in counts.items()},
    }
