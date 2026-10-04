"""
Clinical Cephalometric Calculations.

Landmark indices (from app.core.config.settings.LANDMARK_NAMES), ISBI-2015 convention:
 0 Sella | 1 Nasion | 2 Orbitale | 3 Porion | 4 A-Point | 5 B-Point | 6 Pogonion
 7 Menton | 8 Gnathion | 9 Gonion | 10 Lower Incisal Incision | 11 Upper Incisal Incision
 12 Upper Lip | 13 Lower Lip | 14 Subnasale | 15 Soft Tissue Pogonion
 16 Posterior Nasal Spine | 17 Anterior Nasal Spine | 18 Articulare
"""
import math

IDX = {
    "S": 0, "N": 1, "Or": 2, "Po": 3, "A": 4, "B": 5, "Pog": 6, "Me": 7,
    "Gn": 8, "Go": 9, "L1": 10, "U1": 11, "UL": 12, "LL": 13, "Sn": 14,
    "STPog": 15, "PNS": 16, "ANS": 17, "Ar": 18,
}

# Rough clinical normal ranges, used only to flag interpretation in the UI.
NORMAL_RANGES = {
    "SNA": (80.0, 84.0),
    "SNB": (78.0, 82.0),
    "ANB": (0.0, 4.0),
    "Facial Angle": (85.0, 95.0),
    "Y-Axis": (59.0, 67.0),
}


def _pt(landmarks: dict, key: str):
    """landmarks: {landmark_index(int): (x, y)}"""
    return landmarks[IDX[key]]


def _angle(p1, vertex, p2) -> float:
    """Angle (degrees) at `vertex` between rays vertex->p1 and vertex->p2."""
    v1 = (p1[0] - vertex[0], p1[1] - vertex[1])
    v2 = (p2[0] - vertex[0], p2[1] - vertex[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_theta))


def _line_angle(p1, p2, q1, q2) -> float:
    """Angle (degrees) between line p1-p2 and line q1-q2."""
    v1 = (p2[0] - p1[0], p2[1] - p1[1])
    v2 = (q2[0] - q1[0], q2[1] - q1[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag1 = math.hypot(*v1)
    mag2 = math.hypot(*v2)
    if mag1 < 1e-6 or mag2 < 1e-6:
        return 0.0
    cos_theta = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cos_theta))


def _interpret(name: str, value: float):
    lo, hi = NORMAL_RANGES.get(name, (None, None))
    if lo is None:
        return None, None, None
    if value < lo:
        interp = f"Below normal range ({lo}-{hi})"
    elif value > hi:
        interp = f"Above normal range ({lo}-{hi})"
    else:
        interp = "Within normal range"
    return lo, hi, interp


def compute_all_measurements(landmarks: dict) -> list:
    """
    landmarks: {landmark_index: (x, y)} for all 19 landmarks (in the SAME
    coordinate space, e.g. original pixel space; angles are scale-invariant
    so no mm-conversion is needed here).

    Returns a list of dicts: {name, value, unit, normal_low, normal_high, interpretation}
    """
    S, N, A, B = _pt(landmarks, "S"), _pt(landmarks, "N"), _pt(landmarks, "A"), _pt(landmarks, "B")
    Po, Or_ = _pt(landmarks, "Po"), _pt(landmarks, "Or")
    Pog, Gn = _pt(landmarks, "Pog"), _pt(landmarks, "Gn")

    sna = _angle(S, N, A)
    snb = _angle(S, N, B)
    anb = sna - snb

    # Facial Angle: angle between Frankfort Horizontal (Po-Or) and facial plane (N-Pog)
    facial_angle = _line_angle(Po, Or_, N, Pog)

    # Y-Axis: angle between S-Gn line and Frankfort Horizontal (Po-Or)
    y_axis = _line_angle(Po, Or_, S, Gn)

    results = []
    for name, value in [
        ("SNA", sna), ("SNB", snb), ("ANB", anb),
        ("Facial Angle", facial_angle), ("Y-Axis", y_axis),
    ]:
        lo, hi, interp = _interpret(name, value)
        results.append({
            "name": name, "value": round(value, 2), "unit": "deg",
            "normal_low": lo, "normal_high": hi, "interpretation": interp,
        })
    return results
