"""
Pydantic schemas for request/response validation.
"""
from typing import List, Optional, Dict
from datetime import datetime
from pydantic import BaseModel


class LandmarkPoint(BaseModel):
    landmark_index: int
    name: str
    x: float
    y: float


class PredictedLandmarkOut(LandmarkPoint):
    confidence: float
    uncertainty_std_px: float
    uncertainty_std_mm: float
    reliability: str

    class Config:
        from_attributes = True


class CorrectedLandmarkIn(BaseModel):
    x: float
    y: float
    corrected_by: Optional[str] = "clinician"


class CorrectedLandmarkOut(LandmarkPoint):
    original_x: Optional[float]
    original_y: Optional[float]
    corrected_by: str
    corrected_at: datetime

    class Config:
        from_attributes = True


class CaseCreateOut(BaseModel):
    id: int
    patient_ref: Optional[str]
    image_path: str
    image_width: int
    image_height: int
    pixel_spacing_mm: float
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class MeasurementOut(BaseModel):
    name: str
    value: float
    unit: str
    normal_low: Optional[float]
    normal_high: Optional[float]
    interpretation: Optional[str]

    class Config:
        from_attributes = True


class InferenceResultOut(BaseModel):
    case_id: int
    landmarks: List[PredictedLandmarkOut]
    heatmap_overlay_path: Optional[str] = None
    measurements: List[MeasurementOut]


class EvaluationOut(BaseModel):
    case_id: int
    source: str
    mre_mm: float
    sdr: Dict[str, float]
    per_landmark_error_mm: Dict[str, float]


class BeforeAfterOut(BaseModel):
    case_id: int
    predicted_landmarks: List[PredictedLandmarkOut]
    corrected_landmarks: List[CorrectedLandmarkOut]
    predicted_measurements: List[MeasurementOut]
    corrected_measurements: List[MeasurementOut]
    measurement_deltas: Dict[str, float]
    landmark_shift_mm: Dict[str, float]
