"""
SQLAlchemy ORM models.

Tables:
    Case                 - one row per uploaded X-ray / patient case
    GroundTruthLandmark   - annotated GT landmarks (for cases with known labels, e.g. test set)
    PredictedLandmark     - raw model output per landmark (heatmap + offset + MC-dropout stats)
    CorrectedLandmark     - clinician drag-corrected landmark (versioned, keeps history)
    Measurement           - cephalometric measurements (SNA/SNB/ANB/etc.), tagged by source
    EvaluationResult      - per-case MRE/SDR summary vs ground truth
    CaseEvent             - audit/history log (upload, inference, correction, report...)
"""
import datetime as dt

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, ForeignKey, Text, JSON
)
from sqlalchemy.orm import relationship

from app.db.database import Base


class Case(Base):
    __tablename__ = "cases"

    id = Column(Integer, primary_key=True, index=True)
    patient_ref = Column(String, nullable=True)          # de-identified reference/MRN
    image_path = Column(String, nullable=False)
    image_width = Column(Integer, nullable=False)
    image_height = Column(Integer, nullable=False)
    pixel_spacing_mm = Column(Float, default=0.1)
    status = Column(String, default="uploaded")           # uploaded|inferred|corrected|reported
    has_ground_truth = Column(Integer, default=0)         # 0/1 flag (test-set case)
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    ground_truth = relationship("GroundTruthLandmark", back_populates="case", cascade="all, delete-orphan")
    predictions = relationship("PredictedLandmark", back_populates="case", cascade="all, delete-orphan")
    corrections = relationship("CorrectedLandmark", back_populates="case", cascade="all, delete-orphan")
    measurements = relationship("Measurement", back_populates="case", cascade="all, delete-orphan")
    evaluations = relationship("EvaluationResult", back_populates="case", cascade="all, delete-orphan")
    events = relationship("CaseEvent", back_populates="case", cascade="all, delete-orphan")


class GroundTruthLandmark(Base):
    __tablename__ = "ground_truth_landmarks"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    landmark_index = Column(Integer, nullable=False)      # 0-18
    name = Column(String, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)

    case = relationship("Case", back_populates="ground_truth")


class PredictedLandmark(Base):
    __tablename__ = "predicted_landmarks"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    landmark_index = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)            # heatmap peak probability [0,1]
    uncertainty_std_px = Column(Float, nullable=False)     # MC-dropout std, pixels
    uncertainty_std_mm = Column(Float, nullable=False)     # MC-dropout std, mm
    reliability = Column(String, nullable=False)           # High|Moderate|Low
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    case = relationship("Case", back_populates="predictions")


class CorrectedLandmark(Base):
    __tablename__ = "corrected_landmarks"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    landmark_index = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    original_x = Column(Float, nullable=True)              # predicted x before this correction
    original_y = Column(Float, nullable=True)
    corrected_by = Column(String, default="clinician")
    corrected_at = Column(DateTime, default=dt.datetime.utcnow)

    case = relationship("Case", back_populates="corrections")


class Measurement(Base):
    __tablename__ = "measurements"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    source = Column(String, nullable=False)                # predicted|corrected|ground_truth
    name = Column(String, nullable=False)                  # SNA|SNB|ANB|...
    value = Column(Float, nullable=False)
    unit = Column(String, default="deg")
    normal_low = Column(Float, nullable=True)
    normal_high = Column(Float, nullable=True)
    interpretation = Column(String, nullable=True)          # e.g. "Class II skeletal pattern"
    computed_at = Column(DateTime, default=dt.datetime.utcnow)

    case = relationship("Case", back_populates="measurements")


class EvaluationResult(Base):
    __tablename__ = "evaluation_results"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    source = Column(String, default="predicted")            # predicted|corrected
    mre_mm = Column(Float, nullable=False)
    sdr_json = Column(JSON, nullable=False)                 # {"2.0mm": 0.81, "2.5mm": 0.88, ...}
    per_landmark_error_json = Column(JSON, nullable=False)  # {landmark_name: error_mm}
    created_at = Column(DateTime, default=dt.datetime.utcnow)

    case = relationship("Case", back_populates="evaluations")


class CaseEvent(Base):
    __tablename__ = "case_events"

    id = Column(Integer, primary_key=True, index=True)
    case_id = Column(Integer, ForeignKey("cases.id"), nullable=False)
    event_type = Column(String, nullable=False)   # upload|inference|correction|recalculation|report
    details_json = Column(JSON, nullable=True)
    timestamp = Column(DateTime, default=dt.datetime.utcnow)

    case = relationship("Case", back_populates="events")
