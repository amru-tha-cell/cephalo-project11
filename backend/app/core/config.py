"""
Central configuration for the Cephalometric AI backend.
All paths, model, and clinical thresholds are defined here so the
rest of the codebase never hardcodes a magic number.
"""
import os


class Settings:
    PROJECT_NAME: str = "Cephalometric Landmark Detection & Analysis System"
    VERSION: str = "1.0.0"

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./data/cephalo.db")

    # Storage
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    OVERLAY_DIR: str = os.getenv("OVERLAY_DIR", "./data/processed")
    WEIGHTS_PATH: str = os.getenv("WEIGHTS_PATH", "./weights/swin_attn_unet.pt")

    # Dataset (ISBI 2015 Cephalometric X-ray Dataset)
    NUM_LANDMARKS: int = 19
    LANDMARK_NAMES = [
        "Sella", "Nasion", "Orbitale", "Porion", "A-Point", "B-Point",
        "Pogonion", "Menton", "Gnathion", "Gonion", "Lower Incisal Incision",
        "Upper Incisal Incision", "Upper Lip", "Lower Lip", "Subnasale",
        "Soft Tissue Pogonion", "Posterior Nasal Spine", "Anterior Nasal Spine",
        "Articulare",
    ]

    # Preprocessing / model input
    IMAGE_SIZE: int = 512
    # ISBI images are typically scanned at ~0.1mm per pixel at native resolution;
    # this is recalculated per-image from DB-stored pixel_spacing_mm when available.
    DEFAULT_PIXEL_SPACING_MM: float = 0.1

    # Monte Carlo Dropout
    MC_DROPOUT_PASSES: int = 20
    DROPOUT_RATE: float = 0.2

    # Reliability classification thresholds (uncertainty std, in mm)
    RELIABILITY_HIGH_STD_MM: float = 1.0       # std < this -> High
    RELIABILITY_MODERATE_STD_MM: float = 2.5   # std < this -> Moderate, else Low

    # SDR (Successful Detection Rate) thresholds, mm, standard cephalometric benchmarks
    SDR_THRESHOLDS_MM = [2.0, 2.5, 3.0, 4.0]

    # CORS
    CORS_ORIGINS = ["*"]


settings = Settings()
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.OVERLAY_DIR, exist_ok=True)
os.makedirs(os.path.dirname(settings.WEIGHTS_PATH) or ".", exist_ok=True)
