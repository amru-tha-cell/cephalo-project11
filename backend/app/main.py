"""
FastAPI application entrypoint.

Run with:
    uvicorn app.main:app --reload --port 8000

Flow implemented end-to-end:
Upload -> Preprocess -> Swin Encoder -> Attention U-Net Decoder ->
19 Heatmaps + 2D Offsets -> Coordinate Extraction & Sub-pixel Refinement ->
MC-Dropout Uncertainty -> Reliability Classification -> Clinical Measurements ->
Interactive Dashboard + Heatmaps -> Landmark Correction -> Automatic Recalculation ->
MRE/SDR + Landmark-wise Error Analysis -> Before-vs-After Analysis -> DB + Report.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.db.database import init_db
from app.api.routes import cases, upload, inference, landmarks, measurements, analytics, reports

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cases.router, prefix="/api/cases", tags=["Cases"])
app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(inference.router, prefix="/api/inference", tags=["Inference"])
app.include_router(landmarks.router, prefix="/api/landmarks", tags=["Landmarks"])
app.include_router(measurements.router, prefix="/api/measurements", tags=["Measurements"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])

# Serve uploaded/processed images (for the frontend to load overlays/heatmaps)
app.mount("/static/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")
app.mount("/static/processed", StaticFiles(directory=settings.OVERLAY_DIR), name="processed")


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/")
def root():
    return {"status": "ok", "service": settings.PROJECT_NAME, "version": settings.VERSION}


@app.get("/health")
def health():
    return {"status": "healthy"}
