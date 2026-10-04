# AI-Powered Cephalometric Landmark Detection & Analysis System

ISBI 2015 Cephalometric X-ray Dataset | Swin Transformer + Attention U-Net | FastAPI | SQLite/PostgreSQL

## Folder structure
```
cephalo-project/
  backend/     FastAPI + PyTorch service (fully scaffolded, runnable now -- see backend/README.md)
  frontend/    Dummy placeholder (HTML/CSS/JS) -- build/integrate the real dashboard here later
  database/    Schema reference + notes (SQLite by default, Postgres drop-in)
```

## System flow
```
ISBI Dataset -> Preprocessing (Normalize -> CLAHE -> Resize -> Augment)
  -> Swin Transformer Encoder -> Attention U-Net Decoder
  -> 19 Heatmaps + 2D Offsets -> Coordinate Extraction & Sub-pixel Refinement
  -> Monte Carlo Dropout Uncertainty -> Reliability Classification
  -> Clinical Measurements (SNA/SNB/ANB/...) -> Interactive Dashboard + Heatmaps
  -> Landmark Correction -> Automatic Recalculation
  -> MRE/SDR + Landmark-wise Error Analysis -> Before-vs-After Analysis
  -> Database + Automated Report
```

## Quick start (backend only, for now)
```bash
cd backend
pip install -r requirements.txt --break-system-packages   # or use a venv
uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000/docs for interactive Swagger docs covering every
endpoint (upload, inference, landmarks/correction, measurements, analytics,
reports). The model runs immediately with random weights (full pipeline is
exercised end-to-end); drop a trained checkpoint at
`backend/weights/swin_attn_unet.pt` once you've trained on the ISBI dataset
(see `backend/training/train.py`).

The `frontend/` folder is a placeholder you can open directly
(`frontend/index.html`) to confirm it can reach the backend's `/health`
endpoint -- swap it out for the real dashboard whenever you're ready to wire
up the frontend.
