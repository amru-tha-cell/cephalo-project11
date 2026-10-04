# Cephalometric AI Backend

FastAPI + PyTorch backend for the AI-powered Cephalometric Landmark Detection
and Analysis System (ISBI 2015 Cephalometric X-ray Dataset, 19 landmarks).

## Architecture (unchanged, as specified)
```
Preprocessing: Intensity Normalization -> CLAHE -> 512x512 Resize -> Training Augmentation
Model:         Swin Transformer Encoder -> Attention U-Net Decoder
                 -> 19 Landmark Heatmaps + 2D Offset Regression
                 -> Coordinate Extraction & Sub-pixel Refinement
                 -> Monte Carlo Dropout -> Reliability Classification (High/Moderate/Low)
```

## Folder layout
```
backend/
  app/
    core/config.py            all settings (paths, thresholds, landmark names)
    db/database.py            SQLAlchemy engine/session (SQLite by default, Postgres-ready)
    models/db_models.py       ORM tables: cases, landmarks (pred/gt/corrected), measurements,
                               evaluation results, case history
    schemas/schemas.py        Pydantic request/response models
    ml/
      preprocessing.py        normalize -> CLAHE -> resize -> augmentations
      swin_encoder.py         Swin Transformer encoder (windowed attention, 4 stages)
      attention_unet_decoder.py   Attention-gated U-Net decoder -> heatmap + offset heads
      model.py                combines encoder+decoder, loads/saves checkpoint
      coordinate_extraction.py    argmax + offset sub-pixel refinement
      mc_dropout.py           Monte Carlo Dropout multi-pass inference
      reliability.py          High/Moderate/Low classification from MC-dropout std
      cephalometrics.py       SNA, SNB, ANB, Facial Angle, Y-Axis
      metrics.py               MRE, SDR, landmark-wise error
    services/
      inference_service.py    orchestrates the full detect pipeline, persists to DB
      correction_service.py   clinician correction -> automatic recalculation, before/after
      report_service.py       assembles the automated analysis report
    api/routes/
      cases.py upload.py inference.py landmarks.py measurements.py analytics.py reports.py
    utils/image_utils.py      upload handling, landmark/heatmap overlay rendering
  training/train.py           reference training loop skeleton for the ISBI dataset
  data/{raw,processed,uploads}
  weights/                    put your trained checkpoint here (swin_attn_unet.pt)
  requirements.txt
  .env.example
```

## Run it
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```
Interactive API docs: http://localhost:8000/docs

The model will run with randomly-initialized weights until you train it on the
ISBI dataset and drop a checkpoint at `weights/swin_attn_unet.pt` (see
`training/train.py`) -- the whole API is functional end-to-end either way,
which is what lets the frontend be integrated and tested before training
finishes.

## API surface (matches the six required features)
| Feature | Endpoint(s) |
|---|---|
| Upload & preprocess | `POST /api/upload/` |
| Detect (model inference) | `POST /api/inference/run/{case_id}` |
| 1. Interactive landmark correction | `PUT /api/landmarks/{case_id}/{landmark_index}` |
| 2. Automatic measurement recalculation | `POST /api/measurements/{case_id}/recalculate` (also triggered automatically by the PUT above) |
| Get landmarks (pred/gt/corrected) | `GET /api/landmarks/{case_id}` |
| Get measurements | `GET /api/measurements/{case_id}?source=predicted\|corrected` |
| 3. Heatmap visualization | overlay PNG returned by inference, served at `/static/processed/...` |
| 4. Real-time error/performance analytics | `GET /api/analytics/mre-sdr/{case_id}`, `GET /api/analytics/uncertainty/{case_id}` |
| 5. Landmark-wise error analysis | `GET /api/analytics/landmark-error/{case_id}` |
| 6. Before-vs-after correction analysis | `GET /api/analytics/before-after/{case_id}` |
| Automated report | `GET /api/reports/{case_id}` |
| Case history / audit trail | `GET /api/cases/{case_id}/history` |

## Database
Defaults to SQLite at `data/cephalo.db` (zero setup). To use PostgreSQL,
set `DATABASE_URL=postgresql://user:pass@host:5432/cephalo_db` in `.env` --
no code changes needed, SQLAlchemy handles both. See `../database/schema.sql`
for the raw DDL reference.

## Dataset
Download the ISBI 2015 Cephalometric X-ray Dataset (400 images, 19 landmarks,
junior + senior orthodontist annotations) and place it under `data/raw/`.
Standard split: 150 train / 150 test1 / 100 test2. Ground-truth landmarks for
test-set cases can be loaded into the `ground_truth_landmarks` table to
enable MRE/SDR evaluation via `GET /api/analytics/mre-sdr/{case_id}`.
