"""Trigger model inference (Swin+AttnUNet -> heatmaps/offsets -> coords -> MC-Dropout -> reliability)."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.db_models import Case
from app.services.inference_service import run_inference_for_case

router = APIRouter()


@router.post("/run/{case_id}")
def run_inference(case_id: int, db: Session = Depends(get_db)):
    case = db.query(Case).get(case_id)
    if not case:
        raise HTTPException(status_code=404, detail="Case not found")
    result = run_inference_for_case(db, case)
    return {
        "case_id": case_id,
        "landmarks": result["landmarks"],
        "measurements": result["measurements"],
        "heatmap_overlay_path": result["heatmap_overlay_path"],
    }
