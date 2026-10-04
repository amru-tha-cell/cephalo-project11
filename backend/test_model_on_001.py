import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------
# Make backend importable
# ---------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings
from app.ml.model import build_model
from app.ml.preprocessing import preprocess_image
from app.ml.dataset import parse_annotation_file
from app.ml.coordinate_extraction import extract_coordinates


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DATA_ROOT = BACKEND_DIR / "data" / "raw"

IMAGE_PATH = (
    DATA_ROOT
    / "RawImage"
    / "TrainingData"
    / "001.bmp"
)

ANNOTATION_PATH = (
    DATA_ROOT
    / "AnnotationsByMD"
    / "400_senior"
    / "001.txt"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():

    print("\n" + "=" * 70)
    print("TRAINED MODEL DIAGNOSTIC - IMAGE 001")
    print("=" * 70)

    print(f"Device:      {DEVICE}")
    print(f"Image:       {IMAGE_PATH}")
    print(f"Annotation:  {ANNOTATION_PATH}")
    print(f"Weights:     {settings.WEIGHTS_PATH}")

    # -----------------------------------------------------
    # 1. Check files
    # -----------------------------------------------------

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image not found:\n{IMAGE_PATH}"
        )

    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(
            f"Annotation not found:\n{ANNOTATION_PATH}"
        )

    if not Path(settings.WEIGHTS_PATH).exists():
        raise FileNotFoundError(
            f"Model weights not found:\n{settings.WEIGHTS_PATH}"
        )

    # -----------------------------------------------------
    # 2. Load original image
    # -----------------------------------------------------

    gray = cv2.imread(
        str(IMAGE_PATH),
        cv2.IMREAD_GRAYSCALE
    )

    if gray is None:
        raise RuntimeError(
            f"Could not read image:\n{IMAGE_PATH}"
        )

    orig_h, orig_w = gray.shape

    print("\nOriginal image:")
    print(f"  Width:  {orig_w}")
    print(f"  Height: {orig_h}")

    # -----------------------------------------------------
    # 3. Load ground-truth landmarks
    # -----------------------------------------------------

    gt = parse_annotation_file(
        ANNOTATION_PATH,
        settings.NUM_LANDMARKS
    )

    print(f"\nGround-truth landmarks loaded: {len(gt)}")

    # -----------------------------------------------------
    # 4. Apply SAME preprocessing used by inference
    # -----------------------------------------------------

    tensor_img, scale_x, scale_y = preprocess_image(gray)

    input_tensor = (
        torch.from_numpy(tensor_img)
        .float()
        .unsqueeze(0)
        .unsqueeze(0)
        .to(DEVICE)
    )

    print("\nPreprocessing:")
    print(f"  Tensor shape: {tuple(input_tensor.shape)}")
    print(f"  Scale X:      {scale_x:.6f}")
    print(f"  Scale Y:      {scale_y:.6f}")
    print(f"  Tensor min:   {input_tensor.min().item():.4f}")
    print(f"  Tensor max:   {input_tensor.max().item():.4f}")

    # -----------------------------------------------------
    # 5. Load trained model
    # -----------------------------------------------------

    print("\nLoading trained model...")

    model = build_model(
        device=DEVICE
    )

    model.eval()

    print("Model loaded successfully.")

    # -----------------------------------------------------
    # 6. Run model inference
    # -----------------------------------------------------

    print("\nRunning model inference...")

    with torch.no_grad():
        heatmaps, offsets = model(input_tensor)

    print("Inference complete.")

    print(f"  Heatmaps shape: {tuple(heatmaps.shape)}")
    print(f"  Offsets shape:  {tuple(offsets.shape)}")

    # -----------------------------------------------------
    # 7. Inspect raw model outputs
    # -----------------------------------------------------

    print("\nRaw model output statistics:")

    print(
        f"  Heatmap min:  {heatmaps.min().item():.6f}"
    )

    print(
        f"  Heatmap max:  {heatmaps.max().item():.6f}"
    )

    print(
        f"  Heatmap mean: {heatmaps.mean().item():.6f}"
    )

    print(
        f"  Offset min:   {offsets.min().item():.6f}"
    )

    print(
        f"  Offset max:   {offsets.max().item():.6f}"
    )

    print(
        f"  Offset mean:  {offsets.mean().item():.6f}"
    )

    # -----------------------------------------------------
    # 8. Extract predicted coordinates
    # -----------------------------------------------------

    predictions = extract_coordinates(
        heatmaps[0],
        offsets[0],
        scale_x,
        scale_y
    )

    pred_xy = np.array(
        [
            [p["x"], p["y"]]
            for p in predictions
        ],
        dtype=np.float32
    )

    gt_xy = np.asarray(
        gt,
        dtype=np.float32
    )

    # -----------------------------------------------------
    # 9. Calculate landmark errors
    # -----------------------------------------------------

    errors = np.sqrt(
        np.sum(
            (pred_xy - gt_xy) ** 2,
            axis=1
        )
    )

    # -----------------------------------------------------
    # 10. Print every landmark
    # -----------------------------------------------------

    print("\n" + "=" * 100)
    print("LANDMARK COMPARISON")
    print("=" * 100)

    print(
        f"{'Idx':>3} | "
        f"{'Landmark':<28} | "
        f"{'Ground Truth':<22} | "
        f"{'Prediction':<22} | "
        f"{'Error(px)':>10} | "
        f"{'Confidence':>10}"
    )

    print("-" * 100)

    for i in range(settings.NUM_LANDMARKS):

        name = settings.LANDMARK_NAMES[i]

        gt_x = gt_xy[i, 0]
        gt_y = gt_xy[i, 1]

        pred_x = pred_xy[i, 0]
        pred_y = pred_xy[i, 1]

        error = errors[i]

        confidence = predictions[i]["confidence"]

        print(
            f"{i:3d} | "
            f"{name:<28} | "
            f"({gt_x:7.2f}, {gt_y:7.2f}) | "
            f"({pred_x:7.2f}, {pred_y:7.2f}) | "
            f"{error:10.2f} | "
            f"{confidence:10.4f}"
        )

    # -----------------------------------------------------
    # 11. Summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("MODEL DIAGNOSTIC SUMMARY")
    print("=" * 70)

    print(
        f"Mean error:       {errors.mean():.2f} px"
    )

    print(
        f"Median error:     {np.median(errors):.2f} px"
    )

    print(
        f"Maximum error:    {errors.max():.2f} px"
    )

    print(
        f"Minimum error:    {errors.min():.2f} px"
    )

    print(
        f"Error < 10 px:    {100 * np.mean(errors < 10):.1f}%"
    )

    print(
        f"Error < 20 px:    {100 * np.mean(errors < 20):.1f}%"
    )

    print(
        f"Error < 30 px:    {100 * np.mean(errors < 30):.1f}%"
    )

    print("=" * 70)

    # -----------------------------------------------------
    # 12. Basic diagnosis
    # -----------------------------------------------------

    mean_error = errors.mean()

    print("\nDIAGNOSIS:")

    if mean_error < 10:
        print("  GOOD: Model is locating training landmarks well.")

    elif mean_error < 30:
        print("  MODERATE: Model is learning landmarks but still has noticeable error.")

    else:
        print("  POOR: Model is not accurately locating the training landmarks yet.")

    print()


if __name__ == "__main__":
    main()