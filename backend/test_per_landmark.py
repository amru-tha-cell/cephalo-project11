import sys
from pathlib import Path

import numpy as np
import torch


# ============================================================
# PROJECT ROOT
# ============================================================

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from app.core.config import settings
from app.ml.dataset import ISBICephalometricDataset
from app.ml.model import build_model
from app.ml.coordinate_extraction import extract_coordinates


# ============================================================
# LANDMARK NAMES
# ============================================================

LANDMARK_NAMES = settings.LANDMARK_NAMES


# ============================================================
# EVALUATE ONE SPLIT
# ============================================================

def evaluate_split(model, dataset, device):
    """
    Run inference on every image in a dataset split and
    collect pixel errors separately for each landmark.
    """

    num_landmarks = settings.NUM_LANDMARKS

    # One list of errors for each landmark
    errors = [[] for _ in range(num_landmarks)]

    model.eval()

    for idx in range(len(dataset)):

        sample = dataset[idx]

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        image = sample["image"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image = image.float()

        # Dataset image should be (C,H,W)
        if image.ndim == 3:
            image = image.unsqueeze(0)

        # Final shape should be (1,C,H,W)
        image = image.to(device)

        # ----------------------------------------------------
        # Ground truth
        # ----------------------------------------------------

        gt_xy = np.asarray(
            sample["orig_keypoints"],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Scale used to convert model coordinates back
        # to original image coordinates
        # ----------------------------------------------------

        scale = sample["scale"]

        scale_x = float(scale[0])
        scale_y = float(scale[1])

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        with torch.no_grad():

            heatmaps, offsets = model(image)

        # ----------------------------------------------------
        # Extract predicted coordinates
        # ----------------------------------------------------

        predictions = extract_coordinates(
            heatmaps[0],
            offsets[0],
            scale_x=scale_x,
            scale_y=scale_y
        )

        pred_xy = np.asarray(
            [
                [p["x"], p["y"]]
                for p in predictions
            ],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if pred_xy.shape != gt_xy.shape:

            print(
                f"\n[WARNING] Shape mismatch at sample {idx}"
            )

            print(
                f"Prediction shape: {pred_xy.shape}"
            )

            print(
                f"Ground truth shape: {gt_xy.shape}"
            )

            continue

        # ----------------------------------------------------
        # Euclidean error for every landmark
        # ----------------------------------------------------

        pixel_errors = np.linalg.norm(
            pred_xy - gt_xy,
            axis=1
        )

        # Store each landmark's error separately
        for landmark_idx, error in enumerate(pixel_errors):

            errors[landmark_idx].append(
                float(error)
            )

        # Progress
        if (idx + 1) % 25 == 0 or idx == len(dataset) - 1:

            print(
                f"  Processed {idx + 1}/{len(dataset)} images"
            )

    return errors


# ============================================================
# PRINT RESULTS
# ============================================================

def print_results(
    split_name,
    errors,
    pixel_spacing_mm
):

    print("\n")
    print("=" * 115)
    print(f"{split_name} - PER LANDMARK RESULTS")
    print("=" * 115)

    print(
        f"{'Idx':>3} | "
        f"{'Landmark':<28} | "
        f"{'Mean(px)':>10} | "
        f"{'Median(px)':>11} | "
        f"{'Mean(mm)':>10} | "
        f"{'<10px':>8} | "
        f"{'<20px':>8} | "
        f"{'<30px':>8}"
    )

    print("-" * 115)

    all_errors = []

    for i, landmark_errors in enumerate(errors):

        if len(landmark_errors) == 0:
            continue

        arr = np.asarray(
            landmark_errors,
            dtype=np.float32
        )

        mean_px = float(arr.mean())
        median_px = float(np.median(arr))

        mean_mm = mean_px * pixel_spacing_mm

        under_10 = float(
            (arr < 10).mean() * 100
        )

        under_20 = float(
            (arr < 20).mean() * 100
        )

        under_30 = float(
            (arr < 30).mean() * 100
        )

        all_errors.extend(
            arr.tolist()
        )

        print(
            f"{i:3d} | "
            f"{LANDMARK_NAMES[i]:<28} | "
            f"{mean_px:10.2f} | "
            f"{median_px:11.2f} | "
            f"{mean_mm:10.2f} | "
            f"{under_10:7.1f}% | "
            f"{under_20:7.1f}% | "
            f"{under_30:7.1f}%"
        )

    # --------------------------------------------------------
    # Overall results
    # --------------------------------------------------------

    print("-" * 115)

    if len(all_errors) > 0:

        all_errors = np.asarray(
            all_errors,
            dtype=np.float32
        )

        overall_mean_px = float(
            all_errors.mean()
        )

        overall_median_px = float(
            np.median(all_errors)
        )

        overall_mean_mm = (
            overall_mean_px *
            pixel_spacing_mm
        )

        print(
            f"Overall mean error:   "
            f"{overall_mean_px:.2f} px "
            f"({overall_mean_mm:.2f} mm)"
        )

        print(
            f"Overall median error: "
            f"{overall_median_px:.2f} px"
        )


# ============================================================
# CREATE DATASET
# ============================================================

def create_dataset(split):

    return ISBICephalometricDataset(
        data_root=ROOT / "data" / "raw",
        split=split,
        annotator="senior",
        image_size=settings.IMAGE_SIZE,
        heatmap_sigma=3.0,
        offset_radius=4,
        val_size=15,
        seed=42,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 115)
    print("PER-LANDMARK MODEL DIAGNOSTIC")
    print("=" * 115)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Device:          {device}")
    print(f"Weights:         {settings.WEIGHTS_PATH}")
    print(f"Image size:      {settings.IMAGE_SIZE}")
    print(
        f"Pixel spacing:   "
        f"{settings.DEFAULT_PIXEL_SPACING_MM} mm/px"
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading trained model...")

    model = build_model(
        weights_path=settings.WEIGHTS_PATH,
        device=str(device)
    )

    model.eval()

    print("Model loaded successfully.")

    # ========================================================
    # VALIDATION
    # ========================================================

    print("\n")
    print("=" * 115)
    print("Evaluating VALIDATION split...")
    print("=" * 115)

    val_dataset = create_dataset("val")

    print(
        f"Validation samples: {len(val_dataset)}"
    )

    val_errors = evaluate_split(
        model,
        val_dataset,
        device
    )

    print_results(
        "VALIDATION",
        val_errors,
        settings.DEFAULT_PIXEL_SPACING_MM
    )

    # ========================================================
    # TEST 1
    # ========================================================

    print("\n")
    print("=" * 115)
    print("Evaluating TEST1 split...")
    print("=" * 115)

    test1_dataset = create_dataset("test1")

    print(
        f"Test1 samples: {len(test1_dataset)}"
    )

    test1_errors = evaluate_split(
        model,
        test1_dataset,
        device
    )

    print_results(
        "TEST1",
        test1_errors,
        settings.DEFAULT_PIXEL_SPACING_MM
    )

    # ========================================================
    # TEST 2
    # ========================================================

    print("\n")
    print("=" * 115)
    print("Evaluating TEST2 split...")
    print("=" * 115)

    test2_dataset = create_dataset("test2")

    print(
        f"Test2 samples: {len(test2_dataset)}"
    )

    test2_errors = evaluate_split(
        model,
        test2_dataset,
        device
    )

    print_results(
        "TEST2",
        test2_errors,
        settings.DEFAULT_PIXEL_SPACING_MM
    )

    # ========================================================
    # DONE
    # ========================================================

    print("\n")
    print("=" * 115)
    print("PER-LANDMARK DIAGNOSTIC COMPLETE")
    print("=" * 115)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()