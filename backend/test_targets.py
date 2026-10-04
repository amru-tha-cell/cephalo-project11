import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------
# Make backend importable
# ---------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from app.ml.dataset import ISBICephalometricDataset


def main():

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------

    dataset = ISBICephalometricDataset(
        data_root=BACKEND_DIR / "data" / "raw",
        split="TrainingData",
        annotator="senior",
    )

    # -----------------------------------------------------
    # Get first sample
    # -----------------------------------------------------

    sample = dataset[0]

    landmarks = sample["landmarks"]
    heatmaps = sample["heatmaps"]

    print("\n========================================")
    print("TARGET DIAGNOSTIC")
    print("========================================")

    print(
        f"Image:     {sample['image'].shape}"
    )

    print(
        f"Landmarks: {landmarks.shape}"
    )

    print(
        f"Heatmaps:  {heatmaps.shape}"
    )

    print(
        f"File:      {sample['image_path']}"
    )

    print("\nLandmark comparison:")
    print(
        "Index | Ground Truth | Heatmap Peak | Error(px)"
    )
    print("-" * 65)

    errors = []

    # -----------------------------------------------------
    # Check all 19 heatmaps
    # -----------------------------------------------------

    for i in range(19):

        # Ground truth coordinate
        gt_x = float(landmarks[i, 0])
        gt_y = float(landmarks[i, 1])

        # Heatmap
        heatmap = heatmaps[i].numpy()

        # Find maximum activation
        peak_y, peak_x = np.unravel_index(
            np.argmax(heatmap),
            heatmap.shape
        )

        # Euclidean error
        error = np.sqrt(
            (peak_x - gt_x) ** 2
            +
            (peak_y - gt_y) ** 2
        )

        errors.append(error)

        print(
            f"{i:5d} | "
            f"({gt_x:7.2f}, {gt_y:7.2f}) | "
            f"({peak_x:7.2f}, {peak_y:7.2f}) | "
            f"{error:8.3f}"
        )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    errors = np.asarray(errors)

    print("\n========================================")
    print("SUMMARY")
    print("========================================")

    print(
        f"Mean error:   {errors.mean():.4f} px"
    )

    print(
        f"Max error:    {errors.max():.4f} px"
    )

    print(
        f"Min error:    {errors.min():.4f} px"
    )

    print(
        f"Median error: {np.median(errors):.4f} px"
    )


if __name__ == "__main__":
    main()