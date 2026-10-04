import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# Project path
# ---------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from app.ml.model import SwinAttentionUNet
from app.ml.dataset import ISBICephalometricDataset
from app.ml.preprocessing import get_val_augmentations


# ---------------------------------------------------------
# Losses
# ---------------------------------------------------------
def heatmap_loss(pred_heatmaps, target_heatmaps):
    """
    Compare predicted heatmaps with ground-truth Gaussian heatmaps.
    Model outputs logits, so apply sigmoid first.
    """
    pred = torch.sigmoid(pred_heatmaps)
    return F.mse_loss(pred, target_heatmaps)


def offset_loss(pred_offsets, target_offsets, mask):
    """
    Only calculate offset error near landmark locations.
    """
    loss = F.smooth_l1_loss(
        pred_offsets,
        target_offsets,
        reduction="none"
    )

    masked_loss = loss * mask

    # Avoid division by zero
    denominator = mask.sum().clamp_min(1.0)

    return masked_loss.sum() / denominator


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():

    print("=" * 60)
    print("ONE-IMAGE OVERFIT TEST")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # -----------------------------------------------------
    # Dataset
    # -----------------------------------------------------
    dataset = ISBICephalometricDataset(
        data_root=BACKEND_DIR / "data" / "raw",
        split="TrainingData",
        annotator="senior",

        # IMPORTANT:
        # No random augmentation for this debugging experiment.
        transform=get_val_augmentations(512),
    )

    # Keep ONLY the first image: 001.bmp
    image_path = dataset.image_files[0]

    print(f"Image: {image_path.name}")

    # -----------------------------------------------------
    # Create a one-image dataset
    # -----------------------------------------------------
    class SingleImageDataset(torch.utils.data.Dataset):

        def __init__(self, original_dataset):
            self.original_dataset = original_dataset

        def __len__(self):
            return 1

        def __getitem__(self, index):
            return self.original_dataset[0]

    single_dataset = SingleImageDataset(dataset)

    loader = DataLoader(
        single_dataset,
        batch_size=1,
        shuffle=False,
    )

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = SwinAttentionUNet().to(device)

    model.train()

    print("Model created.")

    # -----------------------------------------------------
    # Optimizer
    # -----------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=0.0,
    )

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    epochs = 100

    best_loss = float("inf")

    print()
    print("Starting training...")
    print()

    for epoch in range(1, epochs + 1):

        for batch in loader:

            image = batch["image"].to(device)
            target_heatmaps = batch["heatmaps"].to(device)
            target_offsets = batch["offsets"].to(device)
            offset_mask = batch["offset_mask"].to(device)

            optimizer.zero_grad()

            # Forward
            pred_heatmaps, pred_offsets = model(image)

            # Losses
            hm_loss = heatmap_loss(
                pred_heatmaps,
                target_heatmaps
            )

            off_loss = offset_loss(
                pred_offsets,
                target_offsets,
                offset_mask
            )

            # Same basic combined objective
            total_loss = hm_loss + 0.1 * off_loss

            # Backpropagation
            total_loss.backward()

            optimizer.step()

        if total_loss.item() < best_loss:
            best_loss = total_loss.item()

        # Print every epoch
        print(
            f"Epoch {epoch:03d} | "
            f"Total: {total_loss.item():.6f} | "
            f"Heatmap: {hm_loss.item():.6f} | "
            f"Offset: {off_loss.item():.6f}"
        )

    # -----------------------------------------------------
    # Final prediction check
    # -----------------------------------------------------
    print()
    print("=" * 60)
    print("FINAL PREDICTION CHECK")
    print("=" * 60)

    model.eval()

    with torch.no_grad():

        batch = next(iter(loader))

        image = batch["image"].to(device)

        pred_heatmaps, pred_offsets = model(image)

        pred_heatmaps = torch.sigmoid(pred_heatmaps)

        target_landmarks = batch["landmarks"][0].numpy()

        pred_heatmaps_np = pred_heatmaps[0].cpu().numpy()

        import numpy as np

        errors = []

        print()
        print(
            "Index | Ground Truth | Predicted Peak | Error(px)"
        )
        print("-" * 65)

        for i in range(19):

            gt_x, gt_y = target_landmarks[i]

            heatmap = pred_heatmaps_np[i]

            peak_y, peak_x = np.unravel_index(
                np.argmax(heatmap),
                heatmap.shape
            )

            error = np.sqrt(
                (peak_x - gt_x) ** 2 +
                (peak_y - gt_y) ** 2
            )

            errors.append(error)

            print(
                f"{i:5d} | "
                f"({gt_x:7.2f}, {gt_y:7.2f}) | "
                f"({peak_x:7.2f}, {peak_y:7.2f}) | "
                f"{error:8.2f}"
            )

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"Best training loss: {best_loss:.6f}")
    print(f"Mean landmark error: {np.mean(errors):.2f} px")
    print(f"Max landmark error:  {np.max(errors):.2f} px")
    print(f"Median error:        {np.median(errors):.2f} px")

    print()
    print("OVERFIT TEST COMPLETE")


if __name__ == "__main__":
    main()