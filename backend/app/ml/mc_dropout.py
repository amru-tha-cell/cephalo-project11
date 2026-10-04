"""
Monte Carlo Dropout Uncertainty Estimation.

Keeps Dropout / Dropout2d layers active during inference (instead of the usual
eval-mode no-op) and runs N stochastic forward passes. The spread (std) of the
resulting landmark coordinates across passes is used as a per-landmark
uncertainty estimate, which then feeds the reliability classifier.
"""
import numpy as np
import torch

from app.core.config import settings
from app.ml.coordinate_extraction import extract_coordinates


def enable_mc_dropout(model: torch.nn.Module):
    """Set the whole model to eval() (so BatchNorm uses running stats) but force
    all Dropout/Dropout2d modules back into train() mode so they keep sampling."""
    model.eval()
    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d)):
            module.train()


@torch.no_grad()
def mc_dropout_predict(model: torch.nn.Module, input_tensor: torch.Tensor,
                        scale_x: float, scale_y: float,
                        passes: int = None, pixel_spacing_mm: float = None):
    """
    input_tensor: (1, 1, H, W) preprocessed image tensor
    Returns per-landmark: mean (x,y), std in px and mm, mean confidence, and
    the raw sampled coordinates (useful for uncertainty visualization).
    """
    passes = passes or settings.MC_DROPOUT_PASSES
    pixel_spacing_mm = pixel_spacing_mm or settings.DEFAULT_PIXEL_SPACING_MM

    enable_mc_dropout(model)

    all_coords = []  # passes x N x 2
    all_conf = []     # passes x N
    for _ in range(passes):
        heatmaps, offsets = model(input_tensor)
        preds = extract_coordinates(heatmaps[0], offsets[0], scale_x, scale_y)
        all_coords.append([[p["x"], p["y"]] for p in preds])
        all_conf.append([p["confidence"] for p in preds])

    coords_arr = np.array(all_coords)   # (passes, N, 2)
    conf_arr = np.array(all_conf)        # (passes, N)

    mean_xy = coords_arr.mean(axis=0)             # (N, 2)
    std_xy = coords_arr.std(axis=0)                # (N, 2) per-axis std, px
    std_px = np.sqrt((std_xy ** 2).sum(axis=1))    # combined radial std, px
    std_mm = std_px * pixel_spacing_mm
    mean_conf = conf_arr.mean(axis=0)              # (N,)

    results = []
    for i in range(mean_xy.shape[0]):
        results.append({
            "landmark_index": i,
            "x": float(mean_xy[i, 0]),
            "y": float(mean_xy[i, 1]),
            "confidence": float(mean_conf[i]),
            "uncertainty_std_px": float(std_px[i]),
            "uncertainty_std_mm": float(std_mm[i]),
        })
    return results, coords_arr
