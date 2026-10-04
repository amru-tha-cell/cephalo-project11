"""
Coordinate Extraction & Sub-pixel Refinement.

For each landmark heatmap channel:
  1. Find the integer-pixel argmax (coarse location) + softmax-normalized peak
     value as the raw "confidence".
  2. Read the regressed (dx, dy) offset at that pixel from the offset head and
     add it to the coarse location for sub-pixel refinement.
  3. Rescale from model input resolution (settings.IMAGE_SIZE) back to the
     original image resolution.
"""
import torch
import torch.nn.functional as F
import numpy as np

from app.core.config import settings


def extract_coordinates(heatmaps: torch.Tensor, offsets: torch.Tensor,
                         scale_x: float = 1.0, scale_y: float = 1.0):
    """
    heatmaps: (N, H, W) single-sample tensor (post-sigmoid or raw logits)
    offsets:  (2N, H, W) single-sample tensor, offsets[2i], offsets[2i+1] = dx, dy for landmark i
    scale_x/scale_y: model-input -> original-image scale factors (>1 upsamples back)

    Returns: list of dicts [{index, x, y, confidence}], coordinates in ORIGINAL image space.
    """
    N, H, W = heatmaps.shape
    probs = torch.sigmoid(heatmaps)  # per-landmark probability map

    results = []
    for i in range(N):
        hmap = probs[i]
        flat_idx = torch.argmax(hmap)
        py = int(flat_idx // W)
        px = int(flat_idx % W)
        confidence = float(hmap[py, px].item())

        dx = float(offsets[2 * i, py, px].item())
        dy = float(offsets[2 * i + 1, py, px].item())

        refined_x = px + dx
        refined_y = py + dy

        # Map back to original image resolution
        orig_x = refined_x / scale_x
        orig_y = refined_y / scale_y

        results.append({
            "landmark_index": i,
            "x": orig_x,
            "y": orig_y,
            "confidence": confidence,
        })
    return results


def soft_argmax_coordinates(heatmaps: torch.Tensor, beta: float = 100.0):
    """Optional differentiable alternative to hard-argmax (useful at training time
    for an auxiliary coordinate-regression loss). Returns (N, 2) tensor of (x, y)
    in heatmap-pixel units, expectation over a softmax of the heatmap."""
    N, H, W = heatmaps.shape
    probs = F.softmax(heatmaps.view(N, -1) * beta, dim=1).view(N, H, W)
    ys = torch.linspace(0, H - 1, H, device=heatmaps.device).view(1, H, 1)
    xs = torch.linspace(0, W - 1, W, device=heatmaps.device).view(1, 1, W)
    exp_x = (probs * xs).sum(dim=(1, 2))
    exp_y = (probs * ys).sum(dim=(1, 2))
    return torch.stack([exp_x, exp_y], dim=1)
