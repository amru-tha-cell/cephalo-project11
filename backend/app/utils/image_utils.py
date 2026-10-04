"""Image I/O and overlay drawing helpers (upload handling, heatmap/landmark overlays)."""
import os
import uuid
import cv2
import numpy as np

from app.core.config import settings


def save_upload_bytes(file_bytes: bytes, original_filename: str) -> str:
    """Persist an uploaded X-ray to disk under a unique name. Returns the saved path."""
    ext = os.path.splitext(original_filename)[1] or ".png"
    unique_name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(settings.UPLOAD_DIR, unique_name)
    with open(path, "wb") as f:
        f.write(file_bytes)
    return path


def load_grayscale(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image at {path}")
    return img


# def draw_landmarks_overlay(gray_image: np.ndarray, landmarks: list, radius: int = 5) -> np.ndarray:
#     """landmarks: list of {x, y, name, reliability(optional)}. Returns a BGR overlay image."""
#     color_map = {"High": (0, 200, 0), "Moderate": (0, 165, 255), "Low": (0, 0, 255)}
#     img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
#     for lm in landmarks:
#         color = color_map.get(lm.get("reliability"), (255, 0, 0))
#         pt = (int(round(lm["x"])), int(round(lm["y"])))
#         cv2.circle(img, pt, radius, color, -1)
#         cv2.putText(img, str(lm.get("landmark_index", "")), (pt[0] + 6, pt[1] - 6),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
#     return img
def draw_landmarks_overlay(gray_image: np.ndarray, landmarks: list, radius: int = 8) -> np.ndarray:
    """landmarks: list of {x, y, name, reliability(optional)}. Returns a BGR overlay image."""
    color_map = {
        "High": (0, 200, 0),
        "Moderate": (0, 165, 255),
        "Low": (0, 0, 255),
    }

    img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

    for lm in landmarks:
        color = color_map.get(
            lm.get("reliability"),
            (255, 0, 0)
        )

        pt = (
            int(round(lm["x"])),
            int(round(lm["y"]))
        )

        cv2.circle(
            img,
            pt,
            radius,
            color,
            -1
        )

        cv2.putText(
            img,
            str(lm.get("landmark_index", "")),
            (pt[0] + 10, pt[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA
        )

    return img

def generate_heatmap_overlay(gray_image: np.ndarray, heatmap_2d: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a single (already summed/max-pooled across landmarks) heatmap on the image."""
    heatmap_norm = cv2.normalize(heatmap_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_resized = cv2.resize(heatmap_norm, (gray_image.shape[1], gray_image.shape[0]))
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(heatmap_color, alpha, base, 1 - alpha, 0)


def save_processed_image(image: np.ndarray, suffix: str) -> str:
    name = f"{uuid.uuid4().hex}_{suffix}.png"
    path = os.path.join(settings.OVERLAY_DIR, name)
    cv2.imwrite(path, image)
    return path
