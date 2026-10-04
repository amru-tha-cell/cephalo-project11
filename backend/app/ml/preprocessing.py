"""
Preprocessing pipeline (exactly as specified):
Intensity Normalization -> CLAHE -> 512x512 Resize -> Training Augmentation
"""
import cv2
import numpy as np
import albumentations as A

from app.core.config import settings


def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """Min-max normalize a grayscale image to [0, 255] uint8."""
    image = image.astype(np.float32)
    lo, hi = image.min(), image.max()
    if hi - lo < 1e-6:
        return np.zeros_like(image, dtype=np.uint8)
    image = (image - lo) / (hi - lo) * 255.0
    return image.astype(np.uint8)


def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization to enhance bone/soft-tissue contrast."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def resize_image(image: np.ndarray, size: int = None):
    """Resize to model input size, returning image + scale factors for coordinate rescaling."""
    size = size or settings.IMAGE_SIZE
    h, w = image.shape[:2]
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    scale_x, scale_y = size / w, size / h
    return resized, scale_x, scale_y


def preprocess_image(raw_gray_image: np.ndarray):
    """Full inference-time pipeline: Normalize -> CLAHE -> Resize. Returns model-ready tensor input (HxW float32 [0,1]) and scale factors to map back to original resolution."""
    norm = normalize_intensity(raw_gray_image)
    enhanced = apply_clahe(norm)
    resized, scale_x, scale_y = resize_image(enhanced)
    tensor_ready = (resized.astype(np.float32) / 255.0)
    return tensor_ready, scale_x, scale_y


def get_train_augmentations(image_size: int = None):
    """Augmentations applied ONLY during training. Kept anatomically safe: no horizontal
    flip (left/right cephalometric landmarks are not symmetric-equivalent)."""
    image_size = image_size or settings.IMAGE_SIZE
    return A.Compose(
        [
            A.Rotate(limit=8, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
            A.GaussNoise(var_limit=(5.0, 20.0), p=0.3),
            # NOTE: ElasticTransform was removed -- it warps the image but does NOT move
            # keypoints (and alpha_affine is rejected by albumentations 1.4.x), which would
            # silently corrupt the landmark labels.
            A.RandomScale(scale_limit=0.05, p=0.3),
            A.Resize(image_size, image_size),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_val_augmentations(image_size: int = None):
    """No augmentation at validation/test time -- only deterministic resize."""
    image_size = image_size or settings.IMAGE_SIZE
    return A.Compose(
        [A.Resize(image_size, image_size)],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )