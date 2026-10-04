"""
PyTorch Dataset for the ISBI 2015 Cephalometric X-ray Dataset.

Expected folder layout under ``data_root`` (this is the standard layout of the
downloaded dataset)::

    <data_root>/
      RawImage/
        TrainingData/   001.bmp ... 150.bmp
        Test1Data/      151.bmp ... 300.bmp
        Test2Data/      301.bmp ... 400.bmp
      AnnotationsByMD/
        400_junior/     001.txt ... 400.txt     (junior orthodontist)
        400_senior/     001.txt ... 400.txt     (senior orthodontist)

Each annotation .txt starts with 19 lines of ``x,y`` (pixel coordinates in the
ORIGINAL image); any lines after those 19 are ignored.

What one sample contains (see ``__getitem__``) matches exactly what the model
and ``coordinate_extraction.extract_coordinates`` expect:

    image        (1, S, S)   float32 [0, 1]   Normalize -> CLAHE -> (aug) -> Resize
    heatmaps     (N, S, S)   Gaussian around each landmark; the nearest pixel is exactly 1.0
    offsets      (2N, S, S)  (dx, dy) = true_xy - pixel_xy, valid near the landmark
    offset_mask  (2N, S, S)  1 where the offsets are supervised, else 0

where S = image size (512 by default) and N = 19 landmarks.
"""
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from app.core.config import settings
from app.ml.preprocessing import (
    apply_clahe,
    get_train_augmentations,
    get_val_augmentations,
    normalize_intensity,
)

# friendly split name -> folder name inside RawImage/
SPLIT_DIRS = {
    "train": "TrainingData",
    "val": "TrainingData",      # carved out of the training folder (see val_size)
    "test1": "Test1Data",
    "test2": "Test2Data",
    # real folder names are accepted as-is too
    "TrainingData": "TrainingData",
    "Test1Data": "Test1Data",
    "Test2Data": "Test2Data",
}
IMAGE_EXTS = (".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
ANNOTATORS = ("junior", "senior", "mean")


def parse_annotation_file(path: Path, num_landmarks: int) -> np.ndarray:
    """Read the first ``num_landmarks`` 'x,y' lines of an ISBI annotation file.
    Returns float32 array (N, 2) in original-image pixel coordinates."""
    points = []
    with open(path, "r") as f:
        for line in f:
            parts = [p for p in line.strip().split(",") if p.strip() != ""]
            if len(parts) != 2:
                continue
            try:
                points.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
            if len(points) == num_landmarks:
                break
    if len(points) != num_landmarks:
        raise ValueError(
            f"{path}: expected {num_landmarks} landmark lines ('x,y'), found {len(points)}"
        )
    return np.asarray(points, dtype=np.float32)


class ISBICephalometricDataset(Dataset):
    def __init__(
        self,
        data_root,
        split: str = "train",
        annotator: str = "senior",
        transform=None,
        image_size: int = None,
        heatmap_sigma: float = 3.0,
        offset_radius: int = 4,
        val_size: int = 15,
        seed: int = 42,
        limit: int = None,
    ):
        """
        split:      'train' | 'val' | 'test1' | 'test2'  (or the raw folder names).
                    'train' and 'val' both come from RawImage/TrainingData: ISBI has no
                    official validation set, so ``val_size`` images are held out
                    (deterministically, via ``seed``) from the 150 training images.
                    Set val_size=0 to train on all 150 and monitor on test1 instead.
        annotator:  'junior' | 'senior' | 'mean' (average of both, a common ground truth).
        transform:  albumentations Compose with keypoint support. Defaults to training
                    augmentations for split='train', and plain resize otherwise.
        limit:      only use the first N images (handy for quick smoke tests).
        """
        if split not in SPLIT_DIRS:
            raise ValueError(f"Unknown split '{split}'. Use one of: train, val, test1, test2")
        if annotator not in ANNOTATORS:
            raise ValueError(f"annotator must be one of {ANNOTATORS}, got '{annotator}'")

        self.data_root = Path(data_root)
        self.split = SPLIT_DIRS[split]            # folder name, e.g. 'TrainingData'
        self.split_name = split                   # what the caller asked for
        self.annotator = annotator
        self.image_size = image_size or settings.IMAGE_SIZE
        self.num_landmarks = settings.NUM_LANDMARKS
        self.heatmap_sigma = heatmap_sigma
        self.offset_radius = offset_radius
        self.pixel_spacing_mm = settings.DEFAULT_PIXEL_SPACING_MM

        self.image_dir = self.data_root / "RawImage" / self.split
        if annotator == "mean":
            self.annotation_dirs = [
                self.data_root / "AnnotationsByMD" / "400_junior",
                self.data_root / "AnnotationsByMD" / "400_senior",
            ]
        else:
            self.annotation_dirs = [self.data_root / "AnnotationsByMD" / f"400_{annotator}"]
        self.annotation_dir = self.annotation_dirs[0]

        self._check_paths()

        stems = sorted(p.stem for p in self.image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
        self._image_files = {p.stem: p for p in self.image_dir.iterdir()
                             if p.suffix.lower() in IMAGE_EXTS}
        if not stems:
            raise FileNotFoundError(f"No images ({', '.join(IMAGE_EXTS)}) found in {self.image_dir}")

        # train / val carve-out of the official training folder
        if split in ("train", "val"):
            if val_size < 0 or val_size >= len(stems):
                raise ValueError(f"val_size must be in [0, {len(stems) - 1}], got {val_size}")
            shuffled = stems[:]
            random.Random(seed).shuffle(shuffled)
            held_out = set(shuffled[:val_size])
            if split == "val":
                if val_size == 0:
                    raise ValueError("split='val' requested but val_size=0")
                stems = [s for s in stems if s in held_out]
            else:
                stems = [s for s in stems if s not in held_out]

        if limit:
            stems = stems[:limit]
        self.stems = stems

        if transform is None:
            transform = (get_train_augmentations(self.image_size) if split == "train"
                         else get_val_augmentations(self.image_size))
        self.transform = transform

    # ------------------------------------------------------------------ helpers
    def _check_paths(self):
        problems = []
        if not self.image_dir.is_dir():
            problems.append(f"images folder not found: {self.image_dir}")
        for d in self.annotation_dirs:
            if not d.is_dir():
                problems.append(f"annotations folder not found: {d}")
        if problems:
            raise FileNotFoundError(
                "\n".join(problems)
                + f"\n\nExpected layout under data_root ({self.data_root}):\n"
                  "  RawImage/{TrainingData,Test1Data,Test2Data}/*.bmp\n"
                  "  AnnotationsByMD/{400_junior,400_senior}/*.txt"
            )

    def _load_landmarks(self, stem: str) -> np.ndarray:
        arrays = []
        for d in self.annotation_dirs:
            path = d / f"{stem}.txt"
            if not path.is_file():
                raise FileNotFoundError(f"Missing annotation file: {path}")
            arrays.append(parse_annotation_file(path, self.num_landmarks))
        return np.mean(arrays, axis=0).astype(np.float32)

    def _render_targets(self, keypoints: np.ndarray):
        """Gaussian heatmaps + (dx, dy) offset maps + offset mask, in model-input pixels."""
        S, N = self.image_size, self.num_landmarks
        heat = np.zeros((N, S, S), np.float32)
        offsets = np.zeros((2 * N, S, S), np.float32)
        mask = np.zeros((2 * N, S, S), np.float32)
        visible = np.zeros(N, np.float32)

        sigma = self.heatmap_sigma
        r_heat = int(math.ceil(3 * sigma))
        r_off = self.offset_radius

        for i, (x, y) in enumerate(keypoints):
            if not (0 <= x <= S - 1 and 0 <= y <= S - 1):
                continue                      # landmark pushed outside the image by augmentation
            visible[i] = 1.0
            cx, cy = int(round(float(x))), int(round(float(y)))

            # Gaussian heatmap centred on the exact (sub-pixel) location
            x0, x1 = max(cx - r_heat, 0), min(cx + r_heat + 1, S)
            y0, y1 = max(cy - r_heat, 0), min(cy + r_heat + 1, S)
            xs = np.arange(x0, x1, dtype=np.float32)
            ys = np.arange(y0, y1, dtype=np.float32)
            heat[i, y0:y1, x0:x1] = np.exp(
                -((xs[None, :] - x) ** 2 + (ys[:, None] - y) ** 2) / (2 * sigma ** 2)
            )
            heat[i, cy, cx] = 1.0     # exact peak pixel (needed by the focal heatmap loss)

            # Offsets: at pixel p near the landmark, target = true_xy - p
            # (coordinate_extraction does refined = argmax_pixel + offset)
            x0, x1 = max(cx - r_off, 0), min(cx + r_off + 1, S)
            y0, y1 = max(cy - r_off, 0), min(cy + r_off + 1, S)
            xs = np.arange(x0, x1, dtype=np.float32)
            ys = np.arange(y0, y1, dtype=np.float32)
            gx, gy = np.meshgrid(xs, ys)
            near = ((gx - cx) ** 2 + (gy - cy) ** 2 <= r_off ** 2).astype(np.float32)
            offsets[2 * i, y0:y1, x0:x1] = (x - gx) * near
            offsets[2 * i + 1, y0:y1, x0:x1] = (y - gy) * near
            mask[2 * i, y0:y1, x0:x1] = near
            mask[2 * i + 1, y0:y1, x0:x1] = near

        return heat, offsets, mask, visible

    # ---------------------------------------------------------- Dataset API
    def __len__(self):
        return len(self.stems)

    def __getitem__(self, idx):
        stem = self.stems[idx]
        gray = cv2.imread(str(self._image_files[stem]), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise IOError(f"Could not read image {self._image_files[stem]}")
        orig_h, orig_w = gray.shape[:2]
        orig_kps = self._load_landmarks(stem)

        # Same first two steps as inference (preprocessing.preprocess_image)
        gray = apply_clahe(normalize_intensity(gray))

        out = self.transform(image=gray, keypoints=[tuple(p) for p in orig_kps])
        image = out["image"]
        kps = np.asarray(out["keypoints"], dtype=np.float32).reshape(-1, 2)[:, :2]
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            raise RuntimeError(f"Transform must output {self.image_size}x{self.image_size}, got {image.shape}")

        heat, offsets, mask, visible = self._render_targets(kps)
        image_t = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)

        return {
            "image": image_t,
            "heatmaps": torch.from_numpy(heat),
            "offsets": torch.from_numpy(offsets),
            "offset_mask": torch.from_numpy(mask),
            "visible": torch.from_numpy(visible),
            # --- metadata for evaluation / debugging ---
            "keypoints": torch.from_numpy(kps),                 # (N, 2) in model-input pixels
            "orig_keypoints": torch.from_numpy(orig_kps),       # (N, 2) in original-image pixels
            "scale": torch.tensor([self.image_size / orig_w, self.image_size / orig_h],
                                  dtype=torch.float32),         # (scale_x, scale_y)
            "pixel_spacing_mm": torch.tensor(self.pixel_spacing_mm, dtype=torch.float32),
            "name": stem,
        }