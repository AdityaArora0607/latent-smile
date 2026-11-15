"""Face detection and CelebA-style alignment for uploaded photos.

An uploaded photo is aligned with a similarity transform (rotation, uniform
scale, translation) that maps the detected 5-point landmarks onto the CelebA
landmark template, producing a 256x256 crop framed like the 148x148 training crop.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

OUT_SIZE = 256
CONFIDENCE_THRESHOLD = 0.9

# Landmarks of the CelebA "aligned" release (178x218 images), in pixel coordinates:
# left eye, right eye, nose, left mouth corner, right mouth corner.
# Approximate mean positions over the aligned dataset.
CELEBA_ALIGNED_LANDMARKS = np.array(
    [[69.5, 111.5], [108.5, 111.5], [88.5, 135.5], [71.5, 152.5], [105.5, 152.5]],
    dtype=np.float64,
)
# The training crop is the 148x148 window at (left=15, top=35), resized to 256.
CROP_LEFT, CROP_TOP, CROP_SIZE = 15, 35, 148


def to_crop_template(aligned: np.ndarray) -> np.ndarray:
    """Convert aligned-CelebA landmark coordinates to the 256x256 output crop."""
    a = np.asarray(aligned, dtype=np.float64)
    return (a - np.array([CROP_LEFT, CROP_TOP])) * (OUT_SIZE / CROP_SIZE)


# Template in 256x256 output-crop coordinates (derived from the constant above).
TEMPLATE_256 = to_crop_template(CELEBA_ALIGNED_LANDMARKS)

_DETECTOR = None


@dataclass
class AlignResult:
    status: str  # "ok" | "no_face" | "multiple_faces"
    n_faces: int
    crop: Image.Image | None
    message: str


def load_template(path: str | Path | None) -> np.ndarray:
    """Template in 256-crop coordinates; optionally overridden from a JSON file.

    The JSON holds a 5x2 list of landmark coordinates in aligned 178x218 CelebA
    pixel space (same convention as CELEBA_ALIGNED_LANDMARKS).
    """
    if path is None:
        return TEMPLATE_256
    arr = np.asarray(json.loads(Path(path).read_text()), dtype=np.float64)
    if arr.shape != (5, 2):
        raise ValueError(f"template JSON must be a 5x2 list, got shape {arr.shape}")
    return to_crop_template(arr)


def get_detector():
    """Lazily create and cache the MTCNN detector (weights ship with the package)."""
    global _DETECTOR
    if _DETECTOR is None:
        from facenet_pytorch import MTCNN

        _DETECTOR = MTCNN(keep_all=True, device="cpu")
    return _DETECTOR


def estimate_similarity(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform mapping src -> dst, as a 3x3 matrix."""
    n = len(src)
    A = np.zeros((2 * n, 4))
    b = np.zeros(2 * n)
    A[0::2] = np.stack([src[:, 0], -src[:, 1], np.ones(n), np.zeros(n)], axis=1)
    A[1::2] = np.stack([src[:, 1], src[:, 0], np.zeros(n), np.ones(n)], axis=1)
    b[0::2], b[1::2] = dst[:, 0], dst[:, 1]
    a, bb, tx, ty = np.linalg.lstsq(A, b, rcond=None)[0]
    return np.array([[a, -bb, tx], [bb, a, ty], [0, 0, 1]])


def prepare_image(image: Image.Image) -> Image.Image:
    """Apply EXIF orientation and convert to RGB."""
    return ImageOps.exif_transpose(image).convert("RGB")


def warp_to_template(image: Image.Image, landmarks: np.ndarray, template: np.ndarray) -> Image.Image:
    from skimage.transform import warp

    M = estimate_similarity(np.asarray(landmarks, dtype=np.float64), template)
    arr = np.asarray(image, dtype=np.float64) / 255.0
    # warp wants the map from output coords to input coords, in (x, y) order.
    out = warp(arr, np.linalg.inv(M), output_shape=(OUT_SIZE, OUT_SIZE), order=1, mode="edge")
    return Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8))


def align_face(image: Image.Image, detector=None, template_path: str | Path | None = None) -> AlignResult:
    """Detect exactly one face and return its CelebA-aligned 256x256 crop."""
    image = prepare_image(image)
    detector = detector if detector is not None else get_detector()
    boxes, probs, landmarks = detector.detect(image, landmarks=True)
    keep = []
    if boxes is not None and probs is not None and landmarks is not None:
        keep = [i for i, p in enumerate(probs) if p is not None and p >= CONFIDENCE_THRESHOLD]
    n = len(keep)
    if n == 0:
        return AlignResult(
            "no_face", 0, None,
            "No face detected. Try a clear, well-lit, front-facing photo of one person.",
        )
    if n > 1:
        return AlignResult(
            "multiple_faces", n, None,
            f"{n} faces detected. Please upload a photo with exactly one face (crop the others out).",
        )
    crop = warp_to_template(image, np.asarray(landmarks[keep[0]]), load_template(template_path))
    return AlignResult("ok", 1, crop, "Face detected and aligned.")
