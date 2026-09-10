"""
Image pre-processing.Handles JPEG/PNG/PDF.
"""
from __future__ import annotations

import io
import os
from typing import List

import cv2
import numpy as np
from PIL import Image


def load_images(path: str) -> List[np.ndarray]:
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        from pdf2image import convert_from_path

        pages = convert_from_path(path, dpi=300)
        return [cv2.cvtColor(np.array(p), cv2.COLOR_RGB2BGR) for p in pages]

    if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Could not read image at {path}")
        return [img]

    raise ValueError(f"Unsupported file type: {ext}")


def _estimate_skew_angle(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
    )
    if lines is None:
        return 0.0
    lines = np.asarray(lines).reshape(-1, 4)

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line
        if x2 - x1 == 0:
            continue
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        # Keep only near-horizontal lines (typical for text baselines)
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return 0.0
    return float(np.median(angles))


def deskew(image: np.ndarray) -> np.ndarray:
    """Rotate the image to correct detected skew."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    angle = _estimate_skew_angle(gray)

    if abs(angle) < 0.5:
        return image  # negligible skew, skip rotation

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        image, rot_mat, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(image, None, 7, 7, 7, 21)


def normalize_contrast(image: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def sharpen(image: np.ndarray) -> np.ndarray:
    """Mild unsharp mask to counter slight blur before OCR."""
    gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=3)
    return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)


def preprocess_pipeline(image: np.ndarray) -> np.ndarray:
    image = denoise(image)
    image = normalize_contrast(image)
    image = deskew(image)
    image = sharpen(image)
    return image


def to_pil(image: np.ndarray) -> Image.Image:
    """Convert a BGR numpy array to a PIL RGB image (for torch/transformers)."""
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
