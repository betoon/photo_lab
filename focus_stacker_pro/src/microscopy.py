"""Original preprocessing tools for transmitted/reflected-light microscope stacks."""
from __future__ import annotations

from collections.abc import Callable
import cv2
import numpy as np

from .models import MicroscopeOptions
from .io import read_image, to_float


def _remove_hot_pixels(image: np.ndarray, strength: float) -> np.ndarray:
    """Replace isolated sensor spikes while leaving coherent specimen edges intact."""
    median = cv2.medianBlur(image.astype(np.float32), 3)
    residual = image - median
    gray_residual = np.max(np.abs(residual), axis=2)
    noise = np.median(np.abs(gray_residual - np.median(gray_residual))) * 1.4826 + 1e-6
    mask = gray_residual > max(2.0, strength) * noise
    # Require a strong isolated deviation; this conservative gate protects detail.
    mask &= gray_residual > 0.015
    return np.where(mask[..., None], median, image)


def _normalize_illumination(image: np.ndarray, sigma: float, preserve_brightness: bool) -> np.ndarray:
    """Divide out a broad illumination field independently per color channel."""
    sigma = max(3.0, float(sigma))
    background = cv2.GaussianBlur(image, (0, 0), sigma, borderType=cv2.BORDER_REFLECT101)
    target = np.median(background, axis=(0, 1), keepdims=True) if preserve_brightness else np.ones((1, 1, 3), np.float32)
    corrected = image / np.maximum(background, 1e-4) * target
    return np.clip(corrected, 0, 1)


def preprocess_microscope_stack(images: list[np.ndarray], options: MicroscopeOptions,
                                progress: Callable[[int, str], None] = lambda *_: None,
                                cancelled: Callable[[], bool] = lambda: False) -> list[np.ndarray]:
    """Apply specimen-safe corrections before registration and focus measurement."""
    dark = flat = None
    target_size = (images[0].shape[1], images[0].shape[0])
    if options.dark_frame_path:
        dark = to_float(read_image(options.dark_frame_path)[0])
        if dark.shape[:2] != images[0].shape[:2]: dark = cv2.resize(dark, target_size, interpolation=cv2.INTER_AREA)
    if options.flat_field_path:
        flat = to_float(read_image(options.flat_field_path)[0])
        if flat.shape[:2] != images[0].shape[:2]: flat = cv2.resize(flat, target_size, interpolation=cv2.INTER_AREA)
        if dark is not None: flat = np.maximum(flat - dark, 1e-5)
        flat /= np.maximum(np.median(flat, axis=(0, 1), keepdims=True), 1e-5)
    output = []
    for index, source in enumerate(images):
        if cancelled():
            raise InterruptedError("Cancelled")
        progress(index, f"Microscope correction {index + 1}/{len(images)}")
        image = source.astype(np.float32, copy=True)
        if dark is not None:
            image = np.maximum(image - dark, 0)
        if flat is not None:
            image = np.clip(image / np.maximum(flat, 1e-4), 0, 1)
        if options.hot_pixel_cleanup:
            image = _remove_hot_pixels(image, options.hot_pixel_strength)
        if options.illumination_normalization:
            image = _normalize_illumination(image, options.background_sigma, options.preserve_brightness)
        if options.contrast_boost > 0:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            local = cv2.createCLAHE(clipLimit=1.0 + options.contrast_boost * 2.0, tileGridSize=(8, 8)).apply(
                np.clip(gray * 65535, 0, 65535).astype(np.uint16)
            ).astype(np.float32) / 65535
            ratio = local / np.maximum(gray, 1e-4)
            image = np.clip(image * ratio[..., None], 0, 1)
        output.append(image)
    return output
