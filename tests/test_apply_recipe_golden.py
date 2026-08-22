"""Golden-image tests: apply_recipe determinism + PSNR/SSIM thresholds.

Synthetic BGR frames (no external files). Skip cleanly if OpenCV missing.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

cv2 = pytest.importorskip("cv2")
from imaging import Recipe, apply_recipe


def _gradient_bgr(h=64, w=96):
    """Smooth gradient + color patches for stable metrics."""
    yy, xx = np.mgrid[0:h, 0:w]
    b = (xx / max(w - 1, 1) * 255).astype(np.uint8)
    g = (yy / max(h - 1, 1) * 255).astype(np.uint8)
    r = ((xx + yy) / (w + h - 2) * 200 + 40).astype(np.uint8)
    img = np.stack([b, g, r], axis=-1)
    # center patch
    img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = (40, 180, 90)
    return img


def _psnr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse < 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0 ** 2) / mse))


def _ssim_simple(a: np.ndarray, b: np.ndarray) -> float:
    """Lightweight grayscale SSIM (enough for regression thresholds)."""
    a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float64)
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = a.mean(), b.mean()
    sig_a, sig_b = a.var(), b.var()
    sig_ab = ((a - mu_a) * (b - mu_b)).mean()
    num = (2 * mu_a * mu_b + C1) * (2 * sig_ab + C2)
    den = (mu_a ** 2 + mu_b ** 2 + C1) * (sig_a + sig_b + C2)
    return float(num / (den + 1e-12))


def test_identity_recipe_near_passthrough():
    """Default recipe should leave a simple image almost unchanged."""
    src = _gradient_bgr()
    out = apply_recipe(src, Recipe())
    assert out.shape == src.shape
    assert out.dtype == np.uint8
    # Float / color-space round-trips lose a little energy on saturated gradients
    assert _psnr(src, out) >= 28.0
    assert _ssim_simple(src, out) >= 0.92


def test_exposure_recipe_deterministic():
    src = _gradient_bgr()
    r = Recipe()
    r.exposure = 0.5
    a = apply_recipe(src, r)
    b = apply_recipe(src, r)
    assert np.array_equal(a, b)
    # Should differ from source
    assert _psnr(src, a) < 40.0 or not np.array_equal(src, a)


def test_exposure_plus_contrast_golden_floor():
    """Regression floor: same recipe on same synthetic image stays similar to itself."""
    src = _gradient_bgr(80, 120)
    r = Recipe()
    r.exposure = 0.25
    r.contrast = 15.0
    r.shadows = 20.0
    r.highlights = -10.0
    out1 = apply_recipe(src, r)
    out2 = apply_recipe(src.copy(), r)
    assert _psnr(out1, out2) >= 50.0
    assert _ssim_simple(out1, out2) >= 0.99


def test_black_and_white_reduces_chroma():
    src = _gradient_bgr()
    r = Recipe()
    r.black_and_white = True
    out = apply_recipe(src, r)
    # Channels nearly equal
    b, g, rch = cv2.split(out)
    assert abs(int(b.mean()) - int(g.mean())) <= 2
    assert abs(int(g.mean()) - int(rch.mean())) <= 2


def test_crop_changes_shape():
    src = _gradient_bgr(100, 100)
    r = Recipe()
    r.crop = (0.2, 0.2, 0.8, 0.8)
    out = apply_recipe(src, r)
    assert out.shape[0] < src.shape[0]
    assert out.shape[1] < src.shape[1]
