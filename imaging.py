"""
imaging.py — the non-destructive edit pipeline.

Everything here is plain NumPy / OpenCV with no Qt dependency, so it can be
unit-tested or reused (e.g. for a CLI batch-export tool) independently of
the GUI. `Recipe` is the per-image "edit stack"; `apply_recipe` always
re-applies it to the *original* pixels, so nothing here mutates in place.
"""

import numpy as np
import cv2
from dataclasses import dataclass, asdict, field

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


# ----------------------------------------------------------------------
# Adjustment recipe
# ----------------------------------------------------------------------

@dataclass
class Recipe:
    """One image's non-destructive edit stack. All values are DxO-ish
    ranges so the sliders in widgets.py can map 1:1 onto them."""
    # -- Light tab --
    exposure: float = 0.0        # EV, -3..3
    smart_light: float = 0.0     # 0..100 (shadow/highlight recovery)
    contrast: float = 0.0        # -100..100
    highlights: float = 0.0      # -100..100
    shadows: float = 0.0         # -100..100
    saturation: float = 0.0      # -100..100
    clarity: float = 0.0         # -100..100 (local contrast / microcontrast)
    gamma: float = 1.0           # 0.3..2.5 tone curve gamma
    # -- Detail tab --
    denoise_luminance: float = 0.0   # 0..100
    denoise_chroma: float = 0.0      # 0..100
    sharpen_intensity: float = 0.0   # 0..200 (%)
    sharpen_radius: float = 1.0      # 0.1..5.0 px sigma
    sharpen_threshold: float = 0.0   # 0..50
    # -- Geometry tab --
    horizon: float = 0.0         # degrees, -45..45 (straighten)
    distortion: float = 0.0      # -100..100 (- pincushion, + barrel)
    perspective: float = 0.0     # -100..100 (vertical keystone)
    crop: tuple = field(default=None)  # (x0, y0, x1, y1) normalized 0..1, or None
    # -- Effects tab --
    vignette: float = 0.0        # 0..100

    def reset(self):
        for k, v in asdict(Recipe()).items():
            setattr(self, k, v)


# ----------------------------------------------------------------------
# Geometry transforms (operate on uint8 BGR, canvas size preserved except
# for crop, which is always applied last)
# ----------------------------------------------------------------------

def apply_horizon(img: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0.0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_distortion(img: np.ndarray, amount: float) -> np.ndarray:
    """Simple radial (barrel/pincushion) correction. amount > 0 pushes the
    image outward (barrel-style), amount < 0 pulls it inward (pincushion)."""
    if amount == 0.0:
        return img
    h, w = img.shape[:2]
    k = amount / 100.0 * 0.6
    fx, fy = w / 2.0, h / 2.0
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x_norm = (xs - fx) / fx
    y_norm = (ys - fy) / fy
    factor = 1.0 + k * (x_norm ** 2 + y_norm ** 2)
    map_x = x_norm * factor * fx + fx
    map_y = y_norm * factor * fy + fy
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)


def apply_perspective(img: np.ndarray, amount: float) -> np.ndarray:
    """Vertical keystone correction, e.g. for converging building verticals.
    amount > 0 widens the top of the frame, amount < 0 widens the bottom."""
    if amount == 0.0:
        return img
    h, w = img.shape[:2]
    f = max(-0.4, min(0.4, amount / 100.0 * 0.4))
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if f >= 0:
        inset = w * f
        dst = np.float32([[inset, 0], [w - inset, 0], [0, h], [w, h]])
    else:
        inset = w * (-f)
        dst = np.float32([[0, 0], [w, 0], [inset, h], [w - inset, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_crop(img: np.ndarray, crop) -> np.ndarray:
    if crop is None:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = crop
    xi0 = max(0, min(int(round(x0 * w)), w - 2))
    yi0 = max(0, min(int(round(y0 * h)), h - 2))
    xi1 = max(xi0 + 1, min(int(round(x1 * w)), w))
    yi1 = max(yi0 + 1, min(int(round(y1 * h)), h))
    return img[yi0:yi1, xi0:xi1]


# ----------------------------------------------------------------------
# Detail: denoising + sharpening
# ----------------------------------------------------------------------

def apply_denoise(img_bgr: np.ndarray, luminance: float, chroma: float) -> np.ndarray:
    """Fast bilateral-filter noise reduction in LAB space, applied
    separately to luminance and chroma channels. This is a lightweight
    stand-in for DxO's trained PRIME/DeepPRIME denoising engines — fast
    enough for a live preview, but not as clever."""
    if luminance <= 0 and chroma <= 0:
        return img_bgr
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    if luminance > 0:
        sigma = luminance / 100.0 * 12 + 1
        l = cv2.bilateralFilter(l, d=7, sigmaColor=sigma * 3, sigmaSpace=sigma)
    if chroma > 0:
        sigma = chroma / 100.0 * 20 + 2
        a = cv2.bilateralFilter(a, d=7, sigmaColor=sigma * 3, sigmaSpace=sigma)
        b = cv2.bilateralFilter(b, d=7, sigmaColor=sigma * 3, sigmaSpace=sigma)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_sharpen(img_float: np.ndarray, intensity: float, radius: float, threshold: float) -> np.ndarray:
    """Classic unsharp mask on a 0..1 float BGR image. `threshold` (0..50,
    treated as an 8-bit delta) suppresses sharpening in flat/noisy areas
    below that edge strength."""
    if intensity <= 0:
        return img_float
    blur = cv2.GaussianBlur(img_float, (0, 0), sigmaX=max(radius, 0.1))
    diff = img_float - blur
    if threshold > 0:
        mag = np.abs(diff).max(axis=2, keepdims=True)
        mask = (mag > (threshold / 255.0)).astype(np.float32)
        diff = diff * mask
    return img_float + diff * (intensity / 100.0)


# ----------------------------------------------------------------------
# Full pipeline
# ----------------------------------------------------------------------

def apply_recipe(img_bgr: np.ndarray, r: Recipe) -> np.ndarray:
    """Apply a Recipe to a full-precision image and return uint8 BGR."""
    # Geometry first (distortion/perspective/horizon keep the canvas size,
    # so the crop's normalized coordinates stay valid regardless of order).
    img_bgr = apply_distortion(img_bgr, r.distortion)
    img_bgr = apply_perspective(img_bgr, r.perspective)
    img_bgr = apply_horizon(img_bgr, r.horizon)
    img_bgr = apply_crop(img_bgr, r.crop)

    # Denoise on uint8, before tonal work amplifies any noise.
    img_bgr = apply_denoise(img_bgr, r.denoise_luminance, r.denoise_chroma)

    img = img_bgr.astype(np.float32) / 255.0

    # Exposure (stops)
    if r.exposure != 0.0:
        img *= (2.0 ** r.exposure)

    # Smart lighting: cheap local shadow/highlight recovery via a blurred
    # luminance mask (mimics DxO Smart Lighting's tone-mapping idea).
    if r.smart_light != 0.0:
        lum = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(lum, (0, 0), sigmaX=img.shape[1] / 20)
        amt = r.smart_light / 100.0
        lift = (0.5 - blur) * amt * 0.6
        img += lift[..., None]

    # Highlights / shadows (simple tonal masks)
    if r.highlights != 0.0 or r.shadows != 0.0:
        lum = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2GRAY)
        hi_mask = np.clip((lum - 0.5) * 2, 0, 1) ** 1.5
        lo_mask = np.clip((0.5 - lum) * 2, 0, 1) ** 1.5
        img += (r.highlights / 100.0) * 0.5 * hi_mask[..., None]
        img += (r.shadows / 100.0) * 0.5 * lo_mask[..., None]

    # Contrast around mid-grey
    if r.contrast != 0.0:
        c = r.contrast / 100.0
        img = (img - 0.5) * (1.0 + c) + 0.5

    # Clarity / microcontrast: unsharp mask on the whole image with a
    # wide radius
    if r.clarity != 0.0:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        img = img + (img - blur) * (r.clarity / 100.0)

    # Sharpening: classic small-radius unsharp mask
    img = apply_sharpen(img, r.sharpen_intensity, r.sharpen_radius, r.sharpen_threshold)

    # Gamma (tone curve)
    img = np.clip(img, 0, 1)
    if r.gamma != 1.0:
        img = img ** (1.0 / r.gamma)

    # Saturation via HSV
    if r.saturation != 0.0:
        hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + r.saturation / 100.0), 0, 1)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Vignette
    if r.vignette != 0.0:
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2, h / 2
        d = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        d = np.clip(d, 0, 1.4) / 1.4
        strength = r.vignette / 100.0
        mask = 1.0 - strength * (d ** 2)
        img *= mask[..., None]

    img = np.clip(img, 0, 1) * 255.0
    return img.astype(np.uint8)
