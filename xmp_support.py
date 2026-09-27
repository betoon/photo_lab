"""PhotoLab implementations of imported Adobe settings (not Adobe's renderer)."""
from __future__ import annotations

import numpy as np
import cv2


GRADE_FIELDS = (
    "color_grade_enabled", "split_shadow_hue", "split_shadow_sat", "split_shadow_lum",
    "grade_midtone_hue", "grade_midtone_sat", "grade_midtone_lum",
    "split_highlight_hue", "split_highlight_sat", "split_highlight_lum",
    "grade_global_hue", "grade_global_sat", "grade_global_lum",
    "split_balance", "grade_blending",
)


def apply_master_curve(img, points):
    """Apply an imported RGB master curve to all channels, not Lab lightness."""
    if not points or len(points) < 2:
        return img
    xy = np.asarray(points, dtype=np.float64)
    # Continuous interpolation avoids banding and makes an identity a true no-op.
    return interpolate_curve(img, xy[:, 0], xy[:, 1])


def interpolate_curve(img, xs, ys):
    """Bound np.interp's float64 temporary when exporting large photographs."""
    out = np.empty_like(img, dtype=np.float32)
    for row in range(0, img.shape[0], 64):
        block = np.clip(img[row:row+64], 0, 1)
        out[row:row+64] = np.interp(block, xs, ys)
    return out


def apply_imported_grade(img, recipe):
    """Four-way grading with tonal overlap/balance and luminance controls.

    This is a bounded, luminance-aware approximation. Adobe's proprietary
    color transforms are not available. Called after monochrome conversion.
    """
    if not getattr(recipe, "color_grade_enabled", False):
        return img
    regions = (
        (recipe.split_shadow_hue, recipe.split_shadow_sat, recipe.split_shadow_lum),
        (recipe.grade_midtone_hue, recipe.grade_midtone_sat, recipe.grade_midtone_lum),
        (recipe.split_highlight_hue, recipe.split_highlight_sat, recipe.split_highlight_lum),
    )
    if not any(abs(s) + abs(l) > 1e-8 for _, s, l in regions) and not (
        abs(recipe.grade_global_sat) + abs(recipe.grade_global_lum) > 1e-8
    ):
        return img
    out = np.clip(img, 0, 1).astype(np.float32)
    lum = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    position = np.clip(lum + recipe.split_balance / 200.0, 0, 1)
    power = 4.0 - 3.0 * np.clip(recipe.grade_blending / 100.0, 0, 1)
    weights = np.stack(((1-position)**power, (4*position*(1-position))**power, position**power))
    weights /= np.maximum(weights.sum(axis=0), 1e-8)
    result = out.copy()
    envelope = (4 * lum * (1-lum))[..., None]

    def chroma(hue):
        hsv = np.array([[[float(hue) % 360, 1.0, 1.0]]], dtype=np.float32)
        color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return color - float(np.dot(color, [0.114, 0.587, 0.299]))

    for (hue, sat, light), weight in zip(regions, weights):
        result += weight[..., None] * envelope * chroma(hue) * (sat / 100.0) * 0.55
        result += weight[..., None] * (light / 100.0) * 0.35
    result += envelope * chroma(recipe.grade_global_hue) * (recipe.grade_global_sat / 100.0) * 0.55
    result += (recipe.grade_global_lum / 100.0) * 0.35
    return np.clip(result, 0, 1)


def estimate_auto_tone(img):
    """Return editable tone settings from normalized, WB-corrected source pixels.

    Analyze a bounded proxy once when applying the preset, never during export.
    This is PhotoLab percentile-based Auto Tone, not Adobe Sensei Auto.
    """
    if img is None or img.size == 0:
        raise ValueError("Auto Tone needs an open image")
    if img.ndim != 3 or img.shape[2] != 3 or not np.isfinite(img).all():
        raise ValueError("Auto Tone requires finite BGR image pixels")
    h, w = img.shape[:2]
    if max(h, w) > 512:
        scale = 512 / max(h, w)
        img = cv2.resize(img, (max(1, round(w*scale)), max(1, round(h*scale))), interpolation=cv2.INTER_AREA)
    lum = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
    low, middle, high = np.percentile(lum, [1, 50, 99])
    result = dict(exposure=0.0, contrast=0.0, highlights=0.0, shadows=0.0, whites=0.0, blacks=0.0)
    if high < 1e-5:  # No captured signal to recover.
        return result
    exposure = float(np.clip(np.log2(0.4 / max(middle, 0.02)), -3, 3))
    if exposure > 0:
        exposure = min(exposure, max(0.0, float(np.log2(0.98 / max(high, 0.02))) + 0.5))
    gain = 2.0 ** exposure
    lo, hi = float(low * gain), float(high * gain)
    result["exposure"] = round(exposure, 3)
    result["highlights"] = round(float(np.clip((0.96-hi) / 0.45 * 100, -100, 0)), 2)
    result["shadows"] = round(float(np.clip((0.08-lo) / 0.45 * 100, 0, 30)), 2)
    # Stretch only a genuinely narrow range, not a uniformly lit flat image.
    span = hi-lo
    if 0.1 < span < 0.55:
        result["contrast"] = round(float(np.clip((0.55/span-1)*20, 0, 15)), 2)
        result["whites"] = round(float(np.clip((0.9-hi)*30, 0, 15)), 2)
        result["blacks"] = round(float(np.clip((0.04-lo)*40, -15, 0)), 2)
    return result
