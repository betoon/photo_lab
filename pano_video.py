"""
Panorama to Video - Brian E. Toon, 2026, V2.6
"""
import os
import time
import json
import uuid
import threading
import subprocess
import dataclasses
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw
Image.MAX_IMAGE_PIXELS = None

# ---------------------------------------------------------------------------
# Feature #1: detect available hardware-accelerated encoders at startup so the
# codec dropdown only shows options the current machine can actually use.
# ---------------------------------------------------------------------------
_HW_ENCODER_CANDIDATES = [
    ("h264_nvenc",        "H.264 NVENC (NVIDIA GPU)"),
    ("hevc_nvenc",        "H.265 NVENC (NVIDIA GPU)"),
    ("h264_videotoolbox", "H.264 VideoToolbox (Apple)"),
    ("hevc_videotoolbox", "H.265 VideoToolbox (Apple)"),
    ("h264_amf",          "H.264 AMF (AMD GPU)"),
    ("hevc_amf",          "H.265 AMF (AMD GPU)"),
]

def _probe_available_encoders() -> list[tuple[str, str]]:
    """Return [(codec_id, label), …] for every encoder ffmpeg can use right now.
    Always includes the two CPU fallbacks first."""
    available = [("libx264", "H.264 (CPU)"), ("libx265", "H.265 (CPU)")]
    try:
        si = None
        if os.name == "nt":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            startupinfo=si, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, timeout=8,
        ).stdout
        for codec_id, label in _HW_ENCODER_CANDIDATES:
            if codec_id in out:
                available.append((codec_id, label))
    except Exception:
        pass
    return available

AVAILABLE_ENCODERS: list[tuple[str, str]] = _probe_available_encoders()

def _ease_smoothstep(t):
    return t * t * (3 - 2 * t)

def _ease_cubic(t):
    # Genuine ease-in-out cubic -- steeper acceleration/deceleration than Smoothstep,
    # a distinct curve rather than an algebraic restatement of it.
    return 4 * t ** 3 if t < 0.5 else 1 - ((-2 * t + 2) ** 3) / 2

def _ease_exponential(t):
    # Pure ease-in: slow, deliberate start that accelerates hard into the finish.
    return t ** 3

def _ease_bounce(t):
    # Genuine ease-out bounce: overshoots past 1.0 and settles with diminishing bounces.
    n1, d1 = 7.5625, 2.75
    x = t
    if x < 1 / d1:
        return n1 * x * x
    elif x < 2 / d1:
        x -= 1.5 / d1
        return n1 * x * x + 0.75
    elif x < 2.5 / d1:
        x -= 2.25 / d1
        return n1 * x * x + 0.9375
    else:
        x -= 2.625 / d1
        return n1 * x * x + 0.984375

def _ease_elastic(t):
    # Genuine ease-out elastic: springy overshoot that decays to a clean stop at 1.0
    # (bounded on [0,1] input; previous version could return large negative/positive
    # values and land back at 0 instead of 1 at the end of the clip).
    if t <= 0.0:
        return 0.0
    if t >= 1.0:
        return 1.0
    c4 = (2 * np.pi) / 3
    return (2 ** (-10 * t)) * np.sin((t * 10 - 0.75) * c4) + 1

EASING_CURVES = {
    "Smoothstep": _ease_smoothstep,
    "Cubic": _ease_cubic,
    "Exponential": _ease_exponential,
    "Bounce": _ease_bounce,
    "Elastic": _ease_elastic,
}

# Fix #13: capped dict used as an LRU-style cache – prevents unbounded memory growth
# when many custom .cube files are loaded in a single session.
_LUT_CACHE: dict = {}
_LUT_CACHE_MAX = 32

FILM_LOOK_CHOICES = ["None", "Leica Look", "Panavision", "Kodak Vision3", "Old Film", "8mm Film", "High Contrast B&W", "Dreamy Soft"]
FILM_EFFECT_CHOICES = ["None", "Subtle Grain", "Dust & Scratches", "Heavy Old Projector"]

def _get_cinematic_lut(look):
    if look in _LUT_CACHE:
        return _LUT_CACHE[look]
    
    identity = np.arange(256, dtype=np.float32) / 255.0
    lut = np.zeros((1, 256, 3), dtype=np.uint8)
    
    if look == "Panavision":
        # Filmic S-curve for punch/depth, plus teal-shadow / warm-highlight split toning --
        # the classic "Hollywood blockbuster" anamorphic color science, rather than a flat tint.
        s_curve = np.clip(identity + 0.18 * (identity - 0.5) * (1 - np.abs(2 * identity - 1)), 0, 1)
        base = np.clip(s_curve * 0.985 - 0.012, 0, 1)  # gentle black deepening instead of lifting
        shadow_w = np.clip(1.0 - identity * 1.6, 0, 1)
        highlight_w = np.clip((identity - 0.4) * 1.6, 0, 1)
        r = np.clip(base - shadow_w * 0.035 + highlight_w * 0.05, 0, 1)
        g = np.clip(base + highlight_w * 0.015, 0, 1)
        b = np.clip(base + shadow_w * 0.06 - highlight_w * 0.03, 0, 1)
        lut[0, :, 0] = (b * 255).astype(np.uint8)
        lut[0, :, 1] = (g * 255).astype(np.uint8)
        lut[0, :, 2] = (r * 255).astype(np.uint8)
        
    elif look == "Kodak Vision3":
        lut_float = np.clip(identity * 0.95 + 0.03, 0, 1)
        lut[0, :, 0] = (lut_float * 255).astype(np.uint8)
        lut[0, :, 1] = (np.clip(lut_float * 1.04, 0, 1) * 255).astype(np.uint8)
        lut[0, :, 2] = (np.clip(lut_float * 1.07, 0, 1) * 255).astype(np.uint8)
        
    elif look == "Old Film":
        lut_float = np.clip(identity * 0.88 + 0.03, 0, 1)
        lut[0, :, 0] = (np.clip(lut_float * 0.86, 0, 1) * 255).astype(np.uint8)
        lut[0, :, 1] = (np.clip(lut_float * 0.96, 0, 1) * 255).astype(np.uint8)
        lut[0, :, 2] = (np.clip(lut_float * 1.12, 0, 1) * 255).astype(np.uint8)

    elif look == "Leica Look":
        leica_curve = np.clip(1.0 / (1.0 + np.exp(-6.5 * (identity - 0.48))), 0, 1)
        lut_float = identity * 0.3 + leica_curve * 0.7
        lut[0, :, 0] = (np.clip(lut_float * 0.98, 0, 1) * 255).astype(np.uint8)
        lut[0, :, 1] = (lut_float * 255).astype(np.uint8)
        lut[0, :, 2] = (np.clip(lut_float * 1.02, 0, 1) * 255).astype(np.uint8)
        
    elif look == "High Contrast B&W":
        lut_float = np.clip((identity ** 1.2) * 1.25, 0, 1)
        val = (lut_float * 255).astype(np.uint8)
        lut[0, :, 0] = val
        lut[0, :, 1] = val
        lut[0, :, 2] = val
        
    elif look == "8mm Film":
        # Punchier contrast + warm amber cast (Super 8 stock), distinct from Old Film's cooler silver fade
        contrast_curve = np.clip(0.5 + (identity - 0.5) * 1.18, 0, 1)
        lut_float = np.clip(contrast_curve * 0.96 + 0.02, 0, 1)
        lut[0, :, 0] = (np.clip(lut_float * 0.70, 0, 1) * 255).astype(np.uint8)
        lut[0, :, 1] = (np.clip(lut_float * 1.02, 0, 1) * 255).astype(np.uint8)
        lut[0, :, 2] = (np.clip(lut_float * 1.30, 0, 1) * 255).astype(np.uint8)
        
    else:
        lut = None

    if len(_LUT_CACHE) >= _LUT_CACHE_MAX:
        # Evict the oldest entry to keep memory bounded.
        _LUT_CACHE.pop(next(iter(_LUT_CACHE)))
    _LUT_CACHE[look] = lut
    return lut

def _generate_organic_grain(shape, intensity=8):
    h, w = shape[:2]
    small_h, small_w = max(1, h // 2), max(1, w // 2)
    noise = np.random.normal(0, intensity, (small_h, small_w, 3)).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (3, 3), 0)
    grain = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
    return grain

def parse_cube_lut(path):
    """Parses a .cube LUT file (1D or 3D). Returns a dict describing the LUT."""
    size_3d = None
    size_1d = None
    domain_min = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    domain_max = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    rows = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            upper = line.upper()
            if upper.startswith('TITLE'):
                continue
            elif upper.startswith('LUT_3D_SIZE'):
                size_3d = int(line.split()[-1])
            elif upper.startswith('LUT_1D_SIZE'):
                size_1d = int(line.split()[-1])
            elif upper.startswith('DOMAIN_MIN'):
                domain_min = np.array(list(map(float, line.split()[1:4])), dtype=np.float32)
            elif upper.startswith('DOMAIN_MAX'):
                domain_max = np.array(list(map(float, line.split()[1:4])), dtype=np.float32)
            else:
                parts = line.split()
                if len(parts) == 3:
                    try:
                        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        pass

    if size_3d is None and size_1d is None:
        raise ValueError("File does not look like a valid .cube LUT (no LUT_3D_SIZE or LUT_1D_SIZE found).")

    table = np.array(rows, dtype=np.float32)

    if size_3d is not None:
        expected = size_3d ** 3
        if table.shape[0] != expected:
            raise ValueError(f"Cube file expects {expected} data rows for LUT_3D_SIZE {size_3d}, found {table.shape[0]}.")
        # File order: R varies fastest, then G, then B.
        grid = table.reshape((size_3d, size_3d, size_3d, 3))  # axes: (B, G, R, channel)
        return {"type": "3d", "size": size_3d, "grid": grid, "domain_min": domain_min, "domain_max": domain_max}
    else:
        if table.shape[0] != size_1d:
            raise ValueError(f"Cube file expects {size_1d} data rows for LUT_1D_SIZE, found {table.shape[0]}.")
        return {"type": "1d", "size": size_1d, "table": table, "domain_min": domain_min, "domain_max": domain_max}

def apply_cube_lut_3d(frame_bgr, lut):
    grid = lut["grid"]
    size = lut["size"]
    domain_min = lut["domain_min"]
    domain_max = lut["domain_max"]

    frame_f = frame_bgr.astype(np.float32) / 255.0
    b_ch, g_ch, r_ch = frame_f[..., 0], frame_f[..., 1], frame_f[..., 2]

    span = np.clip(domain_max - domain_min, 1e-6, None)
    rn = np.clip((r_ch - domain_min[0]) / span[0], 0, 1)
    gn = np.clip((g_ch - domain_min[1]) / span[1], 0, 1)
    bn = np.clip((b_ch - domain_min[2]) / span[2], 0, 1)

    scale = size - 1
    rf, gf, bf = rn * scale, gn * scale, bn * scale
    r0 = np.floor(rf).astype(np.int32); r1 = np.clip(r0 + 1, 0, size - 1); r0 = np.clip(r0, 0, size - 1)
    g0 = np.floor(gf).astype(np.int32); g1 = np.clip(g0 + 1, 0, size - 1); g0 = np.clip(g0, 0, size - 1)
    b0 = np.floor(bf).astype(np.int32); b1 = np.clip(b0 + 1, 0, size - 1); b0 = np.clip(b0, 0, size - 1)
    rd = (rf - r0)[..., None]; gd = (gf - g0)[..., None]; bd = (bf - b0)[..., None]

    def gv(bi, gi, ri):
        return grid[bi, gi, ri]

    c00 = gv(b0, g0, r0) * (1 - rd) + gv(b0, g0, r1) * rd
    c10 = gv(b0, g1, r0) * (1 - rd) + gv(b0, g1, r1) * rd
    c01 = gv(b1, g0, r0) * (1 - rd) + gv(b1, g0, r1) * rd
    c11 = gv(b1, g1, r0) * (1 - rd) + gv(b1, g1, r1) * rd
    c0 = c00 * (1 - gd) + c10 * gd
    c1 = c01 * (1 - gd) + c11 * gd
    out_rgb = np.clip(c0 * (1 - bd) + c1 * bd, 0, 1)

    out_bgr = out_rgb[..., ::-1]
    return (out_bgr * 255).astype(np.uint8)

def apply_cube_lut_1d(frame_bgr, lut):
    table = lut["table"]
    size = lut["size"]
    xs = np.linspace(0, 255, size)
    r_curve = np.interp(np.arange(256), xs, table[:, 0] * 255.0)
    g_curve = np.interp(np.arange(256), xs, table[:, 1] * 255.0)
    b_curve = np.interp(np.arange(256), xs, table[:, 2] * 255.0)
    lut256 = np.zeros((1, 256, 3), dtype=np.uint8)
    lut256[0, :, 0] = np.clip(b_curve, 0, 255).astype(np.uint8)
    lut256[0, :, 1] = np.clip(g_curve, 0, 255).astype(np.uint8)
    lut256[0, :, 2] = np.clip(r_curve, 0, 255).astype(np.uint8)
    return cv2.LUT(frame_bgr, lut256)

def apply_custom_cube_lut(frame_bgr, lut):
    if lut["type"] == "3d":
        return apply_cube_lut_3d(frame_bgr, lut)
    return apply_cube_lut_1d(frame_bgr, lut)


def force_monochrome(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
def apply_cinematic_look(frame, look="None", frame_idx=0, custom_lut=None):
    if look == "Custom LUT":
        if custom_lut is None:
            return frame
        return apply_custom_cube_lut(frame, custom_lut)
    if look == "None":
        return frame
    h, w, c = frame.shape

    if look == "Panavision":
        lut = _get_cinematic_lut(look)
        frame = cv2.LUT(frame, lut)

        # Stylized anamorphic-style horizontal highlight streak (the signature blue flare
        # that horizontally-elongated anamorphic glass produces off bright specular light).
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, highlights = cv2.threshold(gray, 235, 255, cv2.THRESH_BINARY)
        if cv2.countNonZero(highlights) > 0:
            streak = cv2.GaussianBlur(highlights, (101, 1), 0)
            streak = cv2.GaussianBlur(streak, (151, 1), 0)
            streak_norm = (streak.astype(np.float32) / 255.0) * 0.35
            flare = np.zeros_like(frame, dtype=np.float32)
            flare[..., 0] = streak_norm * 255 * 1.0   # blue channel dominant
            flare[..., 1] = streak_norm * 255 * 0.55
            flare[..., 2] = streak_norm * 255 * 0.25
            frame = np.clip(frame.astype(np.float32) + flare, 0, 255).astype(np.uint8)
        return frame

    elif look == "8mm Film":
        lut = _get_cinematic_lut(look)
        frame = cv2.LUT(frame, lut)
        grain = _generate_organic_grain(frame.shape, intensity=10)
        frame = np.clip(frame.astype(np.float32) + grain, 0, 255).astype(np.uint8)

        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max_dist
        vignette = np.clip(1.0 - 0.38 * (dist ** 2), 0.55, 1.0).astype(np.float32)
        frame = (frame.astype(np.float32) * vignette[..., None]).astype(np.uint8)
        return frame

    elif look == "Old Film":
        lut = _get_cinematic_lut(look)
        frame = cv2.LUT(frame, lut)
        grain = _generate_organic_grain(frame.shape, intensity=7)
        frame = np.clip(frame.astype(np.float32) + grain, 0, 255).astype(np.uint8)
        return frame

    elif look == "Dreamy Soft":
        frame_float = frame.astype(np.float32) * 0.85 + 20.4
        blurred = cv2.GaussianBlur(frame, (21, 21), 0).astype(np.float32)
        frame = cv2.addWeighted(frame_float, 0.75, blurred, 0.25, 0)
        return np.clip(frame, 0, 255).astype(np.uint8)

    elif look == "High Contrast B&W":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        lut = _get_cinematic_lut(look)
        gray = cv2.LUT(gray, lut[0, :, 0])
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    lut = _get_cinematic_lut(look)
    if lut is not None:
        return cv2.LUT(frame, lut)
    return frame


def apply_film_effect(frame, effect="None", frame_idx=0, strength=0.0,
                      grain_strength=None, dust_strength=None, scratch_strength=None, flicker_strength=None):
    manual_controls = any(v is not None and float(v) > 0 for v in (grain_strength, dust_strength, scratch_strength, flicker_strength))
    strength = float(np.clip(strength, 0.0, 1.0))
    if not manual_controls and (effect == "None" or strength <= 0):
        return frame

    if manual_controls:
        grain_level = float(np.clip(grain_strength or 0.0, 0.0, 1.0))
        dust_level = float(np.clip(dust_strength or 0.0, 0.0, 1.0))
        scratch_level = float(np.clip(scratch_strength or 0.0, 0.0, 1.0))
        flicker_level = float(np.clip(flicker_strength or 0.0, 0.0, 1.0))
    elif effect == "Subtle Grain":
        grain_level, dust_level, scratch_level, flicker_level = 0.35 + 0.65 * strength, 0.0, 0.0, 0.0
    elif effect == "Dust & Scratches":
        grain_level, dust_level, scratch_level, flicker_level = 0.25 + 0.45 * strength, 0.35 + 0.65 * strength, 0.25 + 0.55 * strength, 0.0
    elif effect == "Heavy Old Projector":
        grain_level, dust_level, scratch_level, flicker_level = 0.55 + 0.45 * strength, 0.65 + 0.35 * strength, 0.55 + 0.40 * strength, 0.45 + 0.55 * strength
    else:
        return frame

    h, w = frame.shape[:2]
    rng = np.random.default_rng((frame_idx + 1) * 7919)
    out = frame.astype(np.float32)

    if grain_level > 0:
        grain_amount = 4 + int(26 * grain_level)
        grain = _generate_organic_grain(frame.shape, intensity=grain_amount)
        out = np.clip(out + grain, 0, 255)

    if scratch_level > 0 and rng.random() < (0.15 + scratch_level * 0.75):
        scratch_count = 1 + int(scratch_level * 5)
        for _ in range(scratch_count):
            x = int(rng.integers(8, max(9, w - 8)))
            drift = int(rng.integers(-5, 6))
            shade = int(rng.integers(30, 105))
            cv2.line(out, (x, 0), (max(0, min(w - 1, x + drift)), h), (shade, shade, shade), 1, cv2.LINE_AA)

    if dust_level > 0:
        speck_count = int((5 + dust_level * 70) * (w * h / (1280 * 720)))
        for _ in range(max(1, speck_count)):
            x = int(rng.integers(0, w))
            y = int(rng.integers(0, h))
            radius = int(rng.integers(1, 2 + max(1, int(3 * dust_level))))
            color = int(rng.integers(190, 245)) if rng.random() < 0.55 else int(rng.integers(8, 55))
            cv2.circle(out, (x, y), radius, (color, color, color), -1, cv2.LINE_AA)

    if flicker_level > 0:
        flicker = 1.0 + np.sin(frame_idx * 0.73) * 0.04 * flicker_level + float(rng.normal(0, 0.03 * flicker_level))
        out *= flicker
        if rng.random() < 0.08 + 0.30 * flicker_level:
            band_y = int(rng.integers(0, h))
            band_h = int(rng.integers(max(2, h // 80), max(3, h // 25)))
            out[band_y:min(h, band_y + band_h), :, :] *= 0.78 + 0.14 * (1.0 - flicker_level)

    return np.clip(out, 0, 255).astype(np.uint8)

# Fix #5: cache the vignette mask keyed on (h, w, strength) so it is only computed
# once per unique combination rather than on every frame during a render.
_VIGNETTE_CACHE: dict = {}

def apply_vignette(frame, strength=0.0):
    if strength <= 0:
        return frame
    h, w = frame.shape[:2]
    key = (h, w, round(strength, 4))
    if key not in _VIGNETTE_CACHE:
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max_dist
        _VIGNETTE_CACHE[key] = np.clip(1.0 - strength * (dist ** 2), 0.0, 1.0).astype(np.float32)
    vignette = _VIGNETTE_CACHE[key]
    return (frame.astype(np.float32) * vignette[..., None]).astype(np.uint8)


# Fix #12: use exact string sets matching the combo box values in _build_widgets so
# that a new style name can never accidentally partial-match a wrong branch.
_STYLE_SMOOTHSTEP = {"Cinematic Drift", "3D Drift", "Slow Zoom In"}
_STYLE_DRAMATIC   = {"Dramatic Push"}
_STYLE_SUBTLE     = {"Subtle Push"}

def ken_burns_zoom_scale(ken_style, zoom_amount, t):
    if ken_style in _STYLE_SMOOTHSTEP:
        return 1.0 + zoom_amount * (3 * t**2 - 2 * t**3)
    if ken_style in _STYLE_DRAMATIC:
        return 1.0 + zoom_amount * (t ** 0.6)
    if ken_style in _STYLE_SUBTLE:
        return 1.0 + zoom_amount * 0.55 * t
    if ken_style == "Zoom Out":
        return 1.0 + zoom_amount * (1.0 - t)
    return 1.0 + zoom_amount * t
def _render_crop_frame(img, img_w, img_h, aspect_ratio, zoom_scale, pan_t, out_w, out_h, motion_blur=0.0, is_preview=False, start_focus=None, start_focus_weight=0.0, end_focus=None, focus_t=None):
    crop_h = img_h / zoom_scale
    crop_w = crop_h * aspect_ratio
    if crop_w > img_w:
        crop_w = img_w
        crop_h = crop_w / aspect_ratio
    max_x = img_w - crop_w
    max_y = img_h - crop_h
    crop_x = pan_t * max_x
    crop_y = max_y / 2.0

    def _crop_for_focus(focus):
        focus_x, focus_y = focus
        fx = (focus_x * img_w) - (crop_w / 2.0)
        fy = (focus_y * img_h) - (crop_h / 2.0)
        return max(0.0, min(fx, max_x)), max(0.0, min(fy, max_y))

    if start_focus is not None and end_focus is not None:
        # Intro zooms use their own focus transition; ordinary pan motion still uses pan_t.
        focus_progress = pan_t if focus_t is None else float(np.clip(focus_t, 0.0, 1.0))
        focus_x = start_focus[0] * (1.0 - focus_progress) + end_focus[0] * focus_progress
        focus_y = start_focus[1] * (1.0 - focus_progress) + end_focus[1] * focus_progress
        crop_x, crop_y = _crop_for_focus((focus_x, focus_y))
    elif start_focus is not None and start_focus_weight > 0:
        focus_crop_x, focus_crop_y = _crop_for_focus(start_focus)
        weight = float(np.clip(start_focus_weight, 0.0, 1.0))
        crop_x = focus_crop_x * weight + crop_x * (1.0 - weight)
        crop_y = focus_crop_y * weight + crop_y * (1.0 - weight)
    elif end_focus is not None:
        focus_crop_x, focus_crop_y = _crop_for_focus(end_focus)
        weight = float(np.clip(pan_t, 0.0, 1.0))
        crop_x = crop_x * (1.0 - weight) + focus_crop_x * weight
        crop_y = crop_y * (1.0 - weight) + focus_crop_y * weight

    x0 = max(0, min(int(round(crop_x)), img_w-1))
    y0 = max(0, min(int(round(crop_y)), img_h-1))
    x1 = min(max(x0+1, int(round(crop_x + crop_w))), img_w)
    y1 = min(max(y0+1, int(round(crop_y + crop_h))), img_h)
    crop = img[y0:y1, x0:x1]
    
    # Use spatial area downsampling for structural sharpness during scaling shifts
    interp = cv2.INTER_AREA if (is_preview or out_w < (x1 - x0)) else cv2.INTER_LANCZOS4
    frame = cv2.resize(crop, (out_w, out_h), interpolation=interp)

    if motion_blur > 0:
        kernel = np.array([[0, 0, 0], [1, 5, 1], [0, 0, 0]], dtype=np.float32) / 7.0
        blurred = cv2.filter2D(frame, -1, kernel)
        frame = cv2.addWeighted(frame, 1.0 - motion_blur, blurred, motion_blur, 0)
    return frame
SETTINGS_PATH = os.path.join(os.path.expanduser("~"), ".pano_video_settings.json")

class RenderCancelled(Exception):
    """Raised internally when the user cancels an in-progress render."""
    pass


# ---------------------------------------------------------------------------
# Fix #7: collect all render parameters into a dataclass so that build_pan_video
# takes one typed object instead of 30+ positional arguments.  A wrong ordering
# at the call site now produces a clear AttributeError rather than silent misuse.
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class RenderSettings:
    duration_sec: float
    fps: int
    out_height: int
    aspect_w: float
    aspect_h: float
    use_ken_burns: bool
    zoom_amount: float
    ken_style: str = "Slow Zoom In"
    easing: str = "Smoothstep"
    use_intro_zoom: bool = False
    intro_duration_sec: float = 2.0
    intro_zoom_level: float = 1.8
    reverse_pan: bool = False
    fade_in_sec: float = 1.0
    fade_out_sec: float = 1.0
    video_fade_in_sec: float = 1.0
    video_fade_out_sec: float = 1.0
    audio_path: str = None
    audio_bitrate: str = "192k"
    crf: int = 23
    motion_blur: float = 0.0
    cinematic_look: str = "None"
    codec: str = "libx264"
    custom_lut_path: str = None
    vignette_amount: float = 0.0
    look_strength: float = 1.0
    film_effect: str = "None"
    film_effect_strength: float = 0.0
    start_focus: tuple = None
    end_focus: tuple = None
    intro_focus: tuple = None
    film_grain: float = 0.0
    film_dust: float = 0.0
    film_scratches: float = 0.0
    film_flicker: float = 0.0


# ---------------------------------------------------------------------------
# Fix #9: single shared function used by both the render pipeline and the GUI
# preview so the look/vignette/film-effect order is always identical.
# ---------------------------------------------------------------------------
def apply_look_pipeline(frame, frame_idx, *,
                        cinematic_look, custom_lut,
                        look_strength, vignette_amount,
                        film_effect, film_effect_strength,
                        film_grain, film_dust, film_scratches, film_flicker):
    """Apply color look → strength blend → vignette → film effects in the
    correct order.  Both the render worker and the GUI preview call this."""
    looked = apply_cinematic_look(frame, cinematic_look, frame_idx=frame_idx, custom_lut=custom_lut)
    if cinematic_look != "None" and look_strength < 1.0:
        blend_source = force_monochrome(frame) if cinematic_look == "High Contrast B&W" else frame
        looked = cv2.addWeighted(looked, look_strength, blend_source, 1.0 - look_strength, 0)
    if vignette_amount > 0:
        looked = apply_vignette(looked, vignette_amount)
    looked = apply_film_effect(
        looked, film_effect, frame_idx, film_effect_strength,
        film_grain, film_dust, film_scratches, film_flicker,
    )
    if cinematic_look == "High Contrast B&W":
        looked = force_monochrome(looked)
    return looked


def get_audio_duration(audio_path):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
    try:
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        res = subprocess.run(cmd, startupinfo=startupinfo, stdout=subprocess.PIPE, text=True, check=True)
        return float(res.stdout.strip())
    except Exception:  # Fix #3: don't swallow SystemExit / KeyboardInterrupt
        return None

def build_pan_video(image_path, output_path, settings: RenderSettings,
                    progress_callback=None, cancel_event=None):
    # Fix #7: unpack RenderSettings so the rest of the function body is unchanged.
    duration_sec       = settings.duration_sec
    fps                = settings.fps
    out_height         = settings.out_height
    aspect_w           = settings.aspect_w
    aspect_h           = settings.aspect_h
    use_ken_burns      = settings.use_ken_burns
    zoom_amount        = settings.zoom_amount
    ken_style          = settings.ken_style
    easing             = settings.easing
    use_intro_zoom     = settings.use_intro_zoom
    intro_duration_sec = settings.intro_duration_sec
    intro_zoom_level   = settings.intro_zoom_level
    reverse_pan        = settings.reverse_pan
    fade_in_sec        = settings.fade_in_sec
    fade_out_sec       = settings.fade_out_sec
    video_fade_in_sec  = settings.video_fade_in_sec
    video_fade_out_sec = settings.video_fade_out_sec
    audio_path         = settings.audio_path
    audio_bitrate      = settings.audio_bitrate
    crf                = settings.crf
    motion_blur        = settings.motion_blur
    cinematic_look     = settings.cinematic_look
    codec              = settings.codec
    custom_lut_path    = settings.custom_lut_path
    vignette_amount    = settings.vignette_amount
    look_strength      = settings.look_strength
    film_effect        = settings.film_effect
    film_effect_strength = settings.film_effect_strength
    start_focus        = settings.start_focus
    end_focus          = settings.end_focus
    intro_focus        = settings.intro_focus
    film_grain         = settings.film_grain
    film_dust          = settings.film_dust
    film_scratches     = settings.film_scratches
    film_flicker       = settings.film_flicker

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    img_h, img_w = img.shape[:2]
    aspect_ratio = aspect_w / aspect_h

    custom_lut_data = None
    if cinematic_look == "Custom LUT" and custom_lut_path:
        custom_lut_data = parse_cube_lut(custom_lut_path)

    # Fix #9: delegate to the shared pipeline so render and preview always match.
    def _look(frame, idx):
        return apply_look_pipeline(
            frame, idx,
            cinematic_look=cinematic_look, custom_lut=custom_lut_data,
            look_strength=look_strength, vignette_amount=vignette_amount,
            film_effect=film_effect, film_effect_strength=film_effect_strength,
            film_grain=film_grain, film_dust=film_dust,
            film_scratches=film_scratches, film_flicker=film_flicker,
        )
    
    out_w = int(round(out_height * aspect_ratio))
    if out_w % 2 != 0: out_w += 1
    if out_height % 2 != 0: out_height += 1

    intro_frames = max(1, int(round(intro_duration_sec * fps))) if use_intro_zoom else 0
    main_frames = max(1, int(round(duration_sec * fps)))
    total_frames = intro_frames + main_frames

    # Fix #11: append a short UUID so two concurrent renders of the same source image
    # never write to the same temp file and corrupt each other's output.
    base_name, _ = os.path.splitext(output_path)
    temp_master_path = base_name + f"_temp_master_{uuid.uuid4().hex[:8]}.mp4"
    
    writer = cv2.VideoWriter(temp_master_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (out_w, out_height))
    if not writer.isOpened():
        raise RuntimeError("Video writer failed to initialize frame pipeline buffer.")

    easing_func = EASING_CURVES.get(easing, EASING_CURVES["Smoothstep"])

    frame_counter = 0
    total_motion_frames = max(1, total_frames - 1)

    def _check_cancelled():
        if cancel_event is not None and cancel_event.is_set():
            writer.release()
            if os.path.exists(temp_master_path):
                try: os.remove(temp_master_path)
                except Exception: pass  # Fix #3
            raise RenderCancelled("Render cancelled by user.")

    def _apply_fade(frame, fc):
        # Apply visual fade after color/film effects so the first frame stays truly black.
        # Uses the video fade duration(s), independent of the audio fade below.
        alpha = 1.0
        # A deep intro must be visible from its very first frame.  Otherwise the
        # default fade-to-black hides the close-up and leaves only a brief glimpse
        # before the normal wide framing begins.
        if video_fade_in_sec > 0 and not (intro_focus is not None and fc < intro_frames):
            fade_in_frames = max(1.0, video_fade_in_sec * fps)
            if fc < fade_in_frames:
                alpha = min(alpha, (fc / fade_in_frames) ** 1.15)
        if video_fade_out_sec > 0:
            fade_out_frames = max(1.0, video_fade_out_sec * fps)
            fade_start = total_frames - fade_out_frames
            if fc >= fade_start:
                fade_progress = max(0.0, (total_frames - 1 - fc) / fade_out_frames)
                alpha = min(alpha, fade_progress ** 1.25)
        if alpha < 1.0:
            frame = cv2.addWeighted(frame, alpha, np.zeros_like(frame), 1 - alpha, 0)
        return frame
    for i in range(intro_frames):
        _check_cancelled()
        intro_t = easing_func(i / (intro_frames - 1) if intro_frames > 1 else 1.0)
        # The intro is a lead-in.  It should not consume the regular Start-to-End motion.
        global_t = 0.0
        base_zoom = ken_burns_zoom_scale(ken_style, zoom_amount, global_t) if use_ken_burns else 1.0
        zoom_scale = base_zoom + max(0.0, intro_zoom_level - base_zoom) * (1.0 - intro_t)
        pan_t = (1.0 - global_t) if reverse_pan else global_t
        # A Deep Intro click is independent from the normal Start click.  At the end of
        # the intro, land on the normal start framing (or the default left/center frame).
        deep_focus = intro_focus or start_focus or (0.0, 0.5)
        intro_target_focus = start_focus or (0.0, 0.5)
        frame = _render_crop_frame(
            img, img_w, img_h, aspect_ratio, zoom_scale, pan_t, out_w, out_height,
            motion_blur, is_preview=False, start_focus=deep_focus, end_focus=intro_target_focus,
            focus_t=intro_t
        )
        frame = _look(frame, frame_counter)
        frame = _apply_fade(frame, frame_counter)
        writer.write(frame)
        frame_counter += 1
        if progress_callback: progress_callback((frame_counter / total_frames) * 0.85)

    for i in range(main_frames):
        _check_cancelled()
        global_t = easing_func(i / (main_frames - 1) if main_frames > 1 else 1.0)

        if use_ken_burns:
            zoom_scale = ken_burns_zoom_scale(ken_style, zoom_amount, global_t)
        else:
            zoom_scale = 1.0

        pan_t = (1.0 - global_t) if reverse_pan else global_t
        frame = _render_crop_frame(
            img, img_w, img_h, aspect_ratio, zoom_scale, pan_t, out_w, out_height,
            motion_blur, is_preview=False, start_focus=start_focus, start_focus_weight=1.0 - global_t, end_focus=end_focus
        )

        frame = _look(frame, frame_counter)
        frame = _apply_fade(frame, frame_counter)
        writer.write(frame)
        frame_counter += 1
        if progress_callback: progress_callback((frame_counter / total_frames) * 0.85)

    writer.release()

    cmd = ['ffmpeg', '-y', '-i', temp_master_path]
    total_video_duration_sec = total_frames / fps
    if audio_path and os.path.isfile(audio_path):
        # Audio fade uses fade_in_sec/fade_out_sec, independent of the video fade above.
        fade_filters = []
        if fade_in_sec > 0:
            fade_filters.append(f"afade=t=in:ss=0:d={fade_in_sec}")
        if fade_out_sec > 0:
            start_fade_out = max(0.0, total_video_duration_sec - fade_out_sec)
            fade_filters.append(f"afade=t=out:st={start_fade_out}:d={fade_out_sec}")
        
        cmd.extend(['-i', audio_path])
        if fade_filters:
            cmd.extend(['-filter_complex', f'[1:a]{",".join(fade_filters)}[aud]', '-map', '0:v:0', '-map', '[aud]'])
        else:
            cmd.extend(['-map', '0:v:0', '-map', '1:a:0'])
        cmd.extend(['-c:a', 'aac', '-b:a', audio_bitrate, '-shortest'])
    else:
        cmd.extend(['-map', '0:v:0'])

    # Feature #1: map any supported codec id to the correct ffmpeg flags.
    _hw_codecs = {"h264_nvenc", "hevc_nvenc", "h264_videotoolbox", "hevc_videotoolbox", "h264_amf", "hevc_amf"}
    if codec in _hw_codecs:
        # HW encoders use -q:v (quality scale) instead of CRF; 18-28 is a sensible range.
        hw_q = max(18, min(28, crf))
        cmd.extend(['-c:v', codec, '-pix_fmt', 'yuv420p', '-q:v', str(hw_q)])
    elif codec == "libx265":
        cmd.extend(['-c:v', 'libx265', '-pix_fmt', 'yuv420p', '-crf', str(crf), '-preset', 'medium'])
    else:
        cmd.extend(['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', str(crf), '-preset', 'medium'])

    cmd.append(output_path)

    try:
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        stderr_log_path = base_name + "_ffmpeg_log.txt"
        with open(stderr_log_path, "w") as log_f:
            proc = subprocess.Popen(cmd, startupinfo=startupinfo, stdout=log_f, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if cancel_event is not None and cancel_event.is_set():
                    proc.terminate()
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise RenderCancelled("Render cancelled by user during encode.")
                time.sleep(0.15)
        if proc.returncode != 0:
            try:
                with open(stderr_log_path, "r") as log_f:
                    tail = log_f.read()[-1500:]
            except Exception:
                tail = "(log unavailable)"
            try: os.remove(stderr_log_path)
            except Exception: pass  # Fix #3
            raise subprocess.CalledProcessError(proc.returncode, cmd, stderr=tail)
        try: os.remove(stderr_log_path)
        except Exception: pass  # Fix #3
        if progress_callback: progress_callback(1.0)
    except RenderCancelled:
        raise
    except Exception as e:
        if os.path.exists(temp_master_path):
            os.rename(temp_master_path, output_path)
            return output_path
        raise RuntimeError(f"FFmpeg transcode pipeline failure: {e}")
    finally:
        if os.path.exists(temp_master_path):
            try: os.remove(temp_master_path)
            except Exception: pass  # Fix #3
    return output_path

class PanoramaToVideoApp:
    RESOLUTION_PRESETS = {
        "720p (HD)": 720, "1080p (Full HD)": 1080, "1440p (2K / QHD)": 1440,
        "2160p (4K UHD)": 2160, "Custom": None,
    }

    def __init__(self, root):
        self.root = root
        self.root.title("PhotoLab — Panorama to Video")
        self.root.geometry("1180x760")
        self.root.minsize(880, 600)
        self.root.configure(bg="#121212")
        try:
            self.root.option_add("*Background", "#1a1a1a")
            self.root.option_add("*Foreground", "#ddd")
            self.root.option_add("*highlightBackground", "#1a1a1a")
            self.root.option_add("*highlightColor", "#2a5080")
            self.root.option_add("*borderWidth", 0)
        except Exception:
            pass
        self._apply_photolab_theme()
        self.preview_running = False
        self.preview_img_cache = None
        self._render_t0 = None

        self.canvas = tk.Canvas(self.root, bg="#121212", highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg="#1a1a1a")

        self.scrollable_frame.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        self.image_path = tk.StringVar()
        self.duration_var = tk.StringVar(value="8")
        self.fps_var = tk.StringVar(value="30")
        self.height_var = tk.StringVar(value="1080")
        self.resolution_var = tk.StringVar(value="1080p (Full HD)")
        self.aspect_var = tk.StringVar(value="16:9")
        self.ken_burns_var = tk.BooleanVar(value=True)
        self.zoom_var = tk.StringVar(value="0.20")
        self.zoom_var_float = tk.DoubleVar(value=0.20)
        self.ken_style_var = tk.StringVar(value="Slow Zoom In")
        self.easing_var = tk.StringVar(value="Smoothstep")
        self.cinematic_look_var = tk.StringVar(value="None")
        self.film_effect_var = tk.StringVar(value="None")
        self.audio_bitrate_var = tk.StringVar(value="192k")
        self.crf_var = tk.IntVar(value=23)
        self.crf_string_var = tk.StringVar(value="23")
        self.motion_blur_var = tk.DoubleVar(value=0.0)
        self.motion_blur_string_var = tk.StringVar(value="0.00")
        self.vignette_var = tk.DoubleVar(value=0.0)
        self.vignette_string_var = tk.StringVar(value="0.00")
        self.look_strength_var = tk.DoubleVar(value=1.0)
        self.look_strength_string_var = tk.StringVar(value="1.00")
        self.film_effect_strength_var = tk.DoubleVar(value=0.0)
        self.film_effect_strength_string_var = tk.StringVar(value="0.00")
        self.film_grain_var = tk.DoubleVar(value=0.0)
        self.film_grain_string_var = tk.StringVar(value="0.00")
        self.film_dust_var = tk.DoubleVar(value=0.0)
        self.film_dust_string_var = tk.StringVar(value="0.00")
        self.film_scratches_var = tk.DoubleVar(value=0.0)
        self.film_scratches_string_var = tk.StringVar(value="0.00")
        self.film_flicker_var = tk.DoubleVar(value=0.0)
        self.film_flicker_string_var = tk.StringVar(value="0.00")
        self.intro_zoom_var = tk.BooleanVar(value=False)
        self.intro_duration_var = tk.StringVar(value="2.0")
        self.intro_duration_float = tk.DoubleVar(value=2.0)
        self.intro_level_var = tk.StringVar(value="1.8")
        self.intro_level_float = tk.DoubleVar(value=1.8)
        self.fade_in_sec_var = tk.DoubleVar(value=1.0)
        self.fade_in_string_var = tk.StringVar(value="1.00")
        self.fade_out_sec_var = tk.DoubleVar(value=2.0)
        self.fade_out_string_var = tk.StringVar(value="2.00")
        self.reverse_pan_var = tk.BooleanVar(value=False)
        self.fade_in_var = tk.BooleanVar(value=True)
        self.fade_out_var = tk.BooleanVar(value=True)
        # Video (visual) fade in/out - separate from the audio fade above so each can be
        # tuned independently.
        self.video_fade_in_sec_var = tk.DoubleVar(value=1.0)
        self.video_fade_in_string_var = tk.StringVar(value="1.00")
        self.video_fade_out_sec_var = tk.DoubleVar(value=2.0)
        self.video_fade_out_string_var = tk.StringVar(value="2.00")
        self.video_fade_in_var = tk.BooleanVar(value=True)
        self.video_fade_out_var = tk.BooleanVar(value=True)
        self.audio_path = tk.StringVar()
        self.codec_var = tk.StringVar(value="libx264")
        self.output_folder = tk.StringVar()
        self.custom_lut_path = tk.StringVar()
        self.custom_lut_data = None
        self._custom_lut_full_path = None
        self.cancel_event = None
        self.rendering = False
        self._preview_placeholder_img = ImageTk.PhotoImage(Image.new("RGB", (720, 180), "#2a2a2a"))
        self._preview_thumb_img = None
        self.preview_display_size = None
        self.start_focus = None
        self.end_focus = None
        self.intro_focus = None
        self.focus_mode_var = tk.StringVar(value="Start")
        self.motion_preset_var = tk.StringVar(value="Gentle Documentary")

        self._build_widgets()
        self._on_resolution_change()
        self.canvas.bind("<Enter>", self._bind_mousewheel)
        self.canvas.bind("<Leave>", self._unbind_mousewheel)

        # Keep every paired Entry+Slider control in sync: the slider's numeric var is the
        # canonical value that render/preview/export/presets all read; typing in the entry
        # box now updates that canonical var too, instead of being silently ignored.
        self._sync_string_to_float(self.zoom_var, self.zoom_var_float, 0.0, 1.0)
        self._sync_string_to_float(self.motion_blur_string_var, self.motion_blur_var, 0.0, 0.8)
        self._sync_string_to_float(self.intro_duration_var, self.intro_duration_float, 0.5, 30.0)
        self._sync_string_to_float(self.intro_level_var, self.intro_level_float, 1.1, 8.0)
        self._sync_string_to_float(self.fade_in_string_var, self.fade_in_sec_var, 0.0, 4.0)
        self._sync_string_to_float(self.fade_out_string_var, self.fade_out_sec_var, 0.0, 5.0)
        self._sync_string_to_float(self.video_fade_in_string_var, self.video_fade_in_sec_var, 0.0, 4.0)
        self._sync_string_to_float(self.video_fade_out_string_var, self.video_fade_out_sec_var, 0.0, 5.0)
        self._sync_string_to_float(self.crf_string_var, self.crf_var, 15, 35)
        self._sync_string_to_float(self.vignette_string_var, self.vignette_var, 0.0, 1.0)
        self._sync_string_to_float(self.look_strength_string_var, self.look_strength_var, 0.0, 1.0)
        self._sync_string_to_float(self.film_effect_strength_string_var, self.film_effect_strength_var, 0.0, 1.0)
        self._sync_string_to_float(self.film_grain_string_var, self.film_grain_var, 0.0, 1.0)
        self._sync_string_to_float(self.film_dust_string_var, self.film_dust_var, 0.0, 1.0)
        self._sync_string_to_float(self.film_scratches_string_var, self.film_scratches_var, 0.0, 1.0)
        self._sync_string_to_float(self.film_flicker_string_var, self.film_flicker_var, 0.0, 1.0)

        # Keep dependent controls enabled/disabled in step with their governing
        # selection/checkbox - e.g. Look Strength only matters when a Film Look is
        # actually selected, and Motion Style/Zoom only matter when Ken Burns is on.
        self.cinematic_look_var.trace_add("write", self._update_film_look_state)
        self.film_effect_var.trace_add("write", self._update_film_damage_state)
        self.ken_burns_var.trace_add("write", self._update_ken_burns_state)
        self._update_film_look_state()
        self._update_film_damage_state()
        self._update_ken_burns_state()

        self._load_session_settings()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _update_film_look_state(self, *_args):
        """Look Strength only has an effect once a Film Look other than None is chosen."""
        state = "normal" if self.cinematic_look_var.get() != "None" else "disabled"
        self.look_strength_scale.config(state=state)
        self.look_strength_entry.config(state=state)

    def _update_film_damage_state(self, *_args):
        """Damage Strength/Grain/Dust/Scratches/Flicker only matter once a Film Damage
        preset other than None is chosen."""
        state = "normal" if self.film_effect_var.get() != "None" else "disabled"
        self.film_effect_strength_scale.config(state=state)
        self.film_effect_strength_entry.config(state=state)
        for w in self.film_damage_detail_widgets:
            w.config(state=state)

    def _update_ken_burns_state(self, *_args):
        """Motion Style and Zoom strength only have an effect while Ken Burns is enabled
        (Easing Curve, Motion Blur, and Reverse Pan still shape the base panning motion
        even with Ken Burns off, so those stay active)."""
        enabled = bool(self.ken_burns_var.get())
        self.motion_style_combo.config(state="readonly" if enabled else "disabled")
        self.zoom_scale.config(state="normal" if enabled else "disabled")
        self.zoom_entry.config(state="normal" if enabled else "disabled")

    def _sync_string_to_float(self, string_var, float_var, minv, maxv):
        """Whenever the text entry is edited directly, parse and push the value into the
        canonical numeric var (which the slider and all render/preview/export code read),
        clamped to the slider's valid range. Silently ignores unparsable in-progress typing
        (e.g. a lone '-' or '.') rather than raising."""
        def _on_write(*_args):
            try:
                val = float(string_var.get())
            except (ValueError, tk.TclError):
                return
            val = max(minv, min(maxv, val))
            if abs(float_var.get() - val) > 1e-9:
                float_var.set(val)
        string_var.trace_add("write", _on_write)

    def _bind_mousewheel(self, event):
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _unbind_mousewheel(self, event):
        self.canvas.unbind_all("<MouseWheel>")
        self.canvas.unbind_all("<Button-4>")
        self.canvas.unbind_all("<Button-5>")

    def _on_mousewheel(self, event):
        if event.num == 4:
            self.canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(3, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_close(self):
        if self.rendering:
            if not messagebox.askyesno("Render in progress", "A render is currently in progress. Quit anyway?\n\n(The render will not be stopped cleanly.)"):
                return
        try:
            with open(SETTINGS_PATH, "w") as f:
                json.dump(self._collect_settings(include_paths=True), f, indent=2)
        except Exception as e:
            print(f"Could not save session settings: {e}")
        self.root.destroy()

    def _load_session_settings(self):
        if os.path.isfile(SETTINGS_PATH):
            try:
                with open(SETTINGS_PATH, "r") as f:
                    d = json.load(f)
                self._apply_settings(d)
            except Exception as e:
                print(f"Could not restore last session: {e}")

    def _on_canvas_configure(self, event):
        self.canvas.itemconfig(self.canvas_window, width=event.width)

    def _apply_photolab_theme(self):
        """Match PhotoLab dark UI (PyQt #121212 / #2a5080 accent)."""
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        bg = "#1a1a1a"
        bg2 = "#121212"
        fg = "#ddd"
        accent = "#2a5080"
        style.configure(".", background=bg, foreground=fg, fieldbackground="#222",
                        troughcolor="#2a2a2a", bordercolor="#2b2b2b")
        style.configure("TFrame", background=bg)
        style.configure("TLabelframe", background=bg, foreground="#8af")
        style.configure("TLabelframe.Label", background=bg, foreground="#8af")
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TButton", background="#2a2a2a", foreground=fg, padding=6)
        style.map("TButton",
                  background=[("active", accent), ("disabled", "#333")],
                  foreground=[("disabled", "#777")])
        style.configure("TEntry", fieldbackground="#222", foreground=fg, insertcolor=fg)
        style.configure("TCombobox", fieldbackground="#222", foreground=fg, background="#222")
        style.map("TCombobox", fieldbackground=[("readonly", "#222")],
                  foreground=[("readonly", fg)])
        style.configure("TCheckbutton", background=bg, foreground=fg)
        style.configure("TRadiobutton", background=bg, foreground=fg)
        style.configure("Horizontal.TScale", background=bg, troughcolor="#2a2a2a")
        style.configure("TProgressbar", background=accent, troughcolor="#2a2a2a",
                        bordercolor="#2b2b2b", lightcolor=accent, darkcolor=accent)
        style.configure("Vertical.TScrollbar", background="#2a2a2a", troughcolor=bg2,
                        arrowcolor=fg)

    def _make_slider_row(self, parent, label, float_var, string_var, from_, to, slide_cmd):
        """Fix #8: factory that builds the label + Scale + Entry triplet used by
        every slider row so the pattern isn't copy-pasted 13 times."""
        r = tk.Frame(parent, bg="#1a1a1a")
        r.pack(fill="x", padx=8, pady=2)
        tk.Label(r, text=label, width=22, bg="#1a1a1a", fg="#ccc").pack(side="left")
        scale = ttk.Scale(r, from_=from_, to=to, variable=float_var,
                          command=slide_cmd, length=200)
        scale.pack(side="left", padx=8)
        entry = ttk.Entry(r, textvariable=string_var, width=8)
        entry.pack(side="left")
        return scale, entry

    def _section_frame(self, text):
        """PhotoLab-style settings group with a subtle gray outline."""
        return tk.LabelFrame(
            self.scrollable_frame,
            text=text,
            bg="#1a1a1a",
            fg="#8faed0",
            font=("Segoe UI", 10, "bold"),
            relief="flat",
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#3a3a3a",
            highlightcolor="#46586d",
            padx=2,
            pady=2,
        )

    def _build_widgets(self):
        self.scrollable_frame.grid_columnconfigure(0, weight=1, uniform="settingscol")
        self.scrollable_frame.grid_columnconfigure(1, weight=1, uniform="settingscol")

        header_wrap = tk.Frame(self.scrollable_frame, bg="#2a5080")
        header_wrap.grid(row=0, column=0, columnspan=2, sticky="ew")
        header = tk.Label(header_wrap, text="Panorama to Video", font=("Segoe UI", 14, "bold"), bg="#1e2a3a", fg="#9cf", pady=8)
        header.pack(fill="x", side="top")
        tk.Frame(header_wrap, bg="#2a5080", height=2).pack(fill="x", side="top")

        f = self._section_frame("Select Panorama Image")
        f.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8, pady=5)
        ttk.Entry(f, textvariable=self.image_path, width=85).pack(side="left", padx=8, pady=6, fill="x", expand=True)
        ttk.Button(f, text="Browse...", command=self.browse_image).pack(side="left", padx=5)

        out = self._section_frame("Output Folder")
        out.grid(row=2, column=0, columnspan=2, sticky="ew", padx=8, pady=5)
        r = tk.Frame(out, bg="#1a1a1a")
        r.pack(fill="x", padx=8, pady=(6, 0))
        ttk.Entry(r, textvariable=self.output_folder, width=85).pack(side="left", fill="x", expand=True)
        ttk.Button(r, text="Browse...", command=self.browse_output_folder).pack(side="left", padx=5)
        r2 = tk.Frame(out, bg="#1a1a1a")
        r2.pack(fill="x", padx=8, pady=(2, 6))
        ttk.Button(r2, text="Same as Source", command=self.use_source_as_output_folder).pack(side="left")

        self.preview_label = ttk.Label(self.scrollable_frame, image=self._preview_placeholder_img, text="No image selected", compound="center", anchor="center", background="#1a1a1a")
        self.preview_label.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        self.preview_label.bind("<Button-1>", self.set_start_focus_from_preview)

        focus_bar = tk.Frame(self.scrollable_frame, bg="#1a1a1a")
        focus_bar.grid(row=4, column=0, columnspan=2, sticky="ew", pady=(0, 4))
        ttk.Radiobutton(focus_bar, text="Click Sets Start", variable=self.focus_mode_var, value="Start").pack(side="left", padx=14)
        ttk.Radiobutton(focus_bar, text="Click Sets End", variable=self.focus_mode_var, value="End").pack(side="left", padx=14)
        ttk.Radiobutton(focus_bar, text="Deep Click Sets Intro Start", variable=self.focus_mode_var, value="Intro").pack(side="left", padx=14)
        ttk.Button(focus_bar, text="Clear Focus Points", command=self.clear_focus_points).pack(side="left", padx=14)

        # ---- Left column: Video Settings / Intro Zoom Out / Audio Track & Fades ----
        s = self._section_frame("Video Settings")
        s.grid(row=5, column=0, sticky="new", padx=8, pady=(5, 1))

        for label, var in [("Duration (seconds):", self.duration_var), ("Frame rate (fps):", self.fps_var)]:
            r = tk.Frame(s, bg="#1a1a1a")
            r.pack(fill="x", padx=8, pady=2)
            tk.Label(r, text=label, width=22, bg="#1a1a1a", fg="#ccc").pack(side="left")
            ttk.Entry(r, textvariable=var, width=10).pack(side="left")

        r = tk.Frame(s, bg="#1a1a1a")
        r.pack(fill="x", padx=8, pady=2)
        tk.Label(r, text="Resolution:", width=22, bg="#1a1a1a").pack(side="left")
        combo = ttk.Combobox(r, textvariable=self.resolution_var, values=list(self.RESOLUTION_PRESETS.keys()), state="readonly", width=20)
        combo.pack(side="left")
        combo.bind("<<ComboboxSelected>>", self._on_resolution_change)
        self.height_entry = ttk.Entry(r, textvariable=self.height_var, width=8, state="disabled")
        self.height_entry.pack(side="left", padx=10)
        tk.Label(r, text="px", bg="#1a1a1a").pack(side="left")

        for label, var, vals in [("Aspect ratio:", self.aspect_var, ["16:9","9:16","4:3","21:9","1:1"]),
                                 ("FFmpeg Encoder:", self.codec_var, ["libx264","libx265"])]:
            r = tk.Frame(s, bg="#1a1a1a")
            r.pack(fill="x", padx=8, pady=2)
            tk.Label(r, text=label, width=22, bg="#1a1a1a", fg="#ccc").pack(side="left")
            ttk.Combobox(r, textvariable=var, values=vals, state="readonly", width=12).pack(side="left")

        intro = self._section_frame("Intro Zoom Out")
        intro.grid(row=6, column=0, sticky="new", padx=8, pady=(1, 1))
        tk.Checkbutton(intro, text="Enable deep-point zoom out (starts at D, then widens)", variable=self.intro_zoom_var, bg="#1a1a1a", fg="#ddd", selectcolor="#2a2a2a", activebackground="#1a1a1a", activeforeground="#ddd").pack(anchor="w", padx=8, pady=1)

        for label, fvar, svar, minv, maxv in [
            ("Deep zoom-out duration (sec):", self.intro_duration_float, self.intro_duration_var, 0.5, 30.0),
            ("Deep starting zoom (x):", self.intro_level_float, self.intro_level_var, 1.1, 8.0)
        ]:
            r = tk.Frame(intro, bg="#1a1a1a")
            r.pack(fill="x", padx=8, pady=1)
            tk.Label(r, text=label, width=22, bg="#1a1a1a", fg="#ccc").pack(side="left")
            ttk.Scale(r, from_=minv, to=maxv, variable=fvar, command=lambda v, sv=svar: sv.set(f"{float(v):.2f}"), length=200).pack(side="left", padx=8)
            ttk.Entry(r, textvariable=svar, width=8).pack(side="left")
        ttk.Button(intro, text="Show Deep Start Frame", command=self.show_deep_start_frame).pack(anchor="w", padx=8, pady=(2, 4))

        vfade = self._section_frame("Video Fade In/Out")
        vfade.grid(row=7, column=0, sticky="new", padx=8, pady=(1, 5))
        tk.Checkbutton(vfade, text="Fade in from black at start", variable=self.video_fade_in_var, bg="#1a1a1a", fg="#ddd", selectcolor="#2a2a2a", activebackground="#1a1a1a", activeforeground="#ddd").pack(anchor="w", padx=8, pady=1)
        r = tk.Frame(vfade, bg="#1a1a1a")
        r.pack(fill="x", padx=8, pady=(0, 2))
        tk.Label(r, text="Fade In Duration:", width=22, bg="#1a1a1a").pack(side="left")
        ttk.Scale(r, from_=0.0, to=4.0, variable=self.video_fade_in_sec_var, command=self._on_video_fade_in_slide, length=200).pack(side="left", padx=8)
        ttk.Entry(r, textvariable=self.video_fade_in_string_var, width=8).pack(side="left")

        tk.Checkbutton(vfade, text="Fade out to black at end", variable=self.video_fade_out_var, bg="#1a1a1a", fg="#ddd", selectcolor="#2a2a2a", activebackground="#1a1a1a", activeforeground="#ddd").pack(anchor="w", padx=8, pady=1)
        r = tk.Frame(vfade, bg="#1a1a1a")
        r.pack(fill="x", padx=8, pady=(0, 2))
        tk.Label(r, text="Fade Out Duration:", width=22, bg="#1a1a1a").pack(side="left")
        ttk.Scale(r, from_=0.0, to=5.0, variable=self.video_fade_out_sec_var, command=self._on_video_fade_out_slide, length=200).pack(side="left", padx=8)
        ttk.Entry(r, textvariable=self.video_fade_out_string_var, width=8).pack(side="left")

        audio = self._section_frame("Audio Track & Fades")
        audio.grid(row=8, column=0, sticky="new", padx=8, pady=5)
        tk.Checkbutton(audio, text="Audio fade in at start", variable=self.fade_in_var, bg="#1a1a1a", fg="#ddd", selectcolor="#2a2a2a", activebackground="#1a1a1a", activeforeground="#ddd").pack(anchor="w", padx=8, pady=1)
        r = tk.Frame(audio, bg="#1a1a1a")
        r.pack(fill="x", padx=8, pady=(0, 2))
        tk.Label(r, text="Fade In Duration:", width=22, bg="#1a1a1a").pack(side="left")
        ttk.Scale(r, from_=0.0, to=4.0, variable=self.fade_in_sec_var, command=self._on_fade_in_slide, length=200).pack(side="left", padx=8)
        ttk.Entry(r, textvariable=self.fade_in_string_var, width=8).pack(side="left")

        tk.Checkbutton(audio, text="Audio fade out at end", variable=self.fade_out_var, bg="#1a1a1a", fg="#ddd", selectcolor="#2a2a2a", activebackground="#1a1a1a", activeforeground="#ddd").pack(anchor="w", padx=8, pady=1)
        r = tk.Frame(audio, bg="#1a1a1a")
        r.pack(fill="x", padx=8, pady=(0, 2))
        tk.Label(r, text="Fade Out Duration:", width=22, bg="#1a1a1a").pack(side="left")
        ttk.Scale(r, from_=0.0, to=5.0, variable=self.fade_out_sec_var, command=self._on_fade_out_slide, length=200).pack(side="left", padx=8)
        ttk.Entry(r, textvariable=self.fade_out_string_var, width=8).pack(side="left")

        r = tk.Frame(audio, bg="#1a1a1a")
        r.pack(fill="x", padx=8, pady=2)
        tk.Label(r, text="Audio Quality:", width=22, bg="#1a1a1a").pack(side="left")
        ttk.Combobox(r, textvariable=self.audio_bitrate_var, values=["64k", "128k", "192k", "256k", "320k"], state="readonly", width=12).pack(side="left")

        r = tk.Frame(audio, bg="#1a1a1a")
        r.pack(fill="x", padx=8, pady=2)
        ttk.Button(r, text="Add Music Track", command=self.browse_audio).pack(side="left", padx=5)
        self.sync_btn = ttk.Button(r, text="Sync Duration to Audio", command=self.sync_duration_to_audio, state="disabled")
        self.sync_btn.pack(side="left", padx=5)
        ttk.Label(r, textvariable=self.audio_path, background="#1a1a1a").pack(side="left", padx=10)

        # ---- Right column: Camera Motion / Film Look / Video Quality Calibration ----
        kb = self._section_frame("Camera Motion (Ken Burns)")
        kb.grid(row=5, column=1, sticky="new", padx=8, pady=5)
        tk.Checkbutton(kb, text="Enable Ken Burns effect", variable=self.ken_burns_var, bg="#1a1a1a", fg="#ddd", selectcolor="#2a2a2a", activebackground="#1a1a1a", activeforeground="#ddd").pack(anchor="w", padx=8, pady=2)

        r = tk.Frame(kb, bg="#1a1a1a")
        r.pack(fill="x", padx=8, pady=2)
        tk.Label(r, text="Motion Preset:", width=22, bg="#1a1a1a").pack(side="left")
        ttk.Combobox(r, textvariable=self.motion_preset_var, values=["Gentle Documentary", "Slow Reveal", "Dramatic Push-In", "Old Film Projector", "Social Media Reel"], state="readonly", width=20).pack(side="left")
        ttk.Button(r, text="Apply", command=self.apply_motion_preset).pack(side="left", padx=8)
        ttk.Button(r, text="Suggest Duration", command=self.suggest_duration_from_motion).pack(side="left", padx=8)

        r = tk.Frame(kb, bg="#1a1a1a")
        r.pack(fill="x", padx=8, pady=2)
        tk.Label(r, text="Motion Style:", width=22, bg="#1a1a1a").pack(side="left")
        self.motion_style_combo = ttk.Combobox(r, textvariable=self.ken_style_var, values=["Slow Zoom In", "Cinematic Drift", "Dramatic Push", "Subtle Push", "Zoom Out", "3D Drift"], state="readonly", width=18)
        self.motion_style_combo.pack(side="left")

        r = tk.Frame(kb, bg="#1a1a1a")
        r.pack(fill="x", padx=8, pady=2)
        tk.Label(r, text="Easing Curve:", width=22, bg="#1a1a1a").pack(side="left")
        ttk.Combobox(r, textvariable=self.easing_var, values=["Smoothstep", "Cubic", "Exponential", "Bounce", "Elastic"], state="readonly", width=18).pack(side="left")

        self.zoom_scale, self.zoom_entry = self._make_slider_row(
            kb, "Zoom strength:", self.zoom_var_float, self.zoom_var,
            0.0, 1.0, self._on_zoom_slide)
        self._make_slider_row(
            kb, "Motion Blur:", self.motion_blur_var, self.motion_blur_string_var,
            0.0, 0.8, self._on_motion_blur_slide)

        tk.Checkbutton(kb, text="Reverse pan direction", variable=self.reverse_pan_var, bg="#1a1a1a", fg="#ddd", selectcolor="#2a2a2a", activebackground="#1a1a1a", activeforeground="#ddd").pack(anchor="w", padx=8, pady=2)

        film = self._section_frame("Film Look & Aging")
        film.grid(row=6, column=1, sticky="new", padx=8, pady=5)

        r = tk.Frame(film, bg="#1a1a1a")
        r.pack(fill="x", padx=8, pady=2)
        tk.Label(r, text="Film Look:", width=22, bg="#1a1a1a").pack(side="left")
        self.cinematic_combo = ttk.Combobox(r, textvariable=self.cinematic_look_var, values=FILM_LOOK_CHOICES, state="readonly", width=20)
        self.cinematic_combo.pack(side="left")

        r = tk.Frame(film, bg="#1a1a1a")
        r.pack(fill="x", padx=8, pady=2)
        tk.Label(r, text="", width=22, bg="#1a1a1a").pack(side="left")
        ttk.Button(r, text="Load Custom LUT (.cube)...", command=self.browse_custom_lut).pack(side="left", padx=(0, 8))
        ttk.Label(r, textvariable=self.custom_lut_path, background="#1a1a1a").pack(side="left")

        self.look_strength_scale, self.look_strength_entry = self._make_slider_row(
            film, "Look Strength:", self.look_strength_var, self.look_strength_string_var,
            0.0, 1.0, self._on_look_strength_slide)

        r = tk.Frame(film, bg="#1a1a1a")
        r.pack(fill="x", padx=8, pady=2)
        tk.Label(r, text="Film Damage:", width=22, bg="#1a1a1a").pack(side="left")
        ttk.Combobox(r, textvariable=self.film_effect_var, values=FILM_EFFECT_CHOICES, state="readonly", width=20).pack(side="left")

        self.film_effect_strength_scale, self.film_effect_strength_entry = self._make_slider_row(
            film, "Damage Strength:", self.film_effect_strength_var, self.film_effect_strength_string_var,
            0.0, 1.0, self._on_film_effect_strength_slide)

        self.film_damage_detail_widgets = []
        for label, var, svar, cmd in [
            ("Grain:", self.film_grain_var, self.film_grain_string_var, self._on_film_grain_slide),
            ("Dust:", self.film_dust_var, self.film_dust_string_var, self._on_film_dust_slide),
            ("Scratches:", self.film_scratches_var, self.film_scratches_string_var, self._on_film_scratches_slide),
            ("Flicker:", self.film_flicker_var, self.film_flicker_string_var, self._on_film_flicker_slide),
        ]:
            scale, entry = self._make_slider_row(film, label, var, svar, 0.0, 1.0, cmd)
            self.film_damage_detail_widgets.extend([scale, entry])

        self._make_slider_row(
            film, "Vignette:", self.vignette_var, self.vignette_string_var,
            0.0, 1.0, self._on_vignette_slide)

        video = self._section_frame("Video Quality Calibration")
        video.grid(row=7, column=1, sticky="new", padx=8, pady=5)

        self._make_slider_row(
            video, "FFmpeg CRF Target:", self.crf_var, self.crf_string_var,
            15, 35, self._on_crf_slide)

        # ---- Bottom: actions, progress, status (full width) ----
        btns = tk.Frame(self.scrollable_frame, bg="#1a1a1a")
        btns.grid(row=9, column=0, columnspan=2, pady=(10, 3))
        self.preview_btn = ttk.Button(btns, text="Live Canvas Preview", command=self.toggle_live_preview)
        self.preview_btn.pack(side="left", padx=10)
        ttk.Button(btns, text="Refresh Still Preview", command=self._refresh_still_preview).pack(side="left", padx=10)
        ttk.Button(btns, text="Export Preview Frame", command=self.export_preview_frame).pack(side="left", padx=10)
        ttk.Button(btns, text="Render 3 Sec Test", command=self.on_generate_test_clip).pack(side="left", padx=10)
        self.generate_btn = ttk.Button(btns, text="Create Video File", command=self.on_generate)
        self.generate_btn.pack(side="left", padx=10)
        self.cancel_btn = ttk.Button(btns, text="Cancel Render", command=self.cancel_render, state="disabled")
        self.cancel_btn.pack(side="left", padx=10)

        btns2 = tk.Frame(self.scrollable_frame, bg="#1a1a1a")
        btns2.grid(row=10, column=0, columnspan=2, pady=(0, 6))
        ttk.Button(btns2, text="Save Preset...", command=self.save_preset).pack(side="left", padx=15)
        ttk.Button(btns2, text="Load Preset...", command=self.load_preset).pack(side="left", padx=15)

        self.progress = ttk.Progressbar(self.scrollable_frame, mode="determinate", length=700)
        self.progress.grid(row=11, column=0, columnspan=2, pady=6, padx=20)

        self.status_label = tk.Label(
            self.scrollable_frame, text="Ready", font=("Segoe UI", 10),
            bg="#1a1a1a", fg="#9cf",
        )
        self.status_label.grid(row=12, column=0, columnspan=2, pady=4)


    def _draw_focus_overlay(self, img):
        if not (self.start_focus or self.end_focus or self.intro_focus):
            return img
        out = img.copy()
        draw = ImageDraw.Draw(out)
        w, h = out.size

        def _pt(focus):
            return int(focus[0] * w), int(focus[1] * h)

        if self.start_focus and self.end_focus:
            draw.line([_pt(self.start_focus), _pt(self.end_focus)], fill="#facc15", width=max(2, w // 220))
        for focus, color, label in (
            (self.start_focus, "#22c55e", "S"),
            (self.end_focus, "#ef4444", "E"),
            (self.intro_focus, "#38bdf8", "D"),
        ):
            if not focus:
                continue
            x, y = _pt(focus)
            r = max(6, min(w, h) // 28)
            draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=max(2, r // 3))
            draw.text((x + r + 2, y - r), label, fill=color)
        return out

    def _refresh_still_preview(self):
        if self.preview_running:
            return
        path = self.image_path.get().strip()
        if not path or not os.path.isfile(path):
            return
        try:
            img = Image.open(path)
            img.thumbnail((720, 180))
            self.preview_display_size = img.size
            # Fix #2: draw overlay only once; previously it was applied twice,
            # causing circles and lines to be rendered on top of each other.
            self._preview_thumb_img = ImageTk.PhotoImage(self._draw_focus_overlay(img))
            self.preview_label.config(image=self._preview_thumb_img, text="")
        except Exception:
            pass

    def clear_focus_points(self):
        self.start_focus = None
        self.end_focus = None
        self.intro_focus = None
        self._refresh_still_preview()
        self.status_label.config(text="Focus points cleared.")

    def set_start_focus_from_preview(self, event):
        # The moving preview is a cropped video frame, not the whole image.  Do not
        # interpret a click on it as source-image coordinates.
        if self.preview_running:
            self.stop_live_preview()
            self.status_label.config(text="Live preview stopped. Click the full image again to set a focus point.")
            return
        if not self.preview_display_size:
            return
        display_w, display_h = self.preview_display_size
        label_w = max(1, self.preview_label.winfo_width())
        label_h = max(1, self.preview_label.winfo_height())
        offset_x = max(0, (label_w - display_w) / 2.0)
        offset_y = max(0, (label_h - display_h) / 2.0)
        x = (event.x - offset_x) / max(1, display_w)
        y = (event.y - offset_y) / max(1, display_h)
        x = float(np.clip(x, 0.0, 1.0))
        y = float(np.clip(y, 0.0, 1.0))
        if self.focus_mode_var.get() == "End":
            self.end_focus = (x, y)
            self.status_label.config(text=f"End focus set: {x:.0%} across, {y:.0%} down")
        elif self.focus_mode_var.get() == "Intro":
            self.intro_focus = (x, y)
            self.intro_zoom_var.set(True)
            self.status_label.config(text=f"Deep intro focus set: {x:.0%} across, {y:.0%} down — intro zoom enabled")
        else:
            self.start_focus = (x, y)
            self.status_label.config(text=f"Start focus set: {x:.0%} across, {y:.0%} down")
        self._refresh_still_preview()

    def show_deep_start_frame(self):
        """Open the exact first deep-intro crop in a separate, non-clickable window."""
        path = self.image_path.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Deep Start", "Please select an image first.")
            return
        if self.intro_focus is None:
            messagebox.showinfo("Deep Start", "Choose 'Deep Click Sets Intro Start' and click the subject first.")
            return
        try:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not load image.")
            img_h, img_w = img.shape[:2]
            aspect_parts = self.aspect_var.get().split(":")
            aspect_ratio = float(aspect_parts[0]) / float(aspect_parts[1])
            preview_h = 360
            preview_w = int(round(preview_h * aspect_ratio))
            frame = _render_crop_frame(
                img, img_w, img_h, aspect_ratio, self.intro_level_float.get(), 0.0,
                preview_w, preview_h, self.motion_blur_var.get(), is_preview=True,
                start_focus=self.intro_focus, start_focus_weight=1.0
            )
            frame = self._apply_look_and_vignette(frame, 0)
            preview = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            window = tk.Toplevel(self.root)
            window.title(f"Deep Start — {self.intro_level_float.get():.1f}x")
            label = ttk.Label(window, image=preview)
            label.image = preview  # Keep the Tk image alive for the lifetime of the window.
            label.pack(padx=10, pady=10)
        except Exception as e:
            messagebox.showerror("Deep Start", f"Could not preview the deep start frame:\n{e}")

    def apply_motion_preset(self):
        name = self.motion_preset_var.get()
        presets = {
            "Gentle Documentary": dict(duration="10", ken=True, style="Slow Zoom In", easing="Smoothstep", zoom=0.16, blur=0.03),
            "Slow Reveal": dict(duration="12", ken=True, style="Cinematic Drift", easing="Cubic", zoom=0.22, blur=0.04),
            "Dramatic Push-In": dict(duration="8", ken=True, style="Dramatic Push", easing="Exponential", zoom=0.34, blur=0.06),
            "Old Film Projector": dict(duration="8", ken=True, style="Subtle Push", easing="Smoothstep", zoom=0.14, blur=0.03, look="Old Film", effect="Heavy Old Projector", damage=0.55, grain=0.55, dust=0.45, scratches=0.42, flicker=0.45, fps="24"),
            "Social Media Reel": dict(duration="6", ken=True, style="Dramatic Push", easing="Cubic", zoom=0.28, blur=0.04, aspect="9:16", resolution="1080p"),
        }
        p = presets.get(name, presets["Gentle Documentary"])
        self.duration_var.set(p["duration"])
        if "fps" in p: self.fps_var.set(p["fps"])
        if "aspect" in p: self.aspect_var.set(p["aspect"])
        if "resolution" in p:
            self.resolution_var.set(p["resolution"])
            self._on_resolution_change()
        self.ken_burns_var.set(p["ken"])
        self.ken_style_var.set(p["style"])
        self.easing_var.set(p["easing"])
        self.zoom_var_float.set(p["zoom"]); self.zoom_var.set(f"{p['zoom']:.2f}")
        self.motion_blur_var.set(p["blur"]); self.motion_blur_string_var.set(f"{p['blur']:.2f}")
        if "look" in p: self.cinematic_look_var.set(p["look"])
        if "effect" in p: self.film_effect_var.set(p["effect"])
        for key, var, svar in (
            ("damage", self.film_effect_strength_var, self.film_effect_strength_string_var),
            ("grain", self.film_grain_var, self.film_grain_string_var),
            ("dust", self.film_dust_var, self.film_dust_string_var),
            ("scratches", self.film_scratches_var, self.film_scratches_string_var),
            ("flicker", self.film_flicker_var, self.film_flicker_string_var),
        ):
            if key in p:
                var.set(p[key]); svar.set(f"{p[key]:.2f}")
        self.status_label.config(text=f"Applied motion preset: {name}")

    def suggest_duration_from_motion(self):
        zoom = self.zoom_var_float.get() if self.ken_burns_var.get() else 0.0
        focus_bonus = 2.0 if self.start_focus and self.end_focus else 0.0
        suggested = max(5.0, min(18.0, 6.0 + zoom * 12.0 + focus_bonus))
        self.duration_var.set(f"{suggested:.1f}")
        self.status_label.config(text=f"Suggested duration: {suggested:.1f}s")
    def browse_audio(self):
        path = filedialog.askopenfilename(filetypes=[("Audio Tracks", "*.mp3 *.wav *.m4a *.aac *.flac")])
        if path:
            self.audio_path.set(path)
            self.sync_btn.config(state="normal")

    def sync_duration_to_audio(self):
        path = self.audio_path.get()
        if path and os.path.isfile(path):
            duration = get_audio_duration(path)
            if duration:
                self.duration_var.set(f"{duration:.2f}")
                self.status_label.config(text=f"Synced duration to match audio exactly: {duration:.2f}s")
            else:
                messagebox.showerror("Error", "Could not query audio track metadata via FFmpeg.")

    def browse_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff")])
        if path:
            self.load_image_path(path)

    def load_image_path(self, path, output_folder=None):
        """Load a source image (used by Browse and by PhotoLab CLI launch)."""
        if not path or not os.path.isfile(path):
            return False
        self.image_path.set(path)
        self.stop_live_preview()
        try:
            img = Image.open(path)
            img.thumbnail((720, 180))
            self.preview_display_size = img.size
            self._preview_thumb_img = ImageTk.PhotoImage(self._draw_focus_overlay(img))
            self.preview_label.config(image=self._preview_thumb_img, text="")
        except Exception:
            pass
        if output_folder and os.path.isdir(output_folder):
            self.output_folder.set(output_folder)
        else:
            self.output_folder.set(os.path.dirname(path))
        self.status_label.config(text=f"Loaded: {os.path.basename(path)}")
        return True

    def browse_output_folder(self):
        folder = filedialog.askdirectory(title="Choose destination folder for rendered videos")
        if folder:
            self.output_folder.set(folder)

    def use_source_as_output_folder(self):
        path = self.image_path.get().strip()
        if not path:
            messagebox.showerror("Error", "Please select a source image first.")
            return
        self.output_folder.set(os.path.dirname(path))

    def browse_custom_lut(self):
        path = filedialog.askopenfilename(filetypes=[("Cube LUT", "*.cube"), ("All Files", "*.*")])
        if not path:
            return
        try:
            self.custom_lut_data = parse_cube_lut(path)
        except Exception as e:
            messagebox.showerror("LUT Load Error", f"Could not load LUT file:\n{e}")
            return
        self.custom_lut_path.set(os.path.basename(path))
        self._custom_lut_full_path = path
        values = list(self.cinematic_combo["values"])
        if "Custom LUT" not in values:
            values.append("Custom LUT")
            self.cinematic_combo["values"] = values
        self.cinematic_look_var.set("Custom LUT")
        self.status_label.config(text=f"Loaded custom LUT: {os.path.basename(path)}")

    def _apply_look_and_vignette(self, frame, frame_idx):
        # Fix #9: delegate to the shared pipeline so preview and render always match.
        return apply_look_pipeline(
            frame, frame_idx,
            cinematic_look=self.cinematic_look_var.get(),
            custom_lut=self.custom_lut_data,
            look_strength=self.look_strength_var.get(),
            vignette_amount=self.vignette_var.get(),
            film_effect=self.film_effect_var.get(),
            film_effect_strength=self.film_effect_strength_var.get(),
            film_grain=self.film_grain_var.get(),
            film_dust=self.film_dust_var.get(),
            film_scratches=self.film_scratches_var.get(),
            film_flicker=self.film_flicker_var.get(),
        )

    def toggle_live_preview(self):
        if self.preview_running:
            self.stop_live_preview()
        else:
            path = self.image_path.get().strip()
            if not path or not os.path.isfile(path):
                messagebox.showerror("Error", "Please select a valid image before triggering preview window.")
                return
            
            self.preview_img_cache = cv2.imread(path, cv2.IMREAD_COLOR)
            if self.preview_img_cache is None:
                messagebox.showerror("Error", "Could not load image for preview pipeline.")
                return
                
            self.preview_running = True
            self.preview_btn.config(text="Stop Preview Loop")
            self.preview_frame_idx = 0
            # Fix #6: pre-generate a pool of grain tiles so the live preview
            # doesn't call np.random.normal + GaussianBlur on every tick.
            self._preview_grain_pool = [
                _generate_organic_grain((240, 480, 3), intensity=8)
                for _ in range(16)
            ]
            self._live_preview_next_frame()

    def stop_live_preview(self):
        self.preview_running = False
        self.preview_img_cache = None
        self.preview_btn.config(text="Live Canvas Preview")
        path = self.image_path.get().strip()
        if path and os.path.isfile(path):
            try:
                img = Image.open(path)
                img.thumbnail((720, 180))
                self.preview_display_size = img.size
                self._preview_thumb_img = ImageTk.PhotoImage(self._draw_focus_overlay(img))
                self.preview_label.config(image=self._preview_thumb_img, text="")
            except Exception: pass  # Fix #3

    def _live_preview_next_frame(self):
        if not self.preview_running or self.preview_img_cache is None:
            return

        try:
            img_h, img_w = self.preview_img_cache.shape[:2]
            aspect_parts = self.aspect_var.get().split(":")
            aspect_ratio = float(aspect_parts[0]) / float(aspect_parts[1])
            
            preview_h = 240
            preview_w = int(preview_h * aspect_ratio)
            
            easing = self.easing_var.get()
            easing_func = EASING_CURVES.get(easing, EASING_CURVES["Smoothstep"])
            
            # Preview the complete sequence, including the deep intro.  Previously this
            # preview rendered only the normal pan, which made a Deep Intro selection
            # look as if it had no effect until after a full export.
            total_frames = 90
            i = self.preview_frame_idx % total_frames
            intro_ratio = 0.0
            if self.intro_zoom_var.get():
                intro_seconds = self.intro_duration_float.get()
                main_seconds = max(0.1, float(self.duration_var.get()))
                intro_ratio = intro_seconds / (intro_seconds + main_seconds)
            intro_preview_frames = max(1, int(round(total_frames * intro_ratio))) if intro_ratio > 0 else 0

            if i < intro_preview_frames:
                intro_t = easing_func(i / (intro_preview_frames - 1) if intro_preview_frames > 1 else 1.0)
                base_zoom = ken_burns_zoom_scale(self.ken_style_var.get(), self.zoom_var_float.get(), 0.0) if self.ken_burns_var.get() else 1.0
                zoom_scale = base_zoom + max(0.0, self.intro_level_float.get() - base_zoom) * (1.0 - intro_t)
                pan_t = 1.0 if self.reverse_pan_var.get() else 0.0
                deep_focus = self.intro_focus or self.start_focus or (0.0, 0.5)
                intro_target_focus = self.start_focus or (0.0, 0.5)
                frame = _render_crop_frame(
                    self.preview_img_cache, img_w, img_h, aspect_ratio,
                    zoom_scale, pan_t, preview_w, preview_h,
                    self.motion_blur_var.get(), is_preview=True,
                    start_focus=deep_focus, end_focus=intro_target_focus, focus_t=intro_t
                )
            else:
                main_preview_frames = max(1, total_frames - intro_preview_frames)
                main_i = i - intro_preview_frames
                t = easing_func(main_i / (main_preview_frames - 1) if main_preview_frames > 1 else 1.0)
                zoom_scale = ken_burns_zoom_scale(self.ken_style_var.get(), self.zoom_var_float.get(), t) if self.ken_burns_var.get() else 1.0
                pan_t = (1.0 - t) if self.reverse_pan_var.get() else t
                frame = _render_crop_frame(
                    self.preview_img_cache, img_w, img_h, aspect_ratio,
                    zoom_scale, pan_t, preview_w, preview_h,
                    self.motion_blur_var.get(), is_preview=True,
                    start_focus=self.start_focus, start_focus_weight=1.0 - t, end_focus=self.end_focus
                )
            frame = self._apply_look_and_vignette(frame, self.preview_frame_idx)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(rgb_frame)
            self.current_preview_tk = ImageTk.PhotoImage(img_pil)
            
            self.preview_display_size = img_pil.size
            self.preview_label.config(image=self.current_preview_tk, text="")
            self.preview_frame_idx += 1
            
            self.root.after(33, self._live_preview_next_frame)
        except Exception as e:
            self.stop_live_preview()
            print(f"Preview clock cycle recovery exception: {e}")

    def export_preview_frame(self):
        path = self.image_path.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Please select a valid image first.")
            return

        img = self.preview_img_cache if self.preview_img_cache is not None else cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            messagebox.showerror("Error", "Could not load image for export.")
            return

        try:
            img_h, img_w = img.shape[:2]
            aspect_parts = self.aspect_var.get().split(":")
            aspect_ratio = float(aspect_parts[0]) / float(aspect_parts[1])

            out_height = int(self.height_var.get())
            out_w = int(round(out_height * aspect_ratio))
            if out_w % 2 != 0: out_w += 1
            if out_height % 2 != 0: out_height += 1

            easing = self.easing_var.get()
            easing_func = EASING_CURVES.get(easing, EASING_CURVES["Smoothstep"])

            if self.preview_running:
                total_frames = 90
                i = self.preview_frame_idx % total_frames
                t_linear = i / (total_frames - 1)
            else:
                t_linear = 0.5

            t = easing_func(t_linear)

            zoom_amount = self.zoom_var_float.get()
            if self.ken_burns_var.get():
                zoom_scale = ken_burns_zoom_scale(self.ken_style_var.get(), zoom_amount, t)
            else:
                zoom_scale = 1.0

            pan_t = (1.0 - t) if self.reverse_pan_var.get() else t

            frame = _render_crop_frame(
                img, img_w, img_h, aspect_ratio,
                zoom_scale, pan_t, out_w, out_height,
                self.motion_blur_var.get(), is_preview=False,
                start_focus=self.start_focus, start_focus_weight=1.0 - t, end_focus=self.end_focus
            )

            frame_idx_for_look = self.preview_frame_idx if self.preview_running else 0
            frame = self._apply_look_and_vignette(frame, frame_idx_for_look)
        except Exception as e:
            messagebox.showerror("Error", f"Could not render preview frame:\n{e}")
            return

        default_name = os.path.splitext(os.path.basename(path))[0] + "_frame.png"
        out_folder = self.output_folder.get().strip()
        dialog_kwargs = dict(
            initialfile=default_name,
            defaultextension=".png",
            filetypes=[("PNG Image", "*.png"), ("JPEG Image", "*.jpg"), ("All Files", "*.*")]
        )
        if out_folder and os.path.isdir(out_folder):
            dialog_kwargs["initialdir"] = out_folder
        save_path = filedialog.asksaveasfilename(**dialog_kwargs)
        if not save_path:
            return

        try:
            if not cv2.imwrite(save_path, frame):
                raise RuntimeError("cv2.imwrite returned False")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save frame:\n{e}")
            return

        self.status_label.config(text=f"Exported preview frame: {os.path.basename(save_path)}")

    def _collect_settings(self, include_paths=False):
        d = {
            "duration": self.duration_var.get(),
            "fps": self.fps_var.get(),
            "resolution": self.resolution_var.get(),
            "height": self.height_var.get(),
            "aspect": self.aspect_var.get(),
            "codec": self.codec_var.get(),
            "ken_burns": bool(self.ken_burns_var.get()),
            "zoom_amount": self.zoom_var_float.get(),
            "ken_style": self.ken_style_var.get(),
            "easing": self.easing_var.get(),
            "cinematic_look": self.cinematic_look_var.get(),
            "custom_lut_full_path": self._custom_lut_full_path,
            "vignette": self.vignette_var.get(),
            "look_strength": self.look_strength_var.get(),
            "motion_blur": self.motion_blur_var.get(),
            "film_effect": self.film_effect_var.get(),
            "film_effect_strength": self.film_effect_strength_var.get(),
            "film_grain": self.film_grain_var.get(),
            "film_dust": self.film_dust_var.get(),
            "film_scratches": self.film_scratches_var.get(),
            "film_flicker": self.film_flicker_var.get(),
            "reverse_pan": bool(self.reverse_pan_var.get()),
            "intro_zoom": bool(self.intro_zoom_var.get()),
            "intro_duration": self.intro_duration_float.get(),
            "intro_level": self.intro_level_float.get(),
            "fade_in_enabled": bool(self.fade_in_var.get()),
            "fade_out_enabled": bool(self.fade_out_var.get()),
            "fade_in_sec": self.fade_in_sec_var.get(),
            "fade_out_sec": self.fade_out_sec_var.get(),
            "video_fade_in_enabled": bool(self.video_fade_in_var.get()),
            "video_fade_out_enabled": bool(self.video_fade_out_var.get()),
            "video_fade_in_sec": self.video_fade_in_sec_var.get(),
            "video_fade_out_sec": self.video_fade_out_sec_var.get(),
            "audio_bitrate": self.audio_bitrate_var.get(),
            "crf": self.crf_var.get(),
            "start_focus": list(self.start_focus) if self.start_focus else None,
            "end_focus": list(self.end_focus) if self.end_focus else None,
            "intro_focus": list(self.intro_focus) if self.intro_focus else None,
        }
        if include_paths:
            d["image_path"] = self.image_path.get()
            d["output_folder"] = self.output_folder.get()
            d["audio_path"] = self.audio_path.get()
        return d

    def _apply_settings(self, d):
        try:
            if "duration" in d: self.duration_var.set(str(d["duration"]))
            if "fps" in d: self.fps_var.set(str(d["fps"]))
            if "resolution" in d: self.resolution_var.set(d["resolution"])
            if "height" in d: self.height_var.set(str(d["height"]))
            if "aspect" in d: self.aspect_var.set(d["aspect"])
            if "codec" in d: self.codec_var.set(d["codec"])
            if "ken_burns" in d: self.ken_burns_var.set(bool(d["ken_burns"]))
            if "zoom_amount" in d:
                v = float(d["zoom_amount"]); self.zoom_var_float.set(v); self.zoom_var.set(f"{v:.2f}")
            if "ken_style" in d: self.ken_style_var.set(d["ken_style"])
            if "easing" in d: self.easing_var.set(d["easing"])
            if "vignette" in d:
                v = float(d["vignette"]); self.vignette_var.set(v); self.vignette_string_var.set(f"{v:.2f}")
            if "look_strength" in d:
                v = float(d["look_strength"]); self.look_strength_var.set(v); self.look_strength_string_var.set(f"{v:.2f}")
            if "motion_blur" in d:
                v = float(d["motion_blur"]); self.motion_blur_var.set(v); self.motion_blur_string_var.set(f"{v:.2f}")
            if "film_effect" in d: self.film_effect_var.set(d["film_effect"])
            if "film_effect_strength" in d:
                v = float(d["film_effect_strength"]); self.film_effect_strength_var.set(v); self.film_effect_strength_string_var.set(f"{v:.2f}")
            for key, var, svar in (
                ("film_grain", self.film_grain_var, self.film_grain_string_var),
                ("film_dust", self.film_dust_var, self.film_dust_string_var),
                ("film_scratches", self.film_scratches_var, self.film_scratches_string_var),
                ("film_flicker", self.film_flicker_var, self.film_flicker_string_var),
            ):
                if key in d:
                    v = float(d[key]); var.set(v); svar.set(f"{v:.2f}")
            if "reverse_pan" in d: self.reverse_pan_var.set(bool(d["reverse_pan"]))
            if "intro_zoom" in d: self.intro_zoom_var.set(bool(d["intro_zoom"]))
            if "intro_duration" in d:
                v = float(d["intro_duration"]); self.intro_duration_float.set(v); self.intro_duration_var.set(f"{v:.2f}")
            if "intro_level" in d:
                v = float(d["intro_level"]); self.intro_level_float.set(v); self.intro_level_var.set(f"{v:.2f}")
            if "fade_in_enabled" in d: self.fade_in_var.set(bool(d["fade_in_enabled"]))
            if "fade_out_enabled" in d: self.fade_out_var.set(bool(d["fade_out_enabled"]))
            if "fade_in_sec" in d:
                v = float(d["fade_in_sec"]); self.fade_in_sec_var.set(v); self.fade_in_string_var.set(f"{v:.2f}")
            if "fade_out_sec" in d:
                v = float(d["fade_out_sec"]); self.fade_out_sec_var.set(v); self.fade_out_string_var.set(f"{v:.2f}")
            if "video_fade_in_enabled" in d: self.video_fade_in_var.set(bool(d["video_fade_in_enabled"]))
            if "video_fade_out_enabled" in d: self.video_fade_out_var.set(bool(d["video_fade_out_enabled"]))
            if "video_fade_in_sec" in d:
                v = float(d["video_fade_in_sec"]); self.video_fade_in_sec_var.set(v); self.video_fade_in_string_var.set(f"{v:.2f}")
            if "video_fade_out_sec" in d:
                v = float(d["video_fade_out_sec"]); self.video_fade_out_sec_var.set(v); self.video_fade_out_string_var.set(f"{v:.2f}")
            if "audio_bitrate" in d: self.audio_bitrate_var.set(d["audio_bitrate"])
            if "crf" in d:
                v = int(d["crf"]); self.crf_var.set(v); self.crf_string_var.set(str(v))

            look = d.get("cinematic_look")
            lut_path = d.get("custom_lut_full_path")
            if look == "Custom LUT" and lut_path and os.path.isfile(lut_path):
                try:
                    self.custom_lut_data = parse_cube_lut(lut_path)
                    self._custom_lut_full_path = lut_path
                    self.custom_lut_path.set(os.path.basename(lut_path))
                    self.cinematic_look_var.set("Custom LUT")
                except Exception:
                    self.cinematic_look_var.set("None")
            elif look:
                self.cinematic_look_var.set(look)

            if "image_path" in d and d["image_path"] and os.path.isfile(d["image_path"]):
                self.image_path.set(d["image_path"])
                try:
                    img = Image.open(d["image_path"])
                    img.thumbnail((720, 180))
                    self.preview_display_size = img.size
                    self._preview_thumb_img = ImageTk.PhotoImage(self._draw_focus_overlay(img))
                    self.preview_label.config(image=self._preview_thumb_img, text="")
                except Exception:
                    pass
            if "output_folder" in d and d["output_folder"] and os.path.isdir(d["output_folder"]):
                self.output_folder.set(d["output_folder"])
            if "audio_path" in d and d["audio_path"] and os.path.isfile(d["audio_path"]):
                self.audio_path.set(d["audio_path"])
                self.sync_btn.config(state="normal")

            if "start_focus" in d and d["start_focus"]:
                self.start_focus = tuple(d["start_focus"])
            else:
                self.start_focus = None
            if "end_focus" in d and d["end_focus"]:
                self.end_focus = tuple(d["end_focus"])
            else:
                self.end_focus = None
            if "intro_focus" in d and d["intro_focus"]:
                self.intro_focus = tuple(d["intro_focus"])
            else:
                self.intro_focus = None
            self._refresh_still_preview()

            self._on_resolution_change()
        except Exception as e:
            print(f"Settings apply warning: {e}")

    def save_preset(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("Preset JSON", "*.json"), ("All Files", "*.*")], initialfile="pano_preset.json")
        if not path:
            return
        try:
            with open(path, "w") as f:
                json.dump(self._collect_settings(include_paths=False), f, indent=2)
            self.status_label.config(text=f"Preset saved: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save preset:\n{e}")

    def load_preset(self):
        path = filedialog.askopenfilename(filetypes=[("Preset JSON", "*.json"), ("All Files", "*.*")])
        if not path:
            return
        try:
            with open(path, "r") as f:
                d = json.load(f)
            self._apply_settings(d)
            self.status_label.config(text=f"Preset loaded: {os.path.basename(path)}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load preset:\n{e}")

    def _validate_inputs(self):
        """Fix #10: validate user-editable text fields before spawning the render
        thread so that bad input produces a clear dialog rather than a cryptic
        exception deep inside the worker."""
        errors = []
        try:
            d = float(self.duration_var.get())
            if d <= 0:
                errors.append("Duration must be a positive number.")
        except ValueError:
            errors.append(f"Duration is not a valid number: {self.duration_var.get()!r}")
        try:
            f = int(self.fps_var.get())
            if f <= 0:
                errors.append("Frame rate must be a positive integer.")
        except ValueError:
            errors.append(f"Frame rate is not a valid integer: {self.fps_var.get()!r}")
        try:
            h = int(self.height_var.get())
            if h <= 0:
                errors.append("Output height must be a positive integer.")
        except ValueError:
            errors.append(f"Output height is not a valid integer: {self.height_var.get()!r}")
        return errors

    def on_generate(self, test_clip=False):
        self.stop_live_preview()
        path = self.image_path.get().strip()
        if not path or not os.path.isfile(path):
            messagebox.showerror("Error", "Please select an image first.")
            return

        # Fix #10: validate before opening the save dialog.
        errors = self._validate_inputs()
        if errors:
            messagebox.showerror("Invalid settings", "\n".join(errors))
            return

        suffix = "_test.mp4" if test_clip else "_pan.mp4"
        default_name = os.path.splitext(os.path.basename(path))[0] + suffix
        out_folder = self.output_folder.get().strip()
        dialog_kwargs = dict(
            initialfile=default_name,
            filetypes=[
                ("MP4 Video", "*.mp4"),
                ("Matroska Video", "*.mkv"),
                ("QuickTime MOV", "*.mov"),
                ("All Files", "*.*")
            ]
        )
        if out_folder and os.path.isdir(out_folder):
            dialog_kwargs["initialdir"] = out_folder
        output_path = filedialog.asksaveasfilename(**dialog_kwargs)
        if not output_path: return

        self.progress["value"] = 0
        self._render_t0 = time.time()
        self.status_label.config(text="Rendering 3 second test..." if test_clip else "Rendering…")
        self.cancel_event = threading.Event()
        self.rendering = True
        self.generate_btn.config(state="disabled")
        self.cancel_btn.config(state="normal", text="Cancel Render")
        threading.Thread(target=self._worker, args=(path, output_path, self.cancel_event, test_clip), daemon=True).start()

    def cancel_render(self):
        if self.cancel_event is not None and self.rendering:
            self.cancel_event.set()
            self.cancel_btn.config(state="disabled", text="Cancelling...")
            self.status_label.config(text="Cancelling render, please wait...")

    def on_generate_test_clip(self):
        self.on_generate(test_clip=True)

    def _worker(self, path, output_path, cancel_event, test_clip=False):
        def progress(p):
            p = float(p or 0)
            eta_s = ""
            t0 = self._render_t0
            if t0 and p > 0.02:
                elapsed = time.time() - t0
                remain = elapsed * (1.0 - p) / p
                if remain < 90:
                    eta_s = f"  ·  ETA {int(remain)}s"
                else:
                    eta_s = f"  ·  ETA {remain / 60.0:.1f} min"
            pct = int(p * 100)
            msg = f"Rendering… {pct}%{eta_s}"

            def _ui():
                self.progress.configure(value=p * 100)
                self.status_label.config(text=msg)

            self.root.after(0, _ui)

        try:
            # Fix #7: build a RenderSettings instead of 30+ positional args.
            aspect_parts = self.aspect_var.get().split(":")
            duration = float(self.duration_var.get())
            settings = RenderSettings(
                duration_sec       = min(duration, 3.0) if test_clip else duration,
                fps                = int(self.fps_var.get()),
                out_height         = int(self.height_var.get()),
                aspect_w           = float(aspect_parts[0]),
                aspect_h           = float(aspect_parts[1]),
                use_ken_burns      = self.ken_burns_var.get(),
                zoom_amount        = self.zoom_var_float.get(),
                ken_style          = self.ken_style_var.get(),
                easing             = self.easing_var.get(),
                use_intro_zoom     = self.intro_zoom_var.get(),
                intro_duration_sec = self.intro_duration_float.get(),
                intro_zoom_level   = self.intro_level_float.get(),
                reverse_pan        = self.reverse_pan_var.get(),
                fade_in_sec        = self.fade_in_sec_var.get() if self.fade_in_var.get() else 0.0,
                fade_out_sec       = self.fade_out_sec_var.get() if self.fade_out_var.get() else 0.0,
                video_fade_in_sec  = self.video_fade_in_sec_var.get() if self.video_fade_in_var.get() else 0.0,
                video_fade_out_sec = self.video_fade_out_sec_var.get() if self.video_fade_out_var.get() else 0.0,
                audio_path         = self.audio_path.get() or None,
                audio_bitrate      = self.audio_bitrate_var.get(),
                crf                = self.crf_var.get(),
                motion_blur        = self.motion_blur_var.get(),
                cinematic_look     = self.cinematic_look_var.get(),
                codec              = self.codec_var.get(),
                custom_lut_path    = self._custom_lut_full_path,
                vignette_amount    = self.vignette_var.get(),
                look_strength      = self.look_strength_var.get(),
                film_effect        = self.film_effect_var.get(),
                film_effect_strength = self.film_effect_strength_var.get(),
                start_focus        = self.start_focus,
                end_focus          = self.end_focus,
                intro_focus        = self.intro_focus,
                film_grain         = self.film_grain_var.get(),
                film_dust          = self.film_dust_var.get(),
                film_scratches     = self.film_scratches_var.get(),
                film_flicker       = self.film_flicker_var.get(),
            )
            final_path = build_pan_video(
                path, output_path, settings,
                progress_callback=progress,
                cancel_event=cancel_event,
            )
            self.root.after(0, lambda: self._success(final_path))
        except RenderCancelled:
            self.root.after(0, self._render_cancelled)
        except Exception as e:
            self.root.after(0, lambda: self._render_failed(str(e)))

    def _reset_render_buttons(self):
        self.rendering = False
        self.cancel_event = None
        self.generate_btn.config(state="normal")
        self.cancel_btn.config(state="disabled", text="Cancel Render")

    def _render_cancelled(self):
        self._reset_render_buttons()
        self.progress["value"] = 0
        self.status_label.config(text="Render cancelled.")

    def _render_failed(self, message):
        self._reset_render_buttons()
        self.status_label.config(text="Render failed.")
        messagebox.showerror("Error", message)

    def _success(self, path):
        self._reset_render_buttons()
        self.status_label.config(text=f"Saved: {os.path.basename(path)}")
        msg = tk.messagebox.askyesno("Success", f"Video saved successfully!\n\n{path}\n\nOpen containing folder?")
        if msg:  # Fix #4: os.startfile is Windows-only
            folder = os.path.dirname(path)
            if os.name == 'nt':
                os.startfile(folder)
            elif os.name == 'posix':
                import shutil
                opener = 'open' if shutil.which('open') else 'xdg-open'
                subprocess.Popen([opener, folder])

    def _on_resolution_change(self, _=None):
        h = self.RESOLUTION_PRESETS.get(self.resolution_var.get())
        if h is None:
            self.height_entry.config(state="normal")
        else:
            self.height_var.set(str(h))
            self.height_entry.config(state="disabled")

    def _on_zoom_slide(self, v): self.zoom_var.set(f"{float(v):.2f}")
    def _on_motion_blur_slide(self, v): self.motion_blur_string_var.set(f"{float(v):.2f}")
    def _on_fade_in_slide(self, v): self.fade_in_string_var.set(f"{float(v):.2f}")
    def _on_fade_out_slide(self, v): self.fade_out_string_var.set(f"{float(v):.2f}")
    def _on_video_fade_in_slide(self, v): self.video_fade_in_string_var.set(f"{float(v):.2f}")
    def _on_video_fade_out_slide(self, v): self.video_fade_out_string_var.set(f"{float(v):.2f}")
    def _on_crf_slide(self, v): self.crf_string_var.set(str(int(float(v))))
    def _on_vignette_slide(self, v): self.vignette_string_var.set(f"{float(v):.2f}")
    def _on_look_strength_slide(self, v): self.look_strength_string_var.set(f"{float(v):.2f}")
    def _on_film_effect_strength_slide(self, v): self.film_effect_strength_string_var.set(f"{float(v):.2f}")
    def _on_film_grain_slide(self, v): self.film_grain_string_var.set(f"{float(v):.2f}")
    def _on_film_dust_slide(self, v): self.film_dust_string_var.set(f"{float(v):.2f}")
    def _on_film_scratches_slide(self, v): self.film_scratches_string_var.set(f"{float(v):.2f}")
    def _on_film_flicker_slide(self, v): self.film_flicker_string_var.set(f"{float(v):.2f}")

def main(argv=None):
    """Entry point. Optional CLI for PhotoLab / automation:

    python pano_video.py --image path/to/pano.jpg [--output-folder dir] [--preset file.json]
    """
    import argparse
    parser = argparse.ArgumentParser(description="Panorama to Video")
    parser.add_argument("--image", "-i", help="Source panorama / image path")
    parser.add_argument("--output-folder", "-o", help="Default folder for rendered videos")
    parser.add_argument("--preset", "-p", help="Optional preset JSON to load")
    parser.add_argument("--title", default=None, help="Window title override")
    args, _unknown = parser.parse_known_args(argv)

    root = tk.Tk()
    app = PanoramaToVideoApp(root)
    if args.title:
        root.title(args.title)
    else:
        root.title("PhotoLab — Panorama to Video")
    if args.image:
        app.load_image_path(args.image, output_folder=args.output_folder)
        try:
            note = "Loaded image"
            low = (args.image or "").replace("\\", "/").lower()
            if "photolab_graded" in low or "photolab_grade" in low:
                note = "Loaded graded still from PhotoLab Develop"
            app.status_label.config(text=note)
        except Exception:
            pass
    elif args.output_folder and os.path.isdir(args.output_folder):
        app.output_folder.set(args.output_folder)
    if args.preset and os.path.isfile(args.preset):
        try:
            with open(args.preset, "r", encoding="utf-8") as f:
                app._apply_settings(json.load(f))
            app.status_label.config(text=f"Preset loaded: {os.path.basename(args.preset)}")
        except Exception as e:
            print(f"Preset load failed: {e}")
    root.mainloop()

if __name__ == "__main__":
    main()
