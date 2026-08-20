"""
imaging.py — the non-destructive edit pipeline.

Everything here is plain NumPy / OpenCV with no Qt dependency.
`Recipe` is the per-image edit stack; `apply_recipe` always re-applies
to the original pixels (non-destructive).
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, asdict, field, fields
from typing import Optional, Tuple

import numpy as np
import cv2

# Suppress OpenCV's noisy TIFF tag warnings (NEF/CR2 false probes)
try:
    if hasattr(cv2, 'setLogLevel'):
        cv2.setLogLevel(3)  # ERROR only
except Exception:
    pass

# rawpy/LibRaw is not thread-safe. ThumbnailWorker and LoadImageWorker (and
# any other background threads) can call into it concurrently, which
# corrupts LibRaw's internal state and raises errors like
# "Out of order call of libraw function". Serialize all RAW decodes through
# this lock so only one thread touches rawpy at a time.
_rawpy_lock = threading.Lock()

IMAGE_EXTS = (
    # RGB
    ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp", ".heic", ".heif",
    # Canon
    ".cr2", ".cr3", ".crw",
    # Nikon
    ".nef", ".nrw",
    # Sony
    ".arw", ".sr2", ".srf",
    # Fujifilm
    ".raf",
    # Olympus / OM System
    ".orf",
    # Panasonic
    ".rw2",
    # Pentax
    ".pef", ".ptx",
    # Samsung
    ".srw",
    # Adobe / Leica / others
    ".dng", ".raw", ".rwl", ".3fr", ".fff", ".mef", ".mos", ".x3f",
)

RAW_EXTS = (
    ".cr2", ".cr3", ".crw",
    ".nef", ".nrw",
    ".arw", ".sr2", ".srf",
    ".raf",
    ".orf",
    ".rw2",
    ".pef", ".ptx",
    ".srw",
    ".dng", ".raw", ".rwl", ".3fr", ".fff", ".mef", ".mos", ".x3f",
)


@dataclass
class Recipe:
    """One image non-destructive edit stack. DxO-ish ranges."""

    exposure: float = 0.0
    smart_light: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0
    clarity: float = 0.0
    gamma: float = 1.0

    temperature: float = 5500.0
    tint: float = 0.0
    # Dual-illuminant WB (blend primary ↔ secondary)
    wb_dual: bool = False
    temperature2: float = 6500.0
    tint2: float = 0.0
    wb_mix: float = 0.0  # 0..100 toward secondary
    wb_as_shot: bool = True

    vibrance: float = 0.0
    saturation: float = 0.0

    # HSL selective (per-channel offsets, -100..100)
    # Channels: red, orange, yellow, green, aqua, blue, purple, magenta
    hsl_hue: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    hsl_sat: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    hsl_lum: tuple = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    hsl_active_channel: int = 0

    soft_proof: bool = False
    soft_proof_profile: str = "sRGB"
    soft_proof_gamut: bool = False
    soft_proof_paper_white: bool = False
    soft_proof_icc_path: str = ""
    soft_proof_intent: str = "relative"  # perceptual | relative | saturation | absolute

    local_points: list = field(default_factory=list)
    gradients: list = field(default_factory=list)  # graduated filters
    brush_masks: list = field(default_factory=list)  # painted local masks
    # Optics (manual / Lensfun-assisted)
    ca_amount: float = 0.0  # lateral chromatic aberration -100..100
    lens_auto: bool = False
    lens_strength: float = 100.0  # 0..100 Lensfun correction strength
    # Four-corner keystone: list of 4 [x,y] normalized dest corners TL,TR,BR,BL
    # None or identity = no warp
    keystone: list | None = None
    perspective_h: float = 0.0  # horizontal keystone -100..100 (simple)

    curve_shadows: float = 0.0
    curve_darks: float = 0.0
    curve_mids: float = 0.0
    curve_lights: float = 0.0
    curve_highlights: float = 0.0
    # Point curves: list of [x, y] in 0..1 (identity if empty / only endpoints)
    curve_points: list = field(default_factory=list)       # luminance / RGB master
    curve_r_points: list = field(default_factory=list)
    curve_g_points: list = field(default_factory=list)
    curve_b_points: list = field(default_factory=list)
    # Split toning
    split_shadow_hue: float = 0.0        # 0..360
    split_shadow_sat: float = 0.0        # 0..100
    split_highlight_hue: float = 0.0     # 0..360
    split_highlight_sat: float = 0.0     # 0..100
    split_balance: float = 0.0           # -100..100 (neg = more shadows, pos = more highlights)

    denoise_luminance: float = 0.0
    denoise_chroma: float = 0.0
    denoise_strength: float = 0.0
    denoise_detail: float = 50.0       # 0..100 preserve fine detail after NR
    denoise_method: str = "auto"       # auto | bilateral | nlm
    sharpen_intensity: float = 0.0     # capture / creative sharpen
    sharpen_radius: float = 1.0
    sharpen_threshold: float = 0.0     # edge masking amount 0..100
    sharpen_detail: float = 0.0        # fine structure (small radius)
    output_sharpen: float = 0.0        # output/print sharpen 0..100
    output_ppi: float = 300.0          # intended output resolution (screen ~96, print 240–360)
    output_media: str = "screen"       # screen | matte | glossy
    protect_skin: float = 0.0          # 0..100 reduce sharpen / vibrance on skin hues

    horizon: float = 0.0
    distortion: float = 0.0
    perspective: float = 0.0
    crop: Optional[Tuple[float, float, float, float]] = field(default=None)

    clearview: float = 0.0
    microcontrast: float = 0.0
    vignette: float = 0.0
    film_grain: float = 0.0
    black_and_white: bool = False
    rotate_90: int = 0  # 0,1,2,3 quarter turns
    hdr_look: float = 0.0  # 0..100 single-image HDR-style tone mapping

    def reset(self):
        blank = Recipe()
        for f in fields(self):
            setattr(self, f.name, getattr(blank, f.name))

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("crop") is not None:
            d["crop"] = list(d["crop"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        r = cls()
        for f in fields(cls):
            if f.name in d:
                val = d[f.name]
                if f.name == "crop" and val is not None:
                    val = tuple(val)
                setattr(r, f.name, val)
        return r

    def save_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str) -> "Recipe":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


def is_raw(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTS


def _silent_imread(path: str, flags=None):
    """cv2.imread with stderr silenced (hides TIFF tag warnings)."""
    if flags is None:
        flags = cv2.IMREAD_COLOR
    devnull = None
    try:
        devnull = open(os.devnull, "w")
        old_err = os.dup(2)
        os.dup2(devnull.fileno(), 2)
        try:
            return cv2.imread(path, flags)
        finally:
            os.dup2(old_err, 2)
            os.close(old_err)
    except Exception:
        return cv2.imread(path, flags)
    finally:
        if devnull is not None:
            try:
                devnull.close()
            except Exception:
                pass


def safe_imread(path: str):
    """Read an image; never call cv2.imread on RAW files."""
    if is_raw(path):
        img, _ = load_image(path, use_camera_wb=True)
        return img
    return cv2.imread(path, cv2.IMREAD_COLOR)


def extract_exif(path: str) -> dict:
    meta = {}
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        with Image.open(path) as img:
            exif = img._getexif()
            if exif:
                for tag, value in exif.items():
                    decoded = TAGS.get(tag, tag)
                    if decoded == "Model":
                        meta["camera"] = str(value)
                    elif decoded == "LensModel":
                        meta["lens"] = str(value)
                    elif decoded == "ExposureTime":
                        try:
                            val = float(value)
                            if val < 1.0:
                                meta["shutter"] = f"1/{int(round(1.0 / val))}"
                            else:
                                meta["shutter"] = f"{val}s"
                        except Exception:
                            meta["shutter"] = str(value)
                    elif decoded == "FNumber":
                        meta["aperture"] = f"f/{value}"
                    elif decoded == "ISOSpeedRatings":
                        meta["iso"] = f"ISO {value}"
                    elif decoded == "FocalLength":
                        meta["focal"] = f"{value}mm"
                    elif decoded == "DateTimeOriginal":
                        meta["datetime_original"] = str(value)
                        meta.setdefault("datetime", str(value))
                    elif decoded == "DateTime":
                        meta.setdefault("datetime", str(value))
    except Exception:
        pass
    return meta


def load_image(path: str, use_camera_wb: bool = True) -> Tuple[np.ndarray, dict]:
    meta = {"is_raw": False, "wb_multipliers": None}
    img_bgr = None
    if is_raw(path):
        try:
            import rawpy
            # Only one thread may touch LibRaw at a time (see _rawpy_lock
            # comment above) — ThumbnailWorker and LoadImageWorker run on
            # separate QThreads and can otherwise call rawpy concurrently.
            with _rawpy_lock:
                with rawpy.imread(path) as raw:
                    # use_camera_wb for as-shot; half_size=False for quality
                    # output_color=rawpy.ColorSpace.sRGB is default
                    try:
                        rgb = raw.postprocess(
                            use_camera_wb=use_camera_wb,
                            no_auto_bright=True,
                            output_bps=8,
                            bright=1.0,
                            gamma=(2.222, 4.5),  # approximate sRGB-ish display gamma
                            demosaic_algorithm=None,  # libraw default (AHD/DHT depending on build)
                        )
                    except Exception:
                        # Fallback for tricky files (some Fuji/X-Trans edge cases)
                        rgb = raw.postprocess(
                            use_camera_wb=True,
                            no_auto_bright=False,
                            output_bps=8,
                        )
                    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    meta["is_raw"] = True
                    try:
                        meta["wb_multipliers"] = list(raw.camera_whitebalance)
                    except Exception:
                        pass
                    try:
                        meta["camera"] = str(getattr(raw, "camera", "") or "")
                    except Exception:
                        pass
                    # Optics fields for Lensfun (best-effort across rawpy versions)
                    try:
                        if hasattr(raw, "lens") and raw.lens:
                            meta["lens"] = str(raw.lens)
                    except Exception:
                        pass
                    try:
                        ed = getattr(raw, "exif_dict", None) or {}
                        flat = {}
                        if isinstance(ed, dict):
                            for section in ed.values():
                                if isinstance(section, dict):
                                    flat.update(section)
                        for k, v in flat.items():
                            ks = str(k).lower().replace(" ", "")
                            if "focallength" in ks and "focal" not in meta:
                                try:
                                    meta["focal"] = f"{float(v)}mm"
                                except Exception:
                                    pass
                            if ks in ("fnumber", "aperturevalue") and "aperture" not in meta:
                                try:
                                    meta["aperture"] = f"f/{float(v)}"
                                except Exception:
                                    pass
                            if "lensmodel" in ks and "lens" not in meta:
                                meta["lens"] = str(v)
                            if ks == "make":
                                meta.setdefault("make", str(v))
                            if ks == "model" and not meta.get("camera"):
                                meta["camera"] = str(v)
                    except Exception:
                        pass
        except Exception as e:
            err = str(e)
            print(f"rawpy failed for {path}: {e}")
            # Retry only for libraw ordering / transient errors — not for true unsupported files
            if "Out of order" in err or "out of order" in err.lower():
                try:
                    import time as _time
                    _time.sleep(0.1)
                    import rawpy as _rawpy
                    with _rawpy_lock:
                        with _rawpy.imread(path) as raw:
                            rgb = raw.postprocess(
                                use_camera_wb=True,
                                no_auto_bright=False,
                                output_bps=8,
                            )
                            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            meta["is_raw"] = True
                            try:
                                meta["wb_multipliers"] = list(raw.camera_whitebalance)
                            except Exception:
                                pass
                except Exception as e2:
                    print(f"rawpy retry failed for {path}: {e2}")
    if img_bgr is None:
        if is_raw(path):
            # Don't fall back to cv2.imread() for RAW files — OpenCV's TIFF
            # reader can't parse the sensor IFD in NEF/CR2/etc. and will
            # just spew misleading TIFF warnings/errors before failing anyway.
            raise RuntimeError(f"Could not decode RAW file: {path}")
        img_bgr = _silent_imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Could not read image: {path}")
    
    # Extract EXIF and merge
    exif_data = extract_exif(path)
    for k, v in exif_data.items():
        if k == "camera" and meta.get("camera"):
            continue
        meta[k] = v
        
    return img_bgr, meta


def apply_horizon(img, angle):
    if abs(angle) < 1e-4:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_distortion(img, amount):
    if abs(amount) < 1e-4:
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
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def apply_perspective(img, amount, horizontal=0.0):
    """Simple vertical and/or horizontal perspective (trapezoid) correction."""
    if abs(amount) < 1e-4 and abs(horizontal) < 1e-4:
        return img
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])  # TL TR BR BL
    dst = src.copy()
    f = max(-0.4, min(0.4, float(amount) / 100.0 * 0.4))
    fh = max(-0.4, min(0.4, float(horizontal) / 100.0 * 0.4))
    if f >= 0:
        inset = w * f
        dst[0, 0] += inset
        dst[1, 0] -= inset
    else:
        inset = w * (-f)
        dst[3, 0] += inset
        dst[2, 0] -= inset
    if fh >= 0:
        inset = h * fh
        dst[0, 1] += inset
        dst[3, 1] -= inset
    else:
        inset = h * (-fh)
        dst[1, 1] += inset
        dst[2, 1] -= inset
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_keystone(img, corners):
    """Four-corner keystone. corners: 4×[x,y] normalized (0..1) dest TL,TR,BR,BL.

    Source is always the full image rectangle. Destination corners define the
    warp (interactive UI stores where the image corners should map to).
    """
    if not corners or len(corners) != 4:
        return img
    try:
        pts = []
        for c in corners:
            pts.append([float(c[0]), float(c[1])])
    except Exception:
        return img
    # Identity check
    identity = [[0, 0], [1, 0], [1, 1], [0, 1]]
    if all(abs(pts[i][0] - identity[i][0]) < 1e-4 and abs(pts[i][1] - identity[i][1]) < 1e-4 for i in range(4)):
        return img
    h, w = img.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[p[0] * (w - 1), p[1] * (h - 1)] for p in pts])
    # Clamp destination slightly inside to avoid degenerate transforms
    dst[:, 0] = np.clip(dst[:, 0], -0.25 * w, 1.25 * w)
    dst[:, 1] = np.clip(dst[:, 1], -0.25 * h, 1.25 * h)
    try:
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    except Exception:
        return img


def detect_horizon_angle(img_bgr, max_side=900):
    """Estimate horizon tilt (degrees) from dominant near-horizontal edges.

    Positive angle = counterclockwise (OpenCV rotation convention).
    Returns 0.0 if detection is weak.
    """
    if img_bgr is None:
        return 0.0
    src = img_bgr
    if src.dtype != np.uint8:
        src = (np.clip(src, 0, 1) * 255).astype(np.uint8) if src.max() <= 1.5 else np.clip(src, 0, 255).astype(np.uint8)
    h, w = src.shape[:2]
    scale = 1.0
    work = src
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        work = cv2.resize(src, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_len = max(work.shape[1] * 0.2, 40)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180.0, threshold=80,
        minLineLength=int(min_len), maxLineGap=20,
    )
    if lines is None or len(lines) == 0:
        return 0.0
    angles = []
    weights = []
    for line in lines[:, 0]:
        x0, y0, x1, y1 = map(float, line)
        dx, dy = x1 - x0, y1 - y0
        length = float(np.hypot(dx, dy))
        if length < 1e-3:
            continue
        ang = float(np.degrees(np.arctan2(dy, dx)))
        # Normalize to [-90, 90]
        while ang > 90:
            ang -= 180
        while ang < -90:
            ang += 180
        # Prefer near-horizontal lines (|ang| < 30°)
        if abs(ang) > 30:
            continue
        angles.append(ang)
        weights.append(length)
    if not angles:
        return 0.0
    angles = np.array(angles, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    # Weighted median
    order = np.argsort(angles)
    angles, weights = angles[order], weights[order]
    cum = np.cumsum(weights)
    mid = cum[-1] * 0.5
    idx = int(np.searchsorted(cum, mid))
    idx = min(idx, len(angles) - 1)
    ang = float(angles[idx])
    # Clamp typical horizon corrections
    return max(-15.0, min(15.0, ang))


def orientation_from_exif(meta: dict | None) -> float:
    """Return additional rotation degrees suggested by EXIF Orientation (if any).

    Most loaders already apply orientation; this returns 0 when already handled.
    """
    if not meta:
        return 0.0
    orient = meta.get("orientation") or meta.get("Orientation")
    if orient is None:
        return 0.0
    try:
        o = int(orient)
    except Exception:
        return 0.0
    # Standard EXIF orientation → CW degrees (only pure rotations)
    mapping = {1: 0.0, 3: 180.0, 6: 90.0, 8: -90.0}
    return mapping.get(o, 0.0)


def apply_crop(img, crop):
    if crop is None:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = crop
    xi0 = max(0, min(int(round(x0 * w)), w - 2))
    yi0 = max(0, min(int(round(y0 * h)), h - 2))
    xi1 = max(xi0 + 1, min(int(round(x1 * w)), w))
    yi1 = max(yi0 + 1, min(int(round(y1 * h)), h))
    return img[yi0:yi1, xi0:xi1]


def kelvin_to_rgb(kelvin):
    temp = np.clip(kelvin, 1000, 40000) / 100.0
    if temp <= 66:
        r = 255.0
    else:
        r = np.clip(329.698727446 * ((temp - 60) ** -0.1332047592), 0, 255)
    if temp <= 66:
        g = 99.4708025861 * np.log(temp) - 161.1195681661
    else:
        g = 288.1221695283 * ((temp - 60) ** -0.0755148492)
    g = np.clip(g, 0, 255)
    if temp >= 66:
        b = 255.0
    elif temp <= 19:
        b = 0.0
    else:
        b = np.clip(138.5177312231 * np.log(temp - 10) - 305.0447927307, 0, 255)
    rgb = np.array([r, g, b], dtype=np.float32) / 255.0
    rgb /= (rgb.max() + 1e-6)
    return rgb


def _wb_gains(temperature, tint):
    rgb = kelvin_to_rgb(temperature)
    tf = tint / 150.0
    rgb[0] *= (1.0 + tf * 0.4)
    rgb[1] *= (1.0 - tf * 0.6)
    rgb[2] *= (1.0 + tf * 0.4)
    rgb = np.clip(rgb, 0.2, 2.5)
    rgb /= (rgb[1] + 1e-6)
    return np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)  # BGR


def apply_white_balance(
    img, temperature, tint, as_shot=False, multipliers=None,
    dual=False, temperature2=6500.0, tint2=0.0, mix=0.0,
):
    """White balance with optional dual-illuminant blend.

    mix 0..100 blends primary (temp/tint) toward secondary (temp2/tint2).
    """
    if as_shot and multipliers is not None and not dual:
        try:
            r_m, g_m, b_m = float(multipliers[0]), float(multipliers[1]), float(multipliers[2])
            gains = np.array([b_m, g_m, r_m], dtype=np.float32)
            gains /= (gains[1] + 1e-6)
            return np.clip(img * gains[None, None, :], 0, 1)
        except Exception:
            pass

    g1 = _wb_gains(temperature, tint)
    if dual and float(mix) > 0.5:
        g2 = _wb_gains(temperature2, tint2)
        t = max(0.0, min(1.0, float(mix) / 100.0))
        gains = g1 * (1.0 - t) + g2 * t
        gains /= (gains[1] + 1e-6)
    else:
        gains = g1
    return np.clip(img * gains[None, None, :], 0, 1)


def apply_vibrance_saturation(img, vibrance, saturation):
    if abs(vibrance) < 1e-4 and abs(saturation) < 1e-4:
        return img
    hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2HSV)
    s = hsv[..., 1]
    if abs(vibrance) > 1e-4:
        amt = vibrance / 100.0
        mask = 1.0 - s
        s = np.clip(s + amt * mask * (1.0 - s) * 0.85, 0, 1)
    if abs(saturation) > 1e-4:
        s = np.clip(s * (1.0 + saturation / 100.0), 0, 1)
    hsv[..., 1] = s
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def apply_tone_curve(img, shadows, darks, mids, lights, highlights):
    """Parametric 5-region tone curve on LAB L channel."""
    if all(abs(v) < 1e-4 for v in (shadows, darks, mids, lights, highlights)):
        return img
    xs = np.array([0.0, 64.0, 128.0, 192.0, 255.0], dtype=np.float32)
    ys = np.clip(xs + np.array([shadows*0.6, darks*0.5, mids*0.4, lights*0.5, highlights*0.6], dtype=np.float32), 0, 255)
    lut = np.interp(np.arange(256, dtype=np.float32), xs, ys).astype(np.float32) / 255.0
    img = np.clip(img, 0, 1)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_idx = (np.clip(lab[..., 0] / 100.0, 0, 1) * 255).astype(np.int32)
    lab[..., 0] = lut[l_idx] * 100.0
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _points_to_lut(points, size=256):
    """Build a 0..1 LUT from sorted [[x,y], ...] control points (0..1)."""
    if not points:
        return np.linspace(0, 1, size, dtype=np.float32)
    pts = []
    for p in points:
        try:
            x, y = float(p[0]), float(p[1])
            pts.append((max(0.0, min(1.0, x)), max(0.0, min(1.0, y))))
        except Exception:
            continue
    if not pts:
        return np.linspace(0, 1, size, dtype=np.float32)
    pts = sorted(pts, key=lambda t: t[0])
    # Ensure endpoints
    if pts[0][0] > 0.0:
        pts = [(0.0, pts[0][1])] + pts
    if pts[-1][0] < 1.0:
        pts = pts + [(1.0, pts[-1][1])]
    xs = np.array([p[0] for p in pts], dtype=np.float32)
    ys = np.array([p[1] for p in pts], dtype=np.float32)
    # Deduplicate x
    uniq_x, uniq_y = [xs[0]], [ys[0]]
    for i in range(1, len(xs)):
        if xs[i] - uniq_x[-1] > 1e-6:
            uniq_x.append(xs[i])
            uniq_y.append(ys[i])
        else:
            uniq_y[-1] = ys[i]
    grid = np.linspace(0, 1, size, dtype=np.float32)
    return np.clip(np.interp(grid, uniq_x, uniq_y), 0, 1).astype(np.float32)


def apply_point_curve_luma(img, points):
    """Apply luminance point curve via LAB L channel."""
    if not points or len(points) < 2:
        return img
    lut = _points_to_lut(points)
    img = np.clip(img, 0, 1).astype(np.float32)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_idx = (np.clip(lab[..., 0] / 100.0, 0, 1) * 255).astype(np.int32)
    lab[..., 0] = lut[l_idx] * 100.0
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_rgb_point_curves(img, r_pts=None, g_pts=None, b_pts=None):
    """Per-channel RGB point curves. Input/output float BGR 0..1."""
    if not r_pts and not g_pts and not b_pts:
        return img
    img = np.clip(img, 0, 1).astype(np.float32)
    out = img.copy()
    # OpenCV BGR order
    channels = [
        (0, b_pts),
        (1, g_pts),
        (2, r_pts),
    ]
    for ch, pts in channels:
        if pts and len(pts) >= 2:
            lut = _points_to_lut(pts)
            idx = (np.clip(out[..., ch], 0, 1) * 255).astype(np.int32)
            out[..., ch] = lut[idx]
    return np.clip(out, 0, 1)


def apply_split_tone(img, sh_hue, sh_sat, hi_hue, hi_sat, balance=0.0):
    """Split toning: colorize shadows and highlights separately.

    sh_hue/hi_hue: 0..360, sat: 0..100, balance: -100..100.
    """
    if abs(sh_sat) < 0.5 and abs(hi_sat) < 0.5:
        return img
    img = np.clip(img, 0, 1).astype(np.float32)
    # Luminance mask
    lum = 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]
    # balance shifts the mid crossover
    mid = 0.5 - float(balance) / 200.0
    mid = max(0.15, min(0.85, mid))
    soft = 0.18
    hi_w = np.clip((lum - (mid - soft)) / (2.0 * soft + 1e-6), 0, 1)
    hi_w = hi_w * hi_w * (3 - 2 * hi_w)
    sh_w = 1.0 - hi_w

    def tint_color(hue_deg, sat):
        # HSV pure color at given hue, sat 0..1 → BGR
        s = max(0.0, min(1.0, float(sat) / 100.0))
        h = float(hue_deg) % 360.0
        hsv = np.array([[[h, s, 1.0]]], dtype=np.float32)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
        return bgr

    out = img.copy()
    if abs(sh_sat) >= 0.5:
        col = tint_color(sh_hue, sh_sat)
        strength = (float(sh_sat) / 100.0) * 0.55
        w = (sh_w * strength)[..., None]
        # Soft light-ish blend toward tint while preserving lum
        out = out * (1.0 - w) + (out * col[None, None, :]) * w + col[None, None, :] * (w * 0.35)
    if abs(hi_sat) >= 0.5:
        col = tint_color(hi_hue, hi_sat)
        strength = (float(hi_sat) / 100.0) * 0.55
        w = (hi_w * strength)[..., None]
        out = out * (1.0 - w) + (out * col[None, None, :]) * w + col[None, None, :] * (w * 0.25)
    return np.clip(out, 0, 1)


def estimate_exposure_stops(img_bgr, target_mid=0.36):
    """Median-luminance based exposure offset in stops."""
    if img_bgr is None:
        return 0.0
    img = img_bgr
    if img.dtype != np.float32 and img.max() > 1.5:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    else:
        g = img if img.ndim == 2 else cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
        gray = g.astype(np.float32)
        if gray.max() > 1.5:
            gray = gray / 255.0
    med = float(np.median(gray))
    if med < 1e-4:
        return 0.0
    stops = float(np.log2(target_mid / med))
    return max(-3.0, min(3.0, stops))


def estimate_wb_temp_tint(img_bgr):
    """Crude gray-world temperature (K) and tint from mean RGB."""
    if img_bgr is None:
        return 5500.0, 0.0
    img = img_bgr.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    b, g, r = float(img[..., 0].mean()), float(img[..., 1].mean()), float(img[..., 2].mean())
    avg = (r + g + b) / 3.0 + 1e-6
    rb = r / (b + 1e-6)
    temp = 5500.0 + (rb - 1.0) * 1500.0
    temp = max(2000.0, min(12000.0, temp))
    tint = (g / avg - 1.0) * 80.0
    tint = max(-150.0, min(150.0, tint))
    return temp, tint


def apply_denoise(img_bgr, luminance, chroma, strength=0.0, detail_preserve=50.0, method="auto"):
    """Edge-aware denoise in LAB with optional detail recovery.

    luminance / chroma / strength: 0..100
    detail_preserve: 0..100 — blend high-frequency residual back after NR
    method: auto | bilateral | nlm
    """
    if luminance <= 0 and chroma <= 0 and strength <= 0:
        return img_bgr
    if max(luminance, chroma, strength) / 100.0 < 0.01:
        return img_bgr

    # Work in uint8 LAB for OpenCV NR filters
    if img_bgr.dtype != np.uint8:
        u8 = np.clip(img_bgr, 0, 255).astype(np.uint8) if img_bgr.max() > 1.5 else \
             np.clip(img_bgr * 255.0, 0, 255).astype(np.uint8)
    else:
        u8 = img_bgr
    original_u8 = u8.copy()
    lab = cv2.cvtColor(u8, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    use_nlm = method == "nlm" or (method == "auto" and strength >= 35)
    use_bilateral = method == "bilateral" or (method == "auto" and strength < 35)

    if luminance > 0 or strength > 0:
        if use_nlm:
            h_l = 2.5 + (luminance / 100.0) * 10.0 + (strength / 100.0) * 9.0
            l = cv2.fastNlMeansDenoising(
                l, None, h=float(h_l), templateWindowSize=7, searchWindowSize=21
            )
        if use_bilateral or not use_nlm:
            sigma = 0.8 + (luminance / 100.0) * 9.0 + (strength / 100.0) * 4.0
            l = cv2.bilateralFilter(l, d=9, sigmaColor=sigma * 2.2, sigmaSpace=sigma * 1.2)

    if chroma > 0 or strength > 0:
        if use_nlm:
            h_c = 3.0 + (chroma / 100.0) * 16.0 + (strength / 100.0) * 10.0
            a = cv2.fastNlMeansDenoising(a, None, h=float(h_c), templateWindowSize=7, searchWindowSize=15)
            b = cv2.fastNlMeansDenoising(b, None, h=float(h_c), templateWindowSize=7, searchWindowSize=15)
        if use_bilateral or not use_nlm:
            sigma = 1.5 + (chroma / 100.0) * 14.0 + (strength / 100.0) * 5.0
            a = cv2.bilateralFilter(a, d=9, sigmaColor=sigma * 2.0, sigmaSpace=sigma)
            b = cv2.bilateralFilter(b, d=9, sigmaColor=sigma * 2.0, sigmaSpace=sigma)

    denoised = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    # Detail recovery: edge-aware blend of original high frequency
    preserve = float(np.clip(detail_preserve, 0, 100)) / 100.0
    if preserve > 0.01:
        # High-frequency residual from original
        blur_o = cv2.GaussianBlur(original_u8, (0, 0), sigmaX=1.2)
        blur_d = cv2.GaussianBlur(denoised, (0, 0), sigmaX=1.2)
        residual = original_u8.astype(np.float32) - blur_o.astype(np.float32)
        # Edge mask so we restore detail on edges more than flat noise
        gray = cv2.cvtColor(original_u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        edge = edge / (edge.max() + 1e-6)
        edge = edge[..., None]
        mix = preserve * (0.35 + 0.65 * edge)
        out = denoised.astype(np.float32) + residual * mix
        denoised = np.clip(out, 0, 255).astype(np.uint8)

    return denoised


def skin_tone_mask(img_float):
    """Soft mask 0..1 for approximate skin hues (OpenCV HSV H in 0..180)."""
    img = np.clip(img_float, 0, 1).astype(np.float32)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # Skin-ish: hue ~0–25 or ~170–180, moderate sat, mid value
    hue_ok = ((h <= 25) | (h >= 165)).astype(np.float32)
    sat_ok = np.clip((s - 0.08) / 0.45, 0, 1) * np.clip((0.75 - s) / 0.25, 0, 1)
    val_ok = np.clip((v - 0.12) / 0.25, 0, 1) * np.clip((0.98 - v) / 0.15, 0, 1)
    m = hue_ok * sat_ok * val_ok
    m = cv2.GaussianBlur(m, (0, 0), sigmaX=2.0)
    return np.clip(m, 0, 1).astype(np.float32)


def apply_sharpen(img_float, intensity, radius, threshold, detail=0.0, protect_skin=0.0):
    """Edge-masked unsharp + optional fine detail boost.

    intensity: 0..200  main USM amount
    radius: blur sigma for USM
    threshold: 0..100  edge masking (higher = only strong edges)
    detail: 0..100  small-radius structure enhancement
    protect_skin: 0..100 reduce sharpening on skin-like hues
    """
    out = img_float
    skin = None
    if float(protect_skin) > 0.5:
        skin = skin_tone_mask(out)
        skin_w = float(protect_skin) / 100.0

    if intensity > 0:
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=max(float(radius), 0.15))
        diff = out - blur
        lum = 0.114 * out[..., 0] + 0.587 * out[..., 1] + 0.299 * out[..., 2]
        gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        edge = edge / (edge.max() + 1e-6)
        if threshold > 0:
            t = float(threshold) / 100.0
            mask = np.clip((edge - t * 0.35) / max(1.0 - t * 0.35, 0.05), 0, 1)
            mask = mask * mask * (3 - 2 * mask)
        else:
            mask = 0.4 + 0.6 * edge
        if skin is not None:
            mask = mask * (1.0 - skin * skin_w)
        diff = diff * mask[..., None]
        out = out + diff * (float(intensity) / 100.0)

    if abs(detail) > 1e-4:
        fine = cv2.GaussianBlur(out, (0, 0), sigmaX=0.6)
        mid = cv2.GaussianBlur(out, (0, 0), sigmaX=1.8)
        residual = fine - mid
        if skin is not None:
            residual = residual * (1.0 - skin * skin_w)[..., None]
        out = out + residual * (float(detail) / 100.0)

    return np.clip(out, 0, 1)


def output_sharpen_params(ppi: float, media: str = "screen"):
    """Map intended output PPI + media to (amount 0..100, radius).

    Higher PPI → slightly stronger / tighter radius for print acuity.
    Media scales overall strength (matte absorbs more ink → milder).
    """
    ppi = max(36.0, min(600.0, float(ppi or 96)))
    media = (media or "screen").lower()
    # Normalize around screen 96 and print 300
    if media == "screen":
        base_amt = 25.0 + (ppi / 96.0) * 15.0
        radius = 0.55 + (96.0 / max(ppi, 48)) * 0.25
        base_amt = min(55.0, base_amt)
    elif media == "matte":
        base_amt = 35.0 + (ppi / 300.0) * 25.0
        radius = 0.7 + (300.0 / max(ppi, 120)) * 0.35
        base_amt = min(70.0, base_amt)
    else:  # glossy
        base_amt = 45.0 + (ppi / 300.0) * 35.0
        radius = 0.6 + (300.0 / max(ppi, 120)) * 0.3
        base_amt = min(85.0, base_amt)
    radius = max(0.35, min(2.5, radius))
    return float(base_amt), float(radius)


def apply_output_sharpen(img_float, amount, radius=0.8, protect_skin=0.0):
    """Output/print sharpening — modest, edge-aware, applied last before grain."""
    if amount <= 0:
        return img_float
    return apply_sharpen(
        img_float,
        intensity=float(amount) * 0.7,
        radius=radius,
        threshold=25.0,
        detail=float(amount) * 0.25,
        protect_skin=protect_skin,
    )



# HSL channel centers in OpenCV hue degrees (0-180)
_HSL_CENTERS = [0, 15, 30, 60, 90, 120, 150, 165]  # R O Y G A B P M (approx)
_HSL_WIDTH = 18  # half-width in hue degrees


def apply_hsl_selective(img, hue_offs, sat_offs, lum_offs):
    """Selective HSL per color channel. Offsets are -100..100 tuples of len 8."""
    if all(abs(v) < 1e-4 for v in list(hue_offs) + list(sat_offs) + list(lum_offs)):
        return img
    hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]  # H is 0..360 in float OpenCV? Actually 0..180 for uint8, 0..360 for float
    # OpenCV float HSV: H in [0, 360)
    for i, center in enumerate(_HSL_CENTERS):
        # Convert center from 0-180 scale to 0-360
        c = center * 2.0
        w = _HSL_WIDTH * 2.0
        # Circular distance
        dh = np.abs(h - c)
        dh = np.minimum(dh, 360.0 - dh)
        weight = np.clip(1.0 - dh / w, 0, 1)
        if abs(hue_offs[i]) > 1e-4:
            h = h + weight * (hue_offs[i] / 100.0) * 30.0  # max ~30 deg shift
        if abs(sat_offs[i]) > 1e-4:
            s = np.clip(s + weight * (sat_offs[i] / 100.0) * 0.5, 0, 1)
        if abs(lum_offs[i]) > 1e-4:
            v = np.clip(v + weight * (lum_offs[i] / 100.0) * 0.4, 0, 1)
    h = np.mod(h, 360.0)
    hsv = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# Intent map for Pillow ImageCms
_PROOF_INTENTS = {
    "perceptual": 0,
    "relative": 1,
    "saturation": 2,
    "absolute": 3,
}


def _find_system_icc(name: str) -> str | None:
    """Best-effort locate a system ICC for common profile names."""
    name = (name or "").lower().replace(" ", "")
    candidates = []
    search_dirs = [
        "/usr/share/color/icc",
        "/usr/share/color/icc/ghostscript",
        "/usr/local/share/color/icc",
        os.path.expanduser("~/.local/share/icc"),
        os.path.expanduser("~/.color/icc"),
    ]
    mapping = {
        "srgb": ["sRGB.icc", "srgb.icc", "sRGB_IEC61966-2-1_black_scaled.icc", "esrgb.icc"],
        "adobergb": ["AdobeRGB1998.icc", "a98.icc", "AdobeRGB.icc"],
        "displayp3": ["DisplayP3.icc", "DCI-P3.icc", "P3.icc"],
        "cmyk": ["default_cmyk.icc", "USWebCoatedSWOP.icc", "CoatedFOGRA39.icc", "ps_cmyk.icc"],
        "gray": ["default_gray.icc", "Gray.icc", "ps_gray.icc"],
    }
    keys = []
    if "adobe" in name:
        keys = mapping["adobergb"]
    elif "p3" in name:
        keys = mapping["displayp3"]
    elif "cmyk" in name or "print" in name:
        keys = mapping["cmyk"]
    elif "gray" in name:
        keys = mapping["gray"]
    else:
        keys = mapping["srgb"]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for root, _dirs, files in os.walk(d):
            lower = {f.lower(): os.path.join(root, f) for f in files}
            for k in keys:
                if k.lower() in lower:
                    return lower[k.lower()]
            for f in files:
                fl = f.lower()
                if name in fl and fl.endswith((".icc", ".icm")):
                    candidates.append(os.path.join(root, f))
    return candidates[0] if candidates else None


def soft_proof_gamut_percent(src: np.ndarray, proofed: np.ndarray, threshold: float = 0.08) -> float:
    """Percentage of pixels that changed more than threshold under soft-proof (OOG proxy)."""
    if src is None or proofed is None:
        return 0.0
    a = np.clip(src, 0, 1).astype(np.float32)
    b = np.clip(proofed, 0, 1).astype(np.float32)
    if a.shape != b.shape:
        return 0.0
    delta = np.max(np.abs(b - a), axis=2)
    return float(100.0 * np.mean(delta > threshold))


def apply_soft_proof(
    img,
    profile: str = "sRGB",
    gamut_warning: bool = False,
    paper_white: bool = False,
    icc_path: str = "",
    intent: str = "relative",
    return_stats: bool = False,
):
    """Soft-proof simulation for common target spaces.

    Prefers a real ICC transform via Pillow ImageCms when a profile is available
    (custom path, system ICC, or built-in sRGB). Falls back to a matrix/tone
    approximation otherwise.

    Optional gamut_warning tints likely out-of-gamut pixels magenta.
    paper_white applies a slight warm paper simulation after conversion.

    If return_stats is True, returns (image, {"gamut_percent": float, "method": str}).
    """
    img = np.clip(img, 0, 1).astype(np.float32)
    name = (profile or "sRGB").strip()
    method = "approximate"
    out = None

    # --- ICC path (Pillow ImageCms) ---
    icc = (icc_path or "").strip()
    if icc and not os.path.isfile(icc):
        icc = ""
    if not icc:
        icc = _find_system_icc(name) or ""

    if name in ("Gray", "Grayscale") and not icc:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        method = "grayscale"
    else:
        try:
            from PIL import Image, ImageCms
            # Source assumed working-space sRGB
            src_prof = ImageCms.createProfile("sRGB")
            if icc:
                dst_prof = ImageCms.getOpenProfile(icc)
            elif name in ("sRGB",):
                dst_prof = ImageCms.createProfile("sRGB")
            else:
                dst_prof = None

            if dst_prof is not None:
                intent_i = _PROOF_INTENTS.get((intent or "relative").lower(), 1)
                # ImageCms soft-proof: src -> dst -> src (display)
                # INTENT flags: use absolute for paper simulation when absolute intent chosen
                u8 = (img[..., ::-1] * 255.0).astype(np.uint8)  # BGR -> RGB
                pil = Image.fromarray(u8, mode="RGB")
                # Transform into destination, then back to sRGB for display
                xform = ImageCms.buildTransformFromOpenProfiles(
                    src_prof, dst_prof, "RGB", "RGB",
                    renderingIntent=intent_i,
                    flags=getattr(ImageCms, "FLAGS", {}).get("SOFTPROOFING", 0) or 0,
                )
                # Pillow may not expose SOFTPROOFING the same on all builds — two-step convert
                try:
                    proofed = ImageCms.applyTransform(pil, xform)
                except Exception:
                    # Manual: convert to dst then back to sRGB
                    to_dst = ImageCms.buildTransformFromOpenProfiles(
                        src_prof, dst_prof, "RGB", "RGB", renderingIntent=intent_i
                    )
                    mid = ImageCms.applyTransform(pil, to_dst)
                    back = ImageCms.buildTransformFromOpenProfiles(
                        dst_prof, src_prof, "RGB", "RGB", renderingIntent=intent_i
                    )
                    proofed = ImageCms.applyTransform(mid, back)
                arr = np.asarray(proofed).astype(np.float32) / 255.0
                out = arr[..., ::-1].copy()  # RGB -> BGR
                method = f"icc:{os.path.basename(icc) if icc else 'sRGB'}"
        except Exception:
            out = None

    # --- Approximate fallback ---
    if out is None:
        def to_linear(x):
            return np.power(np.clip(x, 0, 1), 2.2)

        def to_gamma(x):
            return np.power(np.clip(x, 0, 1), 1.0 / 2.2)

        lin = to_linear(img)
        b, g, r = lin[..., 0], lin[..., 1], lin[..., 2]

        if name in ("DisplayP3", "P3", "Display P3"):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1] * 1.06, 0, 1)
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            out = to_gamma(to_linear(out) * 0.98)
        elif name in ("AdobeRGB", "Adobe RGB"):
            mat = np.array([
                [1.04, -0.02, -0.02],
                [-0.02, 1.03, -0.01],
                [-0.02, -0.01, 1.04],
            ], dtype=np.float32)
            stacked = np.stack([b, g, r], axis=-1)
            conv = stacked @ mat.T
            out = to_gamma(np.clip(conv, 0, 1))
        elif name in ("CMYK", "Printer", "Printer (matte)"):
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1] * 0.88, 0, 1)
            hsv[..., 2] = np.clip(hsv[..., 2] * 0.92 + 0.04, 0, 1)
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            out = np.clip((out - 0.5) * 0.92 + 0.5, 0, 1)
            out = out * np.array([0.97, 0.98, 1.0], dtype=np.float32)
        elif name in ("Gray", "Grayscale"):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 0.96)
            out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        method = "approximate"
        out = np.clip(out, 0, 1).astype(np.float32)

    out = np.clip(out, 0, 1).astype(np.float32)

    # Simulate paper white (slightly warm, not pure 255)
    if paper_white:
        paper = np.array([0.94, 0.96, 0.98], dtype=np.float32)  # BGR warm paper
        out = out * paper

    gamut_pct = soft_proof_gamut_percent(img, out)

    if gamut_warning:
        delta = np.max(np.abs(out - img), axis=2)
        warn = delta > 0.08
        if np.any(warn):
            magenta = np.array([1.0, 0.2, 1.0], dtype=np.float32)
            out = out.copy()
            out[warn] = out[warn] * 0.35 + magenta * 0.65

    if return_stats:
        return out, {"gamut_percent": gamut_pct, "method": method}
    return out


def extract_embedded_preview(path: str, max_side: int = 160):
    """Extract a fast preview from RAW (embedded JPEG) or downsample regular images.

    Returns uint8 BGR or None.
    """
    if is_raw(path):
        try:
            import rawpy
            with _rawpy_lock:
                with rawpy.imread(path) as raw:
                    # Prefer thumb if present
                    try:
                        thumb = raw.extract_thumb()
                        if thumb is not None and getattr(thumb, "format", None) is not None:
                            import numpy as _np
                            data = thumb.data
                            if thumb.format == rawpy.ThumbFormat.JPEG:
                                arr = _np.frombuffer(data, dtype=_np.uint8)
                                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                                if img is not None:
                                    h, w = img.shape[:2]
                                    if max(h, w) > max_side:
                                        scale = max_side / max(h, w)
                                        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
                                    return img
                            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                                # RGB bitmap
                                rgb = _np.array(data)
                                if rgb.ndim == 3:
                                    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                                    h, w = img.shape[:2]
                                    if max(h, w) > max_side:
                                        scale = max_side / max(h, w)
                                        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
                                    return img
                    except Exception:
                        pass
                    # Fallback: half-size quick postprocess
                    try:
                        rgb = raw.postprocess(
                            use_camera_wb=True,
                            half_size=True,
                            no_auto_bright=True,
                            output_bps=8,
                        )
                        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                        h, w = img.shape[:2]
                        if max(h, w) > max_side:
                            scale = max_side / max(h, w)
                            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
                        return img
                    except Exception:
                        return None
        except Exception:
            return None
    img = _silent_imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
    return img


def apply_local_points(img, points):
    """Apply control-point local adjustments with Chroma & Luma selectivity. points: list of dicts."""
    if not points:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    
    # Convert image to HSV once to perform color matching
    hsv_img = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2HSV)
    H_img = hsv_img[..., 0] # Hue (0..360)
    S_img = hsv_img[..., 1] # Saturation (0..1)
    V_img = hsv_img[..., 2] # Value/Luma (0..1)
    
    for pt in points:
        cx_norm = float(pt.get("x", 0.5))
        cy_norm = float(pt.get("y", 0.5))
        cx = cx_norm * w
        cy = cy_norm * h
        radius = float(pt.get("radius", 0.15)) * max(w, h)
        feather = max(float(pt.get("feather", 0.5)), 0.05)
        
        yy, xx = np.mgrid[0:h, 0:w]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        
        # 1. Base distance mask
        inner = radius * (1.0 - feather)
        mask = np.ones((h, w), dtype=np.float32)
        ring = (dist > inner) & (dist < radius)
        mask[dist >= radius] = 0
        if radius > inner:
            mask[ring] = 1.0 - (dist[ring] - inner) / (radius - inner + 1e-6)
            
        # 2. Chroma & Luma Selectivity (similarity to center sample)
        chroma_sel = float(pt.get("chroma", 100.0))
        luma_sel = float(pt.get("luma", 100.0))

        if chroma_sel < 99.5 or luma_sel < 99.5:
            sample_x = max(0, min(int(cx), w - 1))
            sample_y = max(0, min(int(cy), h - 1))
            H_target = H_img[sample_y, sample_x]
            S_target = S_img[sample_y, sample_x]
            V_target = V_img[sample_y, sample_x]
            similarity = np.ones((h, w), dtype=np.float32)

            if chroma_sel < 99.5:
                dist_h = np.abs(H_img - H_target)
                dist_h = np.minimum(dist_h, 360.0 - dist_h) / 180.0
                dist_s = np.abs(S_img - S_target)
                chroma_diff = dist_h * 0.7 + dist_s * 0.3
                sensitivity = max((chroma_sel / 100.0) ** 1.5, 1e-4)
                similarity *= np.clip(1.0 - chroma_diff / sensitivity, 0, 1)

            if luma_sel < 99.5:
                dist_v = np.abs(V_img - V_target)
                sensitivity = max((luma_sel / 100.0) ** 1.5, 1e-4)
                similarity *= np.clip(1.0 - dist_v / sensitivity, 0, 1)

            mask *= similarity

        # 3. Absolute luminance range (0..100 UI → 0..1). Only affect tones in range.
        luma_min = float(pt.get("luma_min", 0.0)) / 100.0
        luma_max = float(pt.get("luma_max", 100.0)) / 100.0
        if luma_min > 0.001 or luma_max < 0.999:
            lo, hi = min(luma_min, luma_max), max(luma_min, luma_max)
            # Soft edges (~4% of range or fixed 0.04)
            soft = 0.04
            range_mask = np.ones((h, w), dtype=np.float32)
            range_mask[V_img < lo - soft] = 0.0
            range_mask[V_img > hi + soft] = 0.0
            mid_lo = (V_img >= lo - soft) & (V_img < lo)
            mid_hi = (V_img > hi) & (V_img <= hi + soft)
            if soft > 1e-6:
                range_mask[mid_lo] = (V_img[mid_lo] - (lo - soft)) / soft
                range_mask[mid_hi] = ((hi + soft) - V_img[mid_hi]) / soft
            mask *= range_mask

        mask = mask[..., None]
        
        # 3. Apply adjustments
        local = out.copy()
        exp = float(pt.get("exposure", 0.0))
        if abs(exp) > 1e-4:
            local = local * (2.0 ** exp)
        sat = float(pt.get("saturation", 0.0))
        if abs(sat) > 1e-4:
            hsv = cv2.cvtColor(np.clip(local, 0, 1).astype(np.float32), cv2.COLOR_BGR2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sat / 100.0), 0, 1)
            local = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        con = float(pt.get("contrast", 0.0))
        if abs(con) > 1e-4:
            local = (local - 0.5) * (1.0 + con / 100.0) + 0.5
        cl = float(pt.get("clarity", 0.0))
        if abs(cl) > 1e-4:
            blur = cv2.GaussianBlur(local, (0, 0), sigmaX=2)
            local = local + (local - blur) * (cl / 100.0)
            
        out = out * (1.0 - mask) + np.clip(local, 0, 1) * mask
        
    return np.clip(out, 0, 1)



def apply_gradients(img, gradients):
    """Apply graduated (linear) filters. gradients: list of dicts with
    x0,y0,x1,y1 (normalized), feather 0..1, and adjustment keys.
    """
    if not gradients:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xx / max(w - 1, 1)
    yn = yy / max(h - 1, 1)

    for g in gradients:
        x0 = float(g.get("x0", 0.5))
        y0 = float(g.get("y0", 0.0))
        x1 = float(g.get("x1", 0.5))
        y1 = float(g.get("y1", 1.0))
        feather = max(float(g.get("feather", 0.5)), 0.05)

        dx = x1 - x0
        dy = y1 - y0
        length = float(np.sqrt(dx * dx + dy * dy)) + 1e-6
        # Project each pixel onto the gradient axis; 0 at start, 1 at end
        proj = ((xn - x0) * dx + (yn - y0) * dy) / (length * length)
        # Soft mask: 0 at start side, 1 at end side
        # Transition width controlled by feather around midpoint
        mid = 0.5
        half = 0.5 * feather + 0.05
        mask = np.clip((proj - (mid - half)) / (2.0 * half + 1e-6), 0, 1)
        # Smoothstep
        mask = mask * mask * (3.0 - 2.0 * mask)
        mask = mask.astype(np.float32)[..., None]

        local = out.copy()
        exp = float(g.get("exposure", 0.0))
        if abs(exp) > 1e-4:
            local = local * (2.0 ** exp)
        con = float(g.get("contrast", 0.0))
        if abs(con) > 1e-4:
            local = (local - 0.5) * (1.0 + con / 100.0) + 0.5
        sat = float(g.get("saturation", 0.0))
        if abs(sat) > 1e-4:
            hsv = cv2.cvtColor(np.clip(local, 0, 1).astype(np.float32), cv2.COLOR_BGR2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sat / 100.0), 0, 1)
            local = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cl = float(g.get("clarity", 0.0))
        if abs(cl) > 1e-4:
            blur = cv2.GaussianBlur(local, (0, 0), sigmaX=2)
            local = local + (local - blur) * (cl / 100.0)
        temp = float(g.get("temperature", 0.0))  # relative -100..100 shift
        if abs(temp) > 1e-4:
            # simple warm/cool: boost R or B
            gains = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # BGR
            gains[2] *= 1.0 + (temp / 100.0) * 0.15  # R
            gains[0] *= 1.0 - (temp / 100.0) * 0.15  # B
            local = local * gains[None, None, :]

        out = out * (1.0 - mask) + np.clip(local, 0, 1) * mask

    return np.clip(out, 0, 1)




def _brush_dab(mask, cx, cy, rad, hardness, flow, mode="add"):
    """Composite one circular dab into mask in-place. mode: add|subtract|intersect."""
    h, w = mask.shape[:2]
    x0 = max(int(cx - rad - 2), 0)
    x1 = min(int(cx + rad + 2), w - 1)
    y0 = max(int(cy - rad - 2), 0)
    y1 = min(int(cy + rad + 2), h - 1)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1 + 1, x0:x1 + 1].astype(np.float32)
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(rad, 1e-6)
    core = max(0.0, min(1.0, hardness))
    fall = np.clip(1.0 - (d - core) / max(1.0 - core, 0.05), 0, 1)
    fall = fall * fall * (3 - 2 * fall)
    fall[d > 1.0] = 0
    fall = fall * float(flow)
    region = mask[y0:y1 + 1, x0:x1 + 1]
    if mode == "subtract":
        mask[y0:y1 + 1, x0:x1 + 1] = np.clip(region - fall, 0, 1)
    elif mode == "intersect":
        # Keep only where dab overlaps existing mask, scaled by flow
        mask[y0:y1 + 1, x0:x1 + 1] = region * fall
    else:
        mask[y0:y1 + 1, x0:x1 + 1] = np.maximum(region, fall)


def apply_brush_masks(img, masks):
    """Apply painted brush local adjustments.

    Each mask: {
      "strokes": [ {"x":0.5,"y":0.5,"r":0.05, "flow":1.0, "mode":"add"}, ... ],
      "exposure", "contrast", "saturation", "clarity", "temperature",
      "hardness": 0..1,
      "flow": 0..1 (default dab strength),
      "opacity": 0..1 (overall mask strength),
      "mode": "add"|"subtract"|"intersect",
      "invert": bool,
    }
    """
    if not masks:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    for m in masks:
        strokes = m.get("strokes") or []
        if not strokes and not m.get("raster"):
            continue
        hardness = float(m.get("hardness", 0.7))
        default_flow = float(m.get("flow", 1.0))
        opacity = float(m.get("opacity", 1.0))
        default_mode = (m.get("mode") or "add").lower()
        mask = np.zeros((h, w), dtype=np.float32)

        # Optional precomputed raster mask (e.g. subject detect) — normalized 0..1, may be smaller
        raster = m.get("raster")
        if raster is not None:
            try:
                arr = np.asarray(raster, dtype=np.float32)
                if arr.ndim == 2:
                    if arr.shape[0] != h or arr.shape[1] != w:
                        arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
                    mask = np.clip(arr, 0, 1)
            except Exception:
                pass

        for s in strokes:
            cx = float(s.get("x", 0.5)) * (w - 1)
            cy = float(s.get("y", 0.5)) * (h - 1)
            rad = max(float(s.get("r", 0.05)) * max(w, h), 1.0)
            flow = float(s.get("flow", default_flow))
            mode = (s.get("mode") or default_mode).lower()
            _brush_dab(mask, cx, cy, rad, hardness, flow, mode=mode)

        if mask.max() < 1e-6:
            continue
        if m.get("invert") or m.get("inverted"):
            mask = 1.0 - mask
        # Overall opacity scales the mask
        mask = np.clip(mask * max(0.0, min(1.0, opacity)), 0, 1)
        mask3 = mask[..., None]
        local = out.copy()
        exp = float(m.get("exposure", 0.0))
        if abs(exp) > 1e-4:
            local = local * (2.0 ** exp)
        con = float(m.get("contrast", 0.0))
        if abs(con) > 1e-4:
            local = (local - 0.5) * (1.0 + con / 100.0) + 0.5
        sat = float(m.get("saturation", 0.0))
        if abs(sat) > 1e-4:
            hsv = cv2.cvtColor(np.clip(local, 0, 1).astype(np.float32), cv2.COLOR_BGR2HSV)
            hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + sat / 100.0), 0, 1)
            local = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cl = float(m.get("clarity", 0.0))
        if abs(cl) > 1e-4:
            blur = cv2.GaussianBlur(local, (0, 0), sigmaX=2)
            local = local + (local - blur) * (cl / 100.0)
        temp = float(m.get("temperature", 0.0))
        if abs(temp) > 1e-4:
            gains = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            gains[2] *= 1.0 + (temp / 100.0) * 0.15
            gains[0] *= 1.0 - (temp / 100.0) * 0.15
            local = local * gains[None, None, :]
        out = out * (1.0 - mask3) + np.clip(local, 0, 1) * mask3
    return np.clip(out, 0, 1)


def generate_subject_mask(img_bgr, max_side: int = 640):
    """Offline subject mask via OpenCV GrabCut (no neural net).

    img_bgr: float 0..1 or uint8 BGR.
    Returns float32 mask 0..1 at the input resolution.
    """
    if img_bgr is None:
        return None
    src = img_bgr
    if src.dtype != np.uint8:
        src_u8 = np.clip(src * 255.0 if src.max() <= 1.5 else src, 0, 255).astype(np.uint8)
    else:
        src_u8 = src
    h, w = src_u8.shape[:2]
    scale = 1.0
    work = src_u8
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        work = cv2.resize(src_u8, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    wh, ww = work.shape[:2]
    # Center rectangle as probable foreground
    margin = 0.12
    rect = (
        int(ww * margin),
        int(wh * margin),
        max(1, int(ww * (1 - 2 * margin))),
        max(1, int(wh * (1 - 2 * margin))),
    )
    mask = np.zeros((wh, ww), np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(work, mask, rect, bgd, fgd, 3, cv2.GC_INIT_WITH_RECT)
        binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 1.0, 0.0).astype(np.float32)
    except Exception:
        # Fallback: center-weighted luminance threshold
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        yy, xx = np.mgrid[0:wh, 0:ww].astype(np.float32)
        cy, cx = (wh - 1) / 2.0, (ww - 1) / 2.0
        dist = np.sqrt(((xx - cx) / max(cx, 1)) ** 2 + ((yy - cy) / max(cy, 1)) ** 2)
        binary = np.clip(1.0 - dist * 0.85, 0, 1) * (0.4 + 0.6 * gray)
        binary = (binary > 0.35).astype(np.float32)
    # Feather edges
    binary = cv2.GaussianBlur(binary, (0, 0), sigmaX=max(ww, wh) * 0.01)
    if scale < 0.999:
        binary = cv2.resize(binary, (w, h), interpolation=cv2.INTER_LINEAR)
    return np.clip(binary, 0, 1).astype(np.float32)


def apply_chromatic_aberration_fix(img, amount):
    """Simple lateral CA correction: shift R/B channels radially. amount -100..100."""
    if abs(amount) < 1e-4:
        return img
    h, w = img.shape[:2]
    amt = float(amount) / 100.0 * 0.008  # subtle
    cy, cx = h / 2.0, w / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # radial distance normalized
    dx = (xx - cx) / max(cx, 1)
    dy = (yy - cy) / max(cy, 1)
    # map R outward, B inward (or opposite based on sign)
    map_x_r = (cx + (xx - cx) * (1.0 + amt)).astype(np.float32)
    map_y_r = (cy + (yy - cy) * (1.0 + amt)).astype(np.float32)
    map_x_b = (cx + (xx - cx) * (1.0 - amt)).astype(np.float32)
    map_y_b = (cy + (yy - cy) * (1.0 - amt)).astype(np.float32)
    b, g, r = cv2.split(np.clip(img, 0, 1).astype(np.float32))
    r2 = cv2.remap(r, map_x_r, map_y_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    b2 = cv2.remap(b, map_x_b, map_y_b, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.merge([b2, g, r2])


def _open_lensfun_database():
    """Open lensfunpy.Database, preferring a local ./lensfun tree next to the app.

    Returns (db, path_or_note). path_or_note is the DB directory used, or a
    short note when falling back to the system/bundled database.
    """
    import lensfunpy  # type: ignore

    try:
        from app_paths import lensfun_db_paths
        paths = lensfun_db_paths()
    except Exception:
        paths = []

    # lensfunpy accepts paths= list of directories containing version_N or xml
    for p in paths:
        try:
            db = lensfunpy.Database(paths=[p], load_common=False, load_bundled=False)
            # Sanity: empty DB is useless
            if hasattr(db, "cameras") and len(getattr(db, "cameras", []) or []) == 0:
                # Some builds use find_cameras only — try a dummy query instead
                pass
            return db, p
        except TypeError:
            # Older lensfunpy may not accept keyword args the same way
            try:
                db = lensfunpy.Database(paths=[p])
                return db, p
            except Exception:
                continue
        except Exception:
            continue

    # System / bundled defaults
    try:
        db = lensfunpy.Database()
        return db, "(system/bundled)"
    except Exception as e:
        raise RuntimeError(f"Could not open Lensfun database: {e}") from e


def probe_lensfun(meta):
    """Return a status dict describing Lensfun availability and EXIF match (no image warp)."""
    info = {
        "installed": False,
        "camera_query": (meta or {}).get("camera") or "",
        "lens_query": (meta or {}).get("lens") or "",
        "camera_match": None,
        "lens_match": None,
        "db_path": None,
        "message": "",
    }
    try:
        import lensfunpy  # type: ignore  # noqa: F401
    except Exception:
        info["message"] = "lensfunpy not installed (pip install lensfunpy)"
        return info
    info["installed"] = True
    try:
        db, db_note = _open_lensfun_database()
        info["db_path"] = db_note
        cam_maker = (info["camera_query"] or "").split()[0] if info["camera_query"] else None
        cams = db.find_cameras(cam_maker, info["camera_query"]) if cam_maker else []
        if not cams and cam_maker:
            cams = db.find_cameras(cam_maker, None)
        if not cams:
            info["message"] = (
                f"No camera match for “{info['camera_query'] or '?'}” "
                f"(DB: {db_note})"
            )
            return info
        cam = cams[0]
        info["camera_match"] = f"{getattr(cam, 'maker', '')} {getattr(cam, 'model', '')}".strip()
        lens_model = info["lens_query"]
        lenses = db.find_lenses(cam, None, lens_model) if lens_model else []
        if not lenses:
            lenses = db.find_lenses(cam)
        if not lenses:
            info["message"] = (
                f"Camera OK ({info['camera_match']}), no lens match for "
                f"“{lens_model or '?'}” (DB: {db_note})"
            )
            return info
        lens = lenses[0]
        info["lens_match"] = f"{getattr(lens, 'maker', '')} {getattr(lens, 'model', '')}".strip()
        info["message"] = f"Match: {info['camera_match']} · {info['lens_match']} · DB: {db_note}"
        return info
    except Exception as e:
        info["message"] = f"Lensfun error: {e}"
        return info


def try_lensfun_correct(img, meta, strength=1.0):
    """Optional Lensfun geometry correction if lensfunpy is installed.
    Returns (img, message). Falls back unchanged if unavailable.
    strength 0..1 blends corrected result with original.
    """
    try:
        import lensfunpy  # type: ignore
    except Exception:
        return img, "Lensfun not installed (pip install lensfunpy)"
    try:
        db, db_note = _open_lensfun_database()
        cam_maker = (meta.get("camera") or "").split()[0] if meta.get("camera") else None
        cam_model = meta.get("camera") or ""
        lens_model = meta.get("lens") or ""
        focal = 50.0
        try:
            focal = float(str(meta.get("focal", "50")).replace("mm", "").strip())
        except Exception:
            pass
        aperture = 4.0
        try:
            aperture = float(str(meta.get("aperture", "4")).replace("f/", "").strip())
        except Exception:
            pass
        cams = db.find_cameras(cam_maker, cam_model) if cam_maker else []
        if not cams and cam_maker:
            cams = db.find_cameras(cam_maker, None)
        if not cams:
            return img, f"No Lensfun camera match ({cam_model or '?'}) DB={db_note}"
        cam = cams[0]
        lenses = db.find_lenses(cam, None, lens_model) if lens_model else []
        if not lenses:
            lenses = db.find_lenses(cam)
        if not lenses:
            return img, f"No Lensfun lens match ({lens_model or '?'}) DB={db_note}"
        lens = lenses[0]
        h, w = img.shape[:2]
        mod = lensfunpy.Modifier(lens, cam.crop_factor, w, h)
        mod.initialize(focal, aperture, 1.0)
        coords = mod.apply_geometry_distortion()
        if coords is not None:
            was_float = img.dtype != np.uint8 and float(np.max(img)) <= 1.5
            if was_float:
                img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            else:
                img_u8 = np.clip(img, 0, 255).astype(np.uint8)
            und = cv2.remap(img_u8, coords[0], coords[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            s = max(0.0, min(1.0, float(strength)))
            if s < 0.999:
                und = cv2.addWeighted(und, s, img_u8, 1.0 - s, 0)
            if was_float:
                img = und.astype(np.float32) / 255.0
            else:
                img = und
        cam_s = f"{getattr(cam, 'maker', '')} {getattr(cam, 'model', '')}".strip()
        lens_s = f"{getattr(lens, 'maker', '')} {getattr(lens, 'model', '')}".strip()
        return img, (
            f"Lensfun OK · {cam_s} · {lens_s} · strength {int(strength * 100)}% · DB: {db_note}"
        )
    except Exception as e:
        return img, f"Lensfun error: {e}"


def apply_recipe(img_bgr, r, wb_multipliers=None, meta=None):
    rot = int(getattr(r, "rotate_90", 0)) % 4
    if rot:
        img_bgr = np.rot90(img_bgr, rot).copy()
    # Lensfun early (works on pixel grid before creative geometry)
    if getattr(r, "lens_auto", False) and meta:
        strength = float(getattr(r, "lens_strength", 100.0) or 100.0) / 100.0
        img_bgr, _msg = try_lensfun_correct(
            img_bgr.astype(np.float32) / 255.0 if img_bgr.dtype == np.uint8 else img_bgr,
            meta, strength=strength,
        )
        if img_bgr.dtype != np.uint8 and float(np.max(img_bgr)) <= 1.5:
            img_bgr = (np.clip(img_bgr, 0, 1) * 255).astype(np.uint8)
    img_bgr = apply_distortion(img_bgr, r.distortion)
    img_bgr = apply_perspective(
        img_bgr, r.perspective, horizontal=getattr(r, "perspective_h", 0.0),
    )
    ks = getattr(r, "keystone", None)
    if ks:
        img_bgr = apply_keystone(img_bgr, ks)
    img_bgr = apply_horizon(img_bgr, r.horizon)
    img_bgr = apply_crop(img_bgr, r.crop)
    img_bgr = apply_denoise(
        img_bgr, r.denoise_luminance, r.denoise_chroma, r.denoise_strength,
        detail_preserve=getattr(r, 'denoise_detail', 50.0),
        method=getattr(r, 'denoise_method', 'auto'),
    )

    img = img_bgr.astype(np.float32) / 255.0
    img = apply_white_balance(
        img, r.temperature, r.tint,
        as_shot=r.wb_as_shot, multipliers=wb_multipliers,
        dual=bool(getattr(r, "wb_dual", False)),
        temperature2=getattr(r, "temperature2", 6500.0),
        tint2=getattr(r, "tint2", 0.0),
        mix=getattr(r, "wb_mix", 0.0),
    )

    if abs(r.exposure) > 1e-4:
        img *= (2.0 ** r.exposure)

    if abs(r.smart_light) > 1e-4:
        lum = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(lum, (0, 0), sigmaX=max(img.shape[1] / 20, 1))
        img += ((0.5 - blur) * (r.smart_light / 100.0) * 0.6)[..., None]

    if any(abs(v) > 1e-4 for v in (r.highlights, r.shadows, r.whites, r.blacks)):
        lum = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2GRAY)
        hi_mask = np.clip((lum - 0.55) * 2.2, 0, 1) ** 1.4
        lo_mask = np.clip((0.45 - lum) * 2.2, 0, 1) ** 1.4
        white_mask = np.clip((lum - 0.75) * 4.0, 0, 1)
        black_mask = np.clip((0.25 - lum) * 4.0, 0, 1)
        img += (r.highlights / 100.0) * 0.45 * hi_mask[..., None]
        img += (r.shadows / 100.0) * 0.45 * lo_mask[..., None]
        img += (r.whites / 100.0) * 0.35 * white_mask[..., None]
        img += (r.blacks / 100.0) * 0.35 * black_mask[..., None]

    if abs(r.contrast) > 1e-4:
        img = (img - 0.5) * (1.0 + r.contrast / 100.0) + 0.5

    if abs(r.clarity) > 1e-4:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        img = img + (img - blur) * (r.clarity / 100.0)

    img = apply_tone_curve(img, r.curve_shadows, r.curve_darks, r.curve_mids, r.curve_lights, r.curve_highlights)
    img = apply_point_curve_luma(img, getattr(r, "curve_points", None) or [])
    img = apply_rgb_point_curves(
        img,
        getattr(r, "curve_r_points", None) or [],
        getattr(r, "curve_g_points", None) or [],
        getattr(r, "curve_b_points", None) or [],
    )
    img = np.clip(img, 0, 1)
    if abs(r.gamma - 1.0) > 1e-4:
        img = img ** (1.0 / r.gamma)

    # Vibrance/sat with optional skin protection (blend unprocessed skin back)
    vib_src = img
    img = apply_vibrance_saturation(img, r.vibrance, r.saturation)
    ps = float(getattr(r, "protect_skin", 0.0) or 0.0)
    if ps > 0.5 and (abs(r.vibrance) > 1e-4 or abs(r.saturation) > 1e-4):
        sm = skin_tone_mask(vib_src)
        w = (sm * (ps / 100.0))[..., None]
        img = img * (1.0 - w) + vib_src * w

    # Selective HSL
    hue_o = r.hsl_hue if r.hsl_hue is not None else (0,) * 8
    sat_o = r.hsl_sat if r.hsl_sat is not None else (0,) * 8
    lum_o = r.hsl_lum if r.hsl_lum is not None else (0,) * 8
    img = apply_hsl_selective(img, hue_o, sat_o, lum_o)

    # Split toning
    img = apply_split_tone(
        img,
        getattr(r, "split_shadow_hue", 0.0),
        getattr(r, "split_shadow_sat", 0.0),
        getattr(r, "split_highlight_hue", 0.0),
        getattr(r, "split_highlight_sat", 0.0),
        getattr(r, "split_balance", 0.0),
    )

    # Local control points
    if r.local_points:
        img = apply_local_points(img, r.local_points)

    # Graduated filters
    grads = getattr(r, "gradients", None) or []
    if grads:
        img = apply_gradients(img, grads)

    # Soft proof (preview only feel — still applied in pipeline for consistency)
    if r.soft_proof:
        img = apply_soft_proof(
            img, r.soft_proof_profile,
            gamut_warning=getattr(r, "soft_proof_gamut", False),
            paper_white=getattr(r, "soft_proof_paper_white", False),
            icc_path=getattr(r, "soft_proof_icc_path", "") or "",
            intent=getattr(r, "soft_proof_intent", "relative") or "relative",
        )
    ps = float(getattr(r, "protect_skin", 0.0) or 0.0)
    img = apply_sharpen(
        img, r.sharpen_intensity, r.sharpen_radius, r.sharpen_threshold,
        detail=getattr(r, 'sharpen_detail', 0.0),
        protect_skin=ps,
    )
    out_amt = float(getattr(r, "output_sharpen", 0.0) or 0.0)
    if out_amt > 1e-4:
        # Prefer explicit amount; radius from PPI/media when available
        media = getattr(r, "output_media", "screen") or "screen"
        ppi = float(getattr(r, "output_ppi", 300.0) or 300.0)
        _sug_amt, sug_radius = output_sharpen_params(ppi, media)
        img = apply_output_sharpen(img, out_amt, radius=sug_radius, protect_skin=ps)

    # ClearView Plus approximation: local contrast / dehaze
    if abs(getattr(r, "clearview", 0.0)) > 1e-4:
        amt = r.clearview / 100.0
        lab = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2LAB)
        l = lab[..., 0] / 100.0
        blur = cv2.GaussianBlur(l, (0, 0), sigmaX=max(img.shape[1] / 12, 2))
        # Unsharp + lift shadows slightly (dehaze-ish)
        l = np.clip(l + (l - blur) * amt * 1.2 + (0.5 - blur) * amt * 0.15, 0, 1)
        lab[..., 0] = l * 100.0
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # Microcontrast (finer than clarity)
    if abs(getattr(r, "microcontrast", 0.0)) > 1e-4:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2)
        img = img + (img - blur) * (r.microcontrast / 100.0)

    if abs(getattr(r, "hdr_look", 0.0)) > 1e-4:
        img = apply_hdr_look(img, r.hdr_look)

    if abs(r.vignette) > 1e-4:
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        d = np.sqrt(((xx - cx) / (cx + 1e-6)) ** 2 + ((yy - cy) / (cy + 1e-6)) ** 2)
        d = np.clip(d, 0, 1.4) / 1.4
        img *= (1.0 - (r.vignette / 100.0) * (d ** 2))[..., None]

    # Black and white
    if getattr(r, "black_and_white", False):
        gray = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Film grain
    if abs(getattr(r, "film_grain", 0.0)) > 1e-4:
        amt = r.film_grain / 100.0
        noise = np.random.randn(*img.shape[:2]).astype(np.float32) * (amt * 0.08)
        img = np.clip(img + noise[..., None], 0, 1)

    return (np.clip(img, 0, 1) * 255.0).astype(np.uint8)


def apply_hdr_look(img, amount):
    """Single-image HDR-style tone mapping on float BGR [0,1]. amount 0..100."""
    if abs(amount) < 1e-4:
        return img
    amt = float(np.clip(amount, 0.0, 100.0)) / 100.0
    img = np.clip(img, 0, 1).astype(np.float32)
    for sigma, weight in ((max(img.shape[1] / 25.0, 3.0), 0.55), (2.5, 0.35)):
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
        img = img + (img - blur) * (amt * weight)
    lum = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2GRAY)
    lo = np.clip((0.45 - lum) * 2.0, 0, 1) ** 1.2
    hi = np.clip((lum - 0.55) * 2.0, 0, 1) ** 1.2
    img = img + (lo * (amt * 0.22))[..., None]
    img = img - (hi * (amt * 0.12))[..., None]
    img = (img - 0.5) * (1.0 + amt * 0.18) + 0.5
    hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + amt * 0.12), 0, 1)
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return np.clip(img, 0, 1)


def merge_hdr_mertens(paths, align=True, max_dim=0,
                      contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.0):
    """Fuse multiple exposures with OpenCV MergeMertens. Returns uint8 BGR."""
    if not paths or len(paths) < 2:
        raise ValueError("HDR merge needs at least 2 images")
    images = []
    for path in paths:
        img, _meta = load_image(path, use_camera_wb=True)
        if img is None:
            raise RuntimeError(f"Could not load: {path}")
        if max_dim and max(img.shape[:2]) > max_dim:
            h, w = img.shape[:2]
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        images.append(img)
    min_h = min(im.shape[0] for im in images)
    min_w = min(im.shape[1] for im in images)
    cropped = []
    for im in images:
        h, w = im.shape[:2]
        y0 = (h - min_h) // 2
        x0 = (w - min_w) // 2
        cropped.append(im[y0:y0 + min_h, x0:x0 + min_w])
    images = cropped
    if align and len(images) >= 2:
        try:
            align_mtb = cv2.createAlignMTB()
            align_mtb.process(images, images)
        except Exception as e:
            print(f"HDR align skipped: {e}")
    merger = cv2.createMergeMertens(
        contrast_weight=contrast_weight,
        saturation_weight=saturation_weight,
        exposure_weight=exposure_weight,
    )
    fused = merger.process(images)
    return np.clip(fused * 255.0, 0, 255).astype(np.uint8)


def recipe_to_dict(r) -> dict:
    """Serialize Recipe to a JSON-friendly dict."""
    from dataclasses import asdict, is_dataclass
    if is_dataclass(r):
        d = asdict(r)
    else:
        d = dict(r.__dict__)
    # tuples → lists
    for k, v in list(d.items()):
        if isinstance(v, tuple):
            d[k] = list(v)
    return d


def recipe_from_dict(d: dict):
    """Create Recipe from dict (sidecar / preset). Unknown keys ignored."""
    r = Recipe()
    if not d:
        return r
    for k, v in d.items():
        if not hasattr(r, k):
            continue
        cur = getattr(r, k)
        if isinstance(cur, tuple) and isinstance(v, (list, tuple)):
            setattr(r, k, tuple(float(x) for x in v))
        else:
            try:
                setattr(r, k, v)
            except Exception:
                pass
    return r


def sidecar_path(image_path: str) -> str:
    return image_path + ".photolab.json"


def load_sidecar_data(image_path: str) -> dict:
    """Load full sidecar JSON (recipe + optional snapshots). Returns {} if missing."""
    import json
    path = sidecar_path(image_path)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"sidecar load failed: {e}")
        return {}


def save_sidecar_data(image_path: str, data: dict) -> str:
    """Write full sidecar JSON. Merges with existing keys when possible."""
    import json
    path = sidecar_path(image_path)
    existing = load_sidecar_data(image_path)
    merged = dict(existing)
    merged.update(data or {})
    merged.setdefault("version", 1)
    merged.setdefault("image", os.path.basename(image_path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return path


def save_recipe_sidecar(image_path: str, recipe, snapshots: list | None = None) -> str:
    """Save recipe (and optional named snapshots) next to the image."""
    data = {
        "version": 1,
        "image": os.path.basename(image_path),
        "recipe": recipe_to_dict(recipe),
    }
    if snapshots is not None:
        data["snapshots"] = list(snapshots)
    else:
        # Preserve existing snapshots when only the recipe is updated
        existing = load_sidecar_data(image_path)
        if existing.get("snapshots"):
            data["snapshots"] = existing["snapshots"]
    return save_sidecar_data(image_path, data)


def load_recipe_sidecar(image_path: str):
    data = load_sidecar_data(image_path)
    if not data:
        return None
    try:
        return recipe_from_dict(data.get("recipe") or data)
    except Exception as e:
        print(f"sidecar recipe parse failed: {e}")
        return None


def load_snapshots_sidecar(image_path: str) -> list:
    """Return list of {name, recipe, ts?} from the sidecar."""
    data = load_sidecar_data(image_path)
    snaps = data.get("snapshots") or []
    if not isinstance(snaps, list):
        return []
    out = []
    for s in snaps:
        if isinstance(s, dict) and s.get("name") and s.get("recipe") is not None:
            out.append(s)
    return out


def save_snapshots_sidecar(image_path: str, snapshots: list) -> str:
    """Update only the snapshots array in the sidecar (creates file if needed)."""
    data = load_sidecar_data(image_path)
    data["snapshots"] = list(snapshots or [])
    data.setdefault("version", 1)
    data.setdefault("image", os.path.basename(image_path))
    if "recipe" not in data:
        data["recipe"] = recipe_to_dict(Recipe())
    return save_sidecar_data(image_path, data)


def apply_watermark(img_bgr, text, opacity=0.45, scale=0.035, margin=0.02):
    """Draw a simple text watermark bottom-right on uint8 BGR image."""
    if not text or img_bgr is None:
        return img_bgr
    out = img_bgr.copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(h, w) * scale / 30.0
    thickness = max(1, int(round(font_scale * 1.5)))
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = int(w - tw - margin * w)
    y = int(h - margin * h - baseline)
    x = max(0, x)
    y = max(th + 2, y)
    overlay = out.copy()
    cv2.putText(overlay, text, (x, y), font, font_scale, (255, 255, 255), thickness + 2, cv2.LINE_AA)
    cv2.putText(overlay, text, (x, y), font, font_scale, (20, 20, 20), thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, float(opacity), out, 1.0 - float(opacity), 0, out)
    return out
