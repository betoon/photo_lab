"""
imaging.py — the non-destructive edit pipeline.

Everything here is plain NumPy / OpenCV with no Qt dependency.
`Recipe` is the per-image edit stack; `apply_recipe` always re-applies
to the original pixels (non-destructive).
"""

from __future__ import annotations

import json
import os
import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass, asdict, field, fields
from typing import Optional, Tuple

import numpy as np
import cv2

log = logging.getLogger(__name__)

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

# Pillow's large-image setting is process-global, so guard and restore it.
_pillow_open_lock = threading.RLock()


@contextmanager
def _pillow_large_image_context():
    try:
        from PIL import Image
    except Exception:
        yield
        return
    with _pillow_open_lock:
        previous = getattr(Image, "MAX_IMAGE_PIXELS", None)
        try:
            Image.MAX_IMAGE_PIXELS = None
            yield
        finally:
            Image.MAX_IMAGE_PIXELS = previous


@contextmanager
def safe_pil_open(path: str, *args, **kwargs):
    """Open a trusted local image while restoring Pillow's safety limit."""
    from PIL import Image
    with _pillow_large_image_context():
        image = Image.open(path, *args, **kwargs)
        try:
            yield image
        finally:
            image.close()

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
    wb_as_shot: bool = True
    # Creative WB is relative and independent of the camera/absolute WB.
    # This is what converted presets with small +/- Temperature values use.
    creative_temperature: float = 0.0  # -100..100 warm/cool shift
    creative_tint: float = 0.0         # -100..100 green/magenta shift

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

    local_points: list = field(default_factory=list)
    gradients: list = field(default_factory=list)  # graduated filters
    brush_masks: list = field(default_factory=list)  # painted local masks
    # Optics (manual / Lensfun-assisted)
    ca_amount: float = 0.0  # lateral chromatic aberration -100..100
    lens_auto: bool = False

    curve_shadows: float = 0.0
    curve_darks: float = 0.0
    curve_mids: float = 0.0
    curve_lights: float = 0.0
    curve_highlights: float = 0.0
    curve_points: list = field(default_factory=list)
    curve_r_points: list = field(default_factory=list)
    curve_g_points: list = field(default_factory=list)
    curve_b_points: list = field(default_factory=list)
    split_shadow_hue: float = 0.0
    split_shadow_sat: float = 0.0
    split_highlight_hue: float = 0.0
    split_highlight_sat: float = 0.0
    split_balance: float = 0.0

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

    horizon: float = 0.0
    distortion: float = 0.0
    perspective: float = 0.0
    crop: Optional[Tuple[float, float, float, float]] = field(default=None)

    clearview: float = 0.0
    microcontrast: float = 0.0
    vignette: float = 0.0
    film_grain: float = 0.0
    black_and_white: bool = False
    # Infrared specialty
    ir_channel_swap: str = "none"  # none | rb | br
    ir_false_color: float = 0.0     # 0..100 blend toward classic false-color IR
    ir_mono: bool = False           # mono IR (weighted toward red/NIR)
    # Astro specialty
    astro_stretch: float = 0.0      # 0..100 asinh / histogram stretch
    astro_bg_remove: float = 0.0    # 0..100 gradient / sky background subtraction
    astro_star_emphasis: float = 0.0  # 0..100 mild star edge boost

    # Ansel Adams zone system (B&W)
    zone_enabled: bool = False
    zone_placement: float = 5.0
    zone_expansion: float = 0.0
    zone_filter: str = "none"
    zone_snap: float = 0.0
    zone_overlay: bool = False
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
        from PIL.ExifTags import TAGS
        with safe_pil_open(path) as img:
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
                    elif decoded == "GPSInfo":
                        try:
                            lat, lon = _parse_gps_info(value)
                            if lat is not None and lon is not None:
                                meta["gps_latitude"] = lat
                                meta["gps_longitude"] = lon
                                meta["gps"] = (lat, lon)
                        except Exception:
                            pass
    except Exception:
        pass
    return meta


def _gps_ratio_to_float(rat):
    try:
        if hasattr(rat, "numerator"):
            return float(rat.numerator) / float(rat.denominator or 1)
        if isinstance(rat, (tuple, list)) and len(rat) >= 2:
            return float(rat[0]) / float(rat[1] or 1)
        return float(rat)
    except Exception:
        return 0.0


def _parse_gps_info(gps_info):
    """Parse PIL GPSInfo dict → (lat, lon) decimal degrees or (None, None)."""
    if not gps_info or not isinstance(gps_info, dict):
        return None, None
    try:
        from PIL.ExifTags import GPSTAGS
        tagged = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}
    except Exception:
        tagged = gps_info

    def _to_deg(values, ref):
        if not values or len(values) < 3:
            return None
        d = _gps_ratio_to_float(values[0])
        m = _gps_ratio_to_float(values[1])
        s = _gps_ratio_to_float(values[2])
        dec = d + m / 60.0 + s / 3600.0
        if str(ref) in ("S", "W"):
            dec = -dec
        return dec

    lat = _to_deg(tagged.get("GPSLatitude"), tagged.get("GPSLatitudeRef") or "N")
    lon = _to_deg(tagged.get("GPSLongitude"), tagged.get("GPSLongitudeRef") or "E")
    return lat, lon


def extract_gps(path: str):
    """Return (lat, lon) or None for an image path."""
    meta = extract_exif(path)
    if meta.get("gps"):
        return meta["gps"]
    if meta.get("gps_latitude") is not None and meta.get("gps_longitude") is not None:
        return float(meta["gps_latitude"]), float(meta["gps_longitude"])
    return None


def format_raw_error(path: str, err: Optional[BaseException] = None) -> str:
    """Human-readable RAW decode failure with actionable hints."""
    name = os.path.basename(path or "") or path
    ext = os.path.splitext(name)[1].lower()
    msg = str(err) if err else "unknown error"
    low = msg.lower()
    hints = []
    if "partial" in low or "truncated" in low or "unexpected end" in low:
        hints.append("File may be incomplete (partial download / interrupted transfer).")
    if "unsupported" in low or "not supported" in low or "compression" in low:
        hints.append("Unsupported compression or camera variant for this LibRaw/rawpy build.")
    if "out of order" in low:
        hints.append("Transient LibRaw lock/order issue — try opening the file again.")
    if "permission" in low or "access" in low:
        hints.append("Check file permissions or that the file is not locked by another app.")
    if not hints:
        if ext in (".nef", ".nrw"):
            hints.append("Nikon NEF tip: very new bodies may need a newer rawpy/LibRaw.")
        elif ext in (".cr2", ".cr3"):
            hints.append("Canon tip: CR3 often needs a recent LibRaw; try updating rawpy.")
        elif ext in (".arw", ".srf"):
            hints.append("Sony tip: ensure rawpy is current for newer ARW versions.")
        elif ext in (".raf",):
            hints.append("Fuji X-Trans tip: some RAF packs need half_size or updated demosaic.")
        elif ext in (".dng",):
            hints.append("DNG tip: non-standard vendor DNGs can fail; try Adobe DNG Converter.")
        else:
            hints.append("See Help / USER_MANUAL for RAW support notes.")
    tip = " ".join(hints)
    return (
        f"Could not decode RAW “{name}”.\n"
        f"Details: {msg}\n"
        f"{tip}"
    )

# Global preference: RAW decode bit depth (8 or 16).
_RAW_OUTPUT_BPS = 8

def load_image(path: str, use_camera_wb: bool = True, output_bps: Optional[int] = None) -> Tuple[np.ndarray, dict]:
    """Decode BGR image data at 8-bit preview or 16-bit export precision."""
    decode_bps = 16 if int(output_bps or 8) >= 16 else 8
    meta = {"is_raw": False, "wb_multipliers": None, "wb_baked": False,
            "decode_bps": decode_bps}
    img_bgr = None
    raw_failure = None
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
                            output_bps=decode_bps,
                            bright=1.0,
                            gamma=(2.222, 4.5),  # approximate sRGB-ish display gamma
                            demosaic_algorithm=None,  # libraw default (AHD/DHT depending on build)
                        )
                    except Exception:
                        # Fallback for tricky files (some Fuji/X-Trans edge cases)
                        rgb = raw.postprocess(
                            use_camera_wb=True,
                            no_auto_bright=False,
                            output_bps=decode_bps,
                        )
                    img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                    meta["is_raw"] = True
                    # LibRaw has already applied the camera multipliers to
                    # this rendered RGB image. apply_recipe must not apply
                    # camera_whitebalance a second time.
                    meta["wb_baked"] = bool(use_camera_wb)
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
            raw_failure = e
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
                                output_bps=decode_bps,
                            )
                            img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                            meta["is_raw"] = True
                            meta["wb_baked"] = True
                            try:
                                meta["wb_multipliers"] = list(raw.camera_whitebalance)
                            except Exception:
                                pass
                except Exception as e2:
                    raw_failure = e2
                    print(f"rawpy retry failed for {path}: {e2}")
    if img_bgr is None:
        if is_raw(path):
            # Don't fall back to cv2.imread() for RAW files — OpenCV's TIFF
            # reader can't parse the sensor IFD in NEF/CR2/etc. and will
            # just spew misleading TIFF warnings/errors before failing anyway.
            raise RuntimeError(format_raw_error(path, raw_failure))
        flags = cv2.IMREAD_UNCHANGED if decode_bps == 16 else cv2.IMREAD_COLOR
        img_bgr = _silent_imread(path, flags)
        if img_bgr is None:
            raise RuntimeError(f"Could not read image: {path}")
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        elif img_bgr.shape[2] == 4:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_BGRA2BGR)
    
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


def apply_perspective(img, amount):
    if abs(amount) < 1e-4:
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


def apply_white_balance(img, temperature, tint, as_shot=False, multipliers=None):
    if as_shot and multipliers is not None:
        try:
            r_m, g_m, b_m = float(multipliers[0]), float(multipliers[1]), float(multipliers[2])
            gains = np.array([b_m, g_m, r_m], dtype=np.float32)
            gains /= (gains[1] + 1e-6)
            return np.clip(img * gains[None, None, :], 0, 1)
        except Exception:
            pass
    rgb = kelvin_to_rgb(temperature)
    tf = tint / 150.0
    rgb[0] *= (1.0 + tf * 0.4)
    rgb[1] *= (1.0 - tf * 0.6)
    rgb[2] *= (1.0 + tf * 0.4)
    rgb = np.clip(rgb, 0.2, 2.5)
    rgb /= (rgb[1] + 1e-6)
    bgr_gain = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32)
    return np.clip(img * bgr_gain[None, None, :], 0, 1)


def apply_creative_white_balance(img, temperature_shift=0.0, tint_shift=0.0):
    """Apply a relative creative WB shift without replacing technical WB."""
    temperature_shift = float(np.clip(temperature_shift or 0.0, -100.0, 100.0))
    tint_shift = float(np.clip(tint_shift or 0.0, -100.0, 100.0))
    if abs(temperature_shift) < 1e-5 and abs(tint_shift) < 1e-5:
        return img
    # Map the relative slider around the neutral reference. Positive is warm.
    kelvin = 5500.0 + temperature_shift * 35.0
    return apply_white_balance(img, kelvin, tint_shift, as_shot=False, multipliers=None)


def _image_to_float01(image):
    """Normalize uint8/uint16/float image data into float32 0..1."""
    if np.issubdtype(image.dtype, np.integer):
        scale = float(np.iinfo(image.dtype).max)
        return image.astype(np.float32) / scale
    out = image.astype(np.float32, copy=False)
    if out.size and float(np.nanmax(out)) > 1.5:
        out = out / 255.0
    return out


def _float01_to_dtype(image, output_dtype):
    clipped = np.clip(image, 0, 1)
    dtype = np.dtype(output_dtype)
    if np.issubdtype(dtype, np.floating):
        return clipped.astype(dtype)
    maximum = float(np.iinfo(dtype).max)
    if dtype == np.dtype(np.uint8):
        # Preserve the established preview/render rounding behavior exactly.
        return (clipped * maximum).astype(dtype)
    return np.rint(clipped * maximum).astype(dtype)


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
    if not points:
        return np.linspace(0, 1, size, dtype=np.float32)
    pts = []
    for point in points:
        try:
            x, y = float(point[0]), float(point[1])
            pts.append((np.clip(x, 0, 1), np.clip(y, 0, 1)))
        except (TypeError, ValueError, IndexError):
            continue
    if not pts:
        return np.linspace(0, 1, size, dtype=np.float32)
    pts.sort(key=lambda item: item[0])
    if pts[0][0] > 0:
        pts.insert(0, (0.0, pts[0][1]))
    if pts[-1][0] < 1:
        pts.append((1.0, pts[-1][1]))
    unique = {}
    for x, y in pts:
        unique[float(x)] = float(y)
    xs = np.array(sorted(unique), dtype=np.float32)
    ys = np.array([unique[float(x)] for x in xs], dtype=np.float32)
    return np.interp(np.linspace(0, 1, size), xs, ys).astype(np.float32)


def apply_point_curve_luma(img, points):
    if not points or len(points) < 2:
        return img
    lut = _points_to_lut(points)
    lab = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2LAB)
    indices = (np.clip(lab[..., 0] / 100.0, 0, 1) * 255).astype(np.int32)
    lab[..., 0] = lut[indices] * 100.0
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def apply_rgb_point_curves(img, r_pts=None, g_pts=None, b_pts=None):
    if not r_pts and not g_pts and not b_pts:
        return img
    out = np.clip(img, 0, 1).astype(np.float32).copy()
    for channel, points in ((0, b_pts), (1, g_pts), (2, r_pts)):
        if points and len(points) >= 2:
            lut = _points_to_lut(points)
            indices = (out[..., channel] * 255).astype(np.int32)
            out[..., channel] = lut[indices]
    return np.clip(out, 0, 1)


def apply_split_tone(img, sh_hue, sh_sat, hi_hue, hi_sat, balance=0.0):
    if abs(sh_sat) < 0.5 and abs(hi_sat) < 0.5:
        return img
    out = np.clip(img, 0, 1).astype(np.float32)
    lum = 0.114 * out[..., 0] + 0.587 * out[..., 1] + 0.299 * out[..., 2]
    midpoint = np.clip(0.5 - float(balance) / 200.0, 0.15, 0.85)
    hi_weight = np.clip((lum - (midpoint - 0.18)) / 0.36, 0, 1)
    hi_weight = hi_weight * hi_weight * (3 - 2 * hi_weight)
    sh_weight = 1.0 - hi_weight

    def tint_color(hue, saturation):
        hsv = np.array([[[float(hue) % 360.0, np.clip(float(saturation) / 100.0, 0, 1), 1.0]]], dtype=np.float32)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]

    if abs(sh_sat) >= 0.5:
        color = tint_color(sh_hue, sh_sat)
        weight = (sh_weight * (float(sh_sat) / 100.0) * 0.55)[..., None]
        out = out * (1 - weight) + (out * color) * weight + color * (weight * 0.35)
    if abs(hi_sat) >= 0.5:
        color = tint_color(hi_hue, hi_sat)
        weight = (hi_weight * (float(hi_sat) / 100.0) * 0.55)[..., None]
        out = out * (1 - weight) + (out * color) * weight + color * (weight * 0.25)
    return np.clip(out, 0, 1)


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
        if np.issubdtype(img_bgr.dtype, np.integer):
            maximum = float(np.iinfo(img_bgr.dtype).max)
            u8 = np.rint(np.clip(img_bgr, 0, maximum) / maximum * 255.0).astype(np.uint8)
        else:
            scale = 255.0 if img_bgr.size and float(np.nanmax(img_bgr)) <= 1.5 else 1.0
            u8 = np.clip(img_bgr * scale, 0, 255).astype(np.uint8)
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


def apply_sharpen(img_float, intensity, radius, threshold, detail=0.0):
    """Edge-masked unsharp + optional fine detail boost.

    intensity: 0..200  main USM amount
    radius: blur sigma for USM
    threshold: 0..100  edge masking (higher = only strong edges)
    detail: 0..100  small-radius structure enhancement
    """
    out = img_float
    if intensity > 0:
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=max(float(radius), 0.15))
        diff = out - blur
        # Edge mask from luminance gradient
        lum = 0.114 * out[..., 0] + 0.587 * out[..., 1] + 0.299 * out[..., 2]
        gx = cv2.Sobel(lum, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(lum, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        edge = edge / (edge.max() + 1e-6)
        if threshold > 0:
            # Soft threshold: suppress low-contrast (noise) regions
            t = float(threshold) / 100.0
            mask = np.clip((edge - t * 0.35) / max(1.0 - t * 0.35, 0.05), 0, 1)
            mask = mask * mask * (3 - 2 * mask)
            diff = diff * mask[..., None]
        else:
            # Still mild edge bias to avoid sharpening noise in flats
            mask = 0.4 + 0.6 * edge
            diff = diff * mask[..., None]
        out = out + diff * (float(intensity) / 100.0)

    if abs(detail) > 1e-4:
        # Fine structure: difference of Gaussians at small scale
        fine = cv2.GaussianBlur(out, (0, 0), sigmaX=0.6)
        mid = cv2.GaussianBlur(out, (0, 0), sigmaX=1.8)
        residual = fine - mid
        out = out + residual * (float(detail) / 100.0)

    return np.clip(out, 0, 1)


def apply_output_sharpen(img_float, amount, radius=0.8):
    """Output/print sharpening — modest, edge-aware, applied last before grain."""
    if amount <= 0:
        return img_float
    return apply_sharpen(img_float, intensity=float(amount) * 0.7, radius=radius,
                         threshold=25.0, detail=float(amount) * 0.25)



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


def apply_soft_proof(img, profile: str, gamut_warning: bool = False):
    """Soft-proof simulation for common target spaces.

    Uses a simple RGB matrix / tone curve approximation (no external ICC required).
    Optional gamut_warning tints out-of-gamut-ish pixels magenta.
    """
    img = np.clip(img, 0, 1).astype(np.float32)
    name = (profile or "sRGB").strip()

    if name in ("Gray", "Grayscale"):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Work in linear-ish space with a cheap gamma expand/compress
    def to_linear(x):
        return np.power(np.clip(x, 0, 1), 2.2)

    def to_gamma(x):
        return np.power(np.clip(x, 0, 1), 1.0 / 2.2)

    lin = to_linear(img)
    b, g, r = lin[..., 0], lin[..., 1], lin[..., 2]

    if name in ("DisplayP3", "P3", "Display P3"):
        # Approximate: slightly expand saturation toward P3-like look, then mild compress
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * 1.06, 0, 1)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        # Soft highlight roll-off
        out = to_gamma(to_linear(out) * 0.98)
    elif name in ("AdobeRGB", "Adobe RGB"):
        # Map toward a wider-gamut feel then clip back (hint of conversion)
        mat = np.array([
            [1.04, -0.02, -0.02],
            [-0.02, 1.03, -0.01],
            [-0.02, -0.01, 1.04],
        ], dtype=np.float32)
        stacked = np.stack([b, g, r], axis=-1)
        conv = stacked @ mat.T
        out = to_gamma(conv)
        out = np.clip(out, 0, 1)
        # Reorder to BGR
        out = out[..., ::-1] if False else out  # already BGR order from stack
        # stacked was B,G,R so keep as BGR
    elif name in ("CMYK", "Printer", "Printer (matte)"):
        # Emulate ink limit: compress contrast + desaturate slightly + warm paper
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * 0.88, 0, 1)
        hsv[..., 2] = np.clip(hsv[..., 2] * 0.92 + 0.04, 0, 1)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out = (out - 0.5) * 0.92 + 0.5
        out = np.clip(out, 0, 1)
        # Paper white tint
        out = out * np.array([0.97, 0.98, 1.0], dtype=np.float32)  # BGR slight warm
    else:
        # sRGB proof: clip extreme saturation, gentle contrast toward display
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 0.96)
        out = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        out = np.clip(out, 0, 1)

    out = np.clip(out, 0, 1).astype(np.float32)

    if gamut_warning:
        # Flag pixels that changed a lot vs original as potential OOG
        delta = np.max(np.abs(out - img), axis=2)
        warn = delta > 0.08
        if np.any(warn):
            magenta = np.array([1.0, 0.0, 1.0], dtype=np.float32)  # BGR? B=1,G=0,R=1 -> magenta-ish in BGR is B+R
            magenta = np.array([1.0, 0.2, 1.0], dtype=np.float32)
            out = out.copy()
            out[warn] = out[warn] * 0.35 + magenta * 0.65

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
            
        # 2. Chroma & Luma Selectivity
        chroma_sel = float(pt.get("chroma", 100.0))
        luma_sel = float(pt.get("luma", 100.0))
        
        if chroma_sel < 99.5 or luma_sel < 99.5:
            # Sample target color at center point (clamped to image bounds)
            sample_x = max(0, min(int(cx), w - 1))
            sample_y = max(0, min(int(cy), h - 1))
            
            H_target = H_img[sample_y, sample_x]
            S_target = S_img[sample_y, sample_x]
            V_target = V_img[sample_y, sample_x]
            
            # Calculate color similarity
            similarity = np.ones((h, w), dtype=np.float32)
            
            if chroma_sel < 99.5:
                # Hue distance (circular 0..180 degrees mapped to 0..1)
                dist_h = np.abs(H_img - H_target)
                dist_h = np.minimum(dist_h, 360.0 - dist_h) / 180.0
                
                # Saturation distance
                dist_s = np.abs(S_img - S_target)
                
                # Combined Chroma difference
                chroma_diff = dist_h * 0.7 + dist_s * 0.3
                
                # Sensitivity mapping: lower selectivity = stricter match
                sensitivity = (chroma_sel / 100.0) ** 1.5
                if sensitivity < 1e-4:
                    sensitivity = 1e-4
                
                # Linear falloff of match based on difference
                color_match = np.clip(1.0 - chroma_diff / sensitivity, 0, 1)
                similarity *= color_match
                
            if luma_sel < 99.5:
                dist_v = np.abs(V_img - V_target)
                
                sensitivity = (luma_sel / 100.0) ** 1.5
                if sensitivity < 1e-4:
                    sensitivity = 1e-4
                    
                luma_match = np.clip(1.0 - dist_v / sensitivity, 0, 1)
                similarity *= luma_match
                
            mask *= similarity
            
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




def build_brush_mask(img, spec):
    """Rasterize one brush spec, including feather and luminance/color ranges."""
    h, w = img.shape[:2]
    strokes = spec.get("strokes") or []
    mask = np.zeros((h, w), dtype=np.float32)
    hardness = float(np.clip(spec.get("hardness", 0.7), 0.0, 1.0))
    edge_refine = float(np.clip(spec.get("edge_refine", 0.0), 0.0, 1.0))
    reference_colors = []
    for s in strokes:
        cx = float(s.get("x", 0.5)) * (w - 1)
        cy = float(s.get("y", 0.5)) * (h - 1)
        rad = max(float(s.get("r", 0.05)) * max(w, h), 1.0)
        x0, x1 = max(int(cx-rad-2), 0), min(int(cx+rad+2), w-1)
        y0, y1 = max(int(cy-rad-2), 0), min(int(cy+rad+2), h-1)
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1].astype(np.float32)
        d = np.sqrt((xx-cx)**2 + (yy-cy)**2) / rad
        fall = np.clip(1.0-(d-hardness)/max(1.0-hardness, 0.05), 0, 1)
        fall = fall*fall*(3-fall*2)
        fall[d > 1.0] = 0
        sample = img[int(np.clip(round(cy), 0, h-1)), int(np.clip(round(cx), 0, w-1))]
        reference_colors.append(sample)
        if edge_refine > 1e-4:
            patch = img[y0:y1+1, x0:x1+1]
            distance = np.linalg.norm(patch-sample, axis=2) / np.sqrt(3.0)
            similarity = np.exp(-distance / max(0.025, 0.35*(1.0-edge_refine)+0.025))
            fall *= (1.0-edge_refine) + edge_refine*similarity
        mask[y0:y1+1, x0:x1+1] = np.maximum(mask[y0:y1+1, x0:x1+1], fall)
    feather = float(np.clip(spec.get("feather", 0.0), 0.0, 100.0))
    if feather > 0 and mask.any():
        sigma = max(0.1, feather/100.0 * max(h, w) * 0.015)
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)
        if mask.max() > 1e-6:
            mask /= mask.max()
    lum = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2GRAY)
    lo = float(np.clip(spec.get("luminance_min", 0.0), 0.0, 1.0))
    hi = float(np.clip(spec.get("luminance_max", 1.0), 0.0, 1.0))
    range_feather = max(0.005, float(spec.get("range_feather", 0.05)))
    if lo > 0 or hi < 1:
        low_gate = np.clip((lum-lo)/range_feather, 0, 1)
        high_gate = np.clip((hi-lum)/range_feather, 0, 1)
        mask *= low_gate*high_gate
    if spec.get("color_range") and reference_colors:
        target = np.asarray(spec.get("color_target", np.mean(reference_colors, axis=0)), np.float32)
        tolerance = max(0.01, float(spec.get("color_tolerance", 0.2)))
        distance = np.linalg.norm(img-target, axis=2) / np.sqrt(3.0)
        mask *= np.clip(1.0-distance/tolerance, 0, 1)
    if spec.get("invert") or spec.get("inverted"):
        mask = 1.0-mask
    return np.clip(mask, 0, 1).astype(np.float32)


def apply_brush_masks(img, masks):
    """Apply painted brush local adjustments.

    Each mask: {
      "strokes": [ {"x":0.5,"y":0.5,"r":0.05}, ... ],  # normalized coords & radius
      "exposure", "contrast", "saturation", "clarity", "temperature",
      "hardness": 0..1
    }
    """
    if not masks:
        return img
    h, w = img.shape[:2]
    out = img.copy()
    raster_masks = {}
    for index, m in enumerate(masks):
        mask = build_brush_mask(img, m)
        raster_masks[str(m.get("id", index))] = mask
        refs = m.get("intersect_with") or []
        for ref in refs:
            if str(ref) in raster_masks:
                mask = np.minimum(mask, raster_masks[str(ref)])
        if mask.max() < 1e-6:
            continue
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


def try_lensfun_correct(img, meta, strength=1.0):
    """Optional Lensfun geometry correction if lensfunpy is installed.
    Returns (img, message). Falls back unchanged if unavailable.
    """
    try:
        import lensfunpy  # type: ignore
    except Exception:
        return img, "Lensfun not installed (pip install lensfunpy)"
    try:
        db = lensfunpy.Database()
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
        if not cams:
            cams = db.find_cameras(None, None)[:1]
        if not cams:
            return img, "No Lensfun camera match"
        cam = cams[0]
        lenses = db.find_lenses(cam, None, lens_model) if lens_model else db.find_lenses(cam)
        if not lenses:
            return img, "No Lensfun lens match"
        lens = lenses[0]
        h, w = img.shape[:2]
        mod = lensfunpy.Modifier(lens, cam.crop_factor, w, h)
        mod.initialize(focal, aperture, 1.0)
        # distortion + vignetting on float image
        coords = mod.apply_geometry_distortion()
        if coords is not None:
            img_u8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            und = cv2.remap(img_u8, coords[0], coords[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            img = und.astype(np.float32) / 255.0
        return img, f"Lensfun: {getattr(lens, 'model', lens_model)}"
    except Exception as e:
        return img, f"Lensfun error: {e}"



def apply_ir_processing(img, r):
    """Infrared specialty looks on float BGR [0,1]. Additive; no-ops when defaults."""
    import numpy as np
    swap = str(getattr(r, "ir_channel_swap", "none") or "none").lower()
    false_amt = float(getattr(r, "ir_false_color", 0.0) or 0.0) / 100.0
    mono = bool(getattr(r, "ir_mono", False))
    if swap == "none" and false_amt < 1e-6 and not mono:
        return img
    out = np.clip(img, 0, 1).astype(np.float32).copy()
    b, g, rch = out[..., 0], out[..., 1], out[..., 2]
    if swap in ("rb", "r-b", "r_b"):
        # Classic IR channel swap: R <-> B
        out[..., 0], out[..., 2] = rch.copy(), b.copy()
        b, g, rch = out[..., 0], out[..., 1], out[..., 2]
    elif swap in ("br", "b-r", "b_r"):
        out[..., 0], out[..., 2] = rch.copy(), b.copy()
        # same physical swap; kept as alias
        b, g, rch = out[..., 0], out[..., 1], out[..., 2]
    if false_amt > 1e-6:
        # Push toward wood-effect-ish false color (cyan sky / warm foliage tendencies)
        # Operate in a mild channel remix
        b2 = np.clip(0.15 * rch + 0.25 * g + 0.60 * b, 0, 1)
        g2 = np.clip(0.25 * rch + 0.55 * g + 0.20 * b, 0, 1)
        r2 = np.clip(0.70 * rch + 0.25 * g + 0.05 * b, 0, 1)
        out[..., 0] = b * (1 - false_amt) + b2 * false_amt
        out[..., 1] = g * (1 - false_amt) + g2 * false_amt
        out[..., 2] = rch * (1 - false_amt) + r2 * false_amt
        b, g, rch = out[..., 0], out[..., 1], out[..., 2]
    if mono:
        # NIR-weighted mono (favor red channel as stand-in for IR-rich signal)
        lum = np.clip(0.15 * b + 0.25 * g + 0.60 * rch, 0, 1)
        out[..., 0] = out[..., 1] = out[..., 2] = lum
    return np.clip(out, 0, 1).astype(np.float32)


def apply_astro_processing(img, r):
    """Astro stretch + background gradient removal on float BGR [0,1]."""
    import numpy as np
    import cv2
    stretch = float(getattr(r, "astro_stretch", 0.0) or 0.0)
    bg = float(getattr(r, "astro_bg_remove", 0.0) or 0.0)
    stars = float(getattr(r, "astro_star_emphasis", 0.0) or 0.0)
    if stretch < 0.5 and bg < 0.5 and stars < 0.5:
        return img
    out = np.clip(img, 0, 1).astype(np.float32).copy()
    h, w = out.shape[:2]
    if bg > 0.5:
        # Large Gaussian as sky model; subtract scaled residual
        k = max(31, int(min(h, w) * 0.25) | 1)
        k = min(k, 251)
        try:
            sky = cv2.GaussianBlur(out, (k, k), sigmaX=k * 0.25)
        except Exception:
            sky = cv2.blur(out, (k, k))
        strength = (bg / 100.0) * 0.85
        out = np.clip(out - sky * strength + np.median(sky) * strength * 0.35, 0, 1)
    if stretch > 0.5:
        # Per-channel asinh stretch anchored near black point
        amt = stretch / 100.0
        # Estimate black from dark percentile
        flat = out.reshape(-1, 3)
        lo = np.percentile(flat, 1.0, axis=0).astype(np.float32)
        work = np.clip(out - lo, 0, 1)
        # soft scale: higher stretch → more aggressive midtone lift
        scale = 1.0 + amt * 12.0
        stretched = np.arcsinh(work * scale) / np.arcsinh(scale)
        # blend so 0 stretch = original
        out = out * (1.0 - amt) + stretched * amt
        out = np.clip(out, 0, 1)
    if stars > 0.5:
        # Mild unsharp on luminance to emphasize point sources
        amt = stars / 100.0 * 0.6
        try:
            blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.2)
            detail = out - blur
            out = np.clip(out + detail * amt, 0, 1)
        except Exception:
            pass
    return out.astype(np.float32)


def apply_recipe(img_bgr, r, wb_multipliers=None, meta=None, output_dtype=np.uint8):
    rot = int(getattr(r, "rotate_90", 0)) % 4
    if rot:
        img_bgr = np.rot90(img_bgr, rot).copy()
    img_bgr = apply_distortion(img_bgr, r.distortion)
    img_bgr = apply_perspective(img_bgr, r.perspective)
    img_bgr = apply_horizon(img_bgr, r.horizon)
    img_bgr = apply_crop(img_bgr, r.crop)
    img_bgr = apply_denoise(
        img_bgr, r.denoise_luminance, r.denoise_chroma, r.denoise_strength,
        detail_preserve=getattr(r, 'denoise_detail', 50.0),
        method=getattr(r, 'denoise_method', 'auto'),
    )

    img = _image_to_float01(img_bgr)
    wb_already_baked = bool(meta and meta.get("wb_baked"))
    if not (r.wb_as_shot and wb_already_baked):
        img = apply_white_balance(
            img, r.temperature, r.tint,
            as_shot=r.wb_as_shot, multipliers=wb_multipliers,
        )
    img = apply_creative_white_balance(
        img,
        getattr(r, "creative_temperature", 0.0),
        getattr(r, "creative_tint", 0.0),
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

    img = apply_vibrance_saturation(img, r.vibrance, r.saturation)

    # Selective HSL
    hue_o = r.hsl_hue if r.hsl_hue is not None else (0,) * 8
    sat_o = r.hsl_sat if r.hsl_sat is not None else (0,) * 8
    lum_o = r.hsl_lum if r.hsl_lum is not None else (0,) * 8
    img = apply_hsl_selective(img, hue_o, sat_o, lum_o)
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
    brushes = getattr(r, "brush_masks", None) or []
    if brushes:
        img = apply_brush_masks(img, brushes)

    # Soft proof (preview only feel — still applied in pipeline for consistency)
    if r.soft_proof:
        img = apply_soft_proof(
            img, r.soft_proof_profile,
            gamut_warning=getattr(r, "soft_proof_gamut", False),
        )
    img = apply_sharpen(
        img, r.sharpen_intensity, r.sharpen_radius, r.sharpen_threshold,
        detail=getattr(r, 'sharpen_detail', 0.0),
    )
    if abs(getattr(r, 'output_sharpen', 0.0)) > 1e-4:
        img = apply_output_sharpen(img, r.output_sharpen)

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



    # Specialty: Infrared + Astro (non-destructive recipe flags)
    img = apply_ir_processing(img, r)
    img = apply_astro_processing(img, r)

    # Zone mapping is opt-in. Plain B&W above must remain a conventional
    # grayscale conversion unless the user explicitly enables zones.
    if getattr(r, "zone_enabled", False):
        img = apply_zone_system(
            img,
            enabled=True,
            placement=float(getattr(r, "zone_placement", 5.0) or 5.0),
            expansion=float(getattr(r, "zone_expansion", 0.0) or 0.0),
            filter_name=str(getattr(r, "zone_filter", "none") or "none"),
            snap=float(getattr(r, "zone_snap", 0.0) or 0.0),
            overlay=bool(getattr(r, "zone_overlay", False)),
        )
    elif getattr(r, "zone_overlay", False):
        img = apply_zone_system(
            img,
            enabled=False,
            placement=float(getattr(r, "zone_placement", 5.0) or 5.0),
            expansion=float(getattr(r, "zone_expansion", 0.0) or 0.0),
            filter_name=str(getattr(r, "zone_filter", "none") or "none"),
            snap=float(getattr(r, "zone_snap", 0.0) or 0.0),
            overlay=True,
        )

    return _float01_to_dtype(img, output_dtype)


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


def save_recipe_sidecar(image_path: str, recipe) -> str:
    import json
    path = sidecar_path(image_path)
    data = {
        "version": 1,
        "image": os.path.basename(image_path),
        "recipe": recipe_to_dict(recipe),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def load_recipe_sidecar(image_path: str):
    import json
    path = sidecar_path(image_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return recipe_from_dict(data.get("recipe") or data)
    except Exception as e:
        print(f"sidecar load failed: {e}")
        return None


def apply_watermark(img_bgr, text, opacity=0.45, scale=0.035, margin=0.02):
    """Draw a text watermark bottom-right while preserving image precision."""
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
    maximum = float(np.iinfo(out.dtype).max) if np.issubdtype(out.dtype, np.integer) else 1.0
    light = (maximum, maximum, maximum)
    dark = (maximum * 20.0 / 255.0,) * 3
    cv2.putText(overlay, text, (x, y), font, font_scale, light, thickness + 2, cv2.LINE_AA)
    cv2.putText(overlay, text, (x, y), font, font_scale, dark, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, float(opacity), out, 1.0 - float(opacity), 0, out)
    return out


# Spectral B&W contrast filters (RGB weights) for zone system
_ZONE_FILTERS = {
    "none": (0.299, 0.587, 0.114),
    "yellow": (0.15, 0.55, 0.30),
    "orange": (0.10, 0.45, 0.45),
    "red": (0.05, 0.25, 0.70),
    "green": (0.20, 0.70, 0.10),
    "blue": (0.55, 0.25, 0.20),
}

def apply_zone_system(
    img,
    enabled: bool = True,
    placement: float = 5.0,
    expansion: float = 0.0,
    filter_name: str = "none",
    snap: float = 0.0,
    overlay: bool = False,
    force_bw: bool = True,
):
    """Ansel Adams–inspired zone mapping on float BGR [0,1].

    placement: which zone (0..10) middle-gray (~18% / zone V) is mapped to.
    expansion: −100 compresses dynamic range, +100 expands (N− / N+ feel).
    filter_name: spectral B&W contrast filter.
    snap: 0..100 pull luminance toward discrete zone centers.
    overlay: paint false-color zones (preview aid).
    """
    if not enabled and not overlay:
        return img
    img = np.clip(img, 0, 1).astype(np.float32)
    filt = _ZONE_FILTERS.get((filter_name or "none").lower(), _ZONE_FILTERS["none"])
    # OpenCV BGR order
    b, g, rch = img[..., 0], img[..., 1], img[..., 2]
    # filt is RGB weights
    lum = np.clip(filt[2] * b + filt[1] * g + filt[0] * rch, 0.0, 1.0)

    # Map: log-ish zone scale. Zone V ≈ 0.18 reflectance → ~0.5 display gamma-ish.
    # Work in linear-ish zone index 0..10
    # Convert luminance to zone via log2 relative to mid-gray 0.18
    mid_ref = 0.18
    # Avoid log0
    safe = np.maximum(lum, 1e-5)
    # Stops relative to mid-gray; zone V = 5
    stops = np.log2(safe / mid_ref)
    zone = 5.0 + stops  # approximately

    place = float(np.clip(placement if placement is not None else 5.0, 0.0, 10.0))
    # Shift so mid-gray lands on placement
    zone = zone + (place - 5.0)

    # Expansion around placement (N+/N−)
    exp = float(expansion or 0.0) / 100.0
    scale = 1.0 + exp * 0.85
    zone = place + (zone - place) * scale
    zone = np.clip(zone, 0.0, 10.0)

    snap_amt = float(np.clip(snap or 0.0, 0.0, 100.0)) / 100.0
    if snap_amt > 0.01:
        centers = np.round(zone)
        zone = zone * (1.0 - snap_amt) + centers * snap_amt

    # Zone → display luminance (approx Adams print scale)
    # Zone 0→0, V→0.18 linear-ish, X→1 with soft shoulder
    out_lin = mid_ref * (2.0 ** (zone - 5.0))
    out_lin = np.clip(out_lin, 0.0, 1.0)
    # Mild display gamma for screen
    out = np.power(out_lin, 1.0 / 2.2)

    if overlay:
        idx = np.clip(np.round(zone).astype(np.int32), 0, 10)
        colors = _ZONE_COLORS[idx]
        if force_bw:
            base = np.stack([out, out, out], axis=-1)
            img = base * 0.35 + colors * 0.65
        else:
            img = img * 0.35 + colors * 0.65
        return np.clip(img, 0, 1)

    if force_bw:
        img = np.stack([out, out, out], axis=-1)
    return np.clip(img, 0, 1)

def deghost_stack(images: List[np.ndarray], strength: float = 50.0) -> List[np.ndarray]:
    """Reduce motion ghosts before fusion.

    strength 0..100: blend each frame toward a robust reference (median of stack).
    Higher strength replaces moving regions more aggressively with the median.
    """
    if not images or len(images) < 2 or float(strength) < 1.0:
        return images
    t = max(0.0, min(1.0, float(strength) / 100.0))
    stack = np.stack([im.astype(np.float32) for im in images], axis=0)
    median = np.median(stack, axis=0)
    # Per-pixel motion amount vs median
    out = []
    for im in images:
        x = im.astype(np.float32)
        diff = np.mean(np.abs(x - median), axis=2)  # H,W
        # Soft motion mask
        dmax = float(diff.max()) + 1e-3
        motion = np.clip(diff / (0.15 * 255.0 + 0.25 * dmax), 0, 1)
        motion = cv2.GaussianBlur(motion, (0, 0), sigmaX=2.0)
        w = (motion * t)[..., None]
        y = x * (1.0 - w) + median * w
        out.append(np.clip(y, 0, 255).astype(np.uint8))
    return out

def merge_hdr_debevec(
    paths,
    align=True,
    max_dim=0,
    deghost=0.0,
    tonemap: str = "reinhard",
    gamma: float = 1.0,
):
    """True HDR via Debevec calibration + tonemap. Returns uint8 BGR.

    Requires varying exposures (EXIF shutter preferred). Falls back to equal
    spaced times if metadata is missing.
    """
    if not paths or len(paths) < 2:
        raise ValueError("HDR merge needs at least 2 images")
    images, metas = _load_hdr_stack(paths, max_dim=max_dim)
    if align:
        images = _align_hdr_stack(images)
    if deghost and float(deghost) > 0:
        images = deghost_stack(images, strength=float(deghost))

    times = []
    for i, meta in enumerate(metas):
        s = _exposure_seconds_from_meta(meta)
        if s is None or s <= 0:
            # synthetic 1-stop steps centered on middle
            s = 2.0 ** (i - (len(paths) - 1) / 2.0) * (1.0 / 60.0)
        times.append(float(s))
    times = np.array(times, dtype=np.float32)

    calibrate = cv2.createCalibrateDebevec()
    response = calibrate.process(images, times)
    merge = cv2.createMergeDebevec()
    hdr = merge.process(images, times, response)

    tonemap = (tonemap or "reinhard").lower()
    if tonemap == "drago":
        mapper = cv2.createTonemapDrago(gamma=float(gamma) if gamma else 1.0, saturation=1.0)
    elif tonemap == "mantiuk":
        mapper = cv2.createTonemapMantiuk(gamma=float(gamma) if gamma else 1.0, scale=0.7)
    else:
        mapper = cv2.createTonemapReinhard(
            gamma=float(gamma) if gamma else 1.0, intensity=0.0, light_adapt=0.1, color_adapt=0.0
        )
    ldr = mapper.process(hdr)
    return np.clip(ldr * 255.0, 0, 255).astype(np.uint8)

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
