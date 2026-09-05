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
import xml.etree.ElementTree as ET
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
    # Luminance-targeted exposure for pixels at/near the zebra threshold.
    zebra_threshold: float = 95.0  # display-referred luma percent, 50..100
    zebra_exposure: float = 0.0    # EV applied only through the zebra mask
    zebra_feather: float = 5.0     # luma transition width in percentage points

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
    mask_library: list = field(default_factory=list)  # named reusable mask specifications
    creative_filters: list = field(default_factory=list)  # ordered post-develop effect blocks
    # Remove Distractions workspace. Coordinates are normalized and operations
    # are replayed at preview or export resolution.
    distraction_operations: list = field(default_factory=list)
    reflection_enabled: bool = False
    reflection_sensitivity: float = 55.0
    reflection_strength: float = 50.0
    reflection_highlights: float = -35.0
    reflection_saturation: float = 0.0
    reflection_neutralize: float = 20.0
    reflection_contrast: float = 10.0
    reflection_blur: float = 8.0
    reflection_mask_strokes: list = field(default_factory=list)
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
    noise_profile: dict = field(default_factory=dict)
    denoise_edge_preserve: float = 70.0
    denoise_deband: float = 0.0
    denoise_deband_orientation: str = "auto"
    denoise_jpeg_artifacts: float = 0.0
    sharpen_intensity: float = 0.0     # capture / creative sharpen
    sharpen_radius: float = 1.0
    sharpen_threshold: float = 0.0     # edge masking amount 0..100
    sharpen_detail: float = 0.0        # fine structure (small radius)
    output_sharpen: float = 0.0        # output/print sharpen 0..100
    output_sharpen_media: str = "custom"
    output_sharpen_ppi: float = 300.0
    output_sharpen_width_in: float = 12.0
    output_sharpen_proof: bool = False
    portrait_detail_enabled: bool = False
    portrait_skin_color: tuple = field(default_factory=lambda: (0.55, 0.62, 0.76))
    portrait_color_reach: float = 28.0
    portrait_small_smooth: float = 20.0
    portrait_medium_smooth: float = 10.0
    portrait_large_smooth: float = 0.0
    portrait_edge_preserve: float = 75.0
    portrait_texture_recovery: float = 45.0
    portrait_mask_id: str = ""

    horizon: float = 0.0
    distortion: float = 0.0
    perspective: float = 0.0
    perspective_horizontal: float = 0.0
    warp_top: float = 0.0
    warp_bottom: float = 0.0
    warp_left: float = 0.0
    warp_right: float = 0.0
    wide_angle: float = 0.0
    diorama_strength: float = 0.0
    diorama_position: float = 50.0
    diorama_width: float = 30.0
    diorama_angle: float = 0.0
    keystone_points: list = field(default_factory=list)  # normalized TL,TR,BR,BL source quad
    geometry_auto_crop: bool = False
    line_reflection_points: list = field(default_factory=list)  # normalized endpoints on developed image
    line_reflection_side: int = -1
    line_reflection_opacity: float = 100.0
    line_reflection_feather: float = 0.0
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
    # Pillow cannot open many RAW containers (including some Nikon NEF
    # variants).  ExifRead reads the metadata IFD without decoding the image,
    # so use it as a best-effort fallback for GPS and the common camera fields.
    if "gps" not in meta:
        try:
            import exifread
            with open(path, "rb") as stream:
                tags = exifread.process_file(stream, details=False)

            def _tag(name):
                value = tags.get(name)
                return value.values if hasattr(value, "values") else value

            lat = _tag("GPS GPSLatitude")
            lon = _tag("GPS GPSLongitude")
            if lat and lon:
                lat_ref = str(tags.get("GPS GPSLatitudeRef", "N"))
                lon_ref = str(tags.get("GPS GPSLongitudeRef", "E"))
                lat, lon = _gps_values_to_degrees(lat, lat_ref), _gps_values_to_degrees(lon, lon_ref)
                if lat is not None and lon is not None:
                    meta["gps_latitude"] = lat
                    meta["gps_longitude"] = lon
                    meta["gps"] = (lat, lon)
            for target, source in (
                ("camera", "Image Model"), ("lens", "EXIF LensModel"),
                ("datetime_original", "EXIF DateTimeOriginal"),
            ):
                if not meta.get(target) and tags.get(source):
                    meta[target] = str(tags[source])
            if meta.get("datetime_original"):
                meta.setdefault("datetime", meta["datetime_original"])
        except Exception:
            pass
    # User-editable descriptive metadata lives in a separate PhotoLab XMP
    # sidecar so camera originals (especially RAW/DNG files) are never
    # rewritten.  Sidecar values intentionally take precedence over EXIF.
    sidecar = read_editable_metadata(path)
    if sidecar.pop("_gps_override", False):
        for key in ("gps", "gps_latitude", "gps_longitude"):
            meta.pop(key, None)
    meta.update(sidecar)
    if meta.get("gps_latitude") is not None and meta.get("gps_longitude") is not None:
        meta["gps"] = (float(meta["gps_latitude"]), float(meta["gps_longitude"]))
    return meta


_PL_METADATA_NS = "https://github.com/betoon/photo_lab/ns/metadata/1.0/"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
_XMP_NS = "adobe:ns:meta/"
_DC_NS = "http://purl.org/dc/elements/1.1/"
_PHOTOSHOP_NS = "http://ns.adobe.com/photoshop/1.0/"
_IPTC_NS = "http://iptc.org/std/Iptc4xmpCore/1.0/xmlns/"


def metadata_sidecar_path(path: str) -> str:
    """Return the non-destructive editable-metadata sidecar for an image."""
    return os.path.splitext(os.path.abspath(path))[0] + ".photolab.xmp"


def read_editable_metadata(path: str) -> dict:
    """Read PhotoLab's editable location/description XMP sidecar."""
    sidecar = metadata_sidecar_path(path)
    if not os.path.isfile(sidecar):
        return {}
    try:
        root = ET.parse(sidecar).getroot()
        description = root.find(f".//{{{_RDF_NS}}}Description")
        if description is None:
            return {}
        get = lambda namespace, name: description.get(f"{{{namespace}}}{name}", "")
        result = {
            "title": get(_PL_METADATA_NS, "title"),
            "description": get(_PL_METADATA_NS, "description"),
            "location": get(_PL_METADATA_NS, "location"),
            "city": get(_PL_METADATA_NS, "city"),
            "state": get(_PL_METADATA_NS, "state"),
            "country": get(_PL_METADATA_NS, "country"),
            "_gps_override": True,
            "metadata_sidecar": sidecar,
        }
        if get(_PL_METADATA_NS, "gpsEnabled").lower() == "true":
            lat = float(get(_PL_METADATA_NS, "gpsLatitude"))
            lon = float(get(_PL_METADATA_NS, "gpsLongitude"))
            if -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0:
                result["gps_latitude"] = lat
                result["gps_longitude"] = lon
        return result
    except (OSError, ValueError, ET.ParseError):
        return {}


def write_editable_metadata(path: str, values: dict) -> str:
    """Atomically write descriptive/GPS metadata without modifying the image."""
    ET.register_namespace("x", _XMP_NS)
    ET.register_namespace("rdf", _RDF_NS)
    ET.register_namespace("pl", _PL_METADATA_NS)
    ET.register_namespace("dc", _DC_NS)
    ET.register_namespace("photoshop", _PHOTOSHOP_NS)
    ET.register_namespace("Iptc4xmpCore", _IPTC_NS)
    root = ET.Element(f"{{{_XMP_NS}}}xmpmeta")
    rdf = ET.SubElement(root, f"{{{_RDF_NS}}}RDF")
    desc = ET.SubElement(rdf, f"{{{_RDF_NS}}}Description")
    text_fields = ("title", "description", "location", "city", "state", "country")
    for name in text_fields:
        desc.set(f"{{{_PL_METADATA_NS}}}{name}", str(values.get(name, "")).strip())
    # Standard location attributes improve interoperability without touching
    # any pre-existing Adobe/Lightroom XMP file.
    desc.set(f"{{{_PHOTOSHOP_NS}}}City", str(values.get("city", "")).strip())
    desc.set(f"{{{_PHOTOSHOP_NS}}}State", str(values.get("state", "")).strip())
    desc.set(f"{{{_PHOTOSHOP_NS}}}Country", str(values.get("country", "")).strip())
    desc.set(f"{{{_IPTC_NS}}}Location", str(values.get("location", "")).strip())
    gps_enabled = bool(values.get("gps_enabled", False))
    desc.set(f"{{{_PL_METADATA_NS}}}gpsEnabled", "true" if gps_enabled else "false")
    if gps_enabled:
        lat = float(values.get("gps_latitude"))
        lon = float(values.get("gps_longitude"))
        if not -90.0 <= lat <= 90.0:
            raise ValueError("Latitude must be between -90 and 90 degrees.")
        if not -180.0 <= lon <= 180.0:
            raise ValueError("Longitude must be between -180 and 180 degrees.")
        desc.set(f"{{{_PL_METADATA_NS}}}gpsLatitude", f"{lat:.8f}")
        desc.set(f"{{{_PL_METADATA_NS}}}gpsLongitude", f"{lon:.8f}")
    sidecar = metadata_sidecar_path(path)
    temporary = sidecar + ".tmp"
    ET.ElementTree(root).write(temporary, encoding="utf-8", xml_declaration=True)
    os.replace(temporary, sidecar)
    return sidecar


def _gps_ratio_to_float(rat):
    try:
        if hasattr(rat, "numerator"):
            return float(rat.numerator) / float(rat.denominator or 1)
        if hasattr(rat, "num"):
            return float(rat.num) / float(rat.den or 1)
        if isinstance(rat, (tuple, list)) and len(rat) >= 2:
            return float(rat[0]) / float(rat[1] or 1)
        return float(rat)
    except Exception:
        return 0.0


def _gps_values_to_degrees(values, ref):
    """Convert three EXIF degrees/minutes/seconds ratios to decimal degrees."""
    if not values or len(values) < 3:
        return None
    result = (_gps_ratio_to_float(values[0])
              + _gps_ratio_to_float(values[1]) / 60.0
              + _gps_ratio_to_float(values[2]) / 3600.0)
    if str(ref).strip().upper() in ("S", "W"):
        result = -result
    return result


def _parse_gps_info(gps_info):
    """Parse PIL GPSInfo dict → (lat, lon) decimal degrees or (None, None)."""
    if not gps_info or not isinstance(gps_info, dict):
        return None, None
    try:
        from PIL.ExifTags import GPSTAGS
        tagged = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}
    except Exception:
        tagged = gps_info

    lat = _gps_values_to_degrees(tagged.get("GPSLatitude"), tagged.get("GPSLatitudeRef") or "N")
    lon = _gps_values_to_degrees(tagged.get("GPSLongitude"), tagged.get("GPSLongitudeRef") or "E")
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
            # Some cameras expose a full-size embedded JPEG even when LibRaw
            # cannot unpack the sensor compression. This preserves an editable
            # image instead of failing completely, but is explicitly tagged so
            # callers never mistake it for a true RAW decode.
            fallback = extract_embedded_preview(path, max_side=0)
            if fallback is not None:
                img_bgr = fallback
                meta.update({
                    "is_raw": True,
                    "wb_baked": True,
                    "decode_bps": 8,
                    "raw_fallback": "embedded_preview",
                    "raw_decode_error": str(raw_failure or "unsupported RAW data"),
                })
            else:
                # Don't fall back to cv2.imread() for RAW files — OpenCV's TIFF
                # reader cannot safely decode the sensor IFD.
                raise RuntimeError(format_raw_error(path, raw_failure))
        else:
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


def apply_advanced_geometry(img, horizontal=0.0, top=0.0, bottom=0.0,
                            left=0.0, right=0.0):
    """Four-edge projective warp; values are signed percentages (-100..100)."""
    values = (horizontal, top, bottom, left, right)
    if all(abs(float(v)) < 1e-4 for v in values):
        return img
    h, w = img.shape[:2]
    # Keep corners inside a conservative 35% envelope so the homography cannot
    # fold over itself. Horizontal perspective moves the left/right edges in
    # opposite vertical directions; the four edge values provide fine control.
    scale_x, scale_y = w * 0.0035, h * 0.0035
    hp = float(horizontal) * scale_y
    t, b = float(top) * scale_x, float(bottom) * scale_x
    l, r = float(left) * scale_y, float(right) * scale_y
    clamp_x = lambda value: float(np.clip(value, 0.0, w - 1.0))
    clamp_y = lambda value: float(np.clip(value, 0.0, h - 1.0))
    dst = np.float32([
        [clamp_x(t), clamp_y(l + hp)],
        [clamp_x(w - 1 + t), clamp_y(r - hp)],
        [clamp_x(w - 1 + b), clamp_y(h - 1 + r - hp)],
        [clamp_x(b), clamp_y(h - 1 + l + hp)],
    ])
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REFLECT_101)


def normalize_keystone_points(points):
    """Validate/canonicalize normalized TL,TR,BR,BL corner points."""
    if not isinstance(points, (list, tuple)) or len(points) != 4:
        return []
    clean = []
    for point in points:
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError, IndexError):
            return []
        clean.append([float(np.clip(x, 0, 1)), float(np.clip(y, 0, 1))])
    # Reject crossed or nearly degenerate quadrilaterals.
    poly = np.asarray(clean, dtype=np.float32)
    area = 0.5 * abs(sum(
        poly[i, 0] * poly[(i + 1) % 4, 1] - poly[(i + 1) % 4, 0] * poly[i, 1]
        for i in range(4)
    ))
    return clean if area >= 0.01 else []


def apply_keystone(img, points):
    """Rectify a normalized TL,TR,BR,BL source quadrilateral to the full canvas."""
    points = normalize_keystone_points(points)
    if not points:
        return img
    h, w = img.shape[:2]
    src = np.float32([[x * (w - 1), y * (h - 1)] for x, y in points])
    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, matrix, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REFLECT_101)


def geometry_auto_crop_bounds(recipe):
    """Return a conservative crop that trims transform-generated edge margins."""
    magnitudes = [
        abs(float(getattr(recipe, "distortion", 0.0))) * 0.0007,
        abs(float(getattr(recipe, "perspective", 0.0))) * 0.0012,
        abs(float(getattr(recipe, "perspective_horizontal", 0.0))) * 0.0012,
        abs(float(getattr(recipe, "wide_angle", 0.0))) * 0.00045,
    ]
    edge = max(
        abs(float(getattr(recipe, name, 0.0)))
        for name in ("warp_top", "warp_bottom", "warp_left", "warp_right")
    ) * 0.0008
    inset = float(np.clip(sum(magnitudes) + edge, 0.0, 0.22))
    if normalize_keystone_points(getattr(recipe, "keystone_points", [])):
        inset = max(inset, 0.012)
    return (inset, inset, 1.0 - inset, 1.0 - inset) if inset > 1e-4 else None


def detect_architectural_upright(img):
    """Estimate horizon rotation and vertical convergence from strong line segments."""
    if img is None or img.size == 0:
        return 0.0, 0.0, 0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    scale = min(1.0, 1200.0 / max(gray.shape[:2]))
    if scale < 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(gray, 60, 180)
    length = max(25, int(min(gray.shape[:2]) * 0.18))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 360, 45, minLineLength=length, maxLineGap=16)
    if lines is None:
        return 0.0, 0.0, 0
    horizontal, vertical = [], []
    width = max(gray.shape[1] - 1, 1)
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        segment = np.hypot(dx, dy)
        if segment < length:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        while angle > 90:
            angle -= 180
        while angle < -90:
            angle += 180
        if abs(angle) <= 25:
            horizontal.append((angle, segment))
        elif abs(abs(angle) - 90) <= 30 and abs(dy) > 1:
            center_x = ((x1 + x2) * 0.5 / width) - 0.5
            vertical.append((dx / dy, center_x, segment))
    horizon = 0.0
    if horizontal:
        horizon = -float(np.median([angle for angle, _ in horizontal]))
    perspective = 0.0
    if len(vertical) >= 2:
        convergence = np.median([slope * np.sign(cx or 1.0) for slope, cx, _ in vertical])
        perspective = float(np.clip(-convergence * 180.0, -60.0, 60.0))
    return float(np.clip(horizon, -15, 15)), perspective, len(horizontal) + len(vertical)


def apply_wide_angle_stretch(img, amount):
    """Adjust horizontal edge stretching without changing the canvas size."""
    amount = float(np.clip(amount, -100.0, 100.0))
    if abs(amount) < 1e-4:
        return img
    h, w = img.shape[:2]
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    strength = abs(amount) / 100.0 * 1.5
    curved = np.arctan(strength * x) / max(np.arctan(strength), 1e-6)
    if amount < 0:
        # Approximate inverse: stretch the center and compress the outer edges.
        curved = np.tan(np.arctan(strength) * x) / max(strength, 1e-6)
    map_x = ((curved + 1.0) * 0.5 * (w - 1))[None, :].repeat(h, axis=0).astype(np.float32)
    map_y = np.arange(h, dtype=np.float32)[:, None].repeat(w, axis=1)
    return cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC,
                     borderMode=cv2.BORDER_REFLECT_101)


def apply_diorama(img, strength=0.0, position=50.0, width=30.0, angle=0.0):
    """Blend a soft tilt-shift blur outside a rotatable in-focus band."""
    amount = float(np.clip(strength, 0.0, 100.0)) / 100.0
    if amount <= 1e-4:
        return img
    h, w = img.shape[:2]
    sigma = 0.8 + amount * 11.0
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) * 0.5, (h - 1) * float(np.clip(position, 0, 100)) / 100.0
    theta = np.deg2rad(float(angle))
    distance = np.abs(-(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta))
    half_band = max(1.0, min(h, w) * float(np.clip(width, 5, 90)) / 200.0)
    feather = max(2.0, min(h, w) * 0.12)
    mask = np.clip((distance - half_band) / feather, 0.0, 1.0)
    mask = (mask * mask * (3.0 - 2.0 * mask) * amount)[..., None]
    out = img.astype(np.float32) * (1.0 - mask) + blurred.astype(np.float32) * mask
    if np.issubdtype(img.dtype, np.integer):
        return np.clip(out, 0, np.iinfo(img.dtype).max).astype(img.dtype)
    return out.astype(img.dtype)


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
    # Build values from the original Python-float keys before narrowing xs to
    # float32. Looking a narrowed 0.18 back up in the dict can become
    # 0.180000007... and fail for otherwise valid hand-authored JSON curves.
    ordered_xs = sorted(unique)
    ys = np.array([unique[x] for x in ordered_xs], dtype=np.float32)
    xs = np.array(ordered_xs, dtype=np.float32)
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


def apply_zebra_highlight_exposure(img, exposure=0.0, threshold=95.0, feather=5.0):
    """Apply exposure only to a soft, display-luminance zebra mask.

    The luma weights intentionally match ImageCanvas' zebra overlay
    (B/G/R = 0.1/0.6/0.3), so the pixels selected by the edit correspond to
    the pixels the photographer sees striped before the adjustment.
    """
    ev = float(exposure or 0.0)
    if abs(ev) < 1e-6:
        return img
    work = np.asarray(img, dtype=np.float32)
    clipped = np.clip(work, 0.0, 1.0)
    lum = clipped[..., 0] * 0.1 + clipped[..., 1] * 0.6 + clipped[..., 2] * 0.3
    high = float(np.clip(float(threshold or 95.0) / 100.0, 0.5, 1.0))
    width = float(np.clip(float(feather or 0.0) / 100.0, 0.0, 0.49))
    if width <= 1e-6:
        mask = (lum >= high).astype(np.float32)
    else:
        low = max(0.0, high - width)
        mask = np.clip((lum - low) / max(high - low, 1e-6), 0.0, 1.0)
        mask = mask * mask * (3.0 - 2.0 * mask)  # smoothstep: halo-resistant edge
    adjusted = work * (2.0 ** ev)
    return work * (1.0 - mask[..., None]) + adjusted * mask[..., None]


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


def measure_noise_profile(img_bgr):
    """Estimate luminance/chroma noise and row/column banding from flat regions."""
    if img_bgr is None or img_bgr.size == 0:
        return {}
    source = img_bgr.astype(np.float32)
    if np.issubdtype(img_bgr.dtype, np.integer):
        source /= float(np.iinfo(img_bgr.dtype).max)
    elif source.max() > 1.5:
        source /= 255.0
    lab = cv2.cvtColor(np.clip(source, 0, 1), cv2.COLOR_BGR2LAB)
    light = lab[..., 0] / 100.0
    local_mean = cv2.GaussianBlur(light, (0, 0), sigmaX=3.0)
    residual = light - local_mean
    gradient = cv2.magnitude(cv2.Sobel(light, cv2.CV_32F, 1, 0), cv2.Sobel(light, cv2.CV_32F, 0, 1))
    flat = gradient < np.percentile(gradient, 35)
    sample = residual[flat] if np.any(flat) else residual.ravel()
    lum_sigma = float(1.4826 * np.median(np.abs(sample - np.median(sample))))
    chroma_residual = lab[..., 1:3] - cv2.GaussianBlur(lab[..., 1:3], (0, 0), sigmaX=3.0)
    chroma_sample = chroma_residual[flat] if np.any(flat) else chroma_residual.reshape(-1, 2)
    chroma_sigma = float(np.sqrt(np.mean(chroma_sample * chroma_sample)) / 128.0)
    row_score = float(np.std(np.mean(residual, axis=1)))
    col_score = float(np.std(np.mean(residual, axis=0)))
    banding = max(row_score, col_score)
    return {
        "luminance_sigma": lum_sigma, "chroma_sigma": chroma_sigma,
        "banding_score": banding,
        "banding_orientation": "horizontal" if row_score >= col_score else "vertical",
        "suggested_luminance": float(np.clip(lum_sigma * 1100.0, 0, 100)),
        "suggested_chroma": float(np.clip(chroma_sigma * 950.0, 0, 100)),
        "suggested_strength": float(np.clip(max(lum_sigma * 700.0, chroma_sigma * 650.0), 0, 100)),
        "suggested_deband": float(np.clip(banding * 2200.0, 0, 100)),
    }


def apply_denoise(img_bgr, luminance, chroma, strength=0.0, detail_preserve=50.0, method="auto",
                  edge_preserve=70.0, deband=0.0, deband_orientation="auto", jpeg_artifacts=0.0):
    """Edge-aware denoise in LAB with optional detail recovery.

    luminance / chroma / strength: 0..100
    detail_preserve: 0..100 — blend high-frequency residual back after NR
    method: auto | bilateral | nlm
    """
    if luminance <= 0 and chroma <= 0 and strength <= 0 and deband <= 0 and jpeg_artifacts <= 0:
        return img_bgr
    if max(luminance, chroma, strength, deband, jpeg_artifacts) / 100.0 < 0.01:
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
    artifact_amount = float(np.clip(jpeg_artifacts, 0, 100)) / 100.0
    if artifact_amount > 1e-6:
        smoothed = cv2.bilateralFilter(u8, d=5, sigmaColor=8.0 + artifact_amount * 32.0, sigmaSpace=2.0)
        smoothed = cv2.GaussianBlur(smoothed, (3, 3), sigmaX=0.45 + artifact_amount * 0.55)
        u8 = cv2.addWeighted(u8, 1.0 - artifact_amount * 0.75, smoothed, artifact_amount * 0.75, 0)
    deband_amount = float(np.clip(deband, 0, 100)) / 100.0
    if deband_amount > 1e-6:
        work = u8.astype(np.float32)
        orientation = str(deband_orientation or "auto").lower()
        gray = cv2.cvtColor(u8, cv2.COLOR_BGR2GRAY).astype(np.float32)
        residual_gray = gray - cv2.GaussianBlur(gray, (0, 0), 5)
        row_score = float(np.std(np.mean(residual_gray, axis=1)))
        col_score = float(np.std(np.mean(residual_gray, axis=0)))
        if orientation == "auto":
            orientation = "horizontal" if row_score >= col_score else "vertical"
        if orientation == "horizontal":
            signal = np.mean(work, axis=1, keepdims=True)
            smooth = cv2.GaussianBlur(signal, (1, 0), sigmaX=0, sigmaY=max(2.0, u8.shape[0] * 0.025))
        else:
            signal = np.mean(work, axis=0, keepdims=True)
            smooth = cv2.GaussianBlur(signal, (0, 1), sigmaX=max(2.0, u8.shape[1] * 0.025), sigmaY=0)
        work -= (signal - smooth) * deband_amount
        u8 = np.clip(work, 0, 255).astype(np.uint8)
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
        edge_setting = float(np.clip(edge_preserve, 0, 100))
        legacy_mix = 0.35 + 0.65 * edge
        if edge_setting >= 70.0:
            emphasis = (edge_setting - 70.0) / 30.0
            recovery_map = legacy_mix * (1.0 - emphasis) + edge * emphasis
        else:
            uniformity = (70.0 - edge_setting) / 70.0
            recovery_map = legacy_mix * (1.0 - uniformity) + 1.0 * uniformity
        mix = preserve * recovery_map
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


def output_sharpen_params(ppi=300.0, media="screen", amount=None):
    """Return an output-sharpen amount/radius suggestion for a delivery condition."""
    ppi = float(np.clip(ppi, 72.0, 720.0))
    media = str(media or "screen").lower()
    profiles = {
        "screen": (28.0, 0.55), "matte": (52.0, 1.05),
        "glossy": (42.0, 0.8), "canvas": (62.0, 1.35), "custom": (35.0, 0.8),
    }
    base_amount, base_radius = profiles.get(media, profiles["custom"])
    scale = np.sqrt(ppi / (96.0 if media == "screen" else 300.0))
    suggested = base_amount * np.clip(scale, 0.7, 1.5)
    radius = base_radius * np.clip(scale, 0.65, 1.8)
    return float(amount if amount is not None else np.clip(suggested, 0, 100)), float(radius)


def build_portrait_skin_mask(img, skin_color=(0.55, 0.62, 0.76), color_reach=28.0,
                             edge_preserve=75.0):
    """Build a feathered color-selected skin mask with edge suppression."""
    source = np.clip(img, 0, 1).astype(np.float32)
    target = np.asarray(skin_color, dtype=np.float32).reshape(1, 1, 3)
    if target.max() > 1.0:
        target /= 255.0
    lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)[0, 0]
    chroma_distance = np.linalg.norm(lab[..., 1:3] - target_lab[1:3], axis=2)
    tolerance = 8.0 + float(np.clip(color_reach, 0, 100)) * 0.55
    mask = np.clip(1.0 - chroma_distance / tolerance, 0, 1)
    hsv = cv2.cvtColor(source, cv2.COLOR_BGR2HSV)
    mask *= np.clip(hsv[..., 1] / 0.12, 0, 1)
    mask *= np.clip((lab[..., 0] - 8.0) / 25.0, 0, 1)
    preserve = float(np.clip(edge_preserve, 0, 100)) / 100.0
    if preserve > 1e-6:
        gray = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        edge /= edge.max() + 1e-6
        mask *= 1.0 - edge * preserve * 0.9
    return np.clip(cv2.GaussianBlur(mask.astype(np.float32), (0, 0), sigmaX=1.5), 0, 1)


def apply_portrait_detail(img, settings, shared_mask=None):
    """Multi-scale skin smoothing with edge protection and texture recovery."""
    if not bool(settings.get("enabled", True)):
        return img
    source = np.clip(img, 0, 1).astype(np.float32)
    mask = build_portrait_skin_mask(
        source, settings.get("skin_color", (0.55, 0.62, 0.76)),
        settings.get("color_reach", 28.0), settings.get("edge_preserve", 75.0),
    )
    if shared_mask is not None:
        mask = np.minimum(mask, np.clip(shared_mask, 0, 1))
    small = float(np.clip(settings.get("small_smooth", 20.0), 0, 100)) / 100.0
    medium = float(np.clip(settings.get("medium_smooth", 10.0), 0, 100)) / 100.0
    large = float(np.clip(settings.get("large_smooth", 0.0), 0, 100)) / 100.0
    if max(small, medium, large) <= 1e-6 or mask.max() <= 1e-6:
        return source
    fine_blur = cv2.bilateralFilter(source, d=5, sigmaColor=0.07, sigmaSpace=2.0)
    medium_blur = cv2.bilateralFilter(source, d=11, sigmaColor=0.12, sigmaSpace=5.0)
    large_blur = cv2.GaussianBlur(source, (0, 0), sigmaX=9.0)
    total = max(small + medium + large, 1e-6)
    smoothed = (fine_blur * small + medium_blur * medium + large_blur * large) / total
    texture = float(np.clip(settings.get("texture_recovery", 45.0), 0, 100)) / 100.0
    high_frequency = source - cv2.GaussianBlur(source, (0, 0), sigmaX=1.0)
    smoothed = np.clip(smoothed + high_frequency * texture * 0.65, 0, 1)
    alpha = (mask * float(np.clip(total / 1.5, 0, 1)))[..., None]
    return np.clip(source * (1.0 - alpha) + smoothed * alpha, 0, 1)



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
                                    if max_side and max(h, w) > max_side:
                                        scale = max_side / max(h, w)
                                        img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
                                    return img
                            elif thumb.format == rawpy.ThumbFormat.BITMAP:
                                # RGB bitmap
                                rgb = _np.array(data)
                                if rgb.ndim == 3:
                                    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                                    h, w = img.shape[:2]
                                    if max_side and max(h, w) > max_side:
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
                        if max_side and max(h, w) > max_side:
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
    if max_side and max(h, w) > max_side:
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


def apply_local_preset_look(img, preset):
    """Apply the tone/color subset of a Recipe dict to float BGR image data.

    Spatial/global operations such as crop, geometry, denoise, sharpening,
    grain, vignette, HDR, local masks, and output proofing are intentionally
    excluded because they cannot be meaningfully blended through a local mask.
    """
    if not preset:
        return img
    p = Recipe.from_dict(preset) if isinstance(preset, dict) else preset
    out = np.asarray(img, dtype=np.float32).copy()
    out = apply_creative_white_balance(
        out, getattr(p, "creative_temperature", 0.0), getattr(p, "creative_tint", 0.0)
    )
    if abs(float(getattr(p, "exposure", 0.0))) > 1e-4:
        out *= 2.0 ** float(p.exposure)
    if any(abs(float(getattr(p, k, 0.0))) > 1e-4 for k in ("highlights", "shadows", "whites", "blacks")):
        lum = cv2.cvtColor(np.clip(out, 0, 1), cv2.COLOR_BGR2GRAY)
        hi_mask = np.clip((lum - 0.55) * 2.2, 0, 1) ** 1.4
        lo_mask = np.clip((0.45 - lum) * 2.2, 0, 1) ** 1.4
        white_mask = np.clip((lum - 0.75) * 4.0, 0, 1)
        black_mask = np.clip((0.25 - lum) * 4.0, 0, 1)
        out += (p.highlights / 100.0) * 0.45 * hi_mask[..., None]
        out += (p.shadows / 100.0) * 0.45 * lo_mask[..., None]
        out += (p.whites / 100.0) * 0.35 * white_mask[..., None]
        out += (p.blacks / 100.0) * 0.35 * black_mask[..., None]
    if abs(float(getattr(p, "contrast", 0.0))) > 1e-4:
        out = (out - 0.5) * (1.0 + p.contrast / 100.0) + 0.5
    if abs(float(getattr(p, "clarity", 0.0))) > 1e-4:
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=3)
        out += (out - blur) * (p.clarity / 100.0)
    out = apply_tone_curve(out, p.curve_shadows, p.curve_darks, p.curve_mids, p.curve_lights, p.curve_highlights)
    out = apply_point_curve_luma(out, getattr(p, "curve_points", None) or [])
    out = apply_rgb_point_curves(out, getattr(p, "curve_r_points", None) or [],
                                 getattr(p, "curve_g_points", None) or [],
                                 getattr(p, "curve_b_points", None) or [])
    out = np.clip(out, 0, 1)
    if abs(float(getattr(p, "gamma", 1.0)) - 1.0) > 1e-4:
        out = out ** (1.0 / max(float(p.gamma), 0.05))
    out = apply_vibrance_saturation(out, p.vibrance, p.saturation)
    out = apply_hsl_selective(out, p.hsl_hue or (0,) * 8, p.hsl_sat or (0,) * 8, p.hsl_lum or (0,) * 8)
    out = apply_split_tone(out, p.split_shadow_hue, p.split_shadow_sat,
                           p.split_highlight_hue, p.split_highlight_sat, p.split_balance)
    if abs(float(getattr(p, "clearview", 0.0))) > 1e-4:
        amount = p.clearview / 100.0
        lab = cv2.cvtColor(np.clip(out, 0, 1), cv2.COLOR_BGR2LAB)
        light = lab[..., 0] / 100.0
        blur = cv2.GaussianBlur(light, (0, 0), sigmaX=max(out.shape[1] / 12, 2))
        lab[..., 0] = np.clip(light + (light - blur) * amount * 1.2 + (0.5 - blur) * amount * 0.15, 0, 1) * 100.0
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if abs(float(getattr(p, "microcontrast", 0.0))) > 1e-4:
        blur = cv2.GaussianBlur(out, (0, 0), sigmaX=1.2)
        out += (out - blur) * (p.microcontrast / 100.0)
    if bool(getattr(p, "black_and_white", False)):
        out = cv2.cvtColor(cv2.cvtColor(np.clip(out, 0, 1), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return np.clip(out, 0, 1)


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
        preset = m.get("local_preset")
        strength = float(np.clip(m.get("preset_strength", 1.0), 0.0, 1.0))
        if preset and strength > 1e-6:
            styled = apply_local_preset_look(local, preset)
            local = local * (1.0 - strength) + styled * strength
        out = out * (1.0 - mask3) + np.clip(local, 0, 1) * mask3
    return np.clip(out, 0, 1)


def build_shared_mask(img, spec, library=None, _seen=None):
    """Rasterize a named reusable mask, including intersections by stable id."""
    if not spec:
        return np.ones(img.shape[:2], dtype=np.float32)
    library = list(library or [])
    by_id = {str(item.get("id")): item for item in library if item.get("id")}
    seen = set(_seen or ())
    mask_id = str(spec.get("id", ""))
    if mask_id and mask_id in seen:
        return np.zeros(img.shape[:2], dtype=np.float32)
    if mask_id:
        seen.add(mask_id)
    kind = str(spec.get("kind", "brush")).lower()
    if kind == "full":
        mask = np.ones(img.shape[:2], dtype=np.float32)
    else:
        source = dict(spec)
        if kind in ("luminance", "color"):
            source.setdefault("strokes", [{"x": 0.5, "y": 0.5, "r": 1.5}])
            source.setdefault("hardness", 1.0)
            if kind == "color":
                source["color_range"] = True
        mask = build_brush_mask(img, source)
    for ref in spec.get("intersect_with") or []:
        other = by_id.get(str(ref))
        if other is not None:
            mask = np.minimum(mask, build_shared_mask(img, other, library, seen))
    if spec.get("library_invert"):
        mask = 1.0 - mask
    return np.clip(mask, 0, 1).astype(np.float32)


def _hue_bgr(hue):
    hsv = np.array([[[float(hue) % 360.0, 1.0, 1.0]]], dtype=np.float32)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]


def apply_four_way_color_grade(img, settings):
    """Grade shadows, midtones, highlights, and global color independently."""
    out = np.clip(img, 0, 1).astype(np.float32)
    lum = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    shadow = np.clip((0.58 - lum) / 0.58, 0, 1) ** 1.5
    highlight = np.clip((lum - 0.42) / 0.58, 0, 1) ** 1.5
    midtone = np.clip(1.0 - np.abs(lum - 0.5) / 0.5, 0, 1) ** 1.8
    result = out.copy()
    for key, weight in (("shadow", shadow), ("midtone", midtone), ("highlight", highlight)):
        saturation = float(settings.get(f"{key}_sat", 0.0)) / 100.0
        if abs(saturation) > 1e-6:
            color = _hue_bgr(settings.get(f"{key}_hue", 0.0))
            tinted = result * color[None, None, :] * 1.35
            result = result * (1.0 - weight[..., None] * abs(saturation)) + tinted * weight[..., None] * abs(saturation)
        luminance = float(settings.get(f"{key}_lum", 0.0)) / 100.0
        if abs(luminance) > 1e-6:
            result += weight[..., None] * luminance * 0.35
    global_sat = float(settings.get("global_sat", 0.0)) / 100.0
    if abs(global_sat) > 1e-6:
        color = _hue_bgr(settings.get("global_hue", 0.0))
        tinted = result * color[None, None, :] * 1.35
        result = result * (1.0 - abs(global_sat)) + tinted * abs(global_sat)
    saturation = float(settings.get("saturation", 0.0))
    contrast = float(settings.get("contrast", 0.0))
    if abs(saturation) > 1e-4:
        result = apply_vibrance_saturation(result, 0.0, saturation)
    if abs(contrast) > 1e-4:
        result = (result - 0.5) * (1.0 + contrast / 100.0) + 0.5
    return np.clip(result, 0, 1)


def apply_monochrome_workspace(img, settings):
    """Expanded monochrome conversion with mixer, virtual filter, structure and toning."""
    source = np.clip(img, 0, 1).astype(np.float32)
    b, g, r = cv2.split(source)
    weights = np.array([
        float(settings.get("mix_blue", 11.0)),
        float(settings.get("mix_green", 59.0)),
        float(settings.get("mix_red", 30.0)),
    ], dtype=np.float32)
    total = max(float(np.sum(np.abs(weights))), 1.0)
    gray = np.clip((b * weights[0] + g * weights[1] + r * weights[2]) / total, 0, 1)
    filter_strength = float(settings.get("filter_strength", 0.0)) / 100.0
    if abs(filter_strength) > 1e-6:
        filter_color = _hue_bgr(settings.get("filter_hue", 45.0))
        affinity = np.clip(1.0 - np.linalg.norm(source - filter_color[None, None, :], axis=2) / np.sqrt(3), 0, 1)
        gray = np.clip(gray + (affinity - 0.5) * filter_strength * 0.7, 0, 1)
    structure = float(settings.get("structure", 0.0)) / 100.0
    if abs(structure) > 1e-6:
        radius = max(0.5, float(settings.get("structure_radius", 3.0)))
        blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=radius)
        gray = np.clip(gray + (gray - blur) * structure * 1.5, 0, 1)
    contrast = float(settings.get("contrast", 0.0)) / 100.0
    brightness = float(settings.get("brightness", 0.0)) / 100.0
    gray = np.clip((gray - 0.5) * (1.0 + contrast) + 0.5 + brightness * 0.35, 0, 1).astype(np.float32)
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = apply_split_tone(
        out,
        float(settings.get("tone_shadow_hue", 30.0)), float(settings.get("tone_shadow_sat", 0.0)),
        float(settings.get("tone_highlight_hue", 45.0)), float(settings.get("tone_highlight_sat", 0.0)),
        float(settings.get("tone_balance", 0.0)),
    )
    grain = float(settings.get("grain", 0.0)) / 100.0
    if grain > 1e-6:
        size = max(0.35, float(settings.get("grain_size", 1.0)))
        noise_h = max(2, int(out.shape[0] / size))
        noise_w = max(2, int(out.shape[1] / size))
        noise = np.random.normal(0, 0.055 * grain, (noise_h, noise_w)).astype(np.float32)
        noise = cv2.resize(noise, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_LINEAR)
        out = np.clip(out + noise[..., None], 0, 1)
    burn = float(settings.get("burn_edges", 0.0)) / 100.0
    border = float(settings.get("border_strength", 0.0)) / 100.0
    if burn > 1e-6 or border > 1e-6:
        h, w = out.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        edge_x = np.minimum(xx, w - 1 - xx) / max(w * 0.5, 1)
        edge_y = np.minimum(yy, h - 1 - yy) / max(h * 0.5, 1)
        edge_distance = np.minimum(edge_x, edge_y)
        if burn > 1e-6:
            out *= (1.0 - burn * np.clip(1.0 - edge_distance, 0, 1) ** 2)[..., None]
        if border > 1e-6:
            width = float(np.clip(settings.get("border_width", 2.0), 0.2, 15.0)) / 100.0 * 2.0
            border_mask = np.clip((width - edge_distance) / max(width * 0.4, 1e-4), 0, 1)
            out *= (1.0 - border * border_mask)[..., None]
    return np.clip(out, 0, 1)


def blend_filter_result(base, filtered, opacity=1.0, mode="normal"):
    """Blend one creative-filter result over its input."""
    a = np.clip(base, 0, 1).astype(np.float32)
    b = np.clip(filtered, 0, 1).astype(np.float32)
    mode = str(mode or "normal").lower()
    if mode == "multiply":
        mixed = a * b
    elif mode == "screen":
        mixed = 1.0 - (1.0 - a) * (1.0 - b)
    elif mode == "soft light":
        mixed = (1.0 - 2.0 * b) * a * a + 2.0 * b * a
    elif mode == "overlay":
        mixed = np.where(a <= 0.5, 2.0 * a * b, 1.0 - 2.0 * (1.0 - a) * (1.0 - b))
    elif mode == "luminosity":
        lab_a = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
        lab_b = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)
        lab_a[..., 0] = lab_b[..., 0]
        mixed = cv2.cvtColor(lab_a, cv2.COLOR_LAB2BGR)
    elif mode == "color":
        lab_a = cv2.cvtColor(a, cv2.COLOR_BGR2LAB)
        lab_b = cv2.cvtColor(b, cv2.COLOR_BGR2LAB)
        lab_a[..., 1:] = lab_b[..., 1:]
        mixed = cv2.cvtColor(lab_a, cv2.COLOR_LAB2BGR)
    else:
        mixed = b
    amount = float(np.clip(opacity, 0, 1))
    return np.clip(a * (1.0 - amount) + mixed * amount, 0, 1)


def apply_analog_effects(img, settings):
    """Composable analog/camera effects used by an Analog creative-filter block."""
    out = np.clip(img, 0, 1).astype(np.float32).copy()
    h, w = out.shape[:2]
    halation = float(settings.get("halation", 0.0)) / 100.0
    if halation > 1e-6:
        lum = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        highlights = np.clip((lum - 0.65) / 0.35, 0, 1)
        glow = cv2.GaussianBlur(highlights, (0, 0), sigmaX=max(2.0, min(h, w) * 0.012))
        warm = np.dstack((glow * 0.12, glow * 0.38, glow))
        out = np.clip(out + warm * halation * 0.38, 0, 1)
    diffusion = float(settings.get("diffusion", 0.0)) / 100.0
    if diffusion > 1e-6:
        radius = max(0.5, float(settings.get("diffusion_radius", 5.0)))
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=radius)
        screened = 1.0 - (1.0 - out) * (1.0 - blurred)
        out = out * (1.0 - diffusion * 0.45) + screened * diffusion * 0.45
    bokeh = float(settings.get("bokeh", 0.0)) / 100.0
    if bokeh > 1e-6:
        cx = float(settings.get("bokeh_x", 50.0)) / 100.0 * (w - 1)
        cy = float(settings.get("bokeh_y", 50.0)) / 100.0 * (h - 1)
        size = max(0.05, float(settings.get("bokeh_size", 35.0)) / 100.0)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        distance = np.sqrt(((xx - cx) / max(w, 1)) ** 2 + ((yy - cy) / max(h, 1)) ** 2)
        mask = np.clip((distance - size * 0.5) / max(size * 0.35, 0.02), 0, 1) * bokeh
        blurred = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0 + bokeh * 14.0)
        out = out * (1.0 - mask[..., None]) + blurred * mask[..., None]
    leak = float(settings.get("light_leak", 0.0)) / 100.0
    if leak > 1e-6:
        x0 = float(settings.get("leak_x", 0.0)) / 100.0 * (w - 1)
        y0 = float(settings.get("leak_y", 45.0)) / 100.0 * (h - 1)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        radius = max(w, h) * max(0.1, float(settings.get("leak_size", 45.0)) / 100.0)
        falloff = np.clip(1.0 - np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2) / radius, 0, 1) ** 1.5
        color = _hue_bgr(settings.get("leak_hue", 18.0))
        out = 1.0 - (1.0 - out) * (1.0 - color[None, None, :] * falloff[..., None] * leak * 0.8)
    shift = float(settings.get("chromatic_shift", 0.0)) / 100.0
    if shift > 1e-6:
        pixels = shift * max(1.0, min(h, w) * 0.018)
        angle = np.deg2rad(float(settings.get("chromatic_angle", 0.0)))
        dx, dy = pixels * np.cos(angle), pixels * np.sin(angle)
        matrix_pos = np.float32([[1, 0, dx], [0, 1, dy]])
        matrix_neg = np.float32([[1, 0, -dx], [0, 1, -dy]])
        b, g, r = cv2.split(out)
        b = cv2.warpAffine(b, matrix_neg, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        r = cv2.warpAffine(r, matrix_pos, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        out = cv2.merge((b, g, r))
    motion = float(settings.get("motion_blur", 0.0)) / 100.0
    if motion > 1e-6:
        length = max(3, int(3 + motion * 35) | 1)
        kernel = np.zeros((length, length), np.float32)
        kernel[length // 2, :] = 1.0 / length
        matrix = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), float(settings.get("motion_angle", 0.0)), 1.0)
        kernel = cv2.warpAffine(kernel, matrix, (length, length))
        kernel /= kernel.sum() + 1e-6
        blurred = cv2.filter2D(out, -1, kernel, borderType=cv2.BORDER_REFLECT_101)
        out = out * (1.0 - motion) + blurred * motion
    zoom = float(settings.get("zoom_blur", 0.0)) / 100.0
    rotation = float(settings.get("rotation_blur", 0.0)) / 100.0
    if zoom > 1e-6 or rotation > 1e-6:
        accum = out.copy()
        samples = 7
        for index in range(1, samples):
            fraction = index / (samples - 1)
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), rotation * 8.0 * fraction, 1.0 + zoom * 0.08 * fraction)
            accum += cv2.warpAffine(out, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        transformed = accum / samples
        amount = max(zoom, rotation)
        out = out * (1.0 - amount) + transformed * amount
    dust = float(settings.get("dust_scratches", 0.0)) / 100.0
    if dust > 1e-6:
        rng = np.random.default_rng(int(settings.get("dust_seed", 1977)))
        overlay = np.zeros((h, w), np.float32)
        count = int(4 + dust * 90)
        for _ in range(count):
            if rng.random() < 0.3:
                x = int(rng.integers(0, max(w, 1)))
                cv2.line(overlay, (x, int(rng.integers(0, max(h // 2, 1)))),
                         (x + int(rng.integers(-3, 4)), int(rng.integers(max(h // 2, 1), max(h, 1)))),
                         float(rng.uniform(0.2, 0.8)), int(rng.integers(1, 3)))
            else:
                cv2.circle(overlay, (int(rng.integers(0, max(w, 1))), int(rng.integers(0, max(h, 1)))),
                           int(rng.integers(1, max(2, int(min(h, w) * 0.008)))), float(rng.uniform(0.2, 1.0)), -1)
        out = np.clip(out * (1.0 - overlay[..., None] * dust * 0.7), 0, 1)
    path = str(settings.get("double_exposure_path", "") or "")
    double_amount = float(settings.get("double_exposure", 0.0)) / 100.0
    if path and double_amount > 1e-6:
        second = cv2.imread(path, cv2.IMREAD_COLOR)
        if second is not None:
            second = cv2.resize(second, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
            mode = str(settings.get("double_blend", "screen")).lower()
            double = blend_filter_result(out, second, 1.0, mode)
            out = out * (1.0 - double_amount) + double * double_amount
    return np.clip(out, 0, 1).astype(np.float32)


def apply_creative_filter_stack(img, filters, mask_library=None):
    """Apply ordered, repeatable creative filters with blend, opacity, and shared masks."""
    if not filters:
        return img
    out = np.clip(img, 0, 1).astype(np.float32)
    library = list(mask_library or [])
    masks = {str(item.get("id")): item for item in library if item.get("id")}
    for item in filters or []:
        if not item.get("enabled", True):
            continue
        kind = str(item.get("type", "basic")).lower()
        settings = item.get("settings") or {}
        if kind == "color_grade":
            filtered = apply_four_way_color_grade(out, settings)
        elif kind == "monochrome":
            filtered = apply_monochrome_workspace(out, settings)
        elif kind == "analog":
            filtered = apply_analog_effects(out, settings)
        elif kind == "basic":
            filtered = out.copy()
            exposure = float(settings.get("exposure", 0.0))
            contrast = float(settings.get("contrast", 0.0))
            saturation = float(settings.get("saturation", 0.0))
            clarity = float(settings.get("clarity", 0.0))
            filtered *= 2.0 ** exposure
            filtered = (filtered - 0.5) * (1.0 + contrast / 100.0) + 0.5
            filtered = apply_vibrance_saturation(np.clip(filtered, 0, 1), 0, saturation)
            if abs(clarity) > 1e-4:
                blur = cv2.GaussianBlur(filtered, (0, 0), sigmaX=3)
                filtered += (filtered - blur) * clarity / 100.0
        else:
            continue
        opacity = float(item.get("opacity", 100.0)) / 100.0
        blended = blend_filter_result(out, filtered, opacity, item.get("blend_mode", "normal"))
        mask_id = str(item.get("mask_id") or "")
        if mask_id and mask_id in masks:
            mask = build_shared_mask(out, masks[mask_id], library)[..., None]
            if item.get("mask_invert"):
                mask = 1.0 - mask
            out = out * (1.0 - mask) + blended * mask
        else:
            out = blended
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
        from app_paths import primary_lensfun_db
        configured_db = primary_lensfun_db()
        db = lensfunpy.Database(paths=[configured_db]) if configured_db else lensfunpy.Database()
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
    img_bgr = apply_advanced_geometry(
        img_bgr,
        getattr(r, "perspective_horizontal", 0.0),
        getattr(r, "warp_top", 0.0), getattr(r, "warp_bottom", 0.0),
        getattr(r, "warp_left", 0.0), getattr(r, "warp_right", 0.0),
    )
    img_bgr = apply_keystone(img_bgr, getattr(r, "keystone_points", []))
    img_bgr = apply_wide_angle_stretch(img_bgr, getattr(r, "wide_angle", 0.0))
    img_bgr = apply_horizon(img_bgr, r.horizon)
    if bool(getattr(r, "geometry_auto_crop", False)):
        img_bgr = apply_crop(img_bgr, geometry_auto_crop_bounds(r))
    img_bgr = apply_crop(img_bgr, r.crop)
    img_bgr = apply_diorama(
        img_bgr, getattr(r, "diorama_strength", 0.0),
        getattr(r, "diorama_position", 50.0), getattr(r, "diorama_width", 30.0),
        getattr(r, "diorama_angle", 0.0),
    )
    img_bgr = apply_denoise(
        img_bgr, r.denoise_luminance, r.denoise_chroma, r.denoise_strength,
        detail_preserve=getattr(r, 'denoise_detail', 50.0),
        method=getattr(r, 'denoise_method', 'auto'),
        edge_preserve=getattr(r, "denoise_edge_preserve", 70.0),
        deband=getattr(r, "denoise_deband", 0.0),
        deband_orientation=getattr(r, "denoise_deband_orientation", "auto"),
        jpeg_artifacts=getattr(r, "denoise_jpeg_artifacts", 0.0),
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

    if bool(getattr(r, "portrait_detail_enabled", False)):
        portrait_shared = None
        portrait_mask_id = str(getattr(r, "portrait_mask_id", "") or "")
        mask_library = getattr(r, "mask_library", None) or []
        if portrait_mask_id:
            match = next((m for m in mask_library if str(m.get("id", "")) == portrait_mask_id), None)
            if match is not None:
                portrait_shared = build_shared_mask(img, match, mask_library)
        img = apply_portrait_detail(img, {
            "enabled": True,
            "skin_color": getattr(r, "portrait_skin_color", (0.55, 0.62, 0.76)),
            "color_reach": getattr(r, "portrait_color_reach", 28.0),
            "small_smooth": getattr(r, "portrait_small_smooth", 20.0),
            "medium_smooth": getattr(r, "portrait_medium_smooth", 10.0),
            "large_smooth": getattr(r, "portrait_large_smooth", 0.0),
            "edge_preserve": getattr(r, "portrait_edge_preserve", 75.0),
            "texture_recovery": getattr(r, "portrait_texture_recovery", 45.0),
        }, portrait_shared)

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
        media = str(getattr(r, "output_sharpen_media", "custom") or "custom")
        radius = 0.8 if media == "custom" else output_sharpen_params(
            getattr(r, "output_sharpen_ppi", 300.0), media
        )[1]
        img = apply_output_sharpen(img, r.output_sharpen, radius=radius)

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

    img = apply_creative_filter_stack(
        img, getattr(r, "creative_filters", None) or [],
        getattr(r, "mask_library", None) or [],
    )

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

    img = apply_zebra_highlight_exposure(
        img,
        exposure=getattr(r, "zebra_exposure", 0.0),
        threshold=getattr(r, "zebra_threshold", 95.0),
        feather=getattr(r, "zebra_feather", 5.0),
    )

    # Distraction corrections deliberately run late: users mark what they see
    # in the developed image, and normalized coordinates reproduce at export.
    operations = getattr(r, "distraction_operations", None) or []
    if operations or bool(getattr(r, "reflection_enabled", False)):
        from distractions import (apply_distraction_operations,
                                   apply_reflection_adjustment,
                                   edit_reflection_mask,
                                   reflection_mask)
        working = np.clip(img, 0, 1).astype(np.float32)
        if bool(getattr(r, "reflection_enabled", False)):
            rmask = reflection_mask(
                working,
                getattr(r, "reflection_sensitivity", 55.0),
                getattr(r, "reflection_blur", 8.0),
            )
            rmask = edit_reflection_mask(
                rmask, getattr(r, "reflection_mask_strokes", None) or []
            )
            working = apply_reflection_adjustment(
                working, rmask,
                getattr(r, "reflection_strength", 50.0),
                getattr(r, "reflection_highlights", -35.0),
                getattr(r, "reflection_saturation", 0.0),
                getattr(r, "reflection_neutralize", 20.0),
                getattr(r, "reflection_contrast", 10.0),
            )
        if operations:
            working = apply_distraction_operations(working, operations)
        img = np.clip(working, 0, 1)

    if getattr(r, "line_reflection_points", None):
        from line_reflection import reflect_under_line
        img = reflect_under_line(
            img, r.line_reflection_points, getattr(r, "line_reflection_side", -1),
            getattr(r, "line_reflection_opacity", 100.0),
            getattr(r, "line_reflection_feather", 0.0),
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


def _load_hdr_stack(paths, max_dim=0):
    """Load and center-crop an HDR bracket to a common size."""
    images, metas = [], []
    for path in paths:
        img, meta = load_image(path, use_camera_wb=True)
        if img is None:
            raise RuntimeError(f"Could not load: {path}")
        if img.dtype != np.uint8:
            maximum = float(np.iinfo(img.dtype).max) if np.issubdtype(img.dtype, np.integer) else 1.0
            img = np.clip(img.astype(np.float32) / max(maximum, 1.0) * 255.0, 0, 255).astype(np.uint8)
        if max_dim and max(img.shape[:2]) > max_dim:
            scale = float(max_dim) / max(img.shape[:2])
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        images.append(img)
        metas.append(meta or {})
    min_h = min(im.shape[0] for im in images)
    min_w = min(im.shape[1] for im in images)
    images = [im[(im.shape[0] - min_h) // 2:(im.shape[0] - min_h) // 2 + min_h,
                 (im.shape[1] - min_w) // 2:(im.shape[1] - min_w) // 2 + min_w]
              for im in images]
    return images, metas


def _align_hdr_stack(images):
    """Return an AlignMTB-aligned copy, leaving input frames untouched."""
    aligned = [im.copy() for im in images]
    if len(aligned) > 1:
        try:
            cv2.createAlignMTB().process(aligned, aligned)
        except Exception as exc:
            print(f"HDR align skipped: {exc}")
    return aligned


def _exposure_seconds_from_meta(meta):
    """Parse PhotoLab's normalized shutter metadata into seconds."""
    value = (meta or {}).get("exposure_seconds") or (meta or {}).get("shutter")
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip().lower().replace("seconds", "").replace("second", "").replace("sec", "").rstrip("s")
        if "/" in text:
            num, den = text.split("/", 1)
            return float(num) / float(den)
        return float(text)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def analyze_hdr_stack(paths, max_dim=900):
    """Return exposure, detail, motion, and alignment diagnostics."""
    images, metas = _load_hdr_stack(paths, max_dim=max_dim)
    grays = [cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32) for im in images]
    medians = [max(float(np.median(gray)), 0.01) for gray in grays]
    reference_index = int(np.argsort(medians)[len(medians) // 2])
    ref = grays[reference_index]
    records = []
    for index, (path, gray, meta) in enumerate(zip(paths, grays, metas)):
        shift, response = ((0.0, 0.0), 1.0)
        if index != reference_index:
            try:
                shift, response = cv2.phaseCorrelate(ref, gray)
            except cv2.error:
                response = 0.0
        records.append({
            "index": index, "path": str(path), "median_luma": medians[index],
            "relative_ev": float(np.log2(medians[index] / medians[reference_index])),
            "sharpness": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            "shift_x": float(shift[0]), "shift_y": float(shift[1]),
            "alignment_confidence": float(np.clip(response, 0.0, 1.0)),
            "exposure_seconds": _exposure_seconds_from_meta(meta),
        })
    return {"reference_index": reference_index, "frames": records}


def hdr_ghost_preview(paths, align=True, max_dim=700, reference_index=None):
    """Build a BGR preview with likely moving areas highlighted in magenta."""
    images, _metas = _load_hdr_stack(paths, max_dim=max_dim)
    if align:
        images = _align_hdr_stack(images)
    stack = np.stack([im.astype(np.float32) for im in images], axis=0)
    median = np.median(stack, axis=0)
    if reference_index is None:
        medians = [float(np.median(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))) for im in images]
        reference_index = int(np.argsort(medians)[len(medians) // 2])
    reference_index = int(np.clip(reference_index, 0, len(images) - 1))
    reference = images[reference_index].astype(np.float32)
    deviation = np.median(np.mean(np.abs(stack - median[None, ...]), axis=3), axis=0)
    mask = np.clip((deviation - 5.0) / 25.0, 0.0, 1.0)
    mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.5)
    overlay = reference.copy()
    magenta = np.zeros_like(overlay)
    magenta[..., 0] = 255
    magenta[..., 2] = 255
    alpha = (mask * 0.72)[..., None]
    overlay = overlay * (1.0 - alpha) + magenta * alpha
    return np.clip(overlay, 0, 255).astype(np.uint8), mask


def _prepare_hdr_images(paths, align=True, max_dim=0, deghost=0.0,
                        reference_index=None, ca_correction=0.0):
    images, metas = _load_hdr_stack(paths, max_dim=max_dim)
    if ca_correction:
        images = [np.clip(apply_chromatic_aberration_fix(
            image.astype(np.float32) / 255.0, ca_correction) * 255.0, 0, 255).astype(np.uint8)
                  for image in images]
    if align:
        images = _align_hdr_stack(images)
    if deghost and float(deghost) > 0:
        images = deghost_stack(images, strength=deghost, reference_index=reference_index)
    return images, metas


def merge_hdr_mertens(paths, align=True, max_dim=0,
                      contrast_weight=1.0, saturation_weight=1.0, exposure_weight=1.0,
                      deghost=0.0, reference_index=None, ca_correction=0.0):
    """Fuse multiple exposures with OpenCV MergeMertens. Returns uint8 BGR."""
    if not paths or len(paths) < 2:
        raise ValueError("HDR merge needs at least 2 images")
    images, _metas = _prepare_hdr_images(
        paths, align=align, max_dim=max_dim, deghost=deghost,
        reference_index=reference_index, ca_correction=ca_correction,
    )
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

def deghost_stack(images: List[np.ndarray], strength: float = 50.0,
                  reference_index=None) -> List[np.ndarray]:
    """Reduce motion ghosts before fusion.

    strength 0..100: blend each frame toward a robust reference (median of stack).
    Higher strength replaces moving regions more aggressively with the median.
    """
    if not images or len(images) < 2 or float(strength) < 1.0:
        return images
    t = max(0.0, min(1.0, float(strength) / 100.0))
    stack = np.stack([im.astype(np.float32) for im in images], axis=0)
    median = np.median(stack, axis=0)
    reference = median
    if reference_index is not None:
        reference = stack[int(np.clip(reference_index, 0, len(images) - 1))]
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
        y = x * (1.0 - w) + reference * w
        out.append(np.clip(y, 0, 255).astype(np.uint8))
    return out

def merge_hdr_debevec(
    paths,
    align=True,
    max_dim=0,
    deghost=0.0,
    tonemap: str = "reinhard",
    gamma: float = 1.0,
    reference_index=None,
    ca_correction: float = 0.0,
):
    """True HDR via Debevec calibration + tonemap. Returns uint8 BGR.

    Requires varying exposures (EXIF shutter preferred). Falls back to equal
    spaced times if metadata is missing.
    """
    if not paths or len(paths) < 2:
        raise ValueError("HDR merge needs at least 2 images")
    images, metas = _prepare_hdr_images(
        paths, align=align, max_dim=max_dim, deghost=deghost,
        reference_index=reference_index, ca_correction=ca_correction,
    )

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
