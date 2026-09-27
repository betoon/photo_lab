"""
presets.py — Load PhotoLab JSON presets and Adobe Lightroom Classic / Camera Raw XMP presets.

Maps common crs:* develop settings into our Recipe fields. Not every LR slider
has a 1:1 equivalent; unmapped values are ignored safely.
"""

from __future__ import annotations

import os
import re
import copy
import math
import numpy as np
import xml.etree.ElementTree as ET
from typing import Optional, Tuple, List, Iterable

from imaging import Recipe, _image_to_float01, apply_white_balance, apply_creative_white_balance
from xmp_support import GRADE_FIELDS, estimate_auto_tone


PRESET_MODULE_FIELDS = {
    "Tone": (
        "exposure", "smart_light", "contrast", "highlights", "shadows", "whites", "blacks",
        "clarity", "gamma", "curve_shadows", "curve_darks", "curve_mids", "curve_lights",
        "curve_highlights", "curve_mode", "curve_points", "curve_r_points", "curve_g_points", "curve_b_points",
        "zone_enabled", "zone_placement", "zone_expansion", "zone_filter", "zone_snap", "zone_overlay",
        "zebra_threshold", "zebra_exposure", "zebra_feather",
    ),
    "Color": (
        "temperature", "tint", "wb_as_shot", "creative_temperature", "creative_tint",
        "vibrance", "saturation", "hsl_hue", "hsl_sat", "hsl_lum", "split_shadow_hue",
        "split_shadow_sat", "split_highlight_hue", "split_highlight_sat", "split_balance",
        "black_and_white", "ir_channel_swap", "ir_false_color", "ir_mono",
    ) + GRADE_FIELDS,
    "Detail": (
        "denoise_luminance", "denoise_chroma", "denoise_strength", "denoise_detail",
        "denoise_method", "noise_profile", "denoise_edge_preserve", "denoise_deband",
        "denoise_deband_orientation", "denoise_jpeg_artifacts",
        "sharpen_intensity", "sharpen_radius", "sharpen_threshold",
        "sharpen_detail", "output_sharpen", "astro_stretch", "astro_bg_remove",
        "astro_star_emphasis", "output_sharpen_media", "output_sharpen_ppi",
        "output_sharpen_width_in", "portrait_detail_enabled", "portrait_skin_color",
        "portrait_color_reach", "portrait_small_smooth", "portrait_medium_smooth",
        "portrait_large_smooth", "portrait_edge_preserve", "portrait_texture_recovery",
        "portrait_mask_id",
    ),
    "Geometry": (
        "horizon", "distortion", "perspective", "perspective_horizontal",
        "warp_top", "warp_bottom", "warp_left", "warp_right", "wide_angle",
        "diorama_strength", "diorama_position", "diorama_width", "diorama_angle",
        "keystone_points", "geometry_auto_crop", "line_reflection_points",
        "line_reflection_side", "line_reflection_opacity", "line_reflection_feather",
        "crop", "ca_amount", "lens_auto", "rotate_90",
    ),
    "Effects": ("clearview", "microcontrast", "vignette", "film_grain", "hdr_look"),
    "Local": ("local_points", "gradients", "brush_masks", "mask_library"),
    "Creative": ("creative_filters",),
}

# Namespaces seen in LR/ACR XMP
_NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "crs": "http://ns.adobe.com/camera-raw-settings/1.0/",
    "xmp": "http://ns.adobe.com/xap/1.0/",
}


def _f(val, default=None) -> Optional[float]:
    if val is None:
        return default
    try:
        s = str(val).strip().replace("+", "")
        number = float(s)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _get_crs(root: ET.Element, local: str) -> Optional[str]:
    """Find crs:LocalName anywhere in the tree (handles default ns and prefixes)."""
    # Try Clark notation with known URI
    uri = _NS["crs"]
    for el in root.iter():
        tag = el.tag
        if tag == f"{{{uri}}}{local}" or tag.endswith("}" + local) or tag == local:
            if el.text and el.text.strip():
                return el.text.strip()
        # attributes on Description
        for ak, av in el.attrib.items():
            if ak == f"{{{uri}}}{local}" or ak.endswith("}" + local) or ak == f"crs:{local}":
                return str(av).strip()
    # Regex fallback on raw-ish serialization
    return None


def _parse_xmp_text(text: str) -> dict:
    """Pull crs:Key=\"value\" pairs with a resilient regex (handles various serializations)."""
    found = {}
    # Attribute style: crs:Exposure2012="+0.50"
    for m in re.finditer(r'crs:([A-Za-z0-9_]+)\s*=\s*"([^"]*)"', text):
        found[m.group(1)] = m.group(2)
    # Element style: <crs:Exposure2012>+0.50</crs:Exposure2012>
    for m in re.finditer(r'<crs:([A-Za-z0-9_]+)[^>]*>([^<*]*)</crs:\1>', text):
        found[m.group(1)] = m.group(2).strip()
    # Sometimes without prefix in default ns
    for m in re.finditer(r'camera-raw-settings[^>]*?([A-Za-z0-9_]+)="([^"]*)"', text):
        found.setdefault(m.group(1), m.group(2))
    return found


def xmp_to_recipe(path: str, base: Optional[Recipe] = None, *, image_bgr=None, meta=None,
                  resolve_auto=True) -> Recipe:
    """Load a Lightroom/ACR .xmp develop preset into a Recipe."""
    # XMP develop presets are partial edits.  Work on a copy so applying one
    # preserves settings that the preset does not mention (especially the
    # image's current/as-shot WB) and a parse failure cannot mutate the live
    # recipe in place.
    r = copy.deepcopy(base) if base is not None else Recipe()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    d = _parse_xmp_text(text)
    # Also try ElementTree for structured files
    root = None
    try:
        root = ET.fromstring(text)
        for el in root.iter():
            tag = el.tag.split("}")[-1] if "}" in el.tag else el.tag
            if el.text and el.text.strip() and tag not in d:
                d[tag] = el.text.strip()
            for ak, av in el.attrib.items():
                local = ak.split("}")[-1] if "}" in ak else ak
                if local not in d:
                    d[local] = av
    except ET.ParseError:
        pass

    def g(*keys, default=None):
        for k in keys:
            if k in d and d[k] not in (None, ""):
                return d[k]
        return default

    # Resolve Auto before explicit values so fixed values in mixed presets win.
    # Materializing these once makes preview/export and history deterministic.
    if resolve_auto and str(g("AutoTone", default="false")).strip().lower() in ("true", "1"):
        if image_bgr is None:
            raise ValueError("This Auto Tone preset needs an open image to analyze")
        step = max(1, math.ceil(max(image_bgr.shape[:2]) / 512))
        source = _image_to_float01(image_bgr[::step, ::step])
        # WB overrides in the same preset are also used for the analysis.
        temp = _f(g("Temperature"))
        tint = _f(g("Tint"))
        absolute = temp is not None and 2000 <= temp <= 50000
        as_shot = r.wb_as_shot and not absolute
        if not (as_shot and meta and meta.get("wb_baked")):
            source = apply_white_balance(
                source, min(temp, 12000) if absolute else r.temperature,
                tint if absolute and tint is not None else r.tint,
                as_shot=as_shot, multipliers=(meta or {}).get("wb_multipliers"),
            )
        source = apply_creative_white_balance(
            source, temp if temp is not None and not absolute else r.creative_temperature,
            tint if tint is not None and not absolute else r.creative_tint,
        )
        for key, value in estimate_auto_tone(source).items():
            setattr(r, key, value)

    # Exposure (stops)
    exp = _f(g("Exposure2012", "Exposure"))
    if exp is not None:
        r.exposure = max(-5.0, min(5.0, exp))

    # Contrast
    con = _f(g("Contrast2012", "Contrast"))
    if con is not None:
        r.contrast = max(-100.0, min(100.0, con))

    # Highlights / Shadows / Whites / Blacks (2012 process)
    for src, dst in (
        ("Highlights2012", "highlights"),
        ("Shadows2012", "shadows"),
        ("Whites2012", "whites"),
        ("Blacks2012", "blacks"),
        ("Highlights", "highlights"),
        ("Shadows", "shadows"),
    ):
        v = _f(g(src))
        if v is not None:
            setattr(r, dst, max(-100.0, min(100.0, v)))

    # Clarity / Texture ~ clarity; Dehaze ~ clearview
    cl = _f(g("Clarity2012", "Clarity"))
    if cl is not None:
        r.clarity = max(-100.0, min(100.0, cl))
    tex = _f(g("Texture"))
    if tex is not None:
        r.microcontrast = max(-100.0, min(100.0, tex))
    dehaze = _f(g("Dehaze"))
    if dehaze is not None:
        r.clearview = max(-100.0, min(100.0, dehaze))

    # Vibrance / Saturation
    vib = _f(g("Vibrance"))
    if vib is not None:
        r.vibrance = max(-100.0, min(100.0, vib))
    sat = _f(g("Saturation"))
    if sat is not None:
        r.saturation = max(-100.0, min(100.0, sat))

    # White balance. Lightroom stores absolute Kelvin values for a custom WB,
    # but many converted creative presets use small signed Temperature/Tint
    # values as relative nudges. They are not Kelvin: clamping +6 to 2000 K
    # creates the severe orange cast this importer used to produce. Because
    # Recipe currently has only absolute WB fields, preserve the existing WB
    # for relative pairs rather than inventing an absolute illuminant.
    temp = _f(g("Temperature"))
    has_absolute_temp = temp is not None and 2000.0 <= temp <= 50000.0
    if has_absolute_temp:
        r.temperature = min(12000.0, temp)
        r.wb_as_shot = False
    elif temp is not None:
        r.creative_temperature = max(-100.0, min(100.0, temp))
    tint = _f(g("Tint"))
    if tint is not None and has_absolute_temp:
        r.tint = max(-150.0, min(150.0, tint))
        r.wb_as_shot = False
    elif tint is not None:
        r.creative_tint = max(-100.0, min(100.0, tint))

    # Sharpening
    sharp = _f(g("Sharpness"))
    if sharp is not None:
        r.sharpen_intensity = max(0.0, min(200.0, sharp * 1.5))
    radius = _f(g("SharpenRadius"))
    if radius is not None:
        r.sharpen_radius = max(0.1, min(5.0, radius))
    detail = _f(g("SharpenDetail"))
    # map detail lightly into threshold inverse
    if detail is not None:
        r.sharpen_threshold = max(0.0, min(50.0, (100.0 - detail) * 0.3))

    # Noise reduction
    lum = _f(g("LuminanceSmoothing", "LuminanceNoiseReduction"))
    if lum is not None:
        r.denoise_luminance = max(0.0, min(100.0, lum))
    chr_ = _f(g("ColorNoiseReduction"))
    if chr_ is not None:
        r.denoise_chroma = max(0.0, min(100.0, chr_))

    # Vignette
    vig = _f(g("PostCropVignetteAmount", "VignetteAmount"))
    if vig is not None:
        # LR negative = darken corners often
        r.vignette = max(0.0, min(100.0, abs(vig)))

    # Grain
    grain = _f(g("GrainAmount"))
    if grain is not None:
        r.film_grain = max(0.0, min(100.0, grain))

    # B&W
    bw = g("ConvertToGrayscale", "Treatment")
    if bw is not None:
        treatment = str(bw).strip().lower()
        if treatment in ("true", "1", "blackandwhite", "black & white"):
            r.black_and_white = True
        elif treatment in ("false", "0", "color", "colour"):
            r.black_and_white = False

    # HSL — HueAdjustmentRed etc. / SaturationAdjustmentRed / LuminanceAdjustmentRed
    hsl_map = [
        ("Red", 0), ("Orange", 1), ("Yellow", 2), ("Green", 3),
        ("Aqua", 4), ("Blue", 5), ("Purple", 6), ("Magenta", 7),
    ]
    hue_l = list(r.hsl_hue)
    sat_l = list(r.hsl_sat)
    lum_l = list(r.hsl_lum)
    for name, idx in hsl_map:
        h = _f(g(f"HueAdjustment{name}"))
        s = _f(g(f"SaturationAdjustment{name}"))
        l = _f(g(f"LuminanceAdjustment{name}"))
        if h is not None:
            hue_l[idx] = max(-100.0, min(100.0, h))
        if s is not None:
            sat_l[idx] = max(-100.0, min(100.0, s))
        if l is not None:
            lum_l[idx] = max(-100.0, min(100.0, l))
    r.hsl_hue = tuple(hue_l)
    r.hsl_sat = tuple(sat_l)
    r.hsl_lum = tuple(lum_l)

    # Structured RDF curve sequences are 0..255 input/output coordinates.
    for tags, field in (
        (("ToneCurvePV2012", "ToneCurve"), "curve_points"),
        (("ToneCurvePV2012Red", "ToneCurveRed"), "curve_r_points"),
        (("ToneCurvePV2012Green", "ToneCurveGreen"), "curve_g_points"),
        (("ToneCurvePV2012Blue", "ToneCurveBlue"), "curve_b_points"),
    ):
        if root is None:
            continue
        node = next((root.find(".//crs:" + tag, _NS) for tag in tags
                     if root.find(".//crs:" + tag, _NS) is not None), None)
        if node is None:
            continue
        points = []
        for item in node.findall(".//rdf:li", _NS):
            pair = (item.text or "").split(",")
            if len(pair) != 2:
                raise ValueError("Invalid XMP tone curve point")
            x, y = _f(pair[0]), _f(pair[1])
            if x is None or y is None or not (0 <= x <= 255 and 0 <= y <= 255):
                raise ValueError("XMP curve coordinates must be finite and in 0..255")
            points.append([x / 255.0, y / 255.0])
        if points:
            points.sort(key=lambda p: p[0])
            if len(points) < 2 or any(a[0] == b[0] for a, b in zip(points, points[1:])):
                raise ValueError("XMP curves require at least two distinct input coordinates")
        elif node.find("rdf:Seq", _NS) is None:
            raise ValueError("XMP tone curve is missing its RDF sequence")
        setattr(r, field, points)
        if field == "curve_points":
            r.curve_mode = "rgb"

    for src, dst in (("ParametricShadows", "curve_shadows"),
                     ("ParametricDarks", "curve_darks"),
                     ("ParametricLights", "curve_lights"),
                     ("ParametricHighlights", "curve_highlights")):
        value = _f(g(src))
        if value is not None:
            setattr(r, dst, max(-100.0, min(100.0, value)))

    grade_map = {
        "SplitToningShadowHue": ("split_shadow_hue", 0, 360),
        "SplitToningShadowSaturation": ("split_shadow_sat", 0, 100),
        "SplitToningHighlightHue": ("split_highlight_hue", 0, 360),
        "SplitToningHighlightSaturation": ("split_highlight_sat", 0, 100),
        "SplitToningBalance": ("split_balance", -100, 100),
        "ColorGradeMidtoneHue": ("grade_midtone_hue", 0, 360),
        "ColorGradeMidtoneSat": ("grade_midtone_sat", 0, 100),
        "ColorGradeMidtoneLum": ("grade_midtone_lum", -100, 100),
        "ColorGradeShadowLum": ("split_shadow_lum", -100, 100),
        "ColorGradeHighlightLum": ("split_highlight_lum", -100, 100),
        "ColorGradeGlobalHue": ("grade_global_hue", 0, 360),
        "ColorGradeGlobalSat": ("grade_global_sat", 0, 100),
        "ColorGradeGlobalLum": ("grade_global_lum", -100, 100),
        "ColorGradeBlending": ("grade_blending", 0, 100),
        "ColorGradeBalance": ("split_balance", -100, 100),
    }
    for src, (dst, low, high) in grade_map.items():
        value = _f(g(src))
        if value is not None:
            setattr(r, dst, max(low, min(high, value)))
            r.color_grade_enabled = True

    return r


def load_preset_file(path: str, base: Optional[Recipe] = None, *, image_bgr=None, meta=None,
                     resolve_auto=True) -> Recipe:
    """Load .json (PhotoLab) or .xmp (Lightroom/ACR) preset."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return Recipe.load_json(path)
    if ext in (".xmp", ".XMP"):
        return xmp_to_recipe(path, base=base, image_bgr=image_bgr, meta=meta, resolve_auto=resolve_auto)
    raise ValueError(f"Unsupported preset format: {ext} (use .json or .xmp)")


def apply_preset_file(path: str, base: Optional[Recipe] = None, strength: float = 1.0,
                      modules: Optional[Iterable[str]] = None, *, image_bgr=None, meta=None) -> Recipe:
    """Apply a preset non-destructively with strength and module filtering."""
    original = copy.deepcopy(base) if base is not None else Recipe()
    amount = max(0.0, min(1.0, float(strength)))
    enabled = set(PRESET_MODULE_FIELDS.keys() if modules is None else modules)
    if amount == 0 or not enabled:
        return original
    target = load_preset_file(path, base=original, image_bgr=image_bgr, meta=meta,
                              resolve_auto="Tone" in enabled)
    result = copy.deepcopy(original)

    for module, names in PRESET_MODULE_FIELDS.items():
        if module not in enabled:
            continue
        for name in names:
            if not hasattr(target, name) or not hasattr(result, name):
                continue
            before = getattr(original, name)
            after = getattr(target, name)
            if isinstance(before, (int, float)) and not isinstance(before, bool) and isinstance(after, (int, float)):
                value = float(before) + (float(after) - float(before)) * amount
                if isinstance(before, int) and isinstance(after, int):
                    value = int(round(value))
            elif isinstance(before, tuple) and isinstance(after, tuple) and len(before) == len(after):
                value = tuple(float(a) + (float(b) - float(a)) * amount for a, b in zip(before, after))
            else:
                value = copy.deepcopy(after if amount >= 0.5 else before)
            setattr(result, name, value)
    if "Tone" in enabled and amount > 0:
        for name in ("curve_points", "curve_r_points", "curve_g_points", "curve_b_points"):
            before, after = getattr(original, name), getattr(target, name)
            if before == after:
                continue
            a = before or [[0, 0], [1, 1]]
            b = after or [[0, 0], [1, 1]]
            xs = sorted({0.0, 1.0, *(p[0] for p in a), *(p[0] for p in b)})
            ya = np.interp(xs, [p[0] for p in a], [p[1] for p in a])
            yb = np.interp(xs, [p[0] for p in b], [p[1] for p in b])
            setattr(result, name, [[float(x), float(y)] for x, y in zip(xs, ya+(yb-ya)*amount)])
        result.curve_mode = target.curve_mode
    if "Color" in enabled and amount > 0 and target.color_grade_enabled:
        result.color_grade_enabled = True
        # Strength changes tint intensity, not its hue when fading from neutral.
        # Between two existing tints, follow the shortest arc around the wheel.
        for prefix in ("split_shadow", "grade_midtone", "split_highlight", "grade_global"):
            a, b = getattr(original, prefix + "_hue"), getattr(target, prefix + "_hue")
            sa, sb = getattr(original, prefix + "_sat"), getattr(target, prefix + "_sat")
            hue = b if sa == 0 else a if sb == 0 else (a + ((b-a+180) % 360-180)*amount) % 360
            setattr(result, prefix + "_hue", hue)
    return result


def list_preset_files(folder: str, recursive: bool = False) -> List[str]:
    """List .xmp and .json presets, optionally including category subfolders."""
    out = []
    if not os.path.isdir(folder):
        return out
    if recursive:
        for root, dirs, names in os.walk(folder):
            dirs.sort(key=str.lower)
            for name in sorted(names, key=str.lower):
                if name.lower().endswith((".xmp", ".json")):
                    out.append(os.path.join(root, name))
    else:
        for name in sorted(os.listdir(folder)):
            if name.lower().endswith((".xmp", ".json")):
                out.append(os.path.join(folder, name))
    return out
