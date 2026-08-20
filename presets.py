"""
presets.py — Load PhotoLab JSON presets and Adobe Lightroom Classic / Camera Raw XMP presets.

Maps common crs:* develop settings into our Recipe fields. Not every LR slider
has a 1:1 equivalent; unmapped values are ignored safely.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Optional, Tuple, List

from imaging import Recipe

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
        return float(s)
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


def xmp_to_recipe(path: str, base: Optional[Recipe] = None) -> Recipe:
    """Load a Lightroom/ACR .xmp develop preset into a Recipe."""
    r = base if base is not None else Recipe()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    d = _parse_xmp_text(text)
    # Also try ElementTree for structured files
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
        r.clearview = max(0.0, min(100.0, dehaze))

    # Vibrance / Saturation
    vib = _f(g("Vibrance"))
    if vib is not None:
        r.vibrance = max(-100.0, min(100.0, vib))
    sat = _f(g("Saturation"))
    if sat is not None:
        r.saturation = max(-100.0, min(100.0, sat))

    # White balance
    temp = _f(g("Temperature"))
    if temp is not None:
        r.temperature = max(2000.0, min(12000.0, temp))
        r.wb_as_shot = False
    tint = _f(g("Tint"))
    if tint is not None:
        r.tint = max(-150.0, min(150.0, tint))
        r.wb_as_shot = False

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
        if str(bw).lower() in ("true", "1", "blackandwhite", "black & white"):
            r.black_and_white = True

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

    # Parametric curve-ish from Lights/Darks if present
    # ToneCurvePV2012 is a list — skip full parse; optional Shadows/Highlights already mapped

    return r


def load_preset_file(path: str, base: Optional[Recipe] = None) -> Recipe:
    """Load .json (PhotoLab) or .xmp (Lightroom/ACR) preset."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        return Recipe.load_json(path)
    if ext in (".xmp", ".XMP"):
        return xmp_to_recipe(path, base=base)
    raise ValueError(f"Unsupported preset format: {ext} (use .json or .xmp)")


def list_preset_files(folder: str) -> List[str]:
    """List .xmp and .json presets in a folder (non-recursive)."""
    out = []
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith((".xmp", ".json")):
            out.append(os.path.join(folder, name))
    return out
