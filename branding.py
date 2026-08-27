"""PhotoLab typography and bundled brand-font support."""
from __future__ import annotations

import logging
from pathlib import Path

from PyQt6.QtGui import QFontDatabase

from app_paths import resource_path


log = logging.getLogger(__name__)
DEFAULT_UI_FAMILY = "Segoe UI"
BRAND_FALLBACKS = ("Michroma", "Eurostile", "Segoe UI")
_brand_family = DEFAULT_UI_FAMILY


def load_brand_font() -> str:
    """Register bundled Michroma and return its actual Qt family name."""
    global _brand_family
    font_path = Path(resource_path("assets", "fonts", "michroma", "Michroma-Regular.ttf"))
    if font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        if font_id >= 0:
            families = QFontDatabase.applicationFontFamilies(font_id)
            if families:
                _brand_family = families[0]
                log.info("Loaded PhotoLab brand font: %s", _brand_family)
                return _brand_family
        log.warning("Could not register bundled brand font: %s", font_path)
    else:
        log.warning("Bundled brand font was not found: %s", font_path)
    installed = set(QFontDatabase.families())
    _brand_family = next((family for family in BRAND_FALLBACKS if family in installed), DEFAULT_UI_FAMILY)
    return _brand_family


def brand_font_family() -> str:
    return _brand_family
