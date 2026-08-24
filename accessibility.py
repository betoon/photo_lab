"""Small, Qt-free helpers for PhotoLab interface accessibility."""
from __future__ import annotations

import re


UI_SCALE_MIN = 0.8
UI_SCALE_MAX = 1.6
UI_SCALE_STEP = 0.1


def clamp_ui_scale(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 1.0
    return round(max(UI_SCALE_MIN, min(UI_SCALE_MAX, value)), 2)


def scale_font_sizes(stylesheet: str, scale: float) -> str:
    """Scale CSS font-size pixel declarations without changing widget geometry."""
    amount = clamp_ui_scale(scale)

    def replace(match):
        pixels = max(8, int(round(float(match.group(1)) * amount)))
        return f"font-size: {pixels}px"

    return re.sub(r"font-size\s*:\s*([0-9]+(?:\.[0-9]+)?)px", replace, stylesheet)
