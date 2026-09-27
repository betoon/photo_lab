"""Shared application typography.

Keep the historical brand-font helpers for callers, but use one readable
family for navigation, editing controls, dialogs and diagnostic text.
"""
from __future__ import annotations

DEFAULT_UI_FAMILY = "Segoe UI"


def load_brand_font() -> str:
    """Return the shared UI family without registering a second display font."""
    return DEFAULT_UI_FAMILY


def brand_font_family() -> str:
    return DEFAULT_UI_FAMILY
