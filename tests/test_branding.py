from pathlib import Path

from app_paths import resource_path


def test_bundled_michroma_and_license_are_present():
    font=Path(resource_path("assets","fonts","michroma","Michroma-Regular.ttf"))
    license_file=font.with_name("OFL.txt")
    assert font.is_file() and font.stat().st_size>60000
    assert "SIL OPEN FONT LICENSE Version 1.1" in license_file.read_text(encoding="utf-8")


def test_brand_font_uses_the_shared_ui_family():
    from branding import DEFAULT_UI_FAMILY, brand_font_family, load_brand_font
    assert load_brand_font() == DEFAULT_UI_FAMILY
    assert brand_font_family() == DEFAULT_UI_FAMILY
