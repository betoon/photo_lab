from pathlib import Path

from app_paths import resource_path


def test_bundled_michroma_and_license_are_present():
    font=Path(resource_path("assets","fonts","michroma","Michroma-Regular.ttf"))
    license_file=font.with_name("OFL.txt")
    assert font.is_file() and font.stat().st_size>60000
    assert "SIL OPEN FONT LICENSE Version 1.1" in license_file.read_text(encoding="utf-8")


def test_brand_font_registers_with_qt():
    from PyQt6.QtWidgets import QApplication
    from branding import load_brand_font
    app=QApplication.instance() or QApplication([])
    assert load_brand_font()=="Michroma"
