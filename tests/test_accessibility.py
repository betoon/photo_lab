from accessibility import clamp_ui_scale, scale_font_sizes


def test_ui_scale_is_clamped_and_handles_invalid_values():
    assert clamp_ui_scale(0.1) == 0.8
    assert clamp_ui_scale(2.5) == 1.6
    assert clamp_ui_scale("1.25") == 1.25
    assert clamp_ui_scale("bad") == 1.0


def test_stylesheet_scaling_changes_fonts_not_geometry():
    css = "QLabel { font-size: 12px; padding: 6px; } QPushButton { font-size:10px; }"
    scaled = scale_font_sizes(css, 1.25)
    assert "font-size: 15px" in scaled
    assert "font-size: 12px" in scaled
    assert "padding: 6px" in scaled


def test_stylesheet_scaling_is_bounded():
    css = "QLabel { font-size: 10px; }"
    assert "font-size: 8px" in scale_font_sizes(css, 0.1)
    assert "font-size: 16px" in scale_font_sizes(css, 9.0)
