"""Exercise the real workspace without accessing the user's catalog/settings."""
import pytest
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

import config
import main_window
from branding import DEFAULT_UI_FAMILY
from catalog import Catalog


@pytest.fixture
def window(monkeypatch):
    class MemoryConfig:
        def get(self, section, key, default=""):
            return default

        def path(self, key):
            return ""

        def set(self, *args):
            pass

        def save_user(self):
            pass

    app = QApplication.instance() or QApplication([])
    app.setFont(QFont(DEFAULT_UI_FAMILY, 10))
    monkeypatch.setattr(config, "get_config", lambda: MemoryConfig())
    monkeypatch.setattr(main_window, "Catalog", lambda: Catalog(":memory:"))
    monkeypatch.setattr(main_window.PhotoLab, "_load_recent_folders", lambda self: None)
    win = main_window.PhotoLab()
    yield win
    win.close()
    app.processEvents()


def test_wide_workspace_gives_extra_width_to_preview(window):
    app = QApplication.instance()
    window.resize(1200, 900)
    window.show()
    app.processEvents()
    before = window.preview.width()
    window.resize(2000, 1000)
    app.processEvents()
    splitter = window.develop_splitter
    assert window.preview.width() >= before + 650
    assert splitter.width() == window.centralWidget().width()
    right = splitter.widget(2)
    assert abs(right.geometry().right() + 1 - splitter.width()) <= 1
    assert not splitter.childrenCollapsible()


@pytest.mark.parametrize("maximized", [False, True])
def test_fullscreen_round_trip_preserves_window_state(window, maximized):
    app = QApplication.instance()
    if maximized:
        window.showMaximized()
    else:
        window.showNormal()
    app.processEvents()
    window.toggle_fullscreen()
    app.processEvents()
    assert window.isFullScreen()
    window.toggle_fullscreen()
    app.processEvents()
    assert not window.isFullScreen()
    assert window.isMaximized() == maximized


def test_scaled_controls_keep_one_font_and_panel_updates(window):
    app = QApplication.instance()
    window.show()
    window.set_interface_scale(1.6, announce=False)
    app.processEvents()
    assert window.corrections_panel.minimumWidth() == 400
    for widget in [window.preview, window.search_edit, window.debug_console, *window._cat_buttons]:
        assert widget.font().family() == DEFAULT_UI_FAMILY
    window.set_interface_scale(1.0, announce=False)
    app.processEvents()
    assert window.corrections_panel.minimumWidth() == 340


def test_correction_search_finds_other_categories_and_restores_view(window):
    app = QApplication.instance()
    window.show()
    app.processEvents()
    original = window.tool_stack.currentIndex()
    window.search_edit.setText("wide-angle")
    app.processEvents()
    assert window.tool_stack.currentIndex() == 3  # Geometry
    assert window._cat_buttons[3].isChecked()
    assert window._cat_buttons[3].isEnabled()
    assert window.correction_search_hint.isHidden()
    window.search_edit.setText("not-a-real-correction-xyz")
    app.processEvents()
    assert not window.correction_search_hint.isHidden()
    assert not any(button.isEnabled() for button in window._cat_buttons)
    window.search_edit.clear()
    app.processEvents()
    assert window.tool_stack.currentIndex() == original
    assert all(button.isEnabled() for button in window._cat_buttons)
    assert window.correction_search_hint.isHidden()


def test_xmp_application_updates_controls_and_reset_clears_imported_edits(window, tmp_path, monkeypatch):
    import numpy as np
    from imaging import Recipe
    from tests.test_xmp_advanced import xmp, curve

    window.current_path = 'test-photo'
    window.original_bgr = np.full((24, 24, 3), 30, dtype=np.uint8)
    window.recipes[window.current_path] = Recipe()
    monkeypatch.setattr(window, 'render_preview', lambda: None)
    path = xmp(tmp_path, 'crs:AutoTone="True" crs:Dehaze="-30" '
               'crs:ColorGradeGlobalHue="40" crs:ColorGradeGlobalSat="25"',
               curve('ToneCurvePV2012', ['0, 10', '255, 245']))
    window._apply_preset_path(path)
    recipe = window.recipes[window.current_path]
    assert recipe.exposure > 0
    assert window.sliders['clearview'].value() == -30
    assert window.sliders['grade_global_sat'].value() == 25
    assert window.color_grade_cb.isChecked()
    window.reset_module('color')
    assert not recipe.color_grade_enabled and recipe.grade_global_sat == 0
    window.reset_module('tone')
    assert recipe.curve_points == [] and recipe.curve_mode == 'luma'
