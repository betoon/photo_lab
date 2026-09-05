"""Use production Geometry widgets and history handlers without loading a catalog."""
import numpy as np
import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QMainWindow

import main_window
from imaging import Recipe, apply_recipe
from main_window import PhotoLab
from widgets import HistoryWidget, ImageCanvas


class GeometryHarness(PhotoLab):
    def __init__(self):
        QMainWindow.__init__(self)
        self.current_path = "test-image"
        self.original_bgr = np.random.default_rng(7).integers(0,256,(31,61,3),dtype=np.uint8)
        self.recipes = {self.current_path: Recipe()}
        self.meta_cache = {}
        self.sliders = {}
        self.preview = ImageCanvas()
        self.history_widget = HistoryWidget()
        self.render_timer = QTimer(self)
        self._pending_history_label = None
        self.rendered = None
        self._push_history("Original")

    def _maybe_autosave(self):
        pass

    def sync_sliders_to_recipe(self):
        pass

    def render_preview(self):
        self.rendered = apply_recipe(self.original_bgr, self.recipes[self.current_path])

    def closeEvent(self, event):
        event.accept()


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def test_apply_cancel_clear_and_actual_undo_redo_handlers(app, monkeypatch):
    window = GeometryHarness()
    def accept(dialog):
        dialog.canvas.line = [[.1,.2],[.9,.7]]
        dialog.opacity.setValue(80)
        dialog.accept()
        return dialog.result()
    monkeypatch.setattr(main_window.ReflectionDialog, "exec", accept)
    window.open_line_reflection()
    assert window.history_widget.list.count() == 2
    applied = window.rendered.copy()
    assert window.recipes[window.current_path].line_reflection_points
    window.undo_edit()
    assert not window.recipes[window.current_path].line_reflection_points
    assert not np.array_equal(window.rendered, applied)
    window.redo_edit()
    assert np.array_equal(window.rendered, applied)
    window.clear_line_reflection()
    assert not window.recipes[window.current_path].line_reflection_points
    window.undo_edit()
    assert np.array_equal(window.rendered, applied)
    count = window.history_widget.list.count()
    def reject(dialog):
        dialog.canvas.line = [[0,0],[1,1]]
        dialog.reject()
        return dialog.result()
    monkeypatch.setattr(main_window.ReflectionDialog, "exec", reject)
    window.open_line_reflection()
    assert window.history_widget.list.count() == count
    assert window.recipes[window.current_path].line_reflection_points == [[.1,.2],[.9,.7]]
    window.reset_module("geometry")
    assert not window.recipes[window.current_path].line_reflection_points
    window.close()


def test_geometry_panel_controls_update_canvas_without_editing_recipe(app):
    window = GeometryHarness()
    panel = window._build_geometry_tab()
    before = window.recipes[window.current_path].to_dict()
    window.show_grid_cb.setChecked(True)
    assert window.preview.show_grid
    for index in range(window.grid_combo.count()):
        window.grid_combo.setCurrentIndex(index)
        assert window.preview.grid_kind == window.grid_combo.currentData()
    window.grid_combo.setCurrentIndex(window.grid_combo.findData("three"))
    window.grid_horizon.setValue(66)
    window.grid_center.setValue(30)
    window.grid_density.setValue(18)
    assert window.preview.grid_horizon == .66
    assert window.preview.grid_center == .3
    assert window.preview.grid_density == 18
    assert window.recipes[window.current_path].to_dict() == before
    assert window.history_widget.list.count() == 1
    panel.close()
    window.close()
