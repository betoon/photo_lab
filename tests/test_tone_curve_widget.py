from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from widgets import ToneCurveWidget


def test_tone_curve_point_curves_are_initialized_and_syncable():
    app = QApplication.instance() or QApplication([])
    widget = ToneCurveWidget()

    widget.set_point_curve("luma", [[0.0, 0.0], [0.5, 0.65], [1.0, 1.0]])
    widget.set_point_curve("r", [])

    assert widget.point_curves["luma"][1] == [0.5, 0.65]
    assert widget.point_curves["r"] == [[0.0, 0.0], [1.0, 1.0]]
    widget.close()
    assert app is not None
