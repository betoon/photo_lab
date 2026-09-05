import numpy as np
import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QDialogButtonBox

from imaging import Recipe
from reflection_dialog import ReflectionDialog


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


def draw(dialog, start, end):
    canvas = dialog.canvas
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=canvas.screen_point(start).toPoint())
    QTest.mouseMove(canvas, canvas.screen_point(end).toPoint())
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=canvas.screen_point(end).toPoint())
    QTest.qWait(150)


def test_draw_refine_preview_side_apply_and_recipe_isolation(app):
    recipe = Recipe()
    image = np.zeros((101, 201, 3), np.uint8)
    image[:50] = (30, 80, 210)
    dialog = ReflectionDialog(None, image, recipe)
    dialog.show()
    app.processEvents()
    assert not dialog.apply_button.isEnabled()
    draw(dialog, [.1,.5], [.9,.5])
    assert dialog.apply_button.isEnabled()
    assert not recipe.line_reflection_points
    reflected = dialog.canvas.pixmap.toImage()
    assert reflected.pixelColor(100, 70).red() > 190, (dialog.canvas.line, dialog.canvas.side, dialog.timer.isActive())
    draw(dialog, dialog.canvas.line[1], [.8,.7])
    assert dialog.canvas.line[1] == pytest.approx([.8,.7], abs=.004)
    dialog.pick.click()
    QTest.mouseClick(dialog.canvas, Qt.MouseButton.LeftButton,
                     pos=dialog.canvas.screen_point([.5,.9]).toPoint())
    QTest.qWait(60)
    assert dialog.canvas.side == 1
    dialog.swap.click()
    assert dialog.canvas.side == -1
    dialog.opacity.setValue(72)
    dialog.feather.setValue(4)
    dialog.apply_button.click()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert dialog.recipe.line_reflection_opacity == 72
    assert dialog.recipe.line_reflection_feather == 4
    assert recipe.line_reflection_points == []


def test_cancel_escape_and_degenerate_line_do_not_apply(app):
    recipe = Recipe(line_reflection_points=[[0,.5],[1,.5]])
    for escape in (False, True):
        dialog = ReflectionDialog(None, np.zeros((31,61,3), np.uint8), recipe)
        dialog.show()
        app.processEvents()
        draw(dialog, [.3,.2], [.3,.2])
        assert not dialog.apply_button.isEnabled()
        dialog.accept()
        assert dialog.result() != QDialog.DialogCode.Accepted
        if escape:
            QTest.keyClick(dialog, Qt.Key.Key_Escape)
        else:
            dialog.buttons.button(QDialogButtonBox.StandardButton.Cancel).click()
        assert dialog.result() == QDialog.DialogCode.Rejected
        assert recipe.line_reflection_points == [[0,.5],[1,.5]]


def test_original_toggle_and_resize_preserve_line(app):
    image = np.zeros((61,121,3), np.uint8)
    image[:30] = 220
    dialog = ReflectionDialog(None, image, Recipe(line_reflection_points=[[0,.5],[1,.5]]))
    dialog.show()
    app.processEvents()
    dialog.original.setChecked(True)
    assert dialog.canvas.pixmap.toImage().pixelColor(60,45).red() == 0
    dialog.resize(750,650)
    app.processEvents()
    assert dialog.canvas.line == [[0,.5],[1,.5]]
    dialog.original.setChecked(False)
    assert dialog.canvas.pixmap.toImage().pixelColor(60,45).red() == 220
    dialog.reject()
