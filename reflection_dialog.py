"""Transactional line-reflection editor: preview changes never touch the recipe."""
import copy

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
)

from line_reflection import line_frame, reflect_under_line
from qt_utils import cv_to_qpixmap


class ReflectionCanvas(QWidget):
    changed = pyqtSignal()

    def __init__(self, image, line=None, side=-1):
        super().__init__()
        self.image = image
        self.line = copy.deepcopy(line or [])
        self.side = side
        self.pick_source = False
        self.drag = None
        self.pixmap = cv_to_qpixmap(image)
        self.setMinimumSize(360, 240)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    def image_rect(self):
        h, w = self.image.shape[:2]
        scale = min(self.width()/w, self.height()/h)
        return QRectF((self.width()-w*scale)/2, (self.height()-h*scale)/2,
                      w*scale, h*scale)

    def screen_point(self, point):
        rect = self.image_rect()
        return QPointF(rect.left()+point[0]*(rect.width()-1),
                       rect.top()+point[1]*(rect.height()-1))

    def normalized(self, pos):
        rect = self.image_rect()
        return [float(np.clip((pos.x()-rect.left())/max(rect.width()-1, 1), 0, 1)),
                float(np.clip((pos.y()-rect.top())/max(rect.height()-1, 1), 0, 1))]

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self.image_rect().contains(event.position()):
            return
        point = self.normalized(event.position())
        frame = line_frame(self.line, self.image.shape[1], self.image.shape[0])
        if self.pick_source and frame is not None:
            a, normal = frame
            distance = np.dot(np.array(point)*[self.image.shape[1]-1, self.image.shape[0]-1]-a, normal)
            if abs(distance) > 1e-6:
                self.side = -1 if distance < 0 else 1
                self.changed.emit()
                self.update()
            return
        for i, endpoint in enumerate(self.line):
            if (self.screen_point(endpoint)-event.position()).manhattanLength() < 16:
                self.drag = i
                return
        self.line = [point, point.copy()]
        self.drag = 1
        self.changed.emit()
        self.update()

    def mouseMoveEvent(self, event):
        if self.drag is not None:
            self.line[self.drag] = self.normalized(event.position())
            self.changed.emit()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.drag is not None:
            self.line[self.drag] = self.normalized(event.position())
            self.drag = None
            self.changed.emit()
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#121212"))
        rect = self.image_rect()
        painter.drawPixmap(rect, self.pixmap, QRectF(self.pixmap.rect()))
        if len(self.line) != 2:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setClipRect(rect)
        p0, p1 = [self.screen_point(p) for p in self.line]
        delta = p1-p0
        length = np.hypot(delta.x(), delta.y())
        if length > .01:
            unit = delta/length
            mid = (p0+p1)/2
            reach = rect.width()+rect.height()
            painter.setPen(QPen(QColor("#ffe36e"), 1, Qt.PenStyle.DashLine))
            painter.drawLine(mid-unit*reach, mid+unit*reach)
            normal = QPointF(-unit.y(), unit.x())*self.side
            painter.setPen(QPen(QColor("#ffffff"), 1))
            for sign, text in ((1, "SOURCE"), (-1, "REFLECTED")):
                point = mid+normal*(sign*30)
                point.setX(max(rect.left()+5, min(point.x(), rect.right()-90)))
                point.setY(max(rect.top()+16, min(point.y(), rect.bottom()-5)))
                painter.drawText(point, text)
        painter.setPen(QPen(QColor("#ffe36e"), 2))
        painter.drawLine(p0, p1)
        painter.setBrush(QColor("#49cfff"))
        for point in (p0, p1):
            painter.drawEllipse(point, 6, 6)


class ReflectionDialog(QDialog):
    def __init__(self, parent, image, recipe):
        super().__init__(parent)
        self.setWindowTitle("Reflection Under a Line")
        self.resize(1000, 760)
        self.recipe = copy.deepcopy(recipe)
        h, w = image.shape[:2]
        if max(h, w) > 1000:
            image = cv2.resize(image, (max(2, round(w*1000/max(h, w))),
                                       max(2, round(h*1000/max(h, w)))), interpolation=cv2.INTER_AREA)
        self.source = image
        layout = QVBoxLayout(self)
        self.status = QLabel("Drag a line on the image. Drag blue endpoints to refine; drag elsewhere to redraw.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.canvas = ReflectionCanvas(image, recipe.line_reflection_points, recipe.line_reflection_side)
        layout.addWidget(self.canvas, 1)
        row = QHBoxLayout()
        self.pick = QPushButton("Click source side")
        self.pick.setCheckable(True)
        self.pick.setToolTip("Enable, then click the half of the image to keep as the source. Disable to edit the line.")
        self.pick.toggled.connect(self._pick_mode)
        row.addWidget(self.pick)
        self.swap = QPushButton("Swap source / reflected")
        self.swap.clicked.connect(self._swap)
        row.addWidget(self.swap)
        self.original = QCheckBox("Show original")
        self.original.toggled.connect(self.refresh)
        row.addWidget(self.original)
        layout.addLayout(row)
        form = QFormLayout()
        self.opacity = QDoubleSpinBox()
        self.opacity.setRange(0, 100)
        self.opacity.setSuffix(" %")
        self.opacity.setValue(recipe.line_reflection_opacity)
        self.opacity.setToolTip("Blend the reflection over the destination; source pixels stay unchanged.")
        form.addRow("Opacity", self.opacity)
        self.feather = QDoubleSpinBox()
        self.feather.setRange(0, 20)
        self.feather.setDecimals(1)
        self.feather.setSuffix(" % of shorter side")
        self.feather.setValue(recipe.line_reflection_feather)
        self.feather.setToolTip("Soften the seam into the reflected side; scales with export resolution.")
        form.addRow("Seam feather", self.feather)
        layout.addLayout(form)
        note = QLabel("The dashed line extends across the canvas. Canvas size stays fixed; reflections outside the source image leave destination pixels unchanged. Apply stores an editable recipe; Cancel / Esc discards changes.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Apply | QDialogButtonBox.StandardButton.Cancel)
        self.apply_button = self.buttons.button(QDialogButtonBox.StandardButton.Apply)
        self.apply_button.clicked.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(35)
        self.timer.timeout.connect(self.refresh)
        self.canvas.changed.connect(self._changed)
        self.opacity.valueChanged.connect(self._changed)
        self.feather.valueChanged.connect(self._changed)
        self.refresh()

    def _pick_mode(self, enabled):
        self.canvas.pick_source = enabled
        self.status.setText("Click the side to KEEP as the source. Disable ‘Click source side’ to edit endpoints."
                            if enabled else "Drag blue endpoints to refine, or drag elsewhere to redraw the line.")

    def _swap(self):
        self.canvas.side *= -1
        self._changed()

    def _changed(self):
        self.apply_button.setEnabled(line_frame(self.canvas.line, self.source.shape[1], self.source.shape[0]) is not None)
        # Throttle rather than debounce: continuous dragging still gets previews.
        if not self.timer.isActive():
            self.timer.start()

    def refresh(self):
        self.timer.stop()
        valid = line_frame(self.canvas.line, self.source.shape[1], self.source.shape[0]) is not None
        self.apply_button.setEnabled(valid)
        self.pick.setEnabled(valid)
        self.swap.setEnabled(valid)
        shown = self.source if self.original.isChecked() else reflect_under_line(
            self.source, self.canvas.line, self.canvas.side, self.opacity.value(), self.feather.value())
        self.canvas.pixmap = cv_to_qpixmap(shown)
        self.canvas.update()

    def accept(self):
        if line_frame(self.canvas.line, self.source.shape[1], self.source.shape[0]) is None:
            return
        self.recipe.line_reflection_points = copy.deepcopy(self.canvas.line)
        self.recipe.line_reflection_side = self.canvas.side
        self.recipe.line_reflection_opacity = self.opacity.value()
        self.recipe.line_reflection_feather = self.feather.value()
        self.timer.stop()
        super().accept()

    def reject(self):
        self.timer.stop()
        super().reject()
