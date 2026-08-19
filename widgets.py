"""widgets.py — small reusable UI pieces: the RGB histogram, the labeled
slider+spinbox row used throughout the adjustment panel, and the preview
canvas that supports an interactive crop rubber-band."""

import cv2
from PyQt6.QtCore import Qt, QRect, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen
from PyQt6.QtWidgets import QWidget, QLabel, QSlider, QGridLayout, QDoubleSpinBox


# ----------------------------------------------------------------------
# Histogram widget
# ----------------------------------------------------------------------

class HistogramWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self.hist = None  # (b, g, r) arrays of length 256

    def set_image(self, img_bgr):
        small = cv2.resize(img_bgr, (256, 256)) if img_bgr.shape[0] > 256 else img_bgr
        chans = cv2.split(small)
        hist = []
        for c in chans:
            h = cv2.calcHist([c], [0], None, [256], [0, 256]).flatten()
            h = h / (h.max() + 1e-6)
            hist.append(h)
        self.hist = hist
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))
        if self.hist is None:
            painter.end()
            return
        w, h = self.width(), self.height()
        colors = [QColor(80, 140, 255, 160), QColor(80, 220, 120, 160), QColor(230, 80, 80, 160)]
        # hist order from cv2.split of BGR is B, G, R
        for data, color in zip(self.hist, colors):
            painter.setPen(QPen(color, 1))
            path_pts = []
            for i, v in enumerate(data):
                x = int(i / 255 * w)
                y = int(h - v * (h - 4) - 2)
                path_pts.append((x, y))
            for i in range(len(path_pts) - 1):
                painter.drawLine(path_pts[i][0], path_pts[i][1], path_pts[i + 1][0], path_pts[i + 1][1])
        painter.end()


# ----------------------------------------------------------------------
# Adjustment slider row
# ----------------------------------------------------------------------

class SliderRow(QWidget):
    def __init__(self, label, lo, hi, value, step, on_change, decimals=0):
        super().__init__()
        self.decimals = decimals
        self.scale = 10 ** decimals
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setColumnStretch(0, 1)

        name_label = QLabel(label)
        name_label.setStyleSheet("color: #ddd;")
        layout.addWidget(name_label, 0, 0)

        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setValue(value)
        self.spin.setFixedWidth(70)
        self.spin.setStyleSheet("background:#2b2b2b; color:#eee; border:1px solid #444;")
        layout.addWidget(self.spin, 0, 1)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(lo * self.scale))
        self.slider.setMaximum(int(hi * self.scale))
        self.slider.setValue(int(value * self.scale))
        layout.addWidget(self.slider, 1, 0, 1, 2)

        self._syncing = False
        self.slider.valueChanged.connect(self._slider_moved)
        self.spin.valueChanged.connect(self._spin_moved)
        self.on_change = on_change

    def _slider_moved(self, v):
        if self._syncing:
            return
        self._syncing = True
        self.spin.setValue(v / self.scale)
        self._syncing = False
        self.on_change(v / self.scale)

    def _spin_moved(self, v):
        if self._syncing:
            return
        self._syncing = True
        self.slider.setValue(int(v * self.scale))
        self._syncing = False
        self.on_change(v)

    def set_value(self, v):
        self._syncing = True
        self.spin.setValue(v)
        self.slider.setValue(int(v * self.scale))
        self._syncing = False


# ----------------------------------------------------------------------
# Preview canvas with an interactive crop rubber-band
# ----------------------------------------------------------------------

class ImageCanvas(QLabel):
    """A QLabel that shows the preview pixmap and, when crop_mode is on,
    lets the user drag out a crop rectangle directly on the image."""
    crop_dragged = pyqtSignal(QRect)

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.crop_mode = False
        self._drag_start = None
        self._drag_rect = None

    def set_crop_mode(self, enabled: bool):
        self.crop_mode = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self._drag_start = None
        self._drag_rect = None
        self.update()

    def mousePressEvent(self, e):
        if self.crop_mode and self.pixmap() is not None:
            self._drag_start = e.position().toPoint()
            self._drag_rect = QRect(self._drag_start, self._drag_start)
            self.update()
        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        if self.crop_mode and self._drag_start is not None:
            self._drag_rect = QRect(self._drag_start, e.position().toPoint()).normalized()
            self.update()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self.crop_mode and self._drag_rect is not None:
            rect = self._drag_rect
            self._drag_start = None
            self._drag_rect = None
            self.update()
            if rect.width() > 5 and rect.height() > 5:
                self.crop_dragged.emit(rect)
        super().mouseReleaseEvent(e)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._drag_rect is not None and self.pixmap() is not None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor(0, 0, 0, 120))
            painter.drawPixmap(self._drag_rect, self.pixmap(), self._drag_rect)
            painter.setPen(QPen(QColor(255, 255, 255), 1, Qt.PenStyle.DashLine))
            painter.drawRect(self._drag_rect)
            painter.end()
