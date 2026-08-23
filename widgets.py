"""widgets.py — histogram, slider rows, tone curve, image canvas with zoom/pan/compare/crop."""

from __future__ import annotations

import math
from typing import Optional
import cv2
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPixmap, QWheelEvent, QMouseEvent, QPainterPath
from PyQt6.QtWidgets import (
    QWidget, QLabel, QSlider, QGridLayout, QDoubleSpinBox, QSizePolicy,
    QVBoxLayout, QHBoxLayout, QPushButton, QButtonGroup, QStackedWidget,
)


class HistogramWidget(QWidget):
    """RGB histogram with R/G/B/L channel toggles (DxO-style)."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(140)
        self.setMinimumWidth(180)
        self.hist = None  # dict r,g,b,l arrays
        self.show_r = True
        self.show_g = True
        self.show_b = True
        self.show_l = True
        # Channel toggle buttons drawn in paint / clickable regions
        self._btn_rects = {}

    def set_image(self, img_bgr):
        if img_bgr is None:
            self.hist = None
            self.update()
            return
        small = cv2.resize(img_bgr, (256, 256)) if max(img_bgr.shape[:2]) > 256 else img_bgr
        b, g, r = cv2.split(small)
        def calc(c):
            h = cv2.calcHist([c], [0], None, [256], [0, 256]).flatten()
            return h / (h.max() + 1e-6)
        # Luminance approx
        lum = (0.299 * r.astype("float32") + 0.587 * g.astype("float32") + 0.114 * b.astype("float32")).astype("uint8")
        self.hist = {"r": calc(r), "g": calc(g), "b": calc(b), "l": calc(lum)}
        self.update()

    def mousePressEvent(self, e):
        pos = e.position().toPoint()
        for key, rect in self._btn_rects.items():
            if rect.contains(pos):
                if key == "r":
                    self.show_r = not self.show_r
                elif key == "g":
                    self.show_g = not self.show_g
                elif key == "b":
                    self.show_b = not self.show_b
                elif key == "l":
                    self.show_l = not self.show_l
                self.update()
                return

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1a1a1a"))
        w, h = self.width(), self.height()
        # Channel buttons at top
        btn_y, btn_h, btn_w = 4, 16, 22
        labels = [("r", "R", QColor(230, 70, 70)), ("g", "G", QColor(70, 200, 90)),
                  ("b", "B", QColor(70, 130, 255)), ("l", "L", QColor(200, 200, 200))]
        self._btn_rects = {}
        x = 6
        for key, lab, col in labels:
            on = getattr(self, f"show_{key}")
            rect = QRect(x, btn_y, btn_w, btn_h)
            self._btn_rects[key] = rect
            painter.fillRect(rect, col if on else QColor(50, 50, 50))
            painter.setPen(QColor("#111") if on else QColor("#888"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, lab)
            x += btn_w + 4

        if self.hist is None:
            painter.setPen(QColor("#555"))
            painter.drawText(QRect(0, 24, w, h - 24), Qt.AlignmentFlag.AlignCenter, "No image")
            painter.end()
            return

        plot_top = 24
        plot_h = h - plot_top - 4
        painter.setPen(QPen(QColor(45, 45, 45), 1))
        for i in range(1, 4):
            yy = plot_top + int(i * plot_h / 4)
            painter.drawLine(0, yy, w, yy)

        channels = []
        if self.show_b:
            channels.append((self.hist["b"], QColor(70, 130, 255, 160)))
        if self.show_g:
            channels.append((self.hist["g"], QColor(70, 210, 110, 160)))
        if self.show_r:
            channels.append((self.hist["r"], QColor(230, 70, 70, 160)))
        if self.show_l:
            channels.append((self.hist["l"], QColor(220, 220, 220, 200)))

        for data, color in channels:
            painter.setPen(QPen(color, 1))
            pts = []
            for i, v in enumerate(data):
                px = int(i / 255 * (w - 1))
                py = int(plot_top + plot_h - 2 - v * (plot_h - 4))
                pts.append((px, py))
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1])
        painter.end()


class NavigatorWidget(QWidget):
    """DxO-style Move/Zoom navigator — shows full image with viewport rectangle."""
    panRequested = pyqtSignal(float, float)  # normalized center 0..1

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self.setMinimumWidth(160)
        self._pixmap = None
        self._view_rect = None  # normalized (x0,y0,x1,y1) in image space
        self._drag = False

    def set_image(self, pixmap: QPixmap | None):
        self._pixmap = pixmap
        self.update()

    def set_viewport(self, x0, y0, x1, y1):
        self._view_rect = (x0, y0, x1, y1)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#141414"))
        if self._pixmap is None or self._pixmap.isNull():
            painter.setPen(QColor("#555"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Navigator")
            painter.end()
            return
        # Fit pixmap in widget
        pw, ph = self._pixmap.width(), self._pixmap.height()
        scale = min((self.width() - 8) / max(pw, 1), (self.height() - 8) / max(ph, 1))
        dw, dh = int(pw * scale), int(ph * scale)
        ox = (self.width() - dw) // 2
        oy = (self.height() - dh) // 2
        painter.drawPixmap(QRect(ox, oy, dw, dh), self._pixmap)
        # Viewport rectangle
        if self._view_rect:
            x0, y0, x1, y1 = self._view_rect
            rx = ox + int(x0 * dw)
            ry = oy + int(y0 * dh)
            rw = max(2, int((x1 - x0) * dw))
            rh = max(2, int((y1 - y0) * dh))
            painter.setPen(QPen(QColor(255, 220, 80), 1))
            painter.setBrush(QColor(255, 220, 80, 30))
            painter.drawRect(rx, ry, rw, rh)
        painter.end()

    def mousePressEvent(self, e):
        self._drag = True
        self._emit_pan(e.position())

    def mouseMoveEvent(self, e):
        if self._drag:
            self._emit_pan(e.position())

    def mouseReleaseEvent(self, e):
        self._drag = False

    def _emit_pan(self, pos):
        if self._pixmap is None:
            return
        pw, ph = self._pixmap.width(), self._pixmap.height()
        scale = min((self.width() - 8) / max(pw, 1), (self.height() - 8) / max(ph, 1))
        dw, dh = int(pw * scale), int(ph * scale)
        ox = (self.width() - dw) // 2
        oy = (self.height() - dh) // 2
        nx = (pos.x() - ox) / max(dw, 1)
        ny = (pos.y() - oy) / max(dh, 1)
        self.panRequested.emit(max(0, min(1, nx)), max(0, min(1, ny)))


class SliderRow(QWidget):
    valueChanged = pyqtSignal(float)

    def __init__(self, label, lo, hi, value, step, on_change=None, decimals=0):
        super().__init__()
        self.decimals = decimals
        self.scale = 10 ** decimals
        self._on_change = on_change
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setColumnStretch(0, 1)
        layout.setSpacing(4)
        name_label = QLabel(label)
        name_label.setStyleSheet("color: #ccc; font-size: 12px;")
        layout.addWidget(name_label, 0, 0)
        self.spin = QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setValue(value)
        self.spin.setFixedWidth(72)
        self.spin.setStyleSheet("background:#2a2a2a; color:#eee; border:1px solid #444; border-radius:3px; padding:2px;")
        layout.addWidget(self.spin, 0, 1)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(int(lo * self.scale))
        self.slider.setMaximum(int(hi * self.scale))
        self.slider.setValue(int(value * self.scale))
        layout.addWidget(self.slider, 1, 0, 1, 2)
        self._syncing = False
        self.slider.valueChanged.connect(self._slider_moved)
        self.spin.valueChanged.connect(self._spin_moved)

    def _slider_moved(self, v):
        if self._syncing:
            return
        self._syncing = True
        val = v / self.scale
        self.spin.setValue(val)
        self._syncing = False
        self.valueChanged.emit(val)
        if self._on_change:
            self._on_change(val)

    def _spin_moved(self, v):
        if self._syncing:
            return
        self._syncing = True
        self.slider.setValue(int(v * self.scale))
        self._syncing = False
        self.valueChanged.emit(v)
        if self._on_change:
            self._on_change(v)

    def set_value(self, v):
        self._syncing = True
        self.spin.setValue(v)
        self.slider.setValue(int(round(v * self.scale)))
        self._syncing = False

    def value(self):
        return self.spin.value()


class _ToneCurveCanvas(QWidget):
    """Interactive curve graph: parametric 5-region handles (channel="param")
    plus free point curves for Luma/R/G/B (double-click to add/remove
    points, drag to move)."""
    curveChanged = pyqtSignal(float, float, float, float, float)
    pointCurveChanged = pyqtSignal(str, list)  # channel key, [[x,y],...]

    CHANNEL_COLORS = {
        "param": QColor(120, 180, 255),
        "luma": QColor(220, 220, 220),
        "r": QColor(230, 90, 90),
        "g": QColor(90, 200, 100),
        "b": QColor(90, 140, 240),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.values = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.point_curves = {
            "luma": [[0.0, 0.0], [1.0, 1.0]],
            "r": [[0.0, 0.0], [1.0, 1.0]],
            "g": [[0.0, 0.0], [1.0, 1.0]],
            "b": [[0.0, 0.0], [1.0, 1.0]],
        }
        self.channel = "param"
        self._drag_idx = None
        self.setMouseTracking(True)
        self.setToolTip(
            "Parametric: drag the 5 region handles (or use the sliders below).\n"
            "Luma/R/G/B: drag points; double-click empty area to add; "
            "double-click a point (not endpoint) to remove."
        )

    def set_channel(self, key: str):
        if key in ("param", "luma", "r", "g", "b"):
            self.channel = key
            self._drag_idx = None
            self.update()

    def set_values(self, shadows, darks, mids, lights, highlights):
        self.values = [float(shadows), float(darks), float(mids), float(lights), float(highlights)]
        self.update()

    def set_point_curve(self, key: str, points: list):
        if key not in self.point_curves:
            return
        pts = []
        for p in points or []:
            try:
                pts.append([float(p[0]), float(p[1])])
            except Exception:
                pass
        if len(pts) < 2:
            pts = [[0.0, 0.0], [1.0, 1.0]]
        pts = sorted(pts, key=lambda t: t[0])
        self.point_curves[key] = pts
        self.update()

    def reset_current(self):
        if self.channel == "param":
            self.values = [0.0, 0.0, 0.0, 0.0, 0.0]
            self.curveChanged.emit(*self.values)
        else:
            self.point_curves[self.channel] = [[0.0, 0.0], [1.0, 1.0]]
            self._emit_points()
        self.update()

    def _margin(self):
        return 12

    def _points(self):
        """Parametric (channel=='param') handle positions — kept as its own
        method name for compatibility with the slider-driven paint path."""
        w, h = self.width(), self.height()
        margin = self._margin()
        xs = [0.0, 0.25, 0.5, 0.75, 1.0]
        pts = []
        for i, xnorm in enumerate(xs):
            x = margin + xnorm * (w - 2 * margin)
            ynorm = max(0.02, min(0.98, 1.0 - (xs[i] + self.values[i] / 100.0 * 0.45)))
            y = margin + ynorm * (h - 2 * margin)
            pts.append(QPoint(int(x), int(y)))
        return pts

    def _curve_widget_pts(self, key):
        w, h = self.width(), self.height()
        margin = self._margin()
        pts = []
        for x, y in self.point_curves.get(key, [[0, 0], [1, 1]]):
            px = margin + float(x) * (w - 2 * margin)
            py = margin + (1.0 - float(y)) * (h - 2 * margin)
            pts.append(QPoint(int(px), int(py)))
        return pts

    def _emit_points(self):
        key = self.channel
        if key in self.point_curves:
            self.pointCurveChanged.emit(key, [list(p) for p in self.point_curves[key]])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        margin = self._margin()
        painter.fillRect(self.rect(), QColor("#1a1a1a"))
        painter.setPen(QPen(QColor(45, 45, 45), 1))
        for i in range(1, 4):
            x = margin + i * (w - 2 * margin) / 4
            y = margin + i * (h - 2 * margin) / 4
            painter.drawLine(int(x), margin, int(x), h - margin)
            painter.drawLine(margin, int(y), w - margin, int(y))
        painter.setPen(QPen(QColor(70, 70, 70), 1, Qt.PenStyle.DashLine))
        painter.drawLine(margin, h - margin, w - margin, margin)

        if self.channel == "param":
            pts = self._points()
            color = self.CHANNEL_COLORS["param"]
        else:
            pts = self._curve_widget_pts(self.channel)
            color = self.CHANNEL_COLORS.get(self.channel, QColor(200, 200, 200))

        painter.setPen(QPen(color, 2))
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])
        for i, p in enumerate(pts):
            painter.setBrush(QBrush(color.lighter(120) if i == self._drag_idx else color))
            painter.setPen(QPen(QColor(220, 220, 220), 1))
            painter.drawEllipse(p, 6, 6)
        painter.setPen(QColor(160, 160, 160))
        painter.drawText(margin, h - 2, self.channel.upper())
        painter.end()

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        pts = self._points() if self.channel == "param" else self._curve_widget_pts(self.channel)
        pos = e.position().toPoint()
        for i, p in enumerate(pts):
            if (pos - p).manhattanLength() < 14:
                self._drag_idx = i
                self.update()
                return

    def mouseDoubleClickEvent(self, e):
        if self.channel == "param":
            return
        pos = e.position().toPoint()
        pts_w = self._curve_widget_pts(self.channel)
        for i, p in enumerate(pts_w):
            if (pos - p).manhattanLength() < 14:
                pts = self.point_curves[self.channel]
                if 0 < i < len(pts) - 1:
                    pts.pop(i)
                    self._emit_points()
                    self.update()
                return
        w, h = self.width(), self.height()
        margin = self._margin()
        xn = max(0.0, min(1.0, (e.position().x() - margin) / max(w - 2 * margin, 1)))
        yn = max(0.0, min(1.0, 1.0 - (e.position().y() - margin) / max(h - 2 * margin, 1)))
        pts = self.point_curves[self.channel]
        pts.append([xn, yn])
        pts.sort(key=lambda t: t[0])
        self._emit_points()
        self.update()

    def mouseMoveEvent(self, e):
        if self._drag_idx is None:
            return
        h = self.height()
        w = self.width()
        margin = self._margin()
        if self.channel == "param":
            ynorm = max(0.02, min(0.98, (e.position().y() - margin) / max(h - 2 * margin, 1)))
            base = [0.0, 0.25, 0.5, 0.75, 1.0][self._drag_idx]
            val = max(-100.0, min(100.0, ((1.0 - ynorm) - base) / 0.45 * 100.0))
            self.values[self._drag_idx] = val
            self.update()
            self.curveChanged.emit(*self.values)
        else:
            pts = self.point_curves[self.channel]
            idx = self._drag_idx
            if idx < 0 or idx >= len(pts):
                return
            xn = max(0.0, min(1.0, (e.position().x() - margin) / max(w - 2 * margin, 1)))
            yn = max(0.0, min(1.0, 1.0 - (e.position().y() - margin) / max(h - 2 * margin, 1)))
            if idx == 0:
                xn = 0.0
            elif idx == len(pts) - 1:
                xn = 1.0
            else:
                lo = pts[idx - 1][0] + 0.01
                hi = pts[idx + 1][0] - 0.01
                xn = max(lo, min(hi, xn))
            pts[idx] = [xn, yn]
            self.update()
            self._emit_points()

    def mouseReleaseEvent(self, e):
        self._drag_idx = None
        self.update()


class ToneCurveWidget(QWidget):

    """Parametric tone curve (graph handles + region sliders, Lightroom-style
    Highlights/Lights/Darks/Shadows) plus free point curves for Luma/R/G/B,
    switchable via the channel buttons above the graph."""
    curveChanged = pyqtSignal(float, float, float, float, float)
    pointCurveChanged = pyqtSignal(str, list)  # channel key, [[x,y],...]

    _REGION_LABELS = (
        ("Highlights", 4),
        ("Lights", 3),
        ("Darks", 1),
        ("Shadows", 0),
    )
    _CHANNELS = (("Param", "param"), ("Luma", "luma"), ("R", "r"), ("G", "g"), ("B", "b"))

    def __init__(self):
        super().__init__()
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.values = [0.0, 0.0, 0.0, 0.0, 0.0]
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        ch_row = QHBoxLayout()
        self._channel_buttons = []
        for label, key in self._CHANNELS:
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(key == "param")
            btn.setMaximumWidth(48)
            btn.clicked.connect(lambda checked=False, k=key: self._on_channel_button(k))
            ch_row.addWidget(btn)
            self._channel_buttons.append((btn, key))
        reset_btn = QPushButton("Reset")
        reset_btn.setMaximumWidth(52)
        reset_btn.clicked.connect(lambda: self.canvas.reset_current())
        ch_row.addWidget(reset_btn)
        lay.addLayout(ch_row)

        self.canvas = _ToneCurveCanvas()
        self.canvas.curveChanged.connect(self._on_canvas)
        self.canvas.pointCurveChanged.connect(self.pointCurveChanged)
        lay.addWidget(self.canvas)

        region_lbl = QLabel("Region")
        region_lbl.setStyleSheet("color:#8af; font-size:11px; font-weight:600;")
        lay.addWidget(region_lbl)

        self._region_sliders = {}
        self._region_labels = {}
        for name, idx in self._REGION_LABELS:
            row = QHBoxLayout()
            lab = QLabel(name)
            lab.setFixedWidth(72)
            lab.setStyleSheet("color:#ccc; font-size:11px;")
            row.addWidget(lab)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(-100, 100)
            sl.setValue(0)
            sl.setToolTip(f"Parametric {name.lower()} region (−100 … +100)")
            val_lab = QLabel("0")
            val_lab.setFixedWidth(32)
            val_lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            val_lab.setStyleSheet("color:#aaa; font-size:11px;")
            sl.valueChanged.connect(lambda v, i=idx, vl=val_lab: self._on_region_slider(i, v, vl))
            row.addWidget(sl, 1)
            row.addWidget(val_lab)
            lay.addLayout(row)
            self._region_sliders[idx] = sl
            self._region_labels[idx] = val_lab

        # Mids keep graph control; optional thin slider for completeness
        row = QHBoxLayout()
        lab = QLabel("Mids")
        lab.setFixedWidth(72)
        lab.setStyleSheet("color:#888; font-size:11px;")
        row.addWidget(lab)
        sl = QSlider(Qt.Orientation.Horizontal)
        sl.setRange(-100, 100)
        sl.setValue(0)
        val_lab = QLabel("0")
        val_lab.setFixedWidth(32)
        val_lab.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        val_lab.setStyleSheet("color:#888; font-size:11px;")
        sl.valueChanged.connect(lambda v, vl=val_lab: self._on_region_slider(2, v, vl))
        row.addWidget(sl, 1)
        row.addWidget(val_lab)
        lay.addLayout(row)
        self._region_sliders[2] = sl
        self._region_labels[2] = val_lab

    def _on_canvas(self, shadows, darks, mids, lights, highlights):
        self.values = [shadows, darks, mids, lights, highlights]
        self._sync_sliders_from_values()
        self.curveChanged.emit(*self.values)

    def _on_region_slider(self, idx, value, val_lab):
        val_lab.setText(f"{int(value):+d}" if value else "0")
        self.values[idx] = float(value)
        self.canvas.set_values(*self.values)
        self.curveChanged.emit(*self.values)

    def _sync_sliders_from_values(self):
        for idx, sl in self._region_sliders.items():
            sl.blockSignals(True)
            sl.setValue(int(round(self.values[idx])))
            sl.blockSignals(False)
            lab = self._region_labels.get(idx)
            if lab is not None:
                v = int(round(self.values[idx]))
                lab.setText(f"{v:+d}" if v else "0")

    def set_values(self, shadows, darks, mids, lights, highlights):
        self.values = [float(shadows), float(darks), float(mids), float(lights), float(highlights)]
        self.canvas.set_values(*self.values)
        self._sync_sliders_from_values()

    def _on_channel_button(self, key):
        for btn, k in self._channel_buttons:
            btn.setChecked(k == key)
        self.canvas.set_channel(key)

    def set_channel(self, key: str):
        """Programmatically switch the visible channel (e.g. on image change)."""
        self._on_channel_button(key)

    def set_point_curve(self, key: str, points: list):
        self.canvas.set_point_curve(key, points)

    def reset_current(self):
        self.canvas.reset_current()


class ImageCanvas(QWidget):
    crop_dragged = pyqtSignal(QRect)
    zoom_changed = pyqtSignal(float)
    
    # Control Point signals
    controlPointSelected = pyqtSignal(int)
    controlPointMoved = pyqtSignal(int, float, float)
    controlPointResized = pyqtSignal(int, float)
    controlPointAdded = pyqtSignal(float, float)
    controlPointDragFinished = pyqtSignal()
    gradientChanged = pyqtSignal()  # any gradient geometry change
    gradientSelected = pyqtSignal(int)
    wbPicked = pyqtSignal(float, float, float)  # b,g,r 0..1 sample
    brushStrokeFinished = pyqtSignal()
    brushMaskChanged = pyqtSignal()
    horizonLineFinished = pyqtSignal(float)  # angle degrees
    
    MODE_NORMAL = 0
    MODE_SPLIT = 1
    MODE_SIDE_BY_SIDE = 2

    def __init__(self):
        super().__init__()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(200, 200)
        self._pixmap = None
        self._original_pixmap = None
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self._fit_mode = True
        self.crop_mode = False
        self._drag_start = None
        self._drag_rect = None
        self._panning = False
        self._pan_start = QPoint()
        self._space_down = False
        self.compare_mode = self.MODE_NORMAL
        self._split_ratio = 0.5
        
        # Control Point state
        self.control_points = []
        self.selected_point_index = -1
        self.local_mode = False
        self._dragging_point = False
        self._resizing_point = False
        self._drag_offset = QPoint()
        self.gradients = []
        self.selected_gradient = -1
        self.gradient_mode = False
        self.wb_picker_mode = False
        self.brush_mode = False
        self.brush_masks = []
        self.selected_brush = -1
        self.brush_radius = 0.05  # normalized
        self.brush_hardness = 0.7
        self._brush_painting = False
        self._brush_current_strokes = []
        self.show_brush_mask = True
        self.brush_erase = False
        self.show_mask_only = False
        self.show_clipping = False
        self.show_peaking = False
        self.horizon_line_mode = False
        self._horizon_line = None  # (x0,y0,x1,y1) widget coords while dragging
        self._clip_lo = 0.005
        self._clip_hi = 0.995
        self._grad_drag = None  # None | 'new' | 'p0' | 'p1' | 'line'
        self._grad_temp = None
        self.show_grid = False
        self.show_spiral = False
        # Composition guide color: 'yellow' | 'white' | 'cyan' | 'black'
        self.guide_color = "yellow"
        # Spiral placement in normalized image coords (0..1)
        self.spiral_cx = 0.5
        self.spiral_cy = 0.5
        self.spiral_scale = 0.85   # fraction of min(image side)
        self.spiral_orient = 0     # 0..7 rotations/mirrors
        self._spiral_drag = None   # None | 'move' | 'resize'
        self._spiral_drag_start = None
        self._spiral_start_vals = None
        
        self.setStyleSheet("background:#121212;")

    def set_image(self, pixmap, original=None):
        self._pixmap = pixmap
        self._original_pixmap = original
        if self._fit_mode:
            self.fit_to_view()
        else:
            self.update()
        self.zoom_changed.emit(self._scale)

    def set_crop_mode(self, enabled):
        self.crop_mode = enabled
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self._drag_start = None
        self._drag_rect = None
        self.update()

    def set_compare_mode(self, mode):
        self.compare_mode = mode
        self.update()

    def set_control_points(self, points, selected_index=-1):
        self.control_points = points if points is not None else []
        self.selected_point_index = selected_index
        self.update()

    def set_gradients(self, gradients, selected_index=-1):
        self.gradients = list(gradients) if gradients else []
        self.selected_gradient = selected_index
        self.update()

    def set_brush_masks(self, masks, selected_index=-1):
        self.brush_masks = list(masks) if masks else []
        self.selected_brush = selected_index
        self.update()

    def _erase_brush_dabs(self, pos):
        """Remove dabs near cursor from selected (or last) brush mask."""
        rect = self.image_rect()
        if rect.isEmpty() or not self.brush_masks:
            return
        ix0, iy0, sw, sh = rect.left(), rect.top(), rect.width(), rect.height()
        nx = max(0.0, min(1.0, (pos.x() - ix0) / max(sw, 1)))
        ny = max(0.0, min(1.0, (pos.y() - iy0) / max(sh, 1)))
        idx = self.selected_brush if 0 <= self.selected_brush < len(self.brush_masks) else len(self.brush_masks) - 1
        if idx < 0:
            return
        m = self.brush_masks[idx]
        rad = self.brush_radius * 1.2
        kept = []
        for s in m.get("strokes") or []:
            dx = float(s.get("x", 0)) - nx
            dy = float(s.get("y", 0)) - ny
            if (dx * dx + dy * dy) ** 0.5 > rad:
                kept.append(s)
        m["strokes"] = kept
        self.brushMaskChanged.emit()
        self.update()

    def _add_brush_dab(self, pos):
        rect = self.image_rect()
        if rect.isEmpty():
            return
        ix0, iy0, sw, sh = rect.left(), rect.top(), rect.width(), rect.height()
        nx = max(0.0, min(1.0, (pos.x() - ix0) / max(sw, 1)))
        ny = max(0.0, min(1.0, (pos.y() - iy0) / max(sh, 1)))
        self._brush_current_strokes.append({"x": nx, "y": ny, "r": self.brush_radius})
        self.update()

    def set_show_grid(self, enabled: bool):
        self.show_grid = bool(enabled)
        self.update()

    def set_guide_color(self, name: str):
        self.guide_color = (name or "yellow").lower()
        self.update()

    def _guide_colors(self):
        """Return (grid_line, grid_center, spiral_box, spiral_arc) QColors."""
        name = getattr(self, "guide_color", "yellow") or "yellow"
        if name == "white":
            return (QColor(255, 255, 255, 90), QColor(255, 255, 255, 200),
                    QColor(255, 255, 255, 70), QColor(255, 255, 255, 230))
        if name == "cyan":
            return (QColor(120, 220, 255, 90), QColor(80, 220, 255, 200),
                    QColor(100, 210, 255, 60), QColor(80, 230, 255, 230))
        if name == "black":
            return (QColor(0, 0, 0, 100), QColor(0, 0, 0, 200),
                    QColor(0, 0, 0, 80), QColor(20, 20, 20, 230))
        # yellow (default)
        return (QColor(255, 255, 255, 70), QColor(255, 220, 80, 160),
                QColor(255, 210, 100, 50), QColor(255, 185, 40, 230))

    def set_show_spiral(self, enabled: bool):
        self.show_spiral = bool(enabled)
        self.update()

    def set_show_clipping(self, enabled: bool):
        self.show_clipping = bool(enabled)
        self.update()

    def set_show_peaking(self, enabled: bool):
        self.show_peaking = bool(enabled)
        self.update()

    def set_spiral_params(self, cx=None, cy=None, scale=None, orient=None):
        if cx is not None:
            self.spiral_cx = float(max(0.0, min(1.0, cx)))
        if cy is not None:
            self.spiral_cy = float(max(0.0, min(1.0, cy)))
        if scale is not None:
            self.spiral_scale = float(max(0.15, min(1.5, scale)))
        if orient is not None:
            self.spiral_orient = int(orient) % 8
        self.update()

    def _spiral_bounds(self):
        """Return (x, y, w, h) of the golden rectangle in widget coords, or None."""
        rect = self.image_rect()
        if rect.isEmpty():
            return None
        ix0, iy0, sw, sh = float(rect.left()), float(rect.top()), float(rect.width()), float(rect.height())
        phi = 1.618033988749895
        base = min(sw, sh) * self.spiral_scale
        # Orientations 0,1,4,5 → landscape golden rect; 2,3,6,7 → portrait
        landscape = (self.spiral_orient % 4) in (0, 1)
        if landscape:
            gw, gh = base * phi, base
        else:
            gw, gh = base, base * phi
        cx = ix0 + self.spiral_cx * sw
        cy = iy0 + self.spiral_cy * sh
        x = cx - gw / 2.0
        y = cy - gh / 2.0
        return x, y, gw, gh, ix0, iy0, sw, sh

    def image_rect(self):
        if self._pixmap is None:
            return QRect()
        sw = int(self._pixmap.width() * self._scale)
        sh = int(self._pixmap.height() * self._scale)
        x = (self.width() - sw) // 2 + self._offset.x()
        y = (self.height() - sh) // 2 + self._offset.y()
        return QRect(x, y, sw, sh)

    def fit_to_view(self):
        if self._pixmap is None:
            return
        self._fit_mode = True
        pw, ph = self._pixmap.width(), self._pixmap.height()
        if pw < 1 or ph < 1:
            return
        self._scale = min(self.width() / pw, self.height() / ph) * 0.98
        self._offset = QPoint(0, 0)
        self.zoom_changed.emit(self._scale)
        self.update()

    def zoom_1_to_1(self):
        self._fit_mode = False
        self._scale = 1.0
        self._offset = QPoint(0, 0)
        self.zoom_changed.emit(self._scale)
        self.update()

    def zoom_by(self, factor, anchor=None):
        if self._pixmap is None:
            return
        self._fit_mode = False
        old = self._scale
        self._scale = max(0.05, min(20.0, self._scale * factor))
        if anchor is not None:
            img_x = (anchor.x() - self.width() / 2 - self._offset.x()) / old
            img_y = (anchor.y() - self.height() / 2 - self._offset.y()) / old
            self._offset = QPoint(
                int(anchor.x() - self.width() / 2 - img_x * self._scale),
                int(anchor.y() - self.height() / 2 - img_y * self._scale),
            )
        self.zoom_changed.emit(self._scale)
        self.update()

    def current_scale(self):
        return self._scale

    def _draw_pixmap(self, painter, pm, x_off=0):
        if pm is None:
            return
        sw, sh = int(pm.width() * self._scale), int(pm.height() * self._scale)
        x = (self.width() - sw) // 2 + self._offset.x() + x_off
        y = (self.height() - sh) // 2 + self._offset.y()
        painter.drawPixmap(QRect(x, y, sw, sh), pm)

    def _draw_exposure_overlays(self, painter):
        """Highlight/shadow clipping warning (show_clipping) + focus
        peaking (show_peaking), over whatever is currently drawn at
        image_rect(). Same logic used inline in Split compare mode;
        pulled out so normal single-image view gets these warnings too
        instead of only Split mode."""
        if self._pixmap is None or self._pixmap.isNull():
            return
        rect = self.image_rect()
        if rect.isEmpty():
            return

        if getattr(self, "show_clipping", False):
            from PyQt6.QtGui import QImage
            qimg = self._pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            w, h = qimg.width(), qimg.height()
            step = max(1, max(w, h) // 400)
            lo = int(255 * getattr(self, "_clip_lo", 0.005))
            hi = int(255 * getattr(self, "_clip_hi", 0.995))
            painter.save()
            painter.setClipRect(rect)
            sx = rect.width() / max(w, 1)
            sy = rect.height() / max(h, 1)
            for y in range(0, h, step):
                for x in range(0, w, step):
                    c = qimg.pixelColor(x, y)
                    yv = (c.red() * 3 + c.green() * 6 + c.blue()) // 10
                    if yv <= lo:
                        painter.fillRect(
                            int(rect.left() + x * sx),
                            int(rect.top() + y * sy),
                            max(1, int(step * sx) + 1),
                            max(1, int(step * sy) + 1),
                            QColor(40, 80, 255, 160),
                        )
                    elif yv >= hi:
                        painter.fillRect(
                            int(rect.left() + x * sx),
                            int(rect.top() + y * sy),
                            max(1, int(step * sx) + 1),
                            max(1, int(step * sy) + 1),
                            QColor(255, 40, 40, 160),
                        )
            painter.restore()

        if getattr(self, "show_peaking", False):
            from PyQt6.QtGui import QImage
            qimg = self._pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
            w, h = qimg.width(), qimg.height()
            step = max(2, max(w, h) // 350)
            painter.save()
            painter.setClipRect(rect)
            sx = rect.width() / max(w, 1)
            sy = rect.height() / max(h, 1)
            painter.setPen(QPen(QColor(0, 255, 90, 210), max(1, int(step * sx * 0.6))))
            for y in range(step, h - step, step):
                for x in range(step, w - step, step):
                    c0 = qimg.pixelColor(x, y)
                    c1 = qimg.pixelColor(x + step, y)
                    c2 = qimg.pixelColor(x, y + step)
                    y0 = (c0.red() + c0.green() + c0.blue()) // 3
                    y1 = (c1.red() + c1.green() + c1.blue()) // 3
                    y2 = (c2.red() + c2.green() + c2.blue()) // 3
                    if abs(y0 - y1) + abs(y0 - y2) > 40:
                        painter.drawPoint(
                            int(rect.left() + x * sx),
                            int(rect.top() + y * sy),
                        )
            painter.restore()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#121212"))
        if self._pixmap is None:
            painter.setPen(QColor("#555"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a folder to load photos")
            painter.end()
            return
        if self.compare_mode == self.MODE_SIDE_BY_SIDE and self._original_pixmap is not None:
            half = self.width() // 2
            painter.setClipRect(0, 0, half - 1, self.height())
            self._draw_pixmap(painter, self._original_pixmap, x_off=-half // 2)
            painter.setClipRect(half + 1, 0, half, self.height())
            self._draw_pixmap(painter, self._pixmap, x_off=half // 2)
            painter.setClipping(False)
            painter.setPen(QPen(QColor(255, 255, 255, 180), 2))
            painter.drawLine(half, 0, half, self.height())
            painter.setPen(QColor("#aaa"))
            painter.drawText(10, 20, "Before")
            painter.drawText(half + 10, 20, "After")
        elif self.compare_mode == self.MODE_SPLIT and self._original_pixmap is not None:
            sx = int(self.width() * self._split_ratio)
            painter.setClipRect(0, 0, sx, self.height())
            self._draw_pixmap(painter, self._original_pixmap)
            painter.setClipRect(sx, 0, self.width() - sx, self.height())
            self._draw_pixmap(painter, self._pixmap)
            if getattr(self, "show_mask_only", False) and self.brush_masks:
                painter.fillRect(self.rect(), QColor(0, 0, 0, 150))

            self._draw_exposure_overlays(painter)
            painter.setClipping(False)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
            painter.drawLine(sx, 0, sx, self.height())
        else:
            self._draw_pixmap(painter, self._pixmap)
            self._draw_exposure_overlays(painter)

        if self.show_grid and self.compare_mode == self.MODE_NORMAL:
            rect = self.image_rect()
            if not rect.isEmpty():
                ix0, iy0, sw, sh = rect.left(), rect.top(), rect.width(), rect.height()
                g_line, g_center, _, _ = self._guide_colors()
                painter.setPen(QPen(g_line, 1, Qt.PenStyle.SolidLine))
                for frac in (1 / 3, 2 / 3):
                    x = ix0 + int(sw * frac)
                    y = iy0 + int(sh * frac)
                    painter.drawLine(x, iy0, x, iy0 + sh)
                    painter.drawLine(ix0, y, ix0 + sw, y)
                painter.setPen(QPen(g_center, 1, Qt.PenStyle.DashLine))
                cx = ix0 + sw // 2
                cy = iy0 + sh // 2
                painter.drawLine(cx, iy0, cx, iy0 + sh)
                painter.drawLine(ix0, cy, ix0 + sw, cy)
                painter.setPen(QPen(g_line, 1))
                painter.drawRect(ix0, iy0, sw - 1, sh - 1)
            

        # Fibonacci / golden spiral — interactive, image-relative
        if getattr(self, "show_spiral", False) and self.compare_mode == self.MODE_NORMAL:
            bounds = self._spiral_bounds()
            if bounds is not None:
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                x, y, gw, gh, ix0, iy0, sw, sh = bounds
                phi = 1.618033988749895
                _, _, c_box, c_arc = self._guide_colors()
                pen_box = QPen(c_box, 1)
                pen_arc = QPen(c_arc, 2)
                # Starting orientation maps which corner the spiral grows from
                orient = self.spiral_orient % 4
                mirror = self.spiral_orient >= 4
                sx, sy, rw, rh = x, y, gw, gh
                for step in range(16):
                    if rw < 4 or rh < 4:
                        break
                    painter.setPen(pen_box)
                    painter.setBrush(Qt.BrushStyle.NoBrush)
                    painter.drawRect(int(round(sx)), int(round(sy)), int(round(rw)), int(round(rh)))
                    painter.setPen(pen_arc)
                    o = (orient + step) % 4
                    if mirror:
                        o = (4 - o) % 4
                    # Qt arc: 0=3 o'clock, positive = counter-clockwise, unit = 1/16 degree
                    if o == 0:  # square on left of remaining
                        side = rh
                        painter.drawArc(int(round(sx)), int(round(sy)),
                                        int(round(side * 2)), int(round(side * 2)),
                                        90 * 16, -90 * 16)
                        sx += side
                        rw -= side
                    elif o == 1:  # square on top
                        side = rw
                        painter.drawArc(int(round(sx - side)), int(round(sy)),
                                        int(round(side * 2)), int(round(side * 2)),
                                        0, -90 * 16)
                        sy += side
                        rh -= side
                    elif o == 2:  # square on right
                        side = rh
                        painter.drawArc(int(round(sx + rw - 2 * side)), int(round(sy - side)),
                                        int(round(side * 2)), int(round(side * 2)),
                                        270 * 16, -90 * 16)
                        rw -= side
                    else:  # square on bottom
                        side = rw
                        painter.drawArc(int(round(sx)), int(round(sy + rh - 2 * side)),
                                        int(round(side * 2)), int(round(side * 2)),
                                        180 * 16, -90 * 16)
                        rh -= side
                # Outer frame + move/resize handles
                _, g_center, _, _ = self._guide_colors()
                painter.setPen(QPen(g_center, 1, Qt.PenStyle.DashLine))
                painter.drawRect(int(round(x)), int(round(y)), int(round(gw)), int(round(gh)))
                # Center move handle
                cx = int(round(x + gw / 2))
                cy = int(round(y + gh / 2))
                painter.setBrush(QBrush(QColor(255, 200, 60)))
                painter.setPen(QPen(QColor(30, 30, 30), 1))
                painter.drawEllipse(cx - 6, cy - 6, 12, 12)
                # Corner resize handle (bottom-right)
                hx = int(round(x + gw))
                hy = int(round(y + gh))
                painter.setBrush(QBrush(QColor(100, 180, 255)))
                painter.drawRect(hx - 5, hy - 5, 10, 10)


        # Horizon line tool preview
        if getattr(self, "_horizon_line", None) is not None:
            x0, y0, x1, y1 = self._horizon_line
            painter.setPen(QPen(QColor(0, 220, 255, 220), 2, Qt.PenStyle.DashLine))
            painter.drawLine(int(x0), int(y0), int(x1), int(y1))
            painter.setBrush(QBrush(QColor(0, 220, 255)))
            painter.setPen(QPen(QColor(0, 100, 150), 1))
            painter.drawEllipse(int(x0) - 4, int(y0) - 4, 8, 8)
            painter.drawEllipse(int(x1) - 4, int(y1) - 4, 8, 8)

        # Graduated filters overlay
        if self.compare_mode == self.MODE_NORMAL and (self.gradients or self._grad_temp):
            rect = self.image_rect()
            if not rect.isEmpty():
                ix0, iy0, sw, sh = rect.left(), rect.top(), rect.width(), rect.height()
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                items = list(self.gradients)
                if self._grad_temp is not None:
                    items = items + [self._grad_temp]
                for idx, g in enumerate(items):
                    x0 = ix0 + float(g.get("x0", 0.5)) * sw
                    y0 = iy0 + float(g.get("y0", 0.0)) * sh
                    x1 = ix0 + float(g.get("x1", 0.5)) * sw
                    y1 = iy0 + float(g.get("y1", 1.0)) * sh
                    is_sel = (idx == self.selected_gradient and self._grad_temp is None)
                    col = QColor(80, 200, 255, 220) if is_sel else QColor(120, 180, 220, 160)
                    painter.setPen(QPen(col, 2 if is_sel else 1.5))
                    painter.drawLine(int(x0), int(y0), int(x1), int(y1))
                    # end handles
                    painter.setBrush(QBrush(QColor(255, 255, 255)))
                    painter.setPen(QPen(col, 1.5))
                    painter.drawEllipse(int(x0) - 5, int(y0) - 5, 10, 10)
                    painter.drawEllipse(int(x1) - 5, int(y1) - 5, 10, 10)
                    # midpoint
                    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
                    painter.setBrush(QBrush(col))
                    painter.drawEllipse(int(mx) - 4, int(my) - 4, 8, 8)


        # Brush mask overlay (semi-transparent red)
        if (self.compare_mode == self.MODE_NORMAL and self.show_brush_mask
                and (self.brush_masks or self._brush_current_strokes)):
            rect = self.image_rect()
            if not rect.isEmpty():
                ix0, iy0, sw, sh = rect.left(), rect.top(), rect.width(), rect.height()
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                painter.setPen(Qt.PenStyle.NoPen)
                # existing masks
                for mi, m in enumerate(self.brush_masks):
                    col = QColor(255, 60, 60, 90 if mi == self.selected_brush else 55)
                    painter.setBrush(QBrush(col))
                    for s in m.get("strokes") or []:
                        cx = ix0 + float(s.get("x", 0.5)) * sw
                        cy = iy0 + float(s.get("y", 0.5)) * sh
                        r = max(float(s.get("r", 0.05)) * max(sw, sh), 2)
                        painter.drawEllipse(QPointF(cx, cy), r, r)
                # current stroke
                if self._brush_current_strokes:
                    painter.setBrush(QBrush(QColor(255, 100, 80, 100)))
                    for s in self._brush_current_strokes:
                        cx = ix0 + float(s.get("x", 0.5)) * sw
                        cy = iy0 + float(s.get("y", 0.5)) * sh
                        r = max(float(s.get("r", self.brush_radius)) * max(sw, sh), 2)
                        painter.drawEllipse(QPointF(cx, cy), r, r)
                # cursor size ring when brush mode
                if self.brush_mode and hasattr(self, "_last_mouse"):
                    pos = self._last_mouse
                    if rect.contains(pos):
                        r = max(self.brush_radius * max(sw, sh), 2)
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.setPen(QPen(QColor(255, 255, 255, 180), 1, Qt.PenStyle.DashLine))
                        painter.drawEllipse(pos, r, r)

        # Draw interactive Control Points overlays if in normal mode
        if self.compare_mode == self.MODE_NORMAL and self.control_points:
            rect = self.image_rect()
            if not rect.isEmpty():
                ix0, iy0, sw, sh = rect.left(), rect.top(), rect.width(), rect.height()
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                for idx, pt in enumerate(self.control_points):
                    cx = ix0 + pt["x"] * sw
                    cy = iy0 + pt["y"] * sh
                    R = pt.get("radius", 0.15) * max(sw, sh)
                    is_sel = (idx == self.selected_point_index)
                    
                    if is_sel:
                        # Draw radius boundary (dashed blue)
                        painter.setPen(QPen(QColor(100, 160, 255, 180), 1.5, Qt.PenStyle.DashLine))
                        painter.setBrush(Qt.BrushStyle.NoBrush)
                        painter.drawEllipse(QPoint(int(cx), int(cy)), int(R), int(R))
                        
                        # Draw feather boundary (dotted blue/gray)
                        feather = pt.get("feather", 0.5)
                        R_feather = R * (1.0 - feather)
                        painter.setPen(QPen(QColor(100, 160, 255, 110), 1, Qt.PenStyle.DotLine))
                        painter.drawEllipse(QPoint(int(cx), int(cy)), int(R_feather), int(R_feather))
                        
                        # Draw outer resize handle at (cx + R, cy)
                        painter.setPen(QPen(QColor(255, 255, 255), 1.2))
                        painter.setBrush(QBrush(QColor(100, 160, 255)))
                        painter.drawEllipse(QPoint(int(cx + R), int(cy)), 5, 5)
                        
                    # Draw center point
                    painter.setPen(QPen(QColor(255, 255, 255), 1.5))
                    painter.setBrush(QBrush(QColor(42, 106, 212) if is_sel else QColor(140, 140, 140, 200)))
                    painter.drawEllipse(QPoint(int(cx), int(cy)), 6, 6)

        if self._drag_rect is not None:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
            painter.setPen(QPen(QColor(255, 220, 80), 1, Qt.PenStyle.DashLine))
            painter.drawRect(self._drag_rect)
        painter.end()

    def mouseDoubleClickEvent(self, e):
        """Double-click toggles fit ↔ 100% at click point."""
        if e.button() != Qt.MouseButton.LeftButton:
            return
        if abs(self._scale - 1.0) < 0.05:
            self.fit_to_view()
        else:
            self._scale = 1.0
            self.update()
            self.zoom_changed.emit(self._scale)

    def wheelEvent(self, e):
        if self._pixmap is None:
            return
        factor = 1.15 if e.angleDelta().y() > 0 else 1 / 1.15
        self.zoom_by(factor, anchor=e.position().toPoint())

    def mousePressEvent(self, e):
        # Horizon line: drag to define level
        if self.horizon_line_mode and e.button() == Qt.MouseButton.LeftButton:
            pos = e.position().toPoint()
            self._horizon_line = (pos.x(), pos.y(), pos.x(), pos.y())
            self._horizon_dragging = True
            self.update()
            return

        # 0. Spiral move / resize (when overlay is on)
        if (self.show_spiral and self.compare_mode == self.MODE_NORMAL
                and e.button() == Qt.MouseButton.LeftButton and not self.crop_mode):
            bounds = self._spiral_bounds()
            if bounds is not None:
                x, y, gw, gh, ix0, iy0, sw, sh = bounds
                pos = e.position().toPoint()
                hx, hy = x + gw, y + gh
                cx, cy = x + gw / 2, y + gh / 2
                if abs(pos.x() - hx) < 12 and abs(pos.y() - hy) < 12:
                    self._spiral_drag = "resize"
                    self._spiral_drag_start = pos
                    self._spiral_start_vals = (self.spiral_cx, self.spiral_cy, self.spiral_scale)
                    return
                if abs(pos.x() - cx) < 14 and abs(pos.y() - cy) < 14:
                    self._spiral_drag = "move"
                    self._spiral_drag_start = pos
                    self._spiral_start_vals = (self.spiral_cx, self.spiral_cy, self.spiral_scale)
                    return
                # Click inside spiral rect → move
                if x <= pos.x() <= x + gw and y <= pos.y() <= y + gh:
                    self._spiral_drag = "move"
                    self._spiral_drag_start = pos
                    self._spiral_start_vals = (self.spiral_cx, self.spiral_cy, self.spiral_scale)
                    return

        # Brush paint / erase
        if self.brush_mode and self._pixmap is not None and e.button() in (
            Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton
        ):
            rect = self.image_rect()
            if not rect.isEmpty() and rect.contains(e.position().toPoint()):
                erase = self.brush_erase or e.button() == Qt.MouseButton.RightButton
                self._brush_painting = True
                self._brush_erasing = erase
                self._brush_current_strokes = []
                if erase:
                    self._erase_brush_dabs(e.position().toPoint())
                else:
                    self._add_brush_dab(e.position().toPoint())
                return

        # WB eyedropper
        if self.wb_picker_mode and e.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            rect = self.image_rect()
            if not rect.isEmpty() and rect.contains(e.position().toPoint()):
                pos = e.position().toPoint()
                # sample from displayed pixmap
                ix0, iy0, sw, sh = rect.left(), rect.top(), rect.width(), rect.height()
                nx = (pos.x() - ix0) / max(sw, 1)
                ny = (pos.y() - iy0) / max(sh, 1)
                nx = max(0.0, min(1.0, nx))
                ny = max(0.0, min(1.0, ny))
                pm = self._original_pixmap or self._pixmap
                if pm is not None and not pm.isNull():
                    px = int(nx * (pm.width() - 1))
                    py = int(ny * (pm.height() - 1))
                    qimg = pm.toImage()
                    c = qimg.pixelColor(px, py)
                    # emit as B,G,R normalized (OpenCV order convenience)
                    self.wbPicked.emit(c.blueF(), c.greenF(), c.redF())
                return

        # Graduated filter create / edit
        if self.gradient_mode and e.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            rect = self.image_rect()
            if not rect.isEmpty():
                pos = e.position().toPoint()
                ix0, iy0, sw, sh = rect.left(), rect.top(), rect.width(), rect.height()
                # hit-test existing handles
                for idx, g in enumerate(self.gradients):
                    for key, hx, hy in (
                        ("p0", g.get("x0", 0.5), g.get("y0", 0.0)),
                        ("p1", g.get("x1", 0.5), g.get("y1", 1.0)),
                    ):
                        px = ix0 + float(hx) * sw
                        py = iy0 + float(hy) * sh
                        if abs(pos.x() - px) < 12 and abs(pos.y() - py) < 12:
                            self.selected_gradient = idx
                            self._grad_drag = key
                            self.gradientSelected.emit(idx)
                            self.update()
                            return
                # start new gradient
                nx = max(0.0, min(1.0, (pos.x() - ix0) / max(sw, 1)))
                ny = max(0.0, min(1.0, (pos.y() - iy0) / max(sh, 1)))
                self._grad_drag = "new"
                self._grad_temp = {"x0": nx, "y0": ny, "x1": nx, "y1": ny, "feather": 0.5,
                                   "exposure": 0.0, "contrast": 0.0, "saturation": 0.0, "clarity": 0.0}
                return

        # 1. Pan check (middle button or space + left click)
        if e.button() == Qt.MouseButton.MiddleButton or (e.button() == Qt.MouseButton.LeftButton and self._space_down):
            self._panning = True
            self._pan_start = e.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        # 2. Control point click check
        if e.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            rect = self.image_rect()
            if not rect.isEmpty():
                ix0, iy0, sw, sh = rect.left(), rect.top(), rect.width(), rect.height()
                click_pos = e.position().toPoint()
                
                # Check resize handle first if a point is selected
                if 0 <= self.selected_point_index < len(self.control_points):
                    pt = self.control_points[self.selected_point_index]
                    cx = ix0 + pt["x"] * sw
                    cy = iy0 + pt["y"] * sh
                    R = pt.get("radius", 0.15) * max(sw, sh)
                    hx, hy = cx + R, cy
                    if math.hypot(click_pos.x() - hx, click_pos.y() - hy) < 10:
                        self._resizing_point = True
                        return
                
                # Check center dots
                for idx, pt in enumerate(reversed(self.control_points)):
                    actual_idx = len(self.control_points) - 1 - idx
                    cx = ix0 + pt["x"] * sw
                    cy = iy0 + pt["y"] * sh
                    if math.hypot(click_pos.x() - cx, click_pos.y() - cy) < 12:
                        self.selected_point_index = actual_idx
                        self._dragging_point = True
                        self._drag_offset = click_pos - QPoint(int(cx), int(cy))
                        self.controlPointSelected.emit(actual_idx)
                        self.update()
                        return

                # Check if click is inside the image bounds for placing a new point
                if self.local_mode and rect.contains(click_pos):
                    nx = (click_pos.x() - ix0) / sw
                    ny = (click_pos.y() - iy0) / sh
                    self.controlPointAdded.emit(nx, ny)
                    return

        # 3. Crop check
        if self.crop_mode and self._pixmap is not None and e.button() == Qt.MouseButton.LeftButton:
            self._drag_start = e.position().toPoint()
            self._drag_rect = QRect(self._drag_start, self._drag_start)
            self.update()
            return
            
        # 4. Split Compare slider check
        if self.compare_mode == self.MODE_SPLIT and e.button() == Qt.MouseButton.LeftButton:
            if abs(e.position().x() - int(self.width() * self._split_ratio)) < 8:
                self._drag_start = e.position().toPoint()
                return

        super().mousePressEvent(e)

    def mouseMoveEvent(self, e):
        self._last_mouse = e.position().toPoint()
        if getattr(self, "_horizon_dragging", False) and self._horizon_line is not None:
            x0, y0, _, _ = self._horizon_line
            pos = e.position().toPoint()
            self._horizon_line = (x0, y0, pos.x(), pos.y())
            self.update()
            return
        if self._brush_painting and self.brush_mode:
            if getattr(self, "_brush_erasing", False):
                self._erase_brush_dabs(e.position().toPoint())
            else:
                self._add_brush_dab(e.position().toPoint())
            return
        if self._grad_drag and self._pixmap is not None:
            rect = self.image_rect()
            if not rect.isEmpty():
                pos = e.position().toPoint()
                ix0, iy0, sw, sh = rect.left(), rect.top(), rect.width(), rect.height()
                nx = max(0.0, min(1.0, (pos.x() - ix0) / max(sw, 1)))
                ny = max(0.0, min(1.0, (pos.y() - iy0) / max(sh, 1)))
                if self._grad_drag == "new" and self._grad_temp is not None:
                    self._grad_temp["x1"] = nx
                    self._grad_temp["y1"] = ny
                    self.update()
                    return
                if self._grad_drag in ("p0", "p1") and 0 <= self.selected_gradient < len(self.gradients):
                    g = self.gradients[self.selected_gradient]
                    if self._grad_drag == "p0":
                        g["x0"], g["y0"] = nx, ny
                    else:
                        g["x1"], g["y1"] = nx, ny
                    self.gradientChanged.emit()
                    self.update()
                    return

        # Spiral drag
        if self._spiral_drag and self._spiral_drag_start is not None:
            bounds = self._spiral_bounds()
            if bounds is not None:
                x, y, gw, gh, ix0, iy0, sw, sh = bounds
                pos = e.position().toPoint()
                dx = pos.x() - self._spiral_drag_start.x()
                dy = pos.y() - self._spiral_drag_start.y()
                scx, scy, sscale = self._spiral_start_vals
                if self._spiral_drag == "move" and sw > 1 and sh > 1:
                    self.spiral_cx = max(0.0, min(1.0, scx + dx / sw))
                    self.spiral_cy = max(0.0, min(1.0, scy + dy / sh))
                    self.update()
                    return
                if self._spiral_drag == "resize" and sw > 1 and sh > 1:
                    # Drag bottom-right corner: change scale from center
                    phi = 1.618033988749895
                    landscape = (self.spiral_orient % 4) in (0, 1)
                    # distance from center to mouse
                    cx = ix0 + self.spiral_cx * sw
                    cy = iy0 + self.spiral_cy * sh
                    dist = max(abs(pos.x() - cx), abs(pos.y() - cy))
                    base = min(sw, sh)
                    if landscape:
                        # half-width = base*scale*phi/2
                        new_scale = (2.0 * dist) / (base * phi)
                    else:
                        new_scale = (2.0 * dist) / (base * phi)
                    self.spiral_scale = max(0.15, min(1.5, new_scale))
                    self.update()
                    return

        # Dragging control point center
        if self._dragging_point:
            rect = self.image_rect()
            if not rect.isEmpty():
                ix0, iy0, sw, sh = rect.left(), rect.top(), rect.width(), rect.height()
                click_pos = e.position().toPoint()
                new_cx = click_pos.x() - self._drag_offset.x()
                new_cy = click_pos.y() - self._drag_offset.y()
                nx = max(0.0, min(1.0, (new_cx - ix0) / sw))
                ny = max(0.0, min(1.0, (new_cy - iy0) / sh))
                self.controlPointMoved.emit(self.selected_point_index, nx, ny)
                self.update()
            return

        # Resizing control point radius
        if self._resizing_point:
            rect = self.image_rect()
            if not rect.isEmpty():
                ix0, iy0, sw, sh = rect.left(), rect.top(), rect.width(), rect.height()
                pt = self.control_points[self.selected_point_index]
                cx = ix0 + pt["x"] * sw
                cy = iy0 + pt["y"] * sh
                click_pos = e.position().toPoint()
                dist_pixels = math.hypot(click_pos.x() - cx, click_pos.y() - cy)
                new_radius = max(0.01, min(1.0, dist_pixels / max(sw, sh)))
                self.controlPointResized.emit(self.selected_point_index, new_radius)
                self.update()
            return

        # Crop, Panning, Split drag handling
        if self.crop_mode and self._drag_start is not None:
            self._drag_rect = QRect(self._drag_start, e.position().toPoint()).normalized()
            self.update()
            return
        if self._panning:
            delta = e.position().toPoint() - self._pan_start
            self._offset += delta
            self._pan_start = e.position().toPoint()
            self._fit_mode = False
            self.update()
            return
        if self.compare_mode == self.MODE_SPLIT and self._drag_start is not None:
            self._split_ratio = max(0.05, min(0.95, e.position().x() / max(self.width(), 1)))
            self.update()
            return
        if self.compare_mode == self.MODE_SPLIT:
            sx = int(self.width() * self._split_ratio)
            self.setCursor(Qt.CursorShape.SizeHorCursor if abs(e.position().x() - sx) < 8 else Qt.CursorShape.ArrowCursor)
            
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if getattr(self, "_horizon_dragging", False):
            self._horizon_dragging = False
            if self._horizon_line is not None:
                x0, y0, x1, y1 = self._horizon_line
                dx, dy = x1 - x0, y1 - y0
                if abs(dx) + abs(dy) > 8:
                    import math
                    # Angle of line from horizontal; negate so drawing a tilted horizon levels the image
                    angle = math.degrees(math.atan2(dy, dx))
                    # Normalize to about -45..45 for typical horizon use
                    while angle > 45:
                        angle -= 90
                    while angle < -45:
                        angle += 90
                    self.horizonLineFinished.emit(-angle)
            self._horizon_line = None
            self.horizon_line_mode = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
            return
        if self._brush_painting:
            self._brush_painting = False
            if getattr(self, "_brush_erasing", False):
                self._brush_erasing = False
                self._brush_current_strokes = []
                self.brushStrokeFinished.emit()  # persist erase
                self.update()
                return
            if self._brush_current_strokes:
                mask = {
                    "strokes": list(self._brush_current_strokes),
                    "hardness": self.brush_hardness,
                    "exposure": 0.0, "contrast": 0.0, "saturation": 0.0,
                    "clarity": 0.0, "temperature": 0.0,
                }
                self.brush_masks = list(self.brush_masks) + [mask]
                self.selected_brush = len(self.brush_masks) - 1
                self._brush_current_strokes = []
                self.brushStrokeFinished.emit()
                self.brushMaskChanged.emit()
            self.update()
            return
        if self._grad_drag:
            if self._grad_drag == "new" and self._grad_temp is not None:
                g = self._grad_temp
                # ignore tiny drags
                if abs(g["x1"] - g["x0"]) + abs(g["y1"] - g["y0"]) > 0.02:
                    self.gradients = list(self.gradients) + [dict(g)]
                    self.selected_gradient = len(self.gradients) - 1
                    self.gradientChanged.emit()
                    self.gradientSelected.emit(self.selected_gradient)
                self._grad_temp = None
            self._grad_drag = None
            self.update()
            return
        if self._spiral_drag:
            self._spiral_drag = None
            self._spiral_drag_start = None
            self._spiral_start_vals = None
            return
        if self._dragging_point or self._resizing_point:
            self._dragging_point = False
            self._resizing_point = False
            self.controlPointDragFinished.emit()
            return
        if self.crop_mode and self._drag_rect is not None:
            rect = self._drag_rect
            self._drag_start = None
            self._drag_rect = None
            self.update()
            if rect.width() > 8 and rect.height() > 8:
                self.crop_dragged.emit(rect)
            return
        if self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            return
        self._drag_start = None
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self._space_down = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        if e.key() == Qt.Key.Key_Space:
            self._space_down = False
            if not self._panning:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        super().keyReleaseEvent(e)

    def resizeEvent(self, e):
        if self._fit_mode:
            self.fit_to_view()
        super().resizeEvent(e)


# ----------------------------------------------------------------------
# Lightroom-style HSL panel: Hue / Saturation / Luminance / All tabs, each
# showing all 8 color-band rows (Red..Magenta) at once.
# ---------------------------------------------------------------------------

class HSLRow(QWidget):
    """One color-band row: colored label, slider (-100..100), numeric value.
    Double-click the slider to reset that band to 0."""
    valueChanged = pyqtSignal(int, float)  # color_index, value

    NAMES = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
    COLORS = [
        QColor(220, 70, 70), QColor(224, 140, 50), QColor(210, 190, 50),
        QColor(90, 185, 90), QColor(60, 190, 190), QColor(90, 130, 224),
        QColor(150, 100, 214), QColor(214, 90, 160),
    ]

    def __init__(self, color_index: int, value: float = 0.0):
        super().__init__()
        self.color_index = color_index
        color = self.COLORS[color_index]
        c = f"rgb({color.red()},{color.green()},{color.blue()})"

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 1, 0, 1)
        layout.setSpacing(6)

        name_lbl = QLabel(self.NAMES[color_index])
        name_lbl.setFixedWidth(56)
        name_lbl.setStyleSheet(f"color: {c}; font-size: 11px;")
        layout.addWidget(name_lbl)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(-100, 100)
        self.slider.setValue(int(round(value)))
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                height: 3px; background: #3a3a3a; border-radius: 1px;
            }}
            QSlider::handle:horizontal {{
                background: {c}; width: 12px; height: 12px; margin: -5px 0;
                border-radius: 6px; border: 1px solid #1a1a1a;
            }}
            QSlider::sub-page:horizontal {{ background: {c}; border-radius: 1px; }}
            QSlider::add-page:horizontal {{ background: #3a3a3a; border-radius: 1px; }}
        """)
        self.slider.valueChanged.connect(self._on_slide)
        self.slider.mouseDoubleClickEvent = self._reset
        layout.addWidget(self.slider, 1)

        self.value_lbl = QLabel(self._fmt(value))
        self.value_lbl.setFixedWidth(30)
        self.value_lbl.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.value_lbl.setStyleSheet("color:#ccc; font-size: 11px;")
        layout.addWidget(self.value_lbl)

    @staticmethod
    def _fmt(v) -> str:
        v = int(round(v))
        return f"+{v}" if v > 0 else str(v)

    def _on_slide(self, val):
        self.value_lbl.setText(self._fmt(val))
        self.valueChanged.emit(self.color_index, float(val))

    def _reset(self, event):
        self.slider.setValue(0)

    def set_value(self, value):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round(value)))
        self.slider.blockSignals(False)
        self.value_lbl.setText(self._fmt(value))


class HSLGroup(QWidget):
    """8 stacked HSLRows (Red..Magenta) for one channel type."""
    valueChanged = pyqtSignal(int, float)  # color_index, value

    def __init__(self, title: Optional[str] = None):
        super().__init__()
        v = QVBoxLayout(self)
        v.setContentsMargins(4, 4, 4, 6)
        v.setSpacing(3)
        if title:
            head = QLabel(title)
            head.setAlignment(Qt.AlignmentFlag.AlignCenter)
            head.setStyleSheet(
                "color:#ddd; font-size:12px; font-weight:600; padding-bottom:2px;"
            )
            v.addWidget(head)
        self.rows = []
        for i in range(8):
            row = HSLRow(i, 0.0)
            row.valueChanged.connect(self.valueChanged)
            v.addWidget(row)
            self.rows.append(row)

    def set_values(self, values):
        for i, row in enumerate(self.rows):
            row.set_value(values[i] if i < len(values) else 0.0)


class HSLPanelWidget(QWidget):
    """Lightroom-style HSL / Color panel: Hue | Saturation | Luminance | All
    tabs, each showing all 8 color-band rows. Matches the classic ACR/LR HSL
    panel layout."""
    hueChanged = pyqtSignal(int, float)  # color_index, value
    satChanged = pyqtSignal(int, float)
    lumChanged = pyqtSignal(int, float)

    def __init__(self):
        super().__init__()
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(4)

        tab_row = QHBoxLayout()
        tab_row.setSpacing(2)
        self._btn_group = QButtonGroup(self)
        self._btn_group.setExclusive(True)
        for i, name in enumerate(["Hue", "Saturation", "Luminance", "All"]):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet("""
                QPushButton {
                    background: transparent; color:#999; border:none;
                    border-bottom: 2px solid transparent; padding: 5px 8px; font-size: 11px;
                }
                QPushButton:checked { color:#fff; border-bottom: 2px solid #2a6ad4; font-weight:600; }
                QPushButton:hover:!checked { color:#ccc; }
            """)
            self._btn_group.addButton(btn, i)
            tab_row.addWidget(btn)
            if i == 0:
                btn.setChecked(True)
        tab_row.addStretch(1)
        outer.addLayout(tab_row)

        self.stack = QStackedWidget()
        outer.addWidget(self.stack)

        self.hue_group = HSLGroup()
        self.sat_group = HSLGroup()
        self.lum_group = HSLGroup()
        self.hue_group.valueChanged.connect(self.hueChanged)
        self.sat_group.valueChanged.connect(self.satChanged)
        self.lum_group.valueChanged.connect(self.lumChanged)
        self.stack.addWidget(self.hue_group)
        self.stack.addWidget(self.sat_group)
        self.stack.addWidget(self.lum_group)

        all_page = QWidget()
        all_v = QVBoxLayout(all_page)
        all_v.setContentsMargins(0, 0, 0, 0)
        all_v.setSpacing(10)
        self.hue_group_all = HSLGroup("Hue")
        self.sat_group_all = HSLGroup("Saturation")
        self.lum_group_all = HSLGroup("Luminance")
        self.hue_group_all.valueChanged.connect(self.hueChanged)
        self.sat_group_all.valueChanged.connect(self.satChanged)
        self.lum_group_all.valueChanged.connect(self.lumChanged)
        all_v.addWidget(self.hue_group_all)
        all_v.addWidget(self.sat_group_all)
        all_v.addWidget(self.lum_group_all)
        self.stack.addWidget(all_page)

        self._btn_group.idClicked.connect(self.stack.setCurrentIndex)

    def set_values(self, hue, sat, lum):
        self.hue_group.set_values(hue)
        self.sat_group.set_values(sat)
        self.lum_group.set_values(lum)
        self.hue_group_all.set_values(hue)
        self.sat_group_all.set_values(sat)
        self.lum_group_all.set_values(lum)


# ----------------------------------------------------------------------
# HSL Color Wheel
# ----------------------------------------------------------------------

class ColorWheelWidget(QWidget):
    """DxO-style HSL color wheel. Click a sector to select channel; ring shows active."""
    channelChanged = pyqtSignal(int)  # 0..7

    CHANNELS = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
    COLORS = [
        QColor(220, 60, 60), QColor(230, 140, 40), QColor(230, 210, 40),
        QColor(60, 180, 60), QColor(40, 200, 200), QColor(50, 100, 220),
        QColor(140, 60, 200), QColor(200, 50, 150),
    ]

    def __init__(self):
        super().__init__()
        self.setMinimumSize(180, 180)
        self.setMaximumHeight(200)
        self.active = 0
        self.setMouseTracking(True)

    def set_channel(self, idx: int):
        self.active = max(0, min(7, idx))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        side = min(self.width(), self.height()) - 10
        cx, cy = self.width() / 2, self.height() / 2
        outer = side / 2
        inner = outer * 0.45

        for i in range(8):
            a0 = math.radians(i * 45 - 90)
            a1 = math.radians((i + 1) * 45 - 90)
            path = QPainterPath()
            path.moveTo(cx + inner * math.cos(a0), cy + inner * math.sin(a0))
            path.arcTo(cx - outer, cy - outer, outer * 2, outer * 2,
                       -i * 45 + 90, -45)
            path.arcTo(cx - inner, cy - inner, inner * 2, inner * 2,
                       -i * 45 + 90 - 45, 45)
            path.closeSubpath()
            color = self.COLORS[i]
            if i == self.active:
                color = QColor(color.red(), color.green(), color.blue(), 255)
                painter.setPen(QPen(QColor(255, 255, 255), 2))
            else:
                color = QColor(color.red(), color.green(), color.blue(), 180)
                painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(color))
            painter.drawPath(path)

        # Center disc
        painter.setBrush(QBrush(QColor("#1a1a1a")))
        painter.setPen(QPen(QColor("#444"), 1))
        painter.drawEllipse(int(cx - inner + 4), int(cy - inner + 4),
                            int((inner - 4) * 2), int((inner - 4) * 2))
        painter.setPen(QColor("#aaa"))
        painter.drawText(QRect(int(cx - inner), int(cy - 10), int(inner * 2), 20),
                         Qt.AlignmentFlag.AlignCenter, self.CHANNELS[self.active])
        painter.end()

    def mousePressEvent(self, e):
        cx, cy = self.width() / 2, self.height() / 2
        dx, dy = e.position().x() - cx, e.position().y() - cy
        dist = math.hypot(dx, dy)
        side = min(self.width(), self.height()) - 10
        outer, inner = side / 2, side / 2 * 0.45
        if inner * 0.5 < dist < outer:
            angle = (math.degrees(math.atan2(dy, dx)) + 90) % 360
            idx = int(angle // 45) % 8
            self.active = idx
            self.update()
            self.channelChanged.emit(idx)


# ----------------------------------------------------------------------
# History list
# ----------------------------------------------------------------------

class HistoryWidget(QWidget):
    """Simple edit history — click an entry to restore that recipe snapshot."""
    restoreRequested = pyqtSignal(int)  # index

    def __init__(self):
        super().__init__()
        from PyQt6.QtWidgets import QListWidget, QVBoxLayout, QLabel
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.list = QListWidget()
        self.list.setStyleSheet(
            "QListWidget { background:#161616; border:1px solid #333; border-radius:3px; "
            "color:#ccc; font-size:12px; letter-spacing:0px; padding:2px; }"
            "QListWidget::item { padding:4px 6px; }"
            "QListWidget::item:selected { background:#2a5080; }"
        )
        self.list.itemClicked.connect(self._on_click)
        layout.addWidget(self.list)
        self._entries = []  # list of (label, recipe_dict)

    def clear(self):
        self._entries.clear()
        self.list.clear()

    def push(self, label: str, recipe_dict: dict):
        # Truncate any redo branch when pushing a new state
        cur = self.list.currentRow()
        if cur >= 0 and cur < len(self._entries) - 1:
            self._entries = self._entries[: cur + 1]
            while self.list.count() > len(self._entries):
                self.list.takeItem(self.list.count() - 1)
        self._entries.append((label, recipe_dict))
        self.list.addItem(f"{len(self._entries)}. {label}")
        self.list.setCurrentRow(len(self._entries) - 1)
        if len(self._entries) > 50:
            self._entries.pop(0)
            self.list.takeItem(0)

    def _on_click(self, item):
        row = self.list.row(item)
        if 0 <= row < len(self._entries):
            self.restoreRequested.emit(row)

    def get_recipe_dict(self, index: int):
        if 0 <= index < len(self._entries):
            return self._entries[index][1]
        return None

    def current_index(self) -> int:
        return self.list.currentRow()

    def can_undo(self) -> bool:
        return self.list.currentRow() > 0

    def can_redo(self) -> bool:
        row = self.list.currentRow()
        return 0 <= row < len(self._entries) - 1

    def undo_index(self):
        row = self.list.currentRow()
        if row > 0:
            return row - 1
        return None

    def redo_index(self):
        row = self.list.currentRow()
        if 0 <= row < len(self._entries) - 1:
            return row + 1
        return None
