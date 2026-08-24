"""widgets.py — histogram, slider rows, tone curve, image canvas with zoom/pan/compare/crop."""

from __future__ import annotations

import math
import uuid
import cv2
import numpy as np
from PyQt6.QtCore import Qt, QRect, QPoint, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QPixmap, QImage, QWheelEvent, QMouseEvent, QPainterPath, QRadialGradient
from PyQt6.QtWidgets import QWidget, QLabel, QSlider, QGridLayout, QDoubleSpinBox, QSizePolicy


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
        # A fixed 72 px field clipped common values such as 5500 K and could
        # paint the text beneath the step arrows.  Keep enough room for the
        # editable value and the arrow-button gutter while still allowing a
        # layout to make the field wider when space is available.
        self.spin.setMinimumWidth(104)
        self.spin.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.spin.setAlignment(Qt.AlignmentFlag.AlignRight)
        # Windows can arrange styled spin buttons side-by-side and otherwise
        # leave the editor beneath the up button.  Define a stacked button
        # gutter explicitly so the value can never cover either arrow.
        self.spin.setStyleSheet("""
            QDoubleSpinBox {
                background: #2a2a2a;
                color: #eee;
                border: 1px solid #444;
                border-radius: 3px;
                padding: 2px 26px 2px 4px;
            }
            QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 22px;
                border-left: 1px solid #444;
                border-bottom: 1px solid #383838;
                border-top-right-radius: 3px;
                background: #303030;
            }
            QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 22px;
                border-left: 1px solid #444;
                border-bottom-right-radius: 3px;
                background: #303030;
            }
            QDoubleSpinBox::up-button:hover,
            QDoubleSpinBox::down-button:hover { background: #414141; }
        """)
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


class ToneCurveWidget(QWidget):
    curveChanged = pyqtSignal(float, float, float, float, float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(160)
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.values = [0.0, 0.0, 0.0, 0.0, 0.0]
        # Point curves are synchronized whenever an image/recipe is loaded.
        # Keep identity curves available even though the current compact widget
        # displays the five-region parametric curve by default.
        identity = [[0.0, 0.0], [1.0, 1.0]]
        self.point_curves = {
            "luma": [p[:] for p in identity],
            "r": [p[:] for p in identity],
            "g": [p[:] for p in identity],
            "b": [p[:] for p in identity],
        }
        self.channel = "param"
        self._drag_idx = None
        self.setMouseTracking(True)

    def set_values(self, shadows, darks, mids, lights, highlights):
        """Update region handles from sliders (shadows…highlights in -100..100)."""
        self.values = [
            float(shadows or 0.0),
            float(darks or 0.0),
            float(mids or 0.0),
            float(lights or 0.0),
            float(highlights or 0.0),
        ]
        self.update()
        self.repaint()

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
        """Five control points: Shadows, Darks, Midtones, Lights, Highlights."""
        w, h = self.width(), self.height()
        margin = 12
        xs = [0.0, 0.25, 0.5, 0.75, 1.0]
        pts = []
        # values[i] is vertical offset from the diagonal, -100..100 → stronger visual (0.55)
        strength = 0.55
        for i, xnorm in enumerate(xs):
            x = margin + xnorm * (w - 2 * margin)
            # diagonal baseline: y goes from bottom-left to top-right in image coords
            # display y increases downward
            base_y_norm = 1.0 - xnorm  # 1 at left (black), 0 at right (white)
            offset = (self.values[i] / 100.0) * strength
            ynorm = max(0.02, min(0.98, base_y_norm - offset))
            y = margin + ynorm * (h - 2 * margin)
            pts.append(QPoint(int(x), int(y)))
        return pts

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        margin = 12
        painter.fillRect(self.rect(), QColor("#1a1a1a"))
        painter.setPen(QPen(QColor(45, 45, 45), 1))
        for i in range(1, 4):
            x = margin + i * (w - 2 * margin) / 4
            y = margin + i * (h - 2 * margin) / 4
            painter.drawLine(int(x), margin, int(x), h - margin)
            painter.drawLine(margin, int(y), w - margin, int(y))
        painter.setPen(QPen(QColor(70, 70, 70), 1, Qt.PenStyle.DashLine))
        painter.drawLine(margin, h - margin, w - margin, margin)
        pts = self._points()
        painter.setPen(QPen(QColor(120, 180, 255), 2))
        for i in range(len(pts) - 1):
            painter.drawLine(pts[i], pts[i + 1])
        for i, p in enumerate(pts):
            painter.setBrush(QBrush(QColor(100, 160, 255) if i == self._drag_idx else QColor(80, 140, 230)))
            painter.setPen(QPen(QColor(220, 220, 220), 1))
            painter.drawEllipse(p, 6, 6)
        painter.end()

    def mousePressEvent(self, e):
        pts = self._points()
        pos = e.position().toPoint()
        for i, p in enumerate(pts):
            if (pos - p).manhattanLength() < 14:
                self._drag_idx = i
                self.update()
                return

    def mouseMoveEvent(self, e):
        if self._drag_idx is None:
            return
        h = self.height()
        margin = 12
        ynorm = max(0.02, min(0.98, (e.position().y() - margin) / max(h - 2 * margin, 1)))
        base = [0.0, 0.25, 0.5, 0.75, 1.0][self._drag_idx]
        val = max(-100.0, min(100.0, ((1.0 - ynorm) - base) / 0.45 * 100.0))
        self.values[self._drag_idx] = val
        self.update()
        self.curveChanged.emit(*self.values)

    def mouseReleaseEvent(self, e):
        self._drag_idx = None
        self.update()


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
    skinColorPicked = pyqtSignal(float, float, float)  # b,g,r 0..1 sample
    brushStrokeFinished = pyqtSignal()
    brushMaskChanged = pyqtSignal()
    horizonLineFinished = pyqtSignal(float)  # angle degrees
    keystoneChanged = pyqtSignal(list)
    keystoneFinished = pyqtSignal(list)
    
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
        self._comparison_pixmap = None
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
        self.hold_original = False
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
        self.skin_picker_mode = False
        self.sharpen_proof = False
        self.sharpen_proof_label = ""
        self.brush_mode = False
        self.brush_masks = []
        self.selected_brush = -1
        self.brush_radius = 0.05  # normalized
        self.brush_hardness = 0.7
        self._brush_painting = False
        self._brush_current_strokes = []
        self.show_brush_mask = True
        self._shared_mask_overlay = None
        self.brush_erase = False
        self.show_mask_only = False
        self.show_clipping = False
        self.show_peaking = False
        self.show_zebras = False
        self._zebra_threshold = 0.95  # luminance 0..1
        self.horizon_line_mode = False
        self._horizon_line = None  # (x0,y0,x1,y1) widget coords while dragging
        self.keystone_mode = False
        self.keystone_points = []
        self._keystone_drag = -1
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
        if mode == self.MODE_NORMAL:
            self._comparison_pixmap = None
        self.update()

    def set_keystone_mode(self, enabled, points=None):
        self.keystone_mode = bool(enabled)
        if points is not None:
            self.keystone_points = [list(p) for p in points]
        if self.keystone_mode and len(self.keystone_points) != 4:
            self.keystone_points = [[0.08, 0.08], [0.92, 0.08], [0.92, 0.92], [0.08, 0.92]]
        self._keystone_drag = -1
        self.setCursor(Qt.CursorShape.CrossCursor if enabled else Qt.CursorShape.ArrowCursor)
        self.update()

    def set_comparison_image(self, pixmap):
        self._comparison_pixmap = pixmap
        self.update()

    def set_hold_original(self, enabled):
        self.hold_original = bool(enabled)
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

    def set_shared_mask_overlay(self, mask=None):
        if mask is None:
            self._shared_mask_overlay = None
        else:
            values = np.clip(np.asarray(mask, dtype=np.float32), 0, 1)
            rgba = np.zeros((values.shape[0], values.shape[1], 4), dtype=np.uint8)
            rgba[..., 0] = 255
            rgba[..., 1] = 55
            rgba[..., 2] = 55
            rgba[..., 3] = (values * 150).astype(np.uint8)
            image = QImage(rgba.data, rgba.shape[1], rgba.shape[0], rgba.strides[0], QImage.Format.Format_RGBA8888)
            self._shared_mask_overlay = QPixmap.fromImage(image.copy())
        self.update()

    def set_sharpen_proof(self, enabled, ppi=300, media="custom", width_in=12.0):
        self.sharpen_proof = bool(enabled)
        self.sharpen_proof_label = f"Sharpening proof · {media.title()} · {float(ppi):.0f} PPI · {float(width_in):.1f} in"
        if enabled:
            self._fit_mode = False
            self._scale = 1.0
            self._offset = QPoint(0, 0)
            self.zoom_changed.emit(self._scale)
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

    def set_show_zebras(self, enabled: bool):
        self.show_zebras = bool(enabled)
        self.update()

    def set_zebra_threshold(self, t: float):
        self._zebra_threshold = float(max(0.5, min(1.0, t)))
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
        """Clipping warning and/or zebra stripes on over/underexposed areas (all view modes)."""
        if self._pixmap is None or self._pixmap.isNull():
            return
        want_clip = getattr(self, "show_clipping", False)
        want_zebra = getattr(self, "show_zebras", False)
        want_peaking = getattr(self, "show_peaking", False)
        if not want_clip and not want_zebra and not want_peaking:
            return
        rect = self.image_rect()
        if rect.isEmpty():
            return
        from PyQt6.QtGui import QImage, QBrush
        qimg = self._pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
        w, h = qimg.width(), qimg.height()
        step = max(1, max(w, h) // 400)
        lo = int(255 * getattr(self, "_clip_lo", 0.005))
        hi = int(255 * getattr(self, "_clip_hi", 0.995))
        z_thr = int(255 * float(getattr(self, "_zebra_threshold", 0.95)))
        painter.save()
        painter.setClipRect(rect)
        sx = rect.width() / max(w, 1)
        sy = rect.height() / max(h, 1)
        for y in range(0, h, step):
            for x in range(0, w, step):
                c = qimg.pixelColor(x, y)
                yv = (c.red() * 3 + c.green() * 6 + c.blue()) // 10
                px = int(rect.left() + x * sx)
                py = int(rect.top() + y * sy)
                pw = max(1, int(step * sx) + 1)
                ph = max(1, int(step * sy) + 1)
                if want_clip:
                    if yv <= lo:
                        painter.fillRect(px, py, pw, ph, QColor(40, 80, 255, 160))
                    elif yv >= hi:
                        painter.fillRect(px, py, pw, ph, QColor(255, 40, 40, 160))
                if want_zebra and yv >= z_thr:
                    # Diagonal zebra stripes (classic video exposure assist)
                    # Alternate black/yellow bands by (x+y)
                    band = ((x + y) // max(4, step * 2)) % 2
                    if band == 0:
                        painter.fillRect(px, py, pw, ph, QColor(0, 0, 0, 180))
                    else:
                        painter.fillRect(px, py, pw, ph, QColor(255, 220, 0, 200))
        painter.restore()

        if getattr(self, "show_peaking", False):
            peak_step = max(2, max(w, h) // 350)
            painter.save()
            painter.setClipRect(rect)
            painter.setPen(QPen(QColor(0, 255, 90, 210), max(1, int(peak_step * sx * 0.6))))
            for y in range(peak_step, h - peak_step, peak_step):
                for x in range(peak_step, w - peak_step, peak_step):
                    c0 = qimg.pixelColor(x, y)
                    c1 = qimg.pixelColor(x + peak_step, y)
                    c2 = qimg.pixelColor(x, y + peak_step)
                    y0 = (c0.red() + c0.green() + c0.blue()) // 3
                    y1 = (c1.red() + c1.green() + c1.blue()) // 3
                    y2 = (c2.red() + c2.green() + c2.blue()) // 3
                    if abs(y0 - y1) + abs(y0 - y2) > 40:
                        painter.drawPoint(int(rect.left() + x * sx), int(rect.top() + y * sy))
            painter.restore()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#121212"))
        if self._pixmap is None:
            painter.setPen(QColor("#555"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Open a folder to load photos")
            painter.end()
            return
        before_pixmap = self._comparison_pixmap if self._comparison_pixmap is not None else self._original_pixmap
        if self.hold_original and self._original_pixmap is not None:
            self._draw_pixmap(painter, self._original_pixmap)
            painter.setPen(QColor("#ddd"))
            painter.drawText(10, 20, "Original")
        elif self.compare_mode == self.MODE_SIDE_BY_SIDE and before_pixmap is not None:
            half = self.width() // 2
            painter.setClipRect(0, 0, half - 1, self.height())
            self._draw_pixmap(painter, before_pixmap, x_off=-half // 2)
            painter.setClipRect(half + 1, 0, half, self.height())
            self._draw_pixmap(painter, self._pixmap, x_off=half // 2)
            painter.setClipping(False)
            painter.setPen(QPen(QColor(255, 255, 255, 180), 2))
            painter.drawLine(half, 0, half, self.height())
            painter.setPen(QColor("#aaa"))
            painter.drawText(10, 20, "Before")
            painter.drawText(half + 10, 20, "After")
        elif self.compare_mode == self.MODE_SPLIT and before_pixmap is not None:
            sx = int(self.width() * self._split_ratio)
            painter.setClipRect(0, 0, sx, self.height())
            self._draw_pixmap(painter, before_pixmap)
            painter.setClipRect(sx, 0, self.width() - sx, self.height())
            self._draw_pixmap(painter, self._pixmap)
            if getattr(self, "show_mask_only", False) and self.brush_masks:
                painter.fillRect(self.rect(), QColor(0, 0, 0, 150))

            if getattr(self, "show_peaking", False) and self._pixmap is not None and not self._pixmap.isNull():
                rect = self.image_rect()
                if not rect.isEmpty():
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
            painter.setClipping(False)
            painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
            painter.drawLine(sx, 0, sx, self.height())
        else:
            self._draw_pixmap(painter, self._pixmap)
            if self.show_mask_only and self.brush_masks:
                rect = self.image_rect()
                if not rect.isEmpty():
                    painter.fillRect(rect, QColor(0, 0, 0, 235))

        # Exposure assists work in every compare mode
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
                _, _, c_box, c_arc = self._guide_colors()
                pen_box = QPen(c_box, 1)
                pen_arc = QPen(c_arc, 2)
                # Draw one continuous logarithmic golden spiral. The previous
                # shrinking-rectangle arc construction could become negative
                # after a few turns, leaving a visibly incomplete curve.
                orient = self.spiral_orient % 4
                mirror = self.spiral_orient >= 4
                path = QPainterPath()
                turns = 3.25
                theta_max = turns * 2.0 * math.pi
                phi = 1.618033988749895
                points = 360
                for index in range(points + 1):
                    theta = theta_max * index / points
                    radius = 0.47 * (phi ** (2.0 * theta / math.pi)) / (phi ** (2.0 * theta_max / math.pi))
                    nx = 0.5 + radius * math.cos(theta)
                    ny = 0.5 + radius * math.sin(theta)
                    if mirror:
                        nx = 1.0 - nx
                    for _ in range(orient):
                        nx, ny = 1.0 - ny, nx
                    point = QPointF(x + nx * gw, y + ny * gh)
                    if index == 0:
                        path.moveTo(point)
                    else:
                        path.lineTo(point)
                painter.setPen(pen_box)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(int(round(x)), int(round(y)), int(round(gw)), int(round(gh)))
                painter.setPen(pen_arc)
                painter.drawPath(path)
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
        if self.keystone_mode and len(self.keystone_points) == 4 and self.compare_mode == self.MODE_NORMAL:
            rect = self.image_rect()
            if not rect.isEmpty():
                painter.setRenderHint(QPainter.RenderHint.Antialiasing)
                screen_points = [
                    QPointF(rect.left() + float(px) * rect.width(), rect.top() + float(py) * rect.height())
                    for px, py in self.keystone_points
                ]
                path = QPainterPath(screen_points[0])
                for point in screen_points[1:]:
                    path.lineTo(point)
                path.closeSubpath()
                painter.setBrush(QBrush(QColor(35, 130, 220, 24)))
                painter.setPen(QPen(QColor(70, 180, 255, 235), 2, Qt.PenStyle.DashLine))
                painter.drawPath(path)
                labels = ("TL", "TR", "BR", "BL")
                for index, point in enumerate(screen_points):
                    painter.setBrush(QBrush(QColor(45, 145, 235)))
                    painter.setPen(QPen(QColor(235, 245, 255), 2))
                    painter.drawEllipse(point, 7, 7)
                    painter.drawText(point + QPointF(9, -7), labels[index])

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
        if self.compare_mode == self.MODE_NORMAL and self._shared_mask_overlay is not None:
            rect = self.image_rect()
            if not rect.isEmpty():
                painter.drawPixmap(rect, self._shared_mask_overlay)

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
                    hardness = float(m.get("hardness", 0.7))
                    for s in m.get("strokes") or []:
                        cx = ix0 + float(s.get("x", 0.5)) * sw
                        cy = iy0 + float(s.get("y", 0.5)) * sh
                        r = max(float(s.get("r", 0.05)) * max(sw, sh), 2)
                        gradient = QRadialGradient(QPointF(cx, cy), r)
                        gradient.setColorAt(0.0, col)
                        gradient.setColorAt(max(0.0, min(0.98, hardness)), col)
                        gradient.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
                        painter.setBrush(QBrush(gradient))
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
        if self.sharpen_proof:
            painter.setPen(QPen(QColor(235, 235, 235), 1))
            painter.setBrush(QBrush(QColor(15, 15, 15, 205)))
            label_rect = QRect(14, self.height() - 45, min(430, self.width() - 28), 30)
            painter.drawRoundedRect(label_rect, 4, 4)
            painter.drawText(label_rect.adjusted(10, 0, -8, 0), Qt.AlignmentFlag.AlignVCenter, self.sharpen_proof_label)
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

        if self.keystone_mode and e.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            rect = self.image_rect()
            if not rect.isEmpty() and len(self.keystone_points) == 4:
                pos = e.position().toPoint()
                distances = []
                for px, py in self.keystone_points:
                    sx = rect.left() + float(px) * rect.width()
                    sy = rect.top() + float(py) * rect.height()
                    distances.append(math.hypot(pos.x() - sx, pos.y() - sy))
                nearest = min(range(4), key=lambda idx: distances[idx])
                if distances[nearest] <= 18:
                    self._keystone_drag = nearest
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

        # Skin-color eyedropper
        if self.skin_picker_mode and e.button() == Qt.MouseButton.LeftButton and self._pixmap is not None:
            rect = self.image_rect()
            if not rect.isEmpty() and rect.contains(e.position().toPoint()):
                pos = e.position().toPoint()
                nx = max(0.0, min(1.0, (pos.x() - rect.left()) / max(rect.width(), 1)))
                ny = max(0.0, min(1.0, (pos.y() - rect.top()) / max(rect.height(), 1)))
                pm = self._original_pixmap or self._pixmap
                px = int(nx * (pm.width() - 1)); py = int(ny * (pm.height() - 1))
                color = pm.toImage().pixelColor(px, py)
                self.skinColorPicked.emit(color.blueF(), color.greenF(), color.redF())
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
        if self._keystone_drag >= 0 and self.keystone_mode:
            rect = self.image_rect()
            if not rect.isEmpty():
                pos = e.position().toPoint()
                nx = max(0.0, min(1.0, (pos.x() - rect.left()) / max(rect.width(), 1)))
                ny = max(0.0, min(1.0, (pos.y() - rect.top()) / max(rect.height(), 1)))
                self.keystone_points[self._keystone_drag] = [nx, ny]
                self.keystoneChanged.emit([list(p) for p in self.keystone_points])
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
        if self._keystone_drag >= 0:
            self._keystone_drag = -1
            self.keystoneFinished.emit([list(p) for p in self.keystone_points])
            self.update()
            return
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
                    "id": uuid.uuid4().hex,
                    "strokes": list(self._brush_current_strokes),
                    "hardness": self.brush_hardness,
                    "feather": 0.0, "edge_refine": 0.0,
                    "luminance_min": 0.0, "luminance_max": 1.0,
                    "color_range": False, "color_tolerance": 0.2,
                    "intersect_with": [],
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
