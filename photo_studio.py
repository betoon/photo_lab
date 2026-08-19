"""
Photo Studio — a lightweight, DxO-style photo editing workbench.

Layout:
    - Left dock:   RGB histogram + thumbnail filmstrip for the open folder
    - Center:      Large image preview (zoom/fit)
    - Right dock:  Non-destructive adjustment stack (Exposure, Smart Light,
                   Contrast, Tone Curve gamma, Saturation, Vignette)
    - Toolbar:     Open folder, Reset, Export

Design notes:
    - Every adjustment is stored as a plain dict of numbers per-image, so
      switching images or exporting always re-applies the *full* pipeline
      to the original pixel data (non-destructive, like DxO's "recipe").
    - Slider drags are debounced with a QTimer so the preview only
      re-renders ~60ms after the user stops moving the slider, keeping
      the UI responsive on large files.
    - Thumbnail generation and full-res export both run on a background
      QThread so the UI never locks up.

Requirements:
    pip install PyQt6 opencv-python numpy Pillow

Run:
    python photo_studio.py
"""

import sys
import os
from dataclasses import dataclass, asdict, field

import numpy as np
import cv2

from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QPen, QIcon, QAction
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QSlider, QVBoxLayout,
    QHBoxLayout, QGridLayout, QScrollArea, QListWidget, QListWidgetItem,
    QFileDialog, QToolBar, QGroupBox, QPushButton, QSizePolicy, QSplitter,
    QStatusBar, QDoubleSpinBox, QFrame, QComboBox, QTabWidget
)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


# ----------------------------------------------------------------------
# Adjustment recipe
# ----------------------------------------------------------------------

@dataclass
class Recipe:
    """One image's non-destructive edit stack. All values are DxO-ish
    ranges so the sliders below can map 1:1 onto them."""
    exposure: float = 0.0        # EV, -3..3
    smart_light: float = 0.0     # 0..100 (shadow/highlight recovery)
    contrast: float = 0.0        # -100..100
    highlights: float = 0.0      # -100..100
    shadows: float = 0.0         # -100..100
    saturation: float = 0.0      # -100..100
    clarity: float = 0.0         # -100..100 (local contrast / microcontrast)
    gamma: float = 1.0           # 0.3..2.5 tone curve gamma
    vignette: float = 0.0        # 0..100
    # -- Geometry tab --
    horizon: float = 0.0         # degrees, -45..45 (straighten)
    distortion: float = 0.0      # -100..100 (- pincushion, + barrel)
    perspective: float = 0.0     # -100..100 (vertical keystone)
    crop: tuple = field(default=None)  # (x0, y0, x1, y1) normalized 0..1, or None

    def reset(self):
        for k, v in asdict(Recipe()).items():
            setattr(self, k, v)


# ----------------------------------------------------------------------
# Geometry transforms (operate on uint8 BGR, canvas size preserved except
# for crop, which is always applied last)
# ----------------------------------------------------------------------

def apply_horizon(img: np.ndarray, angle: float) -> np.ndarray:
    if angle == 0.0:
        return img
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_distortion(img: np.ndarray, amount: float) -> np.ndarray:
    """Simple radial (barrel/pincushion) correction. amount > 0 pushes the
    image outward (barrel-style), amount < 0 pulls it inward (pincushion)."""
    if amount == 0.0:
        return img
    h, w = img.shape[:2]
    k = amount / 100.0 * 0.6
    fx, fy = w / 2.0, h / 2.0
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    x_norm = (xs - fx) / fx
    y_norm = (ys - fy) / fy
    factor = 1.0 + k * (x_norm ** 2 + y_norm ** 2)
    map_x = x_norm * factor * fx + fx
    map_y = y_norm * factor * fy + fy
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)


def apply_perspective(img: np.ndarray, amount: float) -> np.ndarray:
    """Vertical keystone correction, e.g. for converging building verticals.
    amount > 0 widens the top of the frame, amount < 0 widens the bottom."""
    if amount == 0.0:
        return img
    h, w = img.shape[:2]
    f = max(-0.4, min(0.4, amount / 100.0 * 0.4))
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    if f >= 0:
        inset = w * f
        dst = np.float32([[inset, 0], [w - inset, 0], [0, h], [w, h]])
    else:
        inset = w * (-f)
        dst = np.float32([[0, 0], [w, 0], [inset, h], [w - inset, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def apply_crop(img: np.ndarray, crop) -> np.ndarray:
    if crop is None:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = crop
    xi0 = max(0, min(int(round(x0 * w)), w - 2))
    yi0 = max(0, min(int(round(y0 * h)), h - 2))
    xi1 = max(xi0 + 1, min(int(round(x1 * w)), w))
    yi1 = max(yi0 + 1, min(int(round(y1 * h)), h))
    return img[yi0:yi1, xi0:xi1]


def apply_recipe(img_bgr: np.ndarray, r: Recipe) -> np.ndarray:
    """Apply a Recipe to a full-precision float image and return uint8 BGR."""
    # Geometry first (distortion/perspective/horizon keep the canvas size,
    # so the crop's normalized coordinates stay valid regardless of order).
    img_bgr = apply_distortion(img_bgr, r.distortion)
    img_bgr = apply_perspective(img_bgr, r.perspective)
    img_bgr = apply_horizon(img_bgr, r.horizon)
    img_bgr = apply_crop(img_bgr, r.crop)

    img = img_bgr.astype(np.float32) / 255.0

    # Exposure (stops)
    if r.exposure != 0.0:
        img *= (2.0 ** r.exposure)

    # Smart lighting: cheap local shadow/highlight recovery via a blurred
    # luminance mask (mimics DxO Smart Lighting's tone-mapping idea).
    if r.smart_light != 0.0:
        lum = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(lum, (0, 0), sigmaX=img.shape[1] / 20)
        amt = r.smart_light / 100.0
        lift = (0.5 - blur) * amt * 0.6
        img += lift[..., None]

    # Highlights / shadows (simple tonal masks)
    if r.highlights != 0.0 or r.shadows != 0.0:
        lum = cv2.cvtColor(np.clip(img, 0, 1), cv2.COLOR_BGR2GRAY)
        hi_mask = np.clip((lum - 0.5) * 2, 0, 1) ** 1.5
        lo_mask = np.clip((0.5 - lum) * 2, 0, 1) ** 1.5
        img += (r.highlights / 100.0) * 0.5 * hi_mask[..., None]
        img += (r.shadows / 100.0) * 0.5 * lo_mask[..., None]

    # Contrast around mid-grey
    if r.contrast != 0.0:
        c = r.contrast / 100.0
        img = (img - 0.5) * (1.0 + c) + 0.5

    # Clarity / microcontrast: unsharp mask on luminance
    if r.clarity != 0.0:
        blur = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        img = img + (img - blur) * (r.clarity / 100.0)

    # Gamma (tone curve)
    img = np.clip(img, 0, 1)
    if r.gamma != 1.0:
        img = img ** (1.0 / r.gamma)

    # Saturation via HSV
    if r.saturation != 0.0:
        hsv = cv2.cvtColor(np.clip(img, 0, 1).astype(np.float32), cv2.COLOR_BGR2HSV)
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + r.saturation / 100.0), 0, 1)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Vignette
    if r.vignette != 0.0:
        h, w = img.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2, h / 2
        d = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
        d = np.clip(d, 0, 1.4) / 1.4
        strength = r.vignette / 100.0
        mask = 1.0 - strength * (d ** 2)
        img *= mask[..., None]

    img = np.clip(img, 0, 1) * 255.0
    return img.astype(np.uint8)


def cv_to_qpixmap(img_bgr: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())


# ----------------------------------------------------------------------
# Background workers
# ----------------------------------------------------------------------

class ThumbnailWorker(QThread):
    thumb_ready = pyqtSignal(str, QPixmap)

    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def run(self):
        for p in self.paths:
            img = cv2.imread(p)
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = 120 / max(h, w)
            thumb = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
            self.thumb_ready.emit(p, cv_to_qpixmap(thumb))


class ExportWorker(QThread):
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, path, recipe, out_path):
        super().__init__()
        self.path, self.recipe, self.out_path = path, recipe, out_path

    def run(self):
        try:
            img = cv2.imread(self.path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("Could not read source image")
            out = apply_recipe(img, self.recipe)
            cv2.imwrite(self.out_path, out)
            self.finished_ok.emit(self.out_path)
        except Exception as e:
            self.failed.emit(str(e))


# ----------------------------------------------------------------------
# Histogram widget
# ----------------------------------------------------------------------

class HistogramWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self.hist = None  # (r, g, b) arrays of length 256

    def set_image(self, img_bgr: np.ndarray):
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
# Main window
# ----------------------------------------------------------------------

class PhotoStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Photo Studio")
        self.resize(1500, 950)
        self.setStyleSheet(self._stylesheet())

        self.folder = None
        self.image_paths = []
        self.recipes: dict[str, Recipe] = {}
        self.current_path = None
        self.original_bgr = None  # full-res source of current image

        self.render_timer = QTimer()
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(60)
        self.render_timer.timeout.connect(self.render_preview)

        self._build_toolbar()
        self._build_layout()
        self.statusBar().showMessage("Open a folder to begin.")

    # -- UI construction -------------------------------------------------

    def _stylesheet(self):
        return """
            QMainWindow, QWidget { background: #202020; color: #ddd; }
            QGroupBox {
                color: #ddd; border: 1px solid #3a3a3a; border-radius: 4px;
                margin-top: 10px; font-weight: bold;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; color: #ddd; }
            QLabel { color: #ddd; }
            QPushButton {
                background: #3a3a3a; color: #eee; border: 1px solid #4a4a4a;
                padding: 6px 12px; border-radius: 4px;
            }
            QPushButton:hover { background: #4a4a4a; }
            QSlider::groove:horizontal { background: #444; height: 4px; border-radius: 2px; }
            QSlider::handle:horizontal {
                background: #7ab0ff; width: 12px; margin: -5px 0; border-radius: 6px;
            }
            QListWidget { background: #181818; border: none; color: #ddd; }
            QListWidget::item { color: #ddd; }
            QListWidget::item:selected { background: #3a5a8a; color: #fff; }
            QToolBar { background: #262626; border: none; spacing: 6px; }
            QToolButton { color: #ddd; background: transparent; padding: 6px 10px; }
            QToolButton:hover { background: #3a3a3a; border-radius: 4px; }
            QStatusBar { color: #bbb; }
            QStatusBar QLabel { color: #bbb; }
            QScrollArea { border: none; }
        """

    def _build_toolbar(self):
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        open_act = QAction("Open Folder", self)
        open_act.triggered.connect(self.open_folder)
        tb.addAction(open_act)

        reset_act = QAction("Reset", self)
        reset_act.triggered.connect(self.reset_current)
        tb.addAction(reset_act)

        export_act = QAction("Export", self)
        export_act.triggered.connect(self.export_current)
        tb.addAction(export_act)

        self.setStatusBar(QStatusBar())

    def _build_layout(self):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(splitter)

        # --- Left: histogram + filmstrip ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        hist_box = QGroupBox("Histogram")
        hb = QVBoxLayout(hist_box)
        self.histogram = HistogramWidget()
        hb.addWidget(self.histogram)
        left_layout.addWidget(hist_box)

        film_box = QGroupBox("Filmstrip")
        fb = QVBoxLayout(film_box)
        self.filmstrip = QListWidget()
        self.filmstrip.setViewMode(QListWidget.ViewMode.IconMode)
        self.filmstrip.setIconSize(QSize(110, 110))
        self.filmstrip.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.filmstrip.setMovement(QListWidget.Movement.Static)
        self.filmstrip.itemClicked.connect(self.on_thumb_clicked)
        fb.addWidget(self.filmstrip)
        left_layout.addWidget(film_box, stretch=1)

        left.setMinimumWidth(260)
        left.setMaximumWidth(340)
        splitter.addWidget(left)

        # --- Center: preview ---
        center = QWidget()
        center_layout = QVBoxLayout(center)
        self.preview_scroll = QScrollArea()
        self.preview_scroll.setWidgetResizable(True)
        self.preview_scroll.setStyleSheet("background:#151515; border:none;")
        self.preview_label = ImageCanvas()
        self.preview_label.setText("Open a folder to load photos")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet("color:#666; font-size:16px;")
        self.preview_label.crop_dragged.connect(self.on_crop_dragged)
        self.preview_scroll.setWidget(self.preview_label)
        center_layout.addWidget(self.preview_scroll)
        splitter.addWidget(center)

        # --- Right: adjustments, tabbed like DxO's Light / Geometry / Effects ---
        tabs = QTabWidget()
        tabs.setMaximumWidth(340)
        tabs.setDocumentMode(True)

        self.sliders = {}

        def make_tab():
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            inner = QWidget()
            layout = QVBoxLayout(inner)
            scroll.setWidget(inner)
            return scroll, layout

        def add_group(layout, title, rows):
            box = QGroupBox(title)
            v = QVBoxLayout(box)
            for key, label, lo, hi, step, decimals, default in rows:
                row = SliderRow(label, lo, hi, default, step,
                                 lambda val, k=key: self.on_slider(k, val), decimals)
                self.sliders[key] = row
                v.addWidget(row)
            layout.addWidget(box)
            return box

        def combo_row(label_text, options):
            row = QHBoxLayout()
            row.addWidget(QLabel(label_text))
            combo = QComboBox()
            combo.addItems(options)
            combo.setStyleSheet("background:#2b2b2b; color:#eee; border:1px solid #444;")
            row.addWidget(combo, stretch=1)
            return row, combo

        # -- Light tab --
        light_scroll, light_layout = make_tab()
        add_group(light_layout, "Light", [
            ("exposure", "Exposure (EV)", -3.0, 3.0, 0.05, 2, 0.0),
            ("smart_light", "Smart Lighting", 0.0, 100.0, 1, 0, 0.0),
            ("highlights", "Highlights", -100.0, 100.0, 1, 0, 0.0),
            ("shadows", "Shadows", -100.0, 100.0, 1, 0, 0.0),
        ])
        add_group(light_layout, "Contrast & Detail", [
            ("contrast", "Contrast", -100.0, 100.0, 1, 0, 0.0),
            ("clarity", "Clarity", -100.0, 100.0, 1, 0, 0.0),
            ("gamma", "Tone Curve Gamma", 0.3, 2.5, 0.05, 2, 1.0),
        ])
        add_group(light_layout, "Color", [
            ("saturation", "Saturation", -100.0, 100.0, 1, 0, 0.0),
        ])
        light_layout.addStretch(1)
        tabs.addTab(light_scroll, "Light")

        # -- Geometry tab --
        geo_scroll, geo_layout = make_tab()

        horizon_box = QGroupBox("Horizon")
        hv = QVBoxLayout(horizon_box)
        horizon_row = SliderRow("Horizon (\u00b0)", -45.0, 45.0, 0.0, 0.1,
                                 lambda val: self.on_slider("horizon", val), 1)
        self.sliders["horizon"] = horizon_row
        hv.addWidget(horizon_row)
        geo_layout.addWidget(horizon_box)

        crop_box = QGroupBox("Crop")
        cropv = QVBoxLayout(crop_box)
        crop_btn_row = QHBoxLayout()
        self.crop_tool_btn = QPushButton("Crop Tool")
        self.crop_tool_btn.setCheckable(True)
        self.crop_tool_btn.toggled.connect(self.toggle_crop_mode)
        crop_btn_row.addWidget(self.crop_tool_btn)
        self.clear_crop_btn = QPushButton("Clear")
        self.clear_crop_btn.clicked.connect(self.clear_crop)
        crop_btn_row.addWidget(self.clear_crop_btn)
        cropv.addLayout(crop_btn_row)

        ratio_row, self.aspect_combo = combo_row(
            "Aspect ratio", ["Free", "Original", "1:1", "4:3", "3:2", "16:9"])
        cropv.addLayout(ratio_row)

        crop_hint = QLabel("Drag on the preview image to select a crop area.")
        crop_hint.setStyleSheet("color:#888; font-size:11px;")
        crop_hint.setWordWrap(True)
        cropv.addWidget(crop_hint)
        geo_layout.addWidget(crop_box)

        dist_box = QGroupBox("Distortion")
        distv = QVBoxLayout(dist_box)
        dist_combo_row, self.distortion_combo = combo_row("Correction", ["Off", "Manual"])
        self.distortion_combo.currentTextChanged.connect(self.on_distortion_mode)
        distv.addLayout(dist_combo_row)
        distortion_row = SliderRow("Distortion", -100.0, 100.0, 0.0, 1,
                                    lambda val: self.on_slider("distortion", val), 0)
        distortion_row.setEnabled(False)
        self.sliders["distortion"] = distortion_row
        distv.addWidget(distortion_row)
        geo_layout.addWidget(dist_box)

        persp_box = QGroupBox("Perspective")
        perspv = QVBoxLayout(persp_box)
        persp_combo_row, self.perspective_combo = combo_row("Correction", ["Off", "Manual"])
        self.perspective_combo.currentTextChanged.connect(self.on_perspective_mode)
        perspv.addLayout(persp_combo_row)
        perspective_row = SliderRow("Vertical Perspective", -100.0, 100.0, 0.0, 1,
                                     lambda val: self.on_slider("perspective", val), 0)
        perspective_row.setEnabled(False)
        self.sliders["perspective"] = perspective_row
        perspv.addWidget(perspective_row)
        geo_layout.addWidget(persp_box)

        geo_layout.addStretch(1)
        tabs.addTab(geo_scroll, "Geometry")

        # -- Effects tab --
        fx_scroll, fx_layout = make_tab()
        add_group(fx_layout, "Effects", [
            ("vignette", "Vignette", 0.0, 100.0, 1, 0, 0.0),
        ])
        fx_layout.addStretch(1)
        tabs.addTab(fx_scroll, "Effects")

        splitter.addWidget(tabs)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

    # -- Folder / thumbnails ---------------------------------------------

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Choose a photo folder")
        if not folder:
            return
        self.folder = folder
        self.image_paths = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTS)
        )
        self.filmstrip.clear()
        self.recipes = {}
        if not self.image_paths:
            self.statusBar().showMessage("No images found in that folder.")
            return

        for p in self.image_paths:
            item = QListWidgetItem(os.path.basename(p))
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.filmstrip.addItem(item)

        self.thumb_worker = ThumbnailWorker(self.image_paths)
        self.thumb_worker.thumb_ready.connect(self.on_thumb_ready)
        self.thumb_worker.start()

        self.statusBar().showMessage(f"Loaded {len(self.image_paths)} images from {folder}")
        self.load_image(self.image_paths[0])

    def on_thumb_ready(self, path, pixmap):
        for i in range(self.filmstrip.count()):
            item = self.filmstrip.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                item.setIcon(QIcon(pixmap))
                break

    def on_thumb_clicked(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        self.load_image(path)

    # -- Image loading / editing ------------------------------------------

    def load_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            self.statusBar().showMessage(f"Could not open {path}")
            return
        self.current_path = path
        self.original_bgr = img
        if path not in self.recipes:
            self.recipes[path] = Recipe()
        self.sync_sliders_to_recipe()
        self.render_preview()
        self.statusBar().showMessage(f"Editing {os.path.basename(path)} — {img.shape[1]}x{img.shape[0]}")

    def sync_sliders_to_recipe(self):
        r = self.recipes[self.current_path]
        for key, row in self.sliders.items():
            row.set_value(getattr(r, key))

        self.distortion_combo.blockSignals(True)
        self.distortion_combo.setCurrentText("Manual" if r.distortion != 0.0 else "Off")
        self.distortion_combo.blockSignals(False)
        self.sliders["distortion"].setEnabled(r.distortion != 0.0)

        self.perspective_combo.blockSignals(True)
        self.perspective_combo.setCurrentText("Manual" if r.perspective != 0.0 else "Off")
        self.perspective_combo.blockSignals(False)
        self.sliders["perspective"].setEnabled(r.perspective != 0.0)

        self.crop_tool_btn.setChecked(False)
        self.preview_label.set_crop_mode(False)

    def on_slider(self, key, value):
        if self.current_path is None:
            return
        setattr(self.recipes[self.current_path], key, value)
        self.render_timer.start()

    # -- Geometry: distortion / perspective correction toggles ----------

    def on_distortion_mode(self, text):
        enabled = (text == "Manual")
        self.sliders["distortion"].setEnabled(enabled)
        if not enabled:
            self.sliders["distortion"].set_value(0.0)
            self.on_slider("distortion", 0.0)

    def on_perspective_mode(self, text):
        enabled = (text == "Manual")
        self.sliders["perspective"].setEnabled(enabled)
        if not enabled:
            self.sliders["perspective"].set_value(0.0)
            self.on_slider("perspective", 0.0)

    # -- Geometry: crop ---------------------------------------------------

    def toggle_crop_mode(self, checked):
        if self.original_bgr is None:
            self.crop_tool_btn.setChecked(False)
            return
        if checked and self.recipes[self.current_path].crop is not None:
            # Start fresh: a new drag should select against the full image,
            # not the already-cropped preview.
            self.recipes[self.current_path].crop = None
            self.render_preview()
        self.preview_label.set_crop_mode(checked)

    def _aspect_ratio_value(self):
        text = self.aspect_combo.currentText()
        if text == "Free":
            return None
        if text == "Original":
            if self.original_bgr is None:
                return None
            h, w = self.original_bgr.shape[:2]
            return w / h
        return {"1:1": 1.0, "4:3": 4 / 3, "3:2": 3 / 2, "16:9": 16 / 9}.get(text)

    def on_crop_dragged(self, rect: QRect):
        if self.current_path is None or self.preview_label.pixmap() is None:
            return
        pm = self.preview_label.pixmap()
        pw, ph = pm.width(), pm.height()
        x0 = max(0, min(rect.left(), pw))
        y0 = max(0, min(rect.top(), ph))
        x1 = max(0, min(rect.right(), pw))
        y1 = max(0, min(rect.bottom(), ph))
        if x1 - x0 < 5 or y1 - y0 < 5:
            return

        ratio = self._aspect_ratio_value()
        if ratio is not None:
            w = x1 - x0
            h = w / ratio
            if y0 + h > ph:
                h = ph - y0
                w = h * ratio
            x1, y1 = x0 + w, y0 + h

        self.recipes[self.current_path].crop = (x0 / pw, y0 / ph, x1 / pw, y1 / ph)
        self.crop_tool_btn.setChecked(False)
        self.preview_label.set_crop_mode(False)
        self.render_preview()

    def clear_crop(self):
        if self.current_path is None:
            return
        self.recipes[self.current_path].crop = None
        self.crop_tool_btn.setChecked(False)
        self.preview_label.set_crop_mode(False)
        self.render_preview()

    def render_preview(self):
        if self.original_bgr is None or self.current_path is None:
            return
        recipe = self.recipes[self.current_path]

        # Downscale for a responsive live preview; export re-applies at
        # full resolution.
        h, w = self.original_bgr.shape[:2]
        max_dim = 1400
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            preview_src = cv2.resize(self.original_bgr, (int(w * scale), int(h * scale)))
        else:
            preview_src = self.original_bgr

        result = apply_recipe(preview_src, recipe)
        self.histogram.set_image(result)
        pix = cv_to_qpixmap(result)
        self.preview_label.setPixmap(pix)
        self.preview_label.setFixedSize(pix.size())

    def reset_current(self):
        if self.current_path is None:
            return
        self.recipes[self.current_path].reset()
        self.sync_sliders_to_recipe()
        self.render_preview()

    def export_current(self):
        if self.current_path is None:
            self.statusBar().showMessage("No image loaded.")
            return
        base, ext = os.path.splitext(os.path.basename(self.current_path))
        suggested = os.path.join(self.folder or ".", f"{base}_edited{ext}")
        out_path, _ = QFileDialog.getSaveFileName(self, "Export image", suggested)
        if not out_path:
            return
        self.statusBar().showMessage("Exporting...")
        self.export_worker = ExportWorker(self.current_path, self.recipes[self.current_path], out_path)
        self.export_worker.finished_ok.connect(lambda p: self.statusBar().showMessage(f"Exported to {p}"))
        self.export_worker.failed.connect(lambda e: self.statusBar().showMessage(f"Export failed: {e}"))
        self.export_worker.start()


def main():
    app = QApplication(sys.argv)
    win = PhotoStudio()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()