"""main_window.py — the PhotoStudio main window.

Layout:
    - Top splitter: histogram (left) | image preview (center) | tabbed
      adjustments (right): Light, Detail, Geometry, Effects
    - Bottom: a full-width thumbnail filmstrip for browsing the open folder
    - Toolbar: Open folder, Reset, Export
"""

import os
import cv2

from PyQt6.QtCore import Qt, QTimer, QSize, QRect
from PyQt6.QtGui import QIcon, QAction
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QScrollArea, QListWidget, QListWidgetItem, QFileDialog, QToolBar,
    QGroupBox, QPushButton, QSplitter, QStatusBar, QComboBox, QTabWidget
)

from imaging import Recipe, apply_recipe, IMAGE_EXTS
from qt_utils import cv_to_qpixmap
from workers import ThumbnailWorker, ExportWorker
from widgets import HistogramWidget, SliderRow, ImageCanvas


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
            QPushButton:checked { background: #2f6fdb; color: #fff; border-color: #2f6fdb; }
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
            QComboBox {
                background: #2b2b2b; color: #eee; border: 1px solid #444;
                border-radius: 3px; padding: 3px 6px;
            }
            QComboBox QAbstractItemView {
                background: #2b2b2b; color: #eee; selection-background-color: #2f6fdb;
            }
            QTabWidget::pane { border: 1px solid #3a3a3a; background: #202020; top: -1px; }
            QTabBar::tab {
                background: #262626; color: #ccc; padding: 8px 14px;
                border: 1px solid #3a3a3a; border-bottom: none;
                border-top-left-radius: 4px; border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected { background: #2f6fdb; color: #ffffff; }
            QTabBar::tab:hover:!selected { background: #3a3a3a; }
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
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, stretch=1)

        # --- Left: histogram ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(8, 8, 8, 8)

        hist_box = QGroupBox("Histogram")
        hb = QVBoxLayout(hist_box)
        self.histogram = HistogramWidget()
        hb.addWidget(self.histogram)
        left_layout.addWidget(hist_box)
        left_layout.addStretch(1)

        left.setMinimumWidth(220)
        left.setMaximumWidth(300)
        splitter.addWidget(left)

        # --- Center: preview ---
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
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

        # --- Right: adjustments, tabbed like DxO's Light / Detail / Geometry / Effects ---
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

        # -- Detail tab (denoising + sharpening) --
        detail_scroll, detail_layout = make_tab()
        add_group(detail_layout, "Denoising", [
            ("denoise_luminance", "Luminance", 0.0, 100.0, 1, 0, 0.0),
            ("denoise_chroma", "Chrominance", 0.0, 100.0, 1, 0, 0.0),
        ])
        denoise_note = QLabel("Fast bilateral-filter noise reduction — a lighter-weight "
                               "stand-in for DxO's AI denoising engines.")
        denoise_note.setStyleSheet("color:#888; font-size:11px;")
        denoise_note.setWordWrap(True)
        detail_layout.addWidget(denoise_note)

        add_group(detail_layout, "Unsharp Mask", [
            ("sharpen_intensity", "Intensity", 0.0, 200.0, 1, 0, 0.0),
            ("sharpen_radius", "Radius", 0.1, 5.0, 0.1, 1, 1.0),
            ("sharpen_threshold", "Threshold", 0.0, 50.0, 1, 0, 0.0),
        ])
        detail_layout.addStretch(1)
        tabs.addTab(detail_scroll, "Detail")

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

        # --- Bottom: filmstrip, full width ---
        film_box = QGroupBox("Filmstrip")
        film_box.setFixedHeight(190)
        fb = QVBoxLayout(film_box)
        self.filmstrip = QListWidget()
        self.filmstrip.setViewMode(QListWidget.ViewMode.IconMode)
        self.filmstrip.setFlow(QListWidget.Flow.LeftToRight)
        self.filmstrip.setWrapping(False)
        self.filmstrip.setIconSize(QSize(120, 120))
        self.filmstrip.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.filmstrip.setMovement(QListWidget.Movement.Static)
        self.filmstrip.itemClicked.connect(self.on_thumb_clicked)
        fb.addWidget(self.filmstrip)
        outer.addWidget(film_box)

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
