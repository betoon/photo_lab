from __future__ import annotations

import logging
from datetime import datetime
from time import perf_counter
from pathlib import Path
import traceback
import json
import os
import platform
import shutil
import zipfile
import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QThread, QTimer, QSettings, QSize, Signal
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
    QColorDialog, QDoubleSpinBox, QFileDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSlider, QSpinBox, QSplitter,
    QTabWidget, QTableWidget, QTableWidgetItem, QToolBar, QVBoxLayout, QWidget)

from .alignment import align_stack
from .fusion import finish, fuse
from .io import IMAGE_EXTENSIONS, read_image, to_float, write_image
from .microscopy import preprocess_microscope_stack
from .microscope_analysis import (diagnose_focus_order, draw_scale_bar, microscope_fuse, natural_sort,
    parameter_comparison, synthesize_intermediate_planes)
from .processing import build_report, estimate_memory_bytes, human_bytes, normalize_stack, save_report
from .retouch import RetouchSession
from .engine import DiskBackedStack, PauseGate, configure_acceleration, finish_accelerated, fuse_tiled
from .exporting import export_auxiliary, write_advanced
from . import __version__
from .models import Project
from .launch import parse_launch_args

LOG = logging.getLogger("focus_stacker")


class ImageView(QScrollArea):
    imageClicked = Signal(int, int)
    imageHovered = Signal(int, int, object)
    def __init__(self):
        super().__init__(); self.label = QLabel("Import a focus sequence to begin")
        self.label.setAlignment(Qt.AlignCenter); self.setWidget(self.label); self.setWidgetResizable(True)
        self.viewport().setMouseTracking(True); self.label.setMouseTracking(True)
        self._array = None; self._zoom = 1.0

    def set_image(self, image: np.ndarray | None):
        self._array = image
        if image is None: return
        a = np.clip(image, 0, 1); a = (a * 255).astype(np.uint8)
        self._qimage = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format_RGB888).copy()
        self._refresh()

    def _refresh(self):
        if self._array is not None:
            size = self._qimage.size() * self._zoom
            self.label.setPixmap(QPixmap.fromImage(self._qimage).scaled(size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.label.resize(size)

    def wheelEvent(self, event):
        if event.modifiers() & Qt.ControlModifier and self._array is not None:
            self._zoom = max(0.1, min(8.0, self._zoom * (1.2 if event.angleDelta().y() > 0 else 1 / 1.2)))
            self._refresh(); event.accept()
        else: super().wheelEvent(event)

    def mousePressEvent(self, event):
        if self._array is not None and event.button() == Qt.LeftButton:
            x = int((event.position().x() + self.horizontalScrollBar().value()) / self._zoom)
            y = int((event.position().y() + self.verticalScrollBar().value()) / self._zoom)
            if 0 <= x < self._array.shape[1] and 0 <= y < self._array.shape[0]: self.imageClicked.emit(x, y)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._array is not None:
            x = int((event.position().x() + self.horizontalScrollBar().value()) / self._zoom); y = int((event.position().y() + self.verticalScrollBar().value()) / self._zoom)
            if 0 <= x < self._array.shape[1] and 0 <= y < self._array.shape[0]: self.imageHovered.emit(x, y, self._array[y, x].copy())
        super().mouseMoveEvent(event)

    def set_zoom(self, value: float): self._zoom = max(.05, min(8, value)); self._refresh()
    def fit_image(self):
        if self._array is not None:
            self._zoom = min(self.viewport().width() / self._array.shape[1], self.viewport().height() / self._array.shape[0]); self._refresh()


class CollapsibleSection(QWidget):
    def __init__(self, title: str, content: QWidget):
        super().__init__(); layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        self.button = QPushButton(f"▾  {title}"); self.button.setCheckable(True); self.button.setChecked(True); self.button.clicked.connect(self._toggle)
        self.content = content; layout.addWidget(self.button); layout.addWidget(content)
    def _toggle(self, shown): self.content.setVisible(shown); self.button.setText(("▾  " if shown else "▸  ") + self.button.text()[3:])


class StackWorker(QObject):
    progress = Signal(int, str); finished = Signal(object, object, object); failed = Signal(str)
    def __init__(self, project: Project, microscope: bool = False): super().__init__(); self.project = project; self.microscope = microscope; self.gate = PauseGate(); self.cancelled = False
    def cancel(self): self.cancelled = True; self.gate.cancel()
    def pause(self): self.gate.pause()
    def resume(self): self.gate.resume()
    def run(self):
        try:
            timings = {}; started = perf_counter(); warnings = []
            perf = self.project.performance; acceleration = configure_acceleration(perf.use_gpu, perf.cpu_threads)
            arrays, metadata, loaded_paths = [], [], []
            active_paths = [p for p in self.project.images if p not in self.project.disabled_images]
            for i, path in enumerate(active_paths):
                self.gate.checkpoint()
                self.progress.emit(int(10 * i / len(active_paths)), f"Loading {Path(path).name}")
                try:
                    a, m = read_image(path)
                    if arrays and a.shape[:2] != arrays[0].shape[:2]: raise ValueError(f"dimension mismatch {a.shape[:2]} vs {arrays[0].shape[:2]}")
                    m["shape"] = list(a.shape); m["dtype"] = str(a.dtype); arrays.append(to_float(a)); metadata.append(m); loaded_paths.append(path)
                except Exception as exc:
                    if not perf.recover_failed_frames: raise
                    warnings.append(f"Skipped {path}: {exc}"); self.progress.emit(int(10 * i / len(active_paths)), warnings[-1])
            if len(arrays) < 2: raise RuntimeError("Fewer than two usable frames remained after recovery")
            peak_estimate = estimate_memory_bytes(len(arrays), arrays[0].shape[0], arrays[0].shape[1], self.project.stack.algorithm, self.project.stack.pyramid_levels)
            timings["load"] = perf_counter() - started
            if self.microscope:
                stage = perf_counter()
                arrays = preprocess_microscope_stack(arrays, self.project.microscope,
                    lambda i, s: self.progress.emit(10 + int(10 * i / max(1, len(arrays))), s), lambda: self.gate.cancelled)
                timings["microscope_preprocess"] = perf_counter() - stage
            align = self.project.alignment; stack = self.project.stack
            ref = 0 if align.reference == "first" else (len(arrays) - 1 if align.reference == "last" else len(arrays) // 2)
            stage = perf_counter(); arrays, normalization = normalize_stack(arrays, ref, stack.normalize_exposure, stack.normalize_color,
                lambda i, s: self.progress.emit(18 + int(7 * i / max(1, len(arrays))), s))
            timings["normalization"] = perf_counter() - stage; stage = perf_counter()
            result = align_stack(arrays, align.method, ref, align.crop_common, align.ecc_iterations,
                                 align.ecc_epsilon, align.multiscale, align.pyramid_scales,
                                 perf.alignment_proxy_dimension, perf.recover_failed_frames, lambda: self.gate.cancelled,
                                 lambda p, s: self.progress.emit(10 + p, s))
            warnings.extend(result.warnings); loaded_paths = [loaded_paths[i] for i in result.used_indices]; metadata = [metadata[i] for i in result.used_indices]
            timings["alignment"] = perf_counter() - stage; stage = perf_counter()
            cache = DiskBackedStack(result.images, perf.cache_directory) if perf.disk_cache else None
            if cache: result.images = cache.images; arrays = []
            fusion_images = result.images
            microscope_diagnostics = {}; synthetic_provenance = []
            if self.microscope:
                if self.project.microscope.synthesize_intermediate and self.project.microscope.scientific_mode:
                    warnings.append("Intermediate-plane synthesis was disabled because Scientific Mode is active")
                elif self.project.microscope.synthesize_intermediate:
                    fusion_images, synthetic_provenance = synthesize_intermediate_planes(list(fusion_images), self.project.microscope.intermediate_count)
                    warnings.append(f"Presentation-only synthesis added {sum(1 for item in synthetic_provenance if item['synthetic'])} artificial focus planes")
                image, depth, confidence, microscope_diagnostics = microscope_fuse(list(fusion_images), self.project.microscope, stack.smooth_radius)
            elif perf.tiled_fusion:
                image, depth, confidence = fuse_tiled(fusion_images, stack.algorithm, stack.focus_radius, stack.smooth_radius,
                    stack.temperature, stack.pyramid_levels, stack.cleanup_radius, perf.tile_size, self.gate,
                    lambda p, s, eta: self.progress.emit(50 + p // 3, f"{s} — ETA {eta:.0f}s" if eta is not None else s))
            else:
                image, depth = fuse(fusion_images, stack.algorithm, stack.focus_radius, stack.smooth_radius,
                                    stack.temperature, stack.pyramid_levels, stack.cleanup_radius,
                                    lambda p, s: self.progress.emit(p, s)); confidence = np.ones(depth.shape, np.float32)
            image = finish_accelerated(image, stack.sharpen, stack.denoise, acceleration.opencl_enabled)
            timings["fusion_and_finish"] = perf_counter() - stage; timings["total"] = perf_counter() - started
            details = {"alignment": result, "metadata": metadata, "microscope": self.microscope, "normalization": normalization,
                       "timings": timings, "active_paths": loaded_paths, "confidence": confidence, "cache": cache,
                       "acceleration": acceleration.__dict__, "warnings": warnings}
            details["peak_memory_estimate"] = peak_estimate
            details["microscope_diagnostics"] = microscope_diagnostics; details["synthetic_provenance"] = synthetic_provenance
            details["effective_frame_count"] = len(fusion_images)
            details["report"] = build_report(self.project, details, image.shape, timings, warnings)
            self.progress.emit(100, "Complete"); self.finished.emit(image, depth, details)
        except InterruptedError: self.failed.emit("Processing cancelled")
        except Exception: self.failed.emit(traceback.format_exc())


class BatchWorker(QObject):
    progress = Signal(int, str); finished = Signal(int, int); failed = Signal(str)
    def __init__(self, jobs): super().__init__(); self.jobs = jobs; self.cancelled = False; self.current = None
    def cancel(self):
        self.cancelled = True
        if self.current: self.current.cancel()
    def run(self):
        completed = 0
        try:
            for index, (project_path, output_path) in enumerate(self.jobs):
                if self.cancelled: break
                project = Project.load(project_path); holder = {}; errors = []
                self.current = StackWorker(project, project.microscope.enabled)
                self.current.progress.connect(lambda p, s, i=index: self.progress.emit(int((i + p / 100) * 100 / len(self.jobs)), f"Job {i + 1}/{len(self.jobs)}: {s}"))
                self.current.finished.connect(lambda image, depth, details: holder.update(image=image, depth=depth, details=details))
                self.current.failed.connect(errors.append); self.current.run()
                if errors:
                    self.progress.emit(int((index + 1) * 100 / len(self.jobs)), f"Job {index + 1} failed and was skipped: {errors[0].splitlines()[-1]}"); continue
                if self.cancelled: break
                metadata = holder["details"]["metadata"][0] if holder["details"].get("metadata") else {}
                batch_image = draw_scale_bar(holder["image"], project.microscope.microns_per_pixel, project.microscope.scale_bar_microns, project.microscope.scale_bar_color, project.microscope.scale_bar_position) if project.microscope.scale_bar_enabled else holder["image"]
                write_advanced(output_path, batch_image, project.output, metadata, holder["details"].get("confidence") if project.output.include_alpha else None)
                masks = [(holder["depth"] == i).astype(np.float32) for i in range(holder["details"].get("effective_frame_count", len(holder["details"]["alignment"].images)))] if project.output.export_masks else None
                export_auxiliary(output_path, holder["details"]["alignment"].images, holder["depth"], holder["details"].get("confidence"), masks, project.output)
                save_report(str(Path(output_path).with_suffix(".report.json")), holder["details"]["report"]); completed += 1
                if holder["details"].get("cache"): holder["details"]["cache"].close()
            self.finished.emit(completed, len(self.jobs))
        except Exception: self.failed.emit(traceback.format_exc())


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__(); self.project = Project(); self.result = None; self.depth = None; self.details = None
        self.settings = QSettings("Brian E. Toon", "Focus Stacker Pro"); self.history = []; self.setAcceptDrops(True); self._paused = False; self._dark_stylesheet = QApplication.instance().styleSheet()
        self.setWindowTitle(f"Focus Stacker Pro - Brian E. Toon - V{__version__} - 2026"); self.resize(1500, 920); self._build_ui(); self._build_actions()
        self._install_shortcuts()
        self.statusBar().showMessage("Ready")

    def _build_ui(self):
        self.list = QListWidget(); self.list.setSelectionMode(QAbstractItemView.ExtendedSelection); self.list.currentRowChanged.connect(self.preview_source); self.list.setIconSize(QSize(72, 54))
        controls = QWidget(); form = QFormLayout(controls)
        self.alignment = QComboBox(); self.alignment.addItems(["ECC affine", "ECC translation", "ECC rigid", "ECC homography", "Feature affine", "Feature homography", "None"])
        self.reference = QComboBox(); self.reference.addItems(["Middle", "First", "Last"])
        self.crop = QCheckBox(); self.crop.setChecked(True)
        self.multiscale = QCheckBox(); self.multiscale.setChecked(True)
        self.algorithm = QComboBox(); self.algorithm.addItems(["Depth map", "Weighted", "Pyramid", "Average"])
        self.radius = QSpinBox(); self.radius.setRange(1, 50); self.radius.setValue(5)
        self.smooth = QSpinBox(); self.smooth.setRange(0, 50); self.smooth.setValue(7)
        self.levels = QSpinBox(); self.levels.setRange(1, 10); self.levels.setValue(5)
        self.cleanup = QSpinBox(); self.cleanup.setRange(1, 31); self.cleanup.setSingleStep(2); self.cleanup.setValue(5)
        self.sharpen = QDoubleSpinBox(); self.sharpen.setRange(0, 2); self.sharpen.setSingleStep(.05); self.sharpen.setValue(.25)
        self.normalize_exposure = QCheckBox(); self.normalize_color = QCheckBox()
        self.radius.setToolTip("Local sharpness scale. Recommended: 3–8; smaller preserves tiny detail, larger suppresses noise.")
        self.smooth.setToolTip("Focus-mask boundary smoothing. Recommended: 4–10.")
        self.levels.setToolTip("Pyramid spatial scales. Recommended: 4–6 for photographs.")
        self.cleanup.setToolTip("Odd-sized median cleanup for the depth map. Recommended: 3–9.")
        self.sharpen.setToolTip("Post-fusion unsharp amount. Recommended: 0.10–0.35.")
        for name, widget in [("Alignment", self.alignment), ("Reference", self.reference), ("Crop common area", self.crop),
                             ("Coarse-to-fine alignment", self.multiscale), ("Normalize exposure", self.normalize_exposure), ("Normalize color", self.normalize_color),
                             ("Fusion", self.algorithm), ("Focus radius", self.radius), ("Blend smoothing", self.smooth),
                             ("Pyramid levels", self.levels), ("Depth cleanup", self.cleanup), ("Output sharpening", self.sharpen)]: form.addRow(name, widget)
        self.run_btn = QPushButton("✦  Create Focus Stack"); self.run_btn.setObjectName("primaryButton"); self.run_btn.clicked.connect(self.start_stack)
        self.cancel_btn = QPushButton("■  Cancel"); self.cancel_btn.setObjectName("dangerButton"); self.cancel_btn.setEnabled(False); self.cancel_btn.clicked.connect(self.cancel_stack)
        self.pause_btn = QPushButton("Ⅱ  Pause"); self.pause_btn.setEnabled(False); self.pause_btn.clicked.connect(self.toggle_pause)
        form.addRow(self.run_btn, self.cancel_btn); form.addRow(self.pause_btn)
        left = QWidget(); lv = QVBoxLayout(left); lv.addWidget(QLabel("Focus sequence")); lv.addWidget(self.list, 1); lv.addWidget(controls)
        self.view = ImageView(); split = QSplitter(); split.addWidget(left); split.addWidget(self.view); split.setSizes([340, 1100])
        self.tabs = QTabWidget(); self.tabs.setObjectName("workflowTabs"); self.tabs.addTab(split, "◈  General Stacking")
        self._build_microscope_tab(); self._build_parameter_lab(); self._build_z_browser(); self._build_alignment_inspector(); self._build_retouch_tab(); self._build_comparison_tab(); self._build_performance_tab(); self._build_output_tab(); self._build_batch_tab(); self._build_history_tab(); self._build_debug_tab(); self.setCentralWidget(self.tabs)
        self.progress = QProgressBar(); self.progress.setMaximumWidth(260); self.statusBar().addPermanentWidget(self.progress); self.progress.hide()
        self.pixel_label = QLabel("x: —  y: —  RGB: —"); self.statusBar().addPermanentWidget(self.pixel_label); self.view.imageHovered.connect(self.show_pixel)

    def _build_microscope_tab(self):
        page = QWidget(); split = QSplitter(); layout = QVBoxLayout(page); layout.setContentsMargins(0, 0, 0, 0); layout.addWidget(split)
        panel = QWidget(); controls = QVBoxLayout(panel)
        heading = QLabel("MICROSCOPE 2D FOCUS STACKING"); heading.setObjectName("sectionHeading"); controls.addWidget(heading)
        intro = QLabel("A specimen-oriented workflow for brightfield, reflected-light, and similar 2D image sequences.\nUses the same imported sequence shown on the General Stacking tab.")
        intro.setWordWrap(True); intro.setObjectName("helpText"); controls.addWidget(intro)
        self.micro_count = QLabel("0 source images loaded"); self.micro_count.setObjectName("sequenceBadge"); controls.addWidget(self.micro_count)
        import_row = QHBoxLayout(); add_images = QPushButton("＋ Images"); add_images.clicked.connect(self.import_images); add_folder = QPushButton("▣ Folder"); add_folder.clicked.connect(self.import_folder)
        import_row.addWidget(add_images); import_row.addWidget(add_folder); controls.addLayout(import_row)
        form = QFormLayout()
        self.micro_preset = QComboBox(); self.micro_preset.addItems(["Brightfield detail", "Reflected light", "Maximum detail", "Gentle / low noise"]); self.micro_preset.currentTextChanged.connect(self.apply_micro_preset)
        self.micro_illumination = QCheckBox(); self.micro_illumination.setChecked(True)
        self.micro_sigma = QDoubleSpinBox(); self.micro_sigma.setRange(3, 250); self.micro_sigma.setValue(45); self.micro_sigma.setSuffix(" px")
        self.micro_hot_pixels = QCheckBox(); self.micro_hot_pixels.setChecked(True)
        self.micro_hot_strength = QDoubleSpinBox(); self.micro_hot_strength.setRange(2, 12); self.micro_hot_strength.setSingleStep(.5); self.micro_hot_strength.setValue(2.5)
        self.micro_contrast = QDoubleSpinBox(); self.micro_contrast.setRange(0, 2); self.micro_contrast.setSingleStep(.1); self.micro_contrast.setValue(0)
        self.micro_preserve = QCheckBox(); self.micro_preserve.setChecked(True)
        self.micro_flat = QLabel("Not selected"); flat_btn = QPushButton("Choose Flat Field…"); flat_btn.clicked.connect(self.choose_flat_field)
        self.micro_dark = QLabel("Not selected"); dark_btn = QPushButton("Choose Dark Frame…"); dark_btn.clicked.connect(self.choose_dark_frame)
        self.micro_focus_scale = QComboBox(); self.micro_focus_scale.addItems(["Smart", "Fine", "Medium", "Coarse"])
        self.micro_min_structure = QDoubleSpinBox(); self.micro_min_structure.setRange(0, 1); self.micro_min_structure.setSingleStep(.01); self.micro_min_structure.setValue(.08)
        self.micro_min_confidence = QDoubleSpinBox(); self.micro_min_confidence.setRange(0, 1); self.micro_min_confidence.setSingleStep(.01); self.micro_min_confidence.setValue(.12)
        self.micro_uncertain = QComboBox(); self.micro_uncertain.addItems(["Average", "Median", "Reference"])
        self.micro_patch = QSpinBox(); self.micro_patch.setRange(-15, 15); self.micro_patch.setValue(0); self.micro_patch.setToolTip("Negative shrinks focus patches; positive expands them.")
        self.micro_depth_preference = QDoubleSpinBox(); self.micro_depth_preference.setRange(-1, 1); self.micro_depth_preference.setSingleStep(.1); self.micro_depth_preference.setValue(0); self.micro_depth_preference.setToolTip("Negative prefers early/top frames; positive prefers late/bottom frames.")
        self.micro_color_selective = QCheckBox(); self.micro_color_space = QComboBox(); self.micro_color_space.addItems(["Lab", "HSV"])
        self.micro_color_tolerance = QDoubleSpinBox(); self.micro_color_tolerance.setRange(1, 150); self.micro_color_tolerance.setValue(25)
        self.micro_color_mix = QDoubleSpinBox(); self.micro_color_mix.setRange(0, 1); self.micro_color_mix.setSingleStep(.05); self.micro_color_mix.setValue(.5)
        self.micro_target_button = QPushButton("Choose Target Color…"); self.micro_target_button.clicked.connect(self.choose_target_color)
        self.micro_pick_color = QPushButton("Eyedropper: Click Preview"); self.micro_pick_color.clicked.connect(self.arm_target_eyedropper)
        self.micro_synthesize = QCheckBox(); self.micro_intermediates = QSpinBox(); self.micro_intermediates.setRange(1, 3); self.micro_scientific = QCheckBox(); self.micro_scientific.setChecked(True)
        self.micro_um_per_pixel = QDoubleSpinBox(); self.micro_um_per_pixel.setRange(0, 10000); self.micro_um_per_pixel.setDecimals(6); self.micro_um_per_pixel.setSuffix(" µm/px")
        self.micro_bar_length = QDoubleSpinBox(); self.micro_bar_length.setRange(.01, 100000); self.micro_bar_length.setValue(100); self.micro_bar_length.setSuffix(" µm")
        self.micro_bar_enabled = QCheckBox(); self.micro_bar_position = QComboBox(); self.micro_bar_position.addItems(["Bottom-right", "Bottom-left", "Top-right", "Top-left"])
        for label, widget in [("Microscope preset", self.micro_preset), ("Normalize illumination", self.micro_illumination),
                              ("Background scale", self.micro_sigma), ("Remove hot pixels", self.micro_hot_pixels),
                              ("Hot-pixel threshold", self.micro_hot_strength), ("Local contrast", self.micro_contrast),
                              ("Preserve brightness", self.micro_preserve), ("Flat-field calibration", flat_btn), ("Flat field", self.micro_flat),
                              ("Dark-frame calibration", dark_btn), ("Dark frame", self.micro_dark), ("Focus scale", self.micro_focus_scale),
                              ("Minimum structure", self.micro_min_structure), ("Minimum confidence", self.micro_min_confidence), ("Uncertain background", self.micro_uncertain),
                              ("Shrink / expand patches", self.micro_patch), ("Top / bottom preference", self.micro_depth_preference),
                              ("Color-selective fusion", self.micro_color_selective), ("Target color", self.micro_target_button), ("Target eyedropper", self.micro_pick_color), ("Color space", self.micro_color_space),
                              ("Color tolerance", self.micro_color_tolerance), ("Focus/color mix", self.micro_color_mix),
                              ("Synthetic intermediate planes", self.micro_synthesize), ("Planes per gap", self.micro_intermediates), ("Scientific Mode", self.micro_scientific),
                              ("Calibration", self.micro_um_per_pixel), ("Scale-bar length", self.micro_bar_length), ("Add scale bar on export", self.micro_bar_enabled), ("Scale-bar position", self.micro_bar_position)]: form.addRow(label, widget)
        controls.addLayout(form)
        note = QLabel("Recommended processing: ECC translation/affine registration and pyramid or depth-map fusion. Illumination correction estimates a broad background field; choose a scale larger than important specimen structures.")
        note.setWordWrap(True); note.setObjectName("helpCard"); controls.addWidget(note)
        controls.addStretch(1)
        self.micro_run_btn = QPushButton("✧  Build Microscope Stack"); self.micro_run_btn.setObjectName("microscopeButton"); self.micro_run_btn.clicked.connect(lambda: self.start_stack(True)); controls.addWidget(self.micro_run_btn)
        self.micro_view = ImageView(); self.micro_view.label.setText("Import images, then build a microscope stack"); self.micro_view.imageClicked.connect(self.capture_target_color)
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(panel); split.addWidget(scroll); split.addWidget(self.micro_view); split.setSizes([440, 1010]); self.tabs.addTab(page, "⌾  Microscope 2D")

    def _build_parameter_lab(self):
        page = QWidget(); layout = QVBoxLayout(page); tools = QHBoxLayout()
        run = QPushButton("▦  Test Four Parameter Sets"); run.setObjectName("primaryButton"); run.clicked.connect(self.run_parameter_lab)
        self.lab_choice = QComboBox(); self.lab_choice.addItems(["Fine / strict", "Smart / balanced", "Smart / smooth", "Coarse / clean"])
        apply_button = QPushButton("Use Selected Settings"); apply_button.clicked.connect(self.apply_lab_choice)
        tools.addWidget(run); tools.addWidget(QLabel("Promote:")); tools.addWidget(self.lab_choice); tools.addWidget(apply_button); tools.addStretch(1); layout.addLayout(tools)
        grid = QGridLayout(); self.lab_views = []
        for index, name in enumerate(["Fine / strict", "Smart / balanced", "Smart / smooth", "Coarse / clean"]):
            box = QWidget(); box_layout = QVBoxLayout(box); label = QLabel(name); label.setObjectName("sequenceBadge"); view = ImageView(); box_layout.addWidget(label); box_layout.addWidget(view, 1); grid.addWidget(box, index // 2, index % 2); self.lab_views.append(view)
        layout.addLayout(grid, 1); self.tabs.addTab(page, "▦  Microscope Lab")

    def _build_z_browser(self):
        page = QWidget(); layout = QVBoxLayout(page); tools = QHBoxLayout()
        natural = QPushButton("Natural Sort"); natural.clicked.connect(self.natural_sort_sequence); reverse = QPushButton("Reverse Stack"); reverse.clicked.connect(self.reverse_sequence)
        analyze = QPushButton("Analyze Order"); analyze.clicked.connect(self.analyze_focus_order); self.z_play = QPushButton("▶ Play"); self.z_play.clicked.connect(self.toggle_z_play)
        for button in (natural, reverse, analyze, self.z_play): tools.addWidget(button)
        tools.addStretch(1); self.z_status = QLabel("No sequence loaded"); tools.addWidget(self.z_status); layout.addLayout(tools)
        self.z_view = ImageView(); self.z_view.imageHovered.connect(self.show_pixel); layout.addWidget(self.z_view, 1)
        self.z_slider = QSlider(Qt.Horizontal); self.z_slider.setRange(0, 0); self.z_slider.valueChanged.connect(self.show_z_frame); layout.addWidget(self.z_slider)
        self.z_timer = QTimer(self); self.z_timer.setInterval(350); self.z_timer.timeout.connect(self.advance_z_frame); self.tabs.addTab(page, "↕  Z-Stack Browser")

    def _build_alignment_inspector(self):
        page = QWidget(); layout = QHBoxLayout(page); panel = QWidget(); form = QFormLayout(panel)
        self.inspect_frame = QComboBox(); self.inspect_mode = QComboBox(); self.inspect_mode.addItems(["Red / cyan overlay", "Difference heatmap", "Blink comparison"])
        self.inspect_dx = QDoubleSpinBox(); self.inspect_dx.setRange(-500, 500); self.inspect_dx.setDecimals(2); self.inspect_dx.setSuffix(" px")
        self.inspect_dy = QDoubleSpinBox(); self.inspect_dy.setRange(-500, 500); self.inspect_dy.setDecimals(2); self.inspect_dy.setSuffix(" px")
        self.inspect_rotation = QDoubleSpinBox(); self.inspect_rotation.setRange(-20, 20); self.inspect_rotation.setDecimals(3); self.inspect_rotation.setSuffix("°")
        self.inspect_scale = QDoubleSpinBox(); self.inspect_scale.setRange(50, 150); self.inspect_scale.setValue(100); self.inspect_scale.setSuffix(" %")
        self.inspect_enabled = QCheckBox(); self.inspect_enabled.setChecked(True); self.inspect_enabled.toggled.connect(self.set_inspected_frame_enabled)
        self.inspect_score = QLabel("Choose a frame and run inspection"); self.inspect_score.setObjectName("sequenceBadge")
        for label, widget in [("Moving frame", self.inspect_frame), ("View", self.inspect_mode), ("Include in stack", self.inspect_enabled),
                              ("Manual X", self.inspect_dx), ("Manual Y", self.inspect_dy), ("Manual rotation", self.inspect_rotation), ("Manual scale", self.inspect_scale)]: form.addRow(label, widget)
        inspect = QPushButton("◎  Analyze Alignment"); inspect.setObjectName("primaryButton"); inspect.clicked.connect(self.inspect_alignment); form.addRow(inspect)
        apply_manual = QPushButton("Apply Manual Preview"); apply_manual.clicked.connect(self.refresh_inspector_view); form.addRow(apply_manual)
        form.addRow(self.inspect_score); help_text = QLabel("Red/cyan edges or bright difference regions reveal misregistration. Disable a bad frame, or adjust the manual preview values before choosing a better automatic model.")
        help_text.setWordWrap(True); help_text.setObjectName("helpCard"); form.addRow(help_text)
        self.inspect_view = ImageView(); layout.addWidget(panel); layout.addWidget(self.inspect_view, 1); self.tabs.addTab(page, "◎  Alignment Inspector")
        self._blink_state = False; self._blink_timer = QTimer(self); self._blink_timer.setInterval(550); self._blink_timer.timeout.connect(self.refresh_inspector_view); self._blink_timer.start()

    def _build_retouch_tab(self):
        page = QWidget(); layout = QHBoxLayout(page); panel = QWidget(); form = QFormLayout(panel)
        self.retouch_source = QComboBox(); self.brush_size = QSpinBox(); self.brush_size.setRange(2, 300); self.brush_size.setValue(35); self.brush_size.setSuffix(" px")
        self.brush_hardness = QDoubleSpinBox(); self.brush_hardness.setRange(0, 1); self.brush_hardness.setSingleStep(.1); self.brush_hardness.setValue(.7)
        self.brush_opacity = QDoubleSpinBox(); self.brush_opacity.setRange(.05, 1); self.brush_opacity.setSingleStep(.05); self.brush_opacity.setValue(1)
        self.show_retouch_overlay = QCheckBox(); self.show_retouch_overlay.toggled.connect(self.refresh_retouch)
        for label, widget in [("Paint from frame", self.retouch_source), ("Brush size", self.brush_size), ("Hardness", self.brush_hardness), ("Opacity", self.brush_opacity), ("Show ownership overlay", self.show_retouch_overlay)]: form.addRow(label, widget)
        row = QHBoxLayout(); undo = QPushButton("↶ Undo"); undo.clicked.connect(self.retouch_undo); redo = QPushButton("↷ Redo"); redo.clicked.connect(self.retouch_redo); row.addWidget(undo); row.addWidget(redo); form.addRow(row)
        accept = QPushButton("✓ Use Retouched Result"); accept.setObjectName("microscopeButton"); accept.clicked.connect(self.accept_retouch); form.addRow(accept)
        help_text = QLabel("After completing a stack, click the image to paint aligned pixels from the selected source frame. Edits are nondestructive until accepted, with 30 levels of undo.")
        help_text.setWordWrap(True); help_text.setObjectName("helpCard"); form.addRow(help_text)
        self.retouch_view = ImageView(); self.retouch_view.label.setText("Create a stack before retouching"); self.retouch_view.imageClicked.connect(self.paint_retouch)
        layout.addWidget(panel); layout.addWidget(self.retouch_view, 1); self.tabs.addTab(page, "✎  Retouch")

    def _build_comparison_tab(self):
        page = QWidget(); layout = QVBoxLayout(page); tools = QHBoxLayout(); tools.addWidget(QLabel("Before/after comparison — drag the center divider"))
        for label, zoom in (("Fit", 0), ("100%", 1), ("200%", 2), ("400%", 4)):
            button = QPushButton(label); button.clicked.connect(lambda _=False, z=zoom: self.zoom_comparison(z)); tools.addWidget(button)
        full = QPushButton("Full Screen"); full.clicked.connect(self.toggle_fullscreen); tools.addWidget(full); tools.addStretch(1); layout.addLayout(tools)
        split = QSplitter(); self.before_view = ImageView(); self.before_view.label.setText("Select a source frame"); self.after_view = ImageView(); self.after_view.label.setText("Create a stack")
        self.before_view.imageHovered.connect(self.show_pixel); self.after_view.imageHovered.connect(self.show_pixel); split.addWidget(self.before_view); split.addWidget(self.after_view); split.setSizes([700, 700]); layout.addWidget(split, 1)
        self.tabs.addTab(page, "◐  Compare")

    def _build_performance_tab(self):
        page = QWidget(); layout = QVBoxLayout(page)
        processing = QWidget(); form = QFormLayout(processing)
        self.perf_preset = QComboBox(); self.perf_preset.addItems(["Balanced", "Large stack / low RAM", "Maximum speed", "Maximum compatibility"]); self.perf_preset.currentTextChanged.connect(self.apply_performance_preset)
        self.perf_description = QLabel(); self.perf_description.setWordWrap(True); self.perf_description.setObjectName("helpCard")
        self.tile_fusion = QCheckBox(); self.tile_fusion.setChecked(True); self.tile_size = QSpinBox(); self.tile_size.setRange(256, 4096); self.tile_size.setSingleStep(256); self.tile_size.setValue(1024); self.tile_size.setSuffix(" px")
        self.disk_cache = QCheckBox(); self.disk_cache.setChecked(True); self.gpu = QCheckBox(); self.gpu.setChecked(True)
        self.cpu_threads = QSpinBox(); self.cpu_threads.setRange(0, max(1, os.cpu_count() or 1)); self.cpu_threads.setSpecialValueText("Automatic")
        self.proxy_dimension = QSpinBox(); self.proxy_dimension.setRange(500, 6000); self.proxy_dimension.setValue(1800); self.proxy_dimension.setSuffix(" px")
        self.recover_frames = QCheckBox(); self.recover_frames.setChecked(True)
        for label, widget in [("Performance preset", self.perf_preset), ("Tiled fusion", self.tile_fusion), ("Tile size", self.tile_size), ("Disk-backed cache", self.disk_cache),
                              ("GPU/OpenCL when available", self.gpu), ("CPU threads", self.cpu_threads), ("Alignment proxy maximum", self.proxy_dimension), ("Recover failed frames", self.recover_frames)]: form.addRow(label, widget)
        form.addRow(self.perf_description); layout.addWidget(CollapsibleSection("Large-stack processing", processing)); layout.addStretch(1); self.apply_performance_preset("Balanced")
        self.tabs.addTab(page, "⚙  Performance")

    def _build_output_tab(self):
        page = QWidget(); layout = QVBoxLayout(page); settings = QWidget(); form = QFormLayout(settings)
        self.output_preset = QComboBox(); self.output_preset.addItems(["Archival", "Web", "Custom"]); self.output_preset.currentTextChanged.connect(self.apply_output_preset)
        self.output_description = QLabel(); self.output_description.setWordWrap(True); self.output_description.setObjectName("helpCard")
        self.output_bits = QComboBox(); self.output_bits.addItems(["16", "8"]); self.output_gray = QCheckBox(); self.output_alpha = QCheckBox(); self.output_icc = QCheckBox(); self.output_icc.setChecked(True); self.output_dpi = QCheckBox(); self.output_dpi.setChecked(True)
        self.output_resize = QDoubleSpinBox(); self.output_resize.setRange(1, 400); self.output_resize.setValue(100); self.output_resize.setSuffix(" %")
        self.output_bigtiff = QCheckBox(); self.output_bigtiff.setChecked(True); self.export_aligned = QCheckBox(); self.export_masks = QCheckBox(); self.export_depth = QCheckBox(); self.export_confidence = QCheckBox()
        for label, widget in [("Preset", self.output_preset), ("Bit depth", self.output_bits), ("Grayscale TIFF", self.output_gray), ("Optional alpha", self.output_alpha), ("Preserve ICC profile", self.output_icc),
                              ("Preserve DPI", self.output_dpi), ("Resize", self.output_resize), ("BigTIFF", self.output_bigtiff), ("Export aligned frames", self.export_aligned),
                              ("Export individual masks", self.export_masks), ("Export depth map", self.export_depth), ("Export confidence map", self.export_confidence)]: form.addRow(label, widget)
        form.addRow(self.output_description); layout.addWidget(CollapsibleSection("Output and auxiliary assets", settings)); layout.addStretch(1); self.apply_output_preset("Archival")
        self.tabs.addTab(page, "⇩  Output")

    def _build_history_tab(self):
        page = QWidget(); layout = QVBoxLayout(page); self.history_table = QTableWidget(0, 6); self.history_table.setHorizontalHeaderLabels(["Time", "Frames", "Algorithm", "Dimensions", "Duration", "Warnings"]); self.history_table.horizontalHeader().setStretchLastSection(True); layout.addWidget(self.history_table)
        self.tabs.addTab(page, "◷  History")

    def _build_batch_tab(self):
        page = QWidget(); layout = QVBoxLayout(page); self.batch_table = QTableWidget(0, 2); self.batch_table.setHorizontalHeaderLabels(["Project", "16-bit TIFF output"])
        self.batch_table.horizontalHeader().setStretchLastSection(True); layout.addWidget(self.batch_table, 1)
        row = QHBoxLayout(); add = QPushButton("＋ Add Projects"); add.clicked.connect(self.add_batch_projects); remove = QPushButton("− Remove Selected"); remove.clicked.connect(self.remove_batch_jobs)
        self.batch_run = QPushButton("▶ Run Batch"); self.batch_run.setObjectName("primaryButton"); self.batch_run.clicked.connect(self.start_batch)
        self.batch_cancel = QPushButton("■ Cancel Batch"); self.batch_cancel.setObjectName("dangerButton"); self.batch_cancel.setEnabled(False); self.batch_cancel.clicked.connect(self.cancel_batch)
        for widget in (add, remove, self.batch_run, self.batch_cancel): row.addWidget(widget)
        row.addStretch(1); layout.addLayout(row); self.batch_status = QLabel("Queue saved project files to process them sequentially with controlled memory use."); self.batch_status.setObjectName("helpCard"); layout.addWidget(self.batch_status)
        self.tabs.addTab(page, "▤  Batch Queue")

    def _build_debug_tab(self):
        page = QWidget(); layout = QVBoxLayout(page)
        heading = QLabel("PROCESSING & DEBUG CONSOLE"); heading.setObjectName("sectionHeading"); layout.addWidget(heading)
        self.debug_console = QPlainTextEdit(); self.debug_console.setObjectName("debugConsole"); self.debug_console.setReadOnly(True); self.debug_console.setLineWrapMode(QPlainTextEdit.NoWrap); layout.addWidget(self.debug_console, 1)
        row = QHBoxLayout(); clear = QPushButton("Clear"); clear.clicked.connect(self.debug_console.clear); copy = QPushButton("Copy All"); copy.clicked.connect(self.debug_console.selectAll); copy.clicked.connect(self.debug_console.copy)
        save = QPushButton("Save Log…"); save.clicked.connect(self.save_debug_log); diagnostic = QPushButton("Copy Diagnostic Package…"); diagnostic.clicked.connect(self.copy_diagnostic_package)
        row.addWidget(clear); row.addWidget(copy); row.addWidget(save); row.addWidget(diagnostic); row.addStretch(1); layout.addLayout(row)
        self.tabs.addTab(page, ">_  Debug Console"); self._debug("INFO", f"Focus Stacker Pro {__version__} initialized"); self._debug("SYSTEM", f"{platform.platform()} | CPU logical={os.cpu_count()} | OpenCL={cv2.ocl.haveOpenCL()}")

    def _debug(self, level, message):
        if hasattr(self, "debug_console"):
            stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.debug_console.appendPlainText(f"[{stamp}] {level:<7} {message}")
            bar = self.debug_console.verticalScrollBar(); bar.setValue(bar.maximum())

    def save_debug_log(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save debug log", "focus_stacker_debug.log", "Log files (*.log);;Text files (*.txt)")
        if path:
            try:
                Path(path).write_text(self.debug_console.toPlainText(), encoding="utf-8"); self._debug("INFO", f"Debug log saved: {path}")
            except Exception as exc: QMessageBox.critical(self, "Save error", str(exc))

    def apply_micro_preset(self, name):
        presets = {
            "Brightfield detail": (True, 45, True, 2.5, 0.2),
            "Reflected light": (True, 70, True, 3.0, 0.1),
            "Maximum detail": (True, 35, True, 3.5, 0.5),
            "Gentle / low noise": (True, 90, False, 4.0, 0.0),
        }
        if name in presets:
            illum, sigma, hot, strength, contrast = presets[name]
            self.micro_illumination.setChecked(illum); self.micro_sigma.setValue(sigma); self.micro_hot_pixels.setChecked(hot)
            self.micro_hot_strength.setValue(strength); self.micro_contrast.setValue(contrast)

    def choose_target_color(self):
        current = QColor(*self.project.microscope.target_color); color = QColorDialog.getColor(current, self, "Choose microscopy target color")
        if color.isValid():
            self.project.microscope.target_color = [color.red(), color.green(), color.blue()]; self.micro_target_button.setStyleSheet(f"background: {color.name()}; color: {'black' if color.lightness() > 140 else 'white'}")
    def arm_target_eyedropper(self): self._target_eyedropper = True; self.statusBar().showMessage("Click a target dye/fluorescence color in the microscope preview")
    def capture_target_color(self, x, y):
        if not getattr(self, "_target_eyedropper", False) or self.micro_view._array is None: return
        rgb = np.clip(self.micro_view._array[y, x, :3] * 255, 0, 255).astype(int).tolist(); self.project.microscope.target_color = rgb; color = QColor(*rgb); self.micro_target_button.setStyleSheet(f"background: {color.name()}; color: {'black' if color.lightness() > 140 else 'white'}"); self._target_eyedropper = False; self.statusBar().showMessage(f"Target color selected: {rgb}")

    def run_parameter_lab(self):
        if not self.details or not self.details.get("alignment"): QMessageBox.information(self, "No aligned stack", "Build a microscope stack first."); return
        self._sync(); images = [np.asarray(image) for image in self.details["alignment"].images]
        try:
            self.lab_results = parameter_comparison(images, self.project.microscope, lambda p, s: self._debug("LAB", f"{p}% {s}"))
            for view, (_, result, _) in zip(self.lab_views, self.lab_results): view.set_image(result); view.fit_image()
            self._debug("LAB", "Four microscope parameter variants completed")
        except Exception as exc: QMessageBox.critical(self, "Parameter lab failed", str(exc)); self._debug("ERROR", traceback.format_exc())

    def apply_lab_choice(self):
        if not hasattr(self, "lab_results"): return
        _, result, options = self.lab_results[self.lab_choice.currentIndex()]; self.project.microscope = options; self.result = result.copy(); self.micro_view.set_image(result); self.view.set_image(result); self.after_view.set_image(result); self._apply_project(); self._debug("LAB", f"Promoted {self.lab_choice.currentText()}")

    def natural_sort_sequence(self): self._replace_sequence(natural_sort(self.project.images)); self._debug("ORDER", "Applied natural-number filename sorting")
    def reverse_sequence(self): self._replace_sequence(list(reversed(self.project.images))); self._debug("ORDER", "Reversed focus sequence")
    def _replace_sequence(self, paths):
        self.project.images = list(paths); self.list.clear()
        for path in paths: self.list.addItem(self._sequence_item(path))
        self.refresh_sequence_controls(); self.validate_sequence(); self.list.setCurrentRow(0 if paths else -1)
    def analyze_focus_order(self):
        if len(self.project.images) < 3: QMessageBox.information(self, "Need frames", "Load at least three frames."); return
        try:
            images = []
            for path in self.project.images:
                image = to_float(read_image(path)[0]); scale = min(1, 800 / max(image.shape[:2])); images.append(cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else image)
            diagnostic = diagnose_focus_order(images); message = f"Trend: {diagnostic['trend']:.3f}\nLikely reversed: {diagnostic['likely_reversed']}\nIrregular transitions: {diagnostic['irregular_transitions'] or 'none'}"
            self.z_status.setText(message.replace("\n", "  |  ")); QMessageBox.information(self, "Focus-order analysis", message); self._debug("ORDER", json.dumps(diagnostic))
        except Exception as exc: QMessageBox.critical(self, "Order analysis failed", str(exc))
    def show_z_frame(self, index):
        if 0 <= index < len(self.project.images):
            try:
                image = to_float(read_image(self.project.images[index])[0]); old_zoom = self.z_view._zoom; self.z_view.set_image(image); self.z_view.set_zoom(old_zoom); self.z_status.setText(f"Frame {index + 1}/{len(self.project.images)} — {Path(self.project.images[index]).name}")
            except Exception as exc: self.z_status.setText(str(exc))
    def toggle_z_play(self):
        if self.z_timer.isActive(): self.z_timer.stop(); self.z_play.setText("▶ Play")
        else: self.z_timer.start(); self.z_play.setText("Ⅱ Pause")
    def advance_z_frame(self):
        if self.project.images: self.z_slider.setValue((self.z_slider.value() + 1) % len(self.project.images))

    def apply_performance_preset(self, name):
        presets = {
            "Balanced": (True, 1024, True, True, 0, 1800, True, "Recommended default: tiled fusion, disk cache, automatic CPU use, and GPU acceleration when OpenCL is available."),
            "Large stack / low RAM": (True, 512, True, False, max(1, (os.cpu_count() or 2) // 2), 1200, True, "Smaller tiles and disk caching reduce peak RAM at the cost of more disk traffic."),
            "Maximum speed": (True, 2048, False, True, 0, 2400, True, "Larger tiles and automatic threading favor speed when ample RAM is available."),
            "Maximum compatibility": (False, 1024, False, False, 1, 1200, False, "Disables GPU, disk cache, tiling, and recovery for the simplest execution path."),
        }
        if name in presets:
            tiled, size, disk, gpu, threads, proxy, recover, description = presets[name]
            self.tile_fusion.setChecked(tiled); self.tile_size.setValue(size); self.disk_cache.setChecked(disk); self.gpu.setChecked(gpu); self.cpu_threads.setValue(threads); self.proxy_dimension.setValue(proxy); self.recover_frames.setChecked(recover); self.perf_description.setText(description)

    def apply_output_preset(self, name):
        if name == "Archival": self.output_bits.setCurrentText("16"); self.output_resize.setValue(100); self.output_bigtiff.setChecked(True); self.output_description.setText("Lossless 16-bit master output with BigTIFF, ICC, and resolution metadata enabled.")
        elif name == "Web": self.output_bits.setCurrentText("8"); self.output_resize.setValue(50); self.output_bigtiff.setChecked(False); self.output_description.setText("Compact high-quality JPEG-oriented output, resized to 50% by default.")
        else: self.output_description.setText("Every output control is applied as configured.")

    def zoom_comparison(self, zoom):
        for view in (self.before_view, self.after_view): view.fit_image() if zoom == 0 else view.set_zoom(zoom)
    def show_pixel(self, x, y, value):
        values = np.atleast_1d(value); text = ", ".join(f"{float(v):.5f}" for v in values[:4]); self.pixel_label.setText(f"x: {x}  y: {y}  RGB/A: {text}")
    def toggle_fullscreen(self): self.showNormal() if self.isFullScreen() else self.showFullScreen()
    def toggle_pause(self):
        if not hasattr(self, "worker"): return
        self._paused = not self._paused
        if self._paused: self.worker.pause(); self.pause_btn.setText("▶  Resume"); self._debug("PAUSE", "Processing paused at the next safe checkpoint")
        else: self.worker.resume(); self.pause_btn.setText("Ⅱ  Pause"); self._debug("RESUME", "Processing resumed")

    def _install_shortcuts(self):
        for key, callback in (("Ctrl+O", self.import_images), ("Ctrl+Shift+O", self.import_folder), ("Ctrl+S", self.save_project), ("Ctrl+E", self.export_result), ("F11", self.toggle_fullscreen), ("Space", self.toggle_pause)):
            QShortcut(QKeySequence(key), self, callback)
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.acceptProposedAction()
    def dropEvent(self, event):
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls()]
        projects = [p for p in paths if p.suffix.lower() == ".json"]
        images = [str(p) for p in paths if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        for folder in [p for p in paths if p.is_dir()]: images.extend(str(f) for f in sorted(folder.iterdir()) if f.suffix.lower() in IMAGE_EXTENSIONS)
        if projects: self.load_project_path(str(projects[0]))
        if images: self._add(images)
        event.acceptProposedAction()

    def toggle_theme(self):
        light = not getattr(self, "_light_theme", False); self._light_theme = light
        if light:
            QApplication.instance().setStyleSheet("QMainWindow,QWidget{background:#f4f7fb;color:#172033;font-family:'Segoe UI';font-size:10pt} QToolBar,QTabBar::tab{background:#dce8f5;color:#172033;padding:8px} QTabBar::tab:selected{background:#0e7490;color:white} QListWidget,QPlainTextEdit,QScrollArea,QComboBox,QSpinBox,QDoubleSpinBox,QTableWidget{background:white;color:#172033;border:1px solid #9fb3c8} QPushButton{background:#e2e8f0;color:#172033;border:1px solid #829ab1;border-radius:7px;padding:8px} QPushButton:hover{background:#cde6f5} QPushButton#primaryButton{background:#0891b2;color:white}")
        else: QApplication.instance().setStyleSheet(self._dark_stylesheet)
        self.settings.setValue("light_theme", light)

    def open_recent_project(self):
        recent = self.settings.value("recent_projects", [], list)
        if not recent: QMessageBox.information(self, "Recent projects", "No recent projects yet."); return
        choice, ok = QInputDialog.getItem(self, "Recent projects", "Project", recent, 0, False)
        if ok and choice: self.load_project_path(choice)

    def copy_diagnostic_package(self):
        if not self.details: QMessageBox.information(self, "No diagnostics", "Run a stack first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save diagnostic package", "focus_stacker_diagnostics.zip", "ZIP archive (*.zip)")
        if not path: return
        payload = json.dumps(self.details["report"], indent=2)
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("processing_report.json", payload); archive.writestr("debug_console.log", self.debug_console.toPlainText()); archive.writestr("README.txt", "Focus Stacker Pro diagnostic package. Source image pixels are not included; paths and metadata may be present in the report.")
        QApplication.clipboard().setText(path); self._debug("DIAGNOSTIC", f"Package saved and path copied: {path}")

    def choose_flat_field(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose flat-field image", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.dng *.nef *.cr2 *.arw)")
        if path: self.project.microscope.flat_field_path = path; self.micro_flat.setText(Path(path).name); self.micro_flat.setToolTip(path)
    def choose_dark_frame(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose dark-frame image", "", "Images (*.tif *.tiff *.png *.jpg *.jpeg *.dng *.nef *.cr2 *.arw)")
        if path: self.project.microscope.dark_frame_path = path; self.micro_dark.setText(Path(path).name); self.micro_dark.setToolTip(path)

    def inspect_alignment(self):
        row = self.inspect_frame.currentIndex()
        if row < 0 or row >= len(self.project.images): return
        try:
            ref_i = 0 if self.reference.currentText().lower() == "first" else (len(self.project.images) - 1 if self.reference.currentText().lower() == "last" else len(self.project.images) // 2)
            if row == ref_i: self.inspect_score.setText("This is the reference frame"); return
            self._debug("INSPECT", f"Aligning {Path(self.project.images[row]).name} to frame {ref_i + 1}")
            ref = to_float(read_image(self.project.images[ref_i])[0]); moving = to_float(read_image(self.project.images[row])[0])
            result = align_stack([ref, moving], self.alignment.currentText().lower().replace(" ", "_"), 0, False,
                                 self.project.alignment.ecc_iterations, self.project.alignment.ecc_epsilon,
                                 self.multiscale.isChecked(), self.project.alignment.pyramid_scales)
            self._inspect_ref, self._inspect_moving = result.images; self.inspect_score.setText(f"Alignment score: {result.scores[1]:.5f}")
            self.refresh_inspector_view()
        except Exception as exc: self._debug("ERROR", traceback.format_exc()); QMessageBox.critical(self, "Inspection failed", str(exc))

    def refresh_inspector_view(self):
        if not hasattr(self, "_inspect_ref"): return
        ref, moving = self._inspect_ref, self._inspect_moving; h, w = ref.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), self.inspect_rotation.value(), self.inspect_scale.value() / 100)
        matrix[:, 2] += (self.inspect_dx.value(), self.inspect_dy.value()); moving = cv2.warpAffine(moving, matrix, (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT101)
        mode = self.inspect_mode.currentText()
        if mode.startswith("Red"):
            ref_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY); mov_gray = cv2.cvtColor(moving, cv2.COLOR_RGB2GRAY)
            shown = np.stack([ref_gray, mov_gray, mov_gray], axis=2)
        elif mode.startswith("Difference"):
            diff = np.mean(np.abs(ref - moving), axis=2); diff = diff / max(float(np.percentile(diff, 99)), 1e-6)
            shown = cv2.cvtColor(cv2.applyColorMap(np.clip(diff * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB) / 255
        else:
            self._blink_state = not self._blink_state; shown = moving if self._blink_state else ref
        self.inspect_view.set_image(np.clip(shown, 0, 1))

    def set_inspected_frame_enabled(self, enabled):
        row = self.inspect_frame.currentIndex()
        if 0 <= row < len(self.project.images):
            path = self.project.images[row]
            if enabled and path in self.project.disabled_images: self.project.disabled_images.remove(path)
            elif not enabled and path not in self.project.disabled_images: self.project.disabled_images.append(path)
            self._debug("FRAME", f"{'Enabled' if enabled else 'Disabled'} {Path(path).name}")

    def paint_retouch(self, x, y):
        if not hasattr(self, "retouch_session") or self.retouch_source.currentIndex() < 0: return
        self.retouch_session.paint(x, y, self.retouch_source.currentIndex(), self.brush_size.value(), self.brush_hardness.value(), self.brush_opacity.value()); self.refresh_retouch()
    def refresh_retouch(self):
        if hasattr(self, "retouch_session"): self.retouch_view.set_image(self.retouch_session.overlay() if self.show_retouch_overlay.isChecked() else self.retouch_session.result)
    def retouch_undo(self):
        if hasattr(self, "retouch_session"): self.retouch_session.undo(); self.refresh_retouch()
    def retouch_redo(self):
        if hasattr(self, "retouch_session"): self.retouch_session.redo(); self.refresh_retouch()
    def accept_retouch(self):
        if hasattr(self, "retouch_session"):
            self.result = self.retouch_session.result.copy(); self.view.set_image(self.result); self.micro_view.set_image(self.result); self._debug("RETOUCH", "Retouched result accepted")

    def add_batch_projects(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add saved projects", "", "Focus Stacker Projects (*.json)")
        for path in paths:
            row = self.batch_table.rowCount(); self.batch_table.insertRow(row); self.batch_table.setItem(row, 0, QTableWidgetItem(path)); self.batch_table.setItem(row, 1, QTableWidgetItem(str(Path(path).with_name(Path(path).stem + "_stack.tif"))))
    def remove_batch_jobs(self):
        for index in sorted({i.row() for i in self.batch_table.selectedIndexes()}, reverse=True): self.batch_table.removeRow(index)
    def start_batch(self):
        jobs = [(self.batch_table.item(i, 0).text(), self.batch_table.item(i, 1).text()) for i in range(self.batch_table.rowCount()) if self.batch_table.item(i, 0) and self.batch_table.item(i, 1)]
        if not jobs: QMessageBox.information(self, "Empty queue", "Add one or more saved project files."); return
        self.batch_thread = QThread(); self.batch_worker = BatchWorker(jobs); self.batch_worker.moveToThread(self.batch_thread); self.batch_thread.started.connect(self.batch_worker.run)
        self.batch_worker.progress.connect(self.on_batch_progress); self.batch_worker.finished.connect(self.on_batch_finished); self.batch_worker.failed.connect(self.on_batch_failed)
        self.batch_worker.finished.connect(self.batch_thread.quit); self.batch_worker.failed.connect(self.batch_thread.quit); self.batch_thread.finished.connect(self.batch_worker.deleteLater); self.batch_thread.finished.connect(self.batch_thread.deleteLater)
        self.batch_run.setEnabled(False); self.batch_cancel.setEnabled(True); self.progress.show(); self._debug("BATCH", f"Starting {len(jobs)} jobs"); self.batch_thread.start()
    def cancel_batch(self): self.batch_worker.cancel(); self._debug("BATCH", "Cancellation requested")
    def on_batch_progress(self, value, text): self.progress.setValue(value); self.batch_status.setText(text); self._debug("BATCH", text)
    def on_batch_finished(self, completed, total):
        self.batch_status.setText(f"Completed {completed} of {total} jobs"); self.batch_run.setEnabled(True); self.batch_cancel.setEnabled(False); self.progress.hide(); self._debug("BATCH", self.batch_status.text())
    def on_batch_failed(self, error):
        self.batch_run.setEnabled(True); self.batch_cancel.setEnabled(False); self.progress.hide(); self._debug("ERROR", error); QMessageBox.critical(self, "Batch failed", error)

    def _build_actions(self):
        bar = QToolBar("Main"); bar.setMovable(False); self.addToolBar(bar)
        actions = [("＋  Add Images", self.import_images), ("▣  Add Folder", self.import_folder), ("−  Remove", self.remove_images),
                   ("↑  Move Up", lambda: self.move(-1)), ("↓  Move Down", lambda: self.move(1)), ("◇  Save Project", self.save_project),
                   ("◆  Load Project", self.load_project), ("◉  Result", self.show_result), ("▥  Depth Map", self.show_depth),
                   ("▦  Scale Map", self.show_scale_map),
                   ("◷  Recent", self.open_recent_project), ("⇩  Export", self.export_result), ("≋  Alignment Report", self.report), ("{}  Save Report", self.export_report),
                   ("⌕  Diagnostics", self.copy_diagnostic_package), ("◑  Theme", self.toggle_theme), ("⛶  Full Screen", self.toggle_fullscreen)]
        for label, slot in actions:
            action = QAction(label, self); action.triggered.connect(slot); bar.addAction(action)

    def import_images(self):
        filters = "Images (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp *.dng *.nef *.cr2 *.cr3 *.arw *.orf *.rw2 *.raf)"
        paths, _ = QFileDialog.getOpenFileNames(self, "Import focus sequence", "", filters); self._add(paths)
    def import_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Import image folder")
        if folder: self._add([str(p) for p in sorted(Path(folder).iterdir()) if p.suffix.lower() in IMAGE_EXTENSIONS])
    def _add(self, paths):
        existing = set(self.project.images)
        for p in paths:
            if p not in existing:
                self.project.images.append(p); self.list.addItem(self._sequence_item(p)); existing.add(p); self._debug("IMPORT", p)
        self.micro_count.setText(f"{len(self.project.images)} source images loaded")
        self.refresh_sequence_controls()
        if self.list.count() and self.list.currentRow() < 0: self.list.setCurrentRow(0)
        self.validate_sequence()

    def load_initial_images(self, paths, microscope=False):
        """Load an image handoff after the window is visible."""
        valid = [str(Path(path).resolve()) for path in paths
                 if Path(path).is_file() and Path(path).suffix.lower() in IMAGE_EXTENSIONS]
        if valid:
            self._add(valid)
            self.statusBar().showMessage(f"Loaded {len(valid)} image(s) from PhotoLab")
        if microscope:
            self.tabs.setCurrentIndex(1)  # Microscope 2D follows General Stacking.
    def _sequence_item(self, path):
        item = QListWidgetItem(Path(path).name); item.setToolTip(path)
        try:
            image = to_float(read_image(path)[0]); small = cv2.resize(image, (72, 54), interpolation=cv2.INTER_AREA); data = np.clip(small * 255, 0, 255).astype(np.uint8)
            item.setIcon(QIcon(QPixmap.fromImage(QImage(data.data, 72, 54, data.strides[0], QImage.Format_RGB888).copy())))
        except Exception: item.setText("⚠  " + item.text()); item.setForeground(QColor("#f87171"))
        return item
    def validate_sequence(self):
        expected = None
        for i, path in enumerate(self.project.images):
            item = self.list.item(i)
            try:
                if not Path(path).exists(): raise FileNotFoundError("File is missing")
                shape = read_image(path)[0].shape[:2]
                if expected is None: expected = shape
                if shape != expected: raise ValueError(f"Dimensions {shape} do not match {expected}")
                item.setToolTip(f"{path}\n{shape[1]} × {shape[0]}")
            except Exception as exc:
                if not item.text().startswith("⚠"): item.setText("⚠  " + item.text())
                item.setForeground(QColor("#f87171")); item.setToolTip(f"{path}\nWARNING: {exc}")
                self._debug("WARNING", f"{path}: {exc}")
    def remove_images(self):
        for row in sorted([self.list.row(i) for i in self.list.selectedItems()], reverse=True): self.list.takeItem(row); self.project.images.pop(row)
        self.micro_count.setText(f"{len(self.project.images)} source images loaded")
        self.refresh_sequence_controls()
    def move(self, delta):
        row = self.list.currentRow(); new = row + delta
        if row < 0 or not 0 <= new < self.list.count(): return
        item = self.list.takeItem(row); self.list.insertItem(new, item); self.project.images.insert(new, self.project.images.pop(row)); self.list.setCurrentRow(new)
        self.refresh_sequence_controls()
    def refresh_sequence_controls(self):
        current = self.inspect_frame.currentIndex() if hasattr(self, "inspect_frame") else -1
        if hasattr(self, "inspect_frame"):
            self.inspect_frame.blockSignals(True); self.inspect_frame.clear(); self.inspect_frame.addItems([f"{i + 1}: {Path(p).name}" for i, p in enumerate(self.project.images)]); self.inspect_frame.setCurrentIndex(min(max(0, current), len(self.project.images) - 1)); self.inspect_frame.blockSignals(False)
        if hasattr(self, "retouch_source"):
            self.retouch_source.clear(); self.retouch_source.addItems([f"{i + 1}: {Path(p).name}" for i, p in enumerate(self.project.images) if p not in self.project.disabled_images])
        if hasattr(self, "z_slider"):
            self.z_slider.setRange(0, max(0, len(self.project.images) - 1)); self.z_slider.setValue(min(self.z_slider.value(), max(0, len(self.project.images) - 1)))
    def preview_source(self, row):
        if row >= 0:
            try:
                image = to_float(read_image(self.project.images[row])[0]); self.view.set_image(image); self.micro_view.set_image(image)
                self.statusBar().showMessage(self.project.images[row]); self._debug("PREVIEW", f"Frame {row + 1}: {self.project.images[row]} | {image.shape[1]}×{image.shape[0]} {image.dtype}")
            except Exception as e: QMessageBox.warning(self, "Preview error", str(e))
    def _sync(self):
        self.project.alignment.method = self.alignment.currentText().lower().replace(" ", "_")
        self.project.alignment.reference = self.reference.currentText().lower(); self.project.alignment.crop_common = self.crop.isChecked()
        self.project.alignment.multiscale = self.multiscale.isChecked()
        self.project.stack.algorithm = self.algorithm.currentText().lower().replace(" ", "_")
        self.project.stack.focus_radius = self.radius.value(); self.project.stack.smooth_radius = self.smooth.value()
        self.project.stack.pyramid_levels = self.levels.value(); self.project.stack.cleanup_radius = self.cleanup.value(); self.project.stack.sharpen = self.sharpen.value()
        self.project.stack.normalize_exposure = self.normalize_exposure.isChecked(); self.project.stack.normalize_color = self.normalize_color.isChecked()
        micro = self.project.microscope; micro.illumination_normalization = self.micro_illumination.isChecked(); micro.background_sigma = self.micro_sigma.value()
        micro.hot_pixel_cleanup = self.micro_hot_pixels.isChecked(); micro.hot_pixel_strength = self.micro_hot_strength.value()
        micro.contrast_boost = self.micro_contrast.value(); micro.preserve_brightness = self.micro_preserve.isChecked()
        micro.focus_scale_mode = self.micro_focus_scale.currentText().lower(); micro.minimum_structure = self.micro_min_structure.value(); micro.minimum_confidence = self.micro_min_confidence.value(); micro.uncertain_mode = self.micro_uncertain.currentText().lower(); micro.patch_morphology = self.micro_patch.value(); micro.depth_preference = self.micro_depth_preference.value()
        micro.color_selective = self.micro_color_selective.isChecked(); micro.color_space = self.micro_color_space.currentText(); micro.color_tolerance = self.micro_color_tolerance.value(); micro.color_focus_mix = self.micro_color_mix.value(); micro.synthesize_intermediate = self.micro_synthesize.isChecked(); micro.intermediate_count = self.micro_intermediates.value(); micro.scientific_mode = self.micro_scientific.isChecked()
        micro.microns_per_pixel = self.micro_um_per_pixel.value(); micro.scale_bar_microns = self.micro_bar_length.value(); micro.scale_bar_enabled = self.micro_bar_enabled.isChecked(); micro.scale_bar_position = self.micro_bar_position.currentText().lower()
        perf = self.project.performance; perf.tiled_fusion = self.tile_fusion.isChecked(); perf.tile_size = self.tile_size.value(); perf.disk_cache = self.disk_cache.isChecked(); perf.use_gpu = self.gpu.isChecked(); perf.cpu_threads = self.cpu_threads.value(); perf.alignment_proxy_dimension = self.proxy_dimension.value(); perf.recover_failed_frames = self.recover_frames.isChecked()
        output = self.project.output; output.preset = self.output_preset.currentText().lower(); output.bit_depth = int(self.output_bits.currentText()); output.grayscale = self.output_gray.isChecked(); output.include_alpha = self.output_alpha.isChecked(); output.preserve_icc = self.output_icc.isChecked(); output.preserve_dpi = self.output_dpi.isChecked(); output.resize_percent = self.output_resize.value(); output.bigtiff = self.output_bigtiff.isChecked(); output.export_aligned = self.export_aligned.isChecked(); output.export_masks = self.export_masks.isChecked(); output.export_depth = self.export_depth.isChecked(); output.export_confidence = self.export_confidence.isChecked()
    def start_stack(self, microscope=False):
        if self.details and self.details.get("cache"): self.details["cache"].close(); self.details["cache"] = None
        active = [p for p in self.project.images if p not in self.project.disabled_images]
        if len(active) < 2: QMessageBox.information(self, "Need images", "Import and enable at least two images first."); return
        self._sync(); self.project.microscope.enabled = microscope
        try:
            sample = read_image(active[0])[0]; estimate = estimate_memory_bytes(len(active), sample.shape[0], sample.shape[1], self.project.stack.algorithm, self.project.stack.pyramid_levels)
            self._debug("MEMORY", f"Estimated peak working set: {human_bytes(estimate)}")
            megapixel_frames = len(active) * sample.shape[0] * sample.shape[1] / 1e6; seconds_estimate = max(2, megapixel_frames * (0.12 if self.project.performance.use_gpu else 0.2)); self._debug("ETA", f"Initial processing-time estimate: {seconds_estimate:.0f}s for {megapixel_frames:.1f} megapixel-frames"); self.statusBar().showMessage(f"Estimated time: {seconds_estimate:.0f}s — estimated memory: {human_bytes(estimate)}")
            try:
                import psutil
                warning_limit = int(psutil.virtual_memory().available * .70)
            except Exception: warning_limit = 8 * 1024 ** 3
            if estimate > warning_limit and QMessageBox.warning(self, "Large memory requirement", f"This stack may require approximately {human_bytes(estimate)}, above 70% of currently available RAM. Tiled fusion and disk cache are recommended. Continue?", QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes: return
        except Exception as exc: self._debug("WARNING", f"Memory estimate unavailable: {exc}")
        self._debug("START", f"{'Microscope 2D' if microscope else 'General'} stack | {len(self.project.images)} frames")
        self._debug("CONFIG", f"alignment={self.project.alignment.method}, reference={self.project.alignment.reference}, fusion={self.project.stack.algorithm}")
        if microscope: self._debug("MICRO", f"illumination={self.project.microscope.illumination_normalization}, background_sigma={self.project.microscope.background_sigma}, hot_pixels={self.project.microscope.hot_pixel_cleanup}, local_contrast={self.project.microscope.contrast_boost}")
        self.thread = QThread(); self.worker = StackWorker(self.project, microscope); self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run); self.worker.progress.connect(self.on_progress); self.worker.finished.connect(self.on_finished); self.worker.failed.connect(self.on_failed)
        self.worker.finished.connect(self.thread.quit); self.worker.failed.connect(self.thread.quit); self.thread.finished.connect(self.worker.deleteLater); self.thread.finished.connect(self.thread.deleteLater)
        self.run_btn.setEnabled(False); self.micro_run_btn.setEnabled(False); self.cancel_btn.setEnabled(True); self.pause_btn.setEnabled(True); self.progress.show(); self.thread.start()
    def cancel_stack(self):
        if hasattr(self, "worker"): self.worker.cancel(); self.statusBar().showMessage("Cancelling…"); self._debug("CANCEL", "Cancellation requested")
    def on_progress(self, value, text): self.progress.setValue(value); self.statusBar().showMessage(text); self._debug("PROGRESS", f"{value:3d}%  {text}")
    def on_finished(self, result, depth, details):
        self.result, self.depth, self.details = result, depth, details; self.confidence = details.get("confidence"); self.view.set_image(result); self.micro_view.set_image(result); self.after_view.set_image(result)
        if details.get("alignment") and details["alignment"].images: self.before_view.set_image(details["alignment"].images[len(details["alignment"].images) // 2])
        self.retouch_session = RetouchSession(result, details["alignment"].images); self.retouch_view.set_image(result); self.refresh_sequence_controls()
        self.add_history(details)
        self._debug("SUCCESS", f"Stack complete | output={result.shape[1]}×{result.shape[0]} | microscope={details.get('microscope', False)}"); self._done("Stack complete")
        for name, seconds in details.get("timings", {}).items(): self._debug("TIMING", f"{name}={seconds:.4f}s")
        self._debug("ACCEL", json.dumps(details.get("acceleration", {}), sort_keys=True)); self._debug("MEMORY", f"Peak estimate {human_bytes(details.get('peak_memory_estimate', 0))}")
        for warning in details.get("warnings", []): self._debug("WARNING", warning)
        for i, matrix in enumerate(details["alignment"].transforms): self._debug("MATRIX", f"frame={i + 1} score={details['alignment'].scores[i]:.6f} transform={np.array2string(matrix, precision=5)}")
    def on_failed(self, message): self._debug("ERROR", message); self._done("Stopped"); QMessageBox.critical(self, "Processing error", message)
    def _done(self, message): self.run_btn.setEnabled(True); self.micro_run_btn.setEnabled(True); self.cancel_btn.setEnabled(False); self.pause_btn.setEnabled(False); self._paused = False; self.pause_btn.setText("Ⅱ  Pause"); self.progress.hide(); self.statusBar().showMessage(message)
    def add_history(self, details):
        row = self.history_table.rowCount(); self.history_table.insertRow(row); shape = self.result.shape
        values = [datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(len(details.get("active_paths", []))), self.project.stack.algorithm, f"{shape[1]}×{shape[0]}", f"{details['timings'].get('total', 0):.2f}s", str(len(details.get("warnings", [])))]
        for column, value in enumerate(values): self.history_table.setItem(row, column, QTableWidgetItem(value))
    def show_result(self):
        if self.result is not None: self.view.set_image(self.result); self.micro_view.set_image(self.result)
    def show_depth(self):
        if self.depth is not None:
            unlocalized = self.depth == np.iinfo(np.uint16).max; count = self.details.get("effective_frame_count", len(self.project.images)) if self.details else len(self.project.images); d = np.where(unlocalized, 0, self.depth).astype(np.float32) / max(1, count - 1); colored = cv2.cvtColor(cv2.applyColorMap(np.clip(d * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB) / 255; colored[unlocalized] = .45
            self.view.set_image(colored); self.micro_view.set_image(colored)
    def show_scale_map(self):
        diagnostics = self.details.get("microscope_diagnostics", {}) if self.details else {}; scale = diagnostics.get("scale_map")
        if scale is None: QMessageBox.information(self, "No scale map", "Run a Smart microscope stack first."); return
        maximum = max(1, int(np.max(scale))); colored = cv2.cvtColor(cv2.applyColorMap(np.round(scale / maximum * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB) / 255; self.micro_view.set_image(colored); self.view.set_image(colored)
    def export_result(self):
        if self.result is None: QMessageBox.information(self, "No result", "Run a stack first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Export stack", "focus_stack.tif", "TIFF (*.tif *.tiff);;PNG (*.png);;JPEG (*.jpg)")
        if path:
            try:
                self._sync(); metadata = self.details["metadata"][0] if self.details and self.details.get("metadata") else {}
                export_image = draw_scale_bar(self.result, self.project.microscope.microns_per_pixel, self.project.microscope.scale_bar_microns, self.project.microscope.scale_bar_color, self.project.microscope.scale_bar_position) if self.project.microscope.scale_bar_enabled else self.result
                alpha = self.confidence if self.project.output.include_alpha else None; write_advanced(path, export_image, self.project.output, metadata, alpha)
                masks = None
                if self.project.output.export_masks and self.depth is not None: masks = [(self.depth == i).astype(np.float32) for i in range(self.details.get("effective_frame_count", len(self.details["alignment"].images)))]
                assets = export_auxiliary(path, self.details["alignment"].images, self.depth, self.confidence, masks, self.project.output)
                self.statusBar().showMessage(f"Saved {path}"); self._debug("EXPORT", f"{path}; {len(assets)} auxiliary assets")
            except Exception as e: QMessageBox.critical(self, "Export error", str(e))
    def export_report(self):
        if not self.details or "report" not in self.details: QMessageBox.information(self, "No report", "Run a stack first."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save processing report", "focus_stack_report.json", "JSON report (*.json)")
        if path: save_report(path, self.details["report"]); self._debug("REPORT", f"Saved {path}")
    def save_project(self):
        self._sync(); path, _ = QFileDialog.getSaveFileName(self, "Save project", "stack.focusstack.json", "Focus Stacker Project (*.json)")
        if path:
            self.project.save(path); self._debug("PROJECT", f"Saved {path}"); recent = [path] + [p for p in self.settings.value("recent_projects", [], list) if p != path]; self.settings.setValue("recent_projects", recent[:10])
    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load project", "", "Focus Stacker Project (*.json)")
        if path:
            self.load_project_path(path)
    def load_project_path(self, path):
        self.project = Project.load(path); self.list.clear()
        for p in self.project.images: self.list.addItem(self._sequence_item(p))
        self.micro_count.setText(f"{len(self.project.images)} source images loaded"); self.refresh_sequence_controls(); self._apply_project(); self.validate_sequence(); self._debug("PROJECT", f"Loaded {path}")
        recent = [path] + [p for p in self.settings.value("recent_projects", [], list) if p != path and Path(p).exists()]; self.settings.setValue("recent_projects", recent[:10])
    def _apply_project(self):
        def choose(combo, value):
            i = combo.findText(value.replace("_", " ").title()); combo.setCurrentIndex(max(0, i))
        choose(self.alignment, self.project.alignment.method); choose(self.reference, self.project.alignment.reference); choose(self.algorithm, self.project.stack.algorithm)
        self.crop.setChecked(self.project.alignment.crop_common); self.radius.setValue(self.project.stack.focus_radius); self.smooth.setValue(self.project.stack.smooth_radius)
        self.multiscale.setChecked(self.project.alignment.multiscale); self.normalize_exposure.setChecked(self.project.stack.normalize_exposure); self.normalize_color.setChecked(self.project.stack.normalize_color)
        self.levels.setValue(self.project.stack.pyramid_levels); self.cleanup.setValue(self.project.stack.cleanup_radius); self.sharpen.setValue(self.project.stack.sharpen)
        micro = self.project.microscope; self.micro_illumination.setChecked(micro.illumination_normalization); self.micro_sigma.setValue(micro.background_sigma)
        self.micro_hot_pixels.setChecked(micro.hot_pixel_cleanup); self.micro_hot_strength.setValue(micro.hot_pixel_strength); self.micro_contrast.setValue(micro.contrast_boost); self.micro_preserve.setChecked(micro.preserve_brightness)
        self.micro_flat.setText(Path(micro.flat_field_path).name if micro.flat_field_path else "Not selected"); self.micro_dark.setText(Path(micro.dark_frame_path).name if micro.dark_frame_path else "Not selected")
        self.micro_focus_scale.setCurrentText(micro.focus_scale_mode.title()); self.micro_min_structure.setValue(micro.minimum_structure); self.micro_min_confidence.setValue(micro.minimum_confidence); self.micro_uncertain.setCurrentText(micro.uncertain_mode.title()); self.micro_patch.setValue(micro.patch_morphology); self.micro_depth_preference.setValue(micro.depth_preference)
        self.micro_color_selective.setChecked(micro.color_selective); self.micro_color_space.setCurrentText(micro.color_space); self.micro_color_tolerance.setValue(micro.color_tolerance); self.micro_color_mix.setValue(micro.color_focus_mix); self.micro_synthesize.setChecked(micro.synthesize_intermediate); self.micro_intermediates.setValue(micro.intermediate_count); self.micro_scientific.setChecked(micro.scientific_mode)
        self.micro_um_per_pixel.setValue(micro.microns_per_pixel); self.micro_bar_length.setValue(micro.scale_bar_microns); self.micro_bar_enabled.setChecked(micro.scale_bar_enabled); self.micro_bar_position.setCurrentText(micro.scale_bar_position.title())
        perf = self.project.performance; self.tile_fusion.setChecked(perf.tiled_fusion); self.tile_size.setValue(perf.tile_size); self.disk_cache.setChecked(perf.disk_cache); self.gpu.setChecked(perf.use_gpu); self.cpu_threads.setValue(perf.cpu_threads); self.proxy_dimension.setValue(perf.alignment_proxy_dimension); self.recover_frames.setChecked(perf.recover_failed_frames)
        output = self.project.output; self.output_preset.setCurrentText(output.preset.title()); self.output_bits.setCurrentText(str(output.bit_depth)); self.output_gray.setChecked(output.grayscale); self.output_alpha.setChecked(output.include_alpha); self.output_icc.setChecked(output.preserve_icc); self.output_dpi.setChecked(output.preserve_dpi); self.output_resize.setValue(output.resize_percent); self.output_bigtiff.setChecked(output.bigtiff); self.export_aligned.setChecked(output.export_aligned); self.export_masks.setChecked(output.export_masks); self.export_depth.setChecked(output.export_depth); self.export_confidence.setChecked(output.export_confidence)
    def report(self):
        if not self.details: QMessageBox.information(self, "No report", "Run a stack first."); return
        a = self.details["alignment"]; text = "Alignment quality (higher is better):\n" + "\n".join(f"{Path(p).name}: {s:.4f}" for p, s in zip(self.project.images, a.scores)) + f"\n\nCommon crop: {a.common_roi}"
        QMessageBox.information(self, "Alignment report", text)

    def closeEvent(self, event):
        if self.details and self.details.get("cache"): self.details["cache"].close()
        event.accept()


def main(argv=None) -> int:
    args = parse_launch_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = QApplication.instance() or QApplication([]); app.setApplicationName("Focus Stacker Pro")
    app.setStyleSheet("""
        QMainWindow, QWidget {
            background-color: #0f172a;
            color: #f1f5f9;
            font-family: "Segoe UI", sans-serif;
            font-size: 10pt;
        }
        QToolBar {
            background: #16213a;
            border: none;
            border-bottom: 1px solid #334155;
            spacing: 5px;
            padding: 7px;
        }
        QToolButton {
            background: transparent;
            color: #dbeafe;
            border: 1px solid transparent;
            border-radius: 7px;
            padding: 7px 9px;
            font-weight: 600;
        }
        QToolButton:hover { background: #243656; border-color: #38bdf8; color: white; }
        QToolButton:pressed { background: #0e7490; }
        QTabWidget::pane { border: 1px solid #334155; border-radius: 8px; top: -1px; }
        QTabBar::tab {
            background: #16213a; color: #94a3b8; border: 1px solid #334155;
            border-bottom: none; padding: 10px 20px; margin-right: 3px;
            border-top-left-radius: 8px; border-top-right-radius: 8px; font-weight: 700;
        }
        QTabBar::tab:hover { background: #243656; color: white; }
        QTabBar::tab:selected { background: #0e7490; color: white; border-color: #22d3ee; }
        QLabel { color: #e2e8f0; background: transparent; }
        QLabel#sectionHeading { color: #67e8f9; font-size: 14pt; font-weight: 800; padding: 8px 0; }
        QLabel#helpText { color: #cbd5e1; padding-bottom: 8px; }
        QLabel#sequenceBadge { background: #1e3a5f; color: #bae6fd; border: 1px solid #0284c7; border-radius: 8px; padding: 8px; font-weight: 700; }
        QLabel#helpCard { background: #172554; color: #bfdbfe; border-left: 4px solid #38bdf8; border-radius: 6px; padding: 10px; }
        QListWidget, QScrollArea {
            background: #111c31;
            color: #f8fafc;
            border: 1px solid #334155;
            border-radius: 9px;
            selection-background-color: #0e7490;
            selection-color: white;
            outline: none;
        }
        QListWidget::item { padding: 7px; border-bottom: 1px solid #1e293b; }
        QListWidget::item:hover { background: #1e3a5f; }
        QComboBox, QSpinBox, QDoubleSpinBox {
            background: #1e293b;
            color: #f8fafc;
            border: 1px solid #475569;
            border-radius: 6px;
            padding: 6px 8px;
            min-height: 20px;
        }
        QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: #38bdf8; }
        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 2px solid #22d3ee; }
        QComboBox QAbstractItemView { background: #1e293b; color: white; selection-background-color: #0e7490; }
        QCheckBox { spacing: 8px; color: #f1f5f9; }
        QCheckBox::indicator { width: 18px; height: 18px; border: 1px solid #64748b; border-radius: 5px; background: #1e293b; }
        QCheckBox::indicator:checked { background: #14b8a6; border-color: #5eead4; }
        QPushButton {
            background: #334155;
            color: white;
            border: 1px solid #475569;
            border-radius: 8px;
            padding: 9px 14px;
            font-weight: 700;
        }
        QPushButton:hover { background: #475569; border-color: #7dd3fc; }
        QPushButton:pressed { background: #1e293b; padding-top: 10px; padding-bottom: 8px; }
        QPushButton#primaryButton {
            background: #0891b2;
            border: 1px solid #67e8f9;
            color: white;
            font-size: 11pt;
            padding: 11px 16px;
        }
        QPushButton#primaryButton:hover { background: #06b6d4; border-color: #cffafe; }
        QPushButton#primaryButton:pressed { background: #0e7490; }
        QPushButton#dangerButton { background: #7f1d1d; border-color: #ef4444; }
        QPushButton#dangerButton:hover { background: #b91c1c; border-color: #fca5a5; }
        QPushButton#microscopeButton { background: #6d28d9; border: 1px solid #c4b5fd; color: white; font-size: 11pt; padding: 12px; }
        QPushButton#microscopeButton:hover { background: #7c3aed; border-color: #ede9fe; }
        QPushButton:disabled { background: #253047; color: #64748b; border-color: #334155; }
        QPlainTextEdit#debugConsole {
            background: #050b14; color: #86efac; border: 1px solid #0e7490; border-radius: 8px;
            padding: 12px; font-family: "Cascadia Mono", "Consolas", monospace; font-size: 10pt;
            selection-background-color: #164e63; selection-color: white;
        }
        QProgressBar {
            background: #1e293b;
            color: white;
            border: 1px solid #475569;
            border-radius: 7px;
            text-align: center;
            min-height: 18px;
        }
        QProgressBar::chunk { background: #14b8a6; border-radius: 6px; }
        QStatusBar { background: #0b1220; color: #cbd5e1; border-top: 1px solid #334155; }
        QSplitter::handle { background: #334155; width: 2px; }
        QScrollBar:vertical { background: #111827; width: 12px; margin: 0; }
        QScrollBar::handle:vertical { background: #475569; border-radius: 6px; min-height: 28px; }
        QScrollBar::handle:vertical:hover { background: #0891b2; }
        QToolTip { background: #f8fafc; color: #0f172a; border: 1px solid #38bdf8; padding: 5px; }
    """)
    window = MainWindow(); window.show()
    if args.images or args.microscope:
        QTimer.singleShot(0, lambda: window.load_initial_images(args.images, args.microscope))
    return app.exec()


if __name__ == "__main__": raise SystemExit(main())
