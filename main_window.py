"""main_window.py — PhotoLab main window, DxO-inspired layout.

Right panel category tabs (icons): Light | Color | Detail | Geometry | Effects
Collapsible correction groups matching DxO PhotoLab structure.
"""

from __future__ import annotations

import os
import json
import cv2
import numpy as np

from PyQt6.QtCore import Qt, QTimer, QSize, QRect, QDir, QSettings
from PyQt6.QtGui import QIcon, QAction, QKeySequence, QFont, QFileSystemModel, QColor
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QScrollArea, QListWidget, QListWidgetItem, QFileDialog, QToolBar,
    QGroupBox, QPushButton, QSplitter, QStatusBar, QComboBox, QTabWidget,
    QCheckBox, QFrame, QLineEdit, QToolButton, QSizePolicy, QMessageBox, QStackedWidget, QButtonGroup, QMenu,
    QTreeView, QTextEdit, QTextBrowser, QDockWidget, QPlainTextEdit, QApplication,
    QDialog, QDialogButtonBox, QFormLayout, QInputDialog, QProgressDialog,
    QSpinBox, QDoubleSpinBox, QSlider,
)

VIDEO_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v",
    ".mpeg", ".mpg", ".wmv", ".3gp", ".insv", ".lrv",
}

from imaging import (Recipe, apply_recipe, IMAGE_EXTS, load_image, is_raw,
                     load_recipe_sidecar, save_recipe_sidecar, apply_watermark,
                     detect_architectural_upright, normalize_keystone_points)
from presets import load_preset_file, apply_preset_file, list_preset_files, PRESET_MODULE_FIELDS
from qt_utils import cv_to_qpixmap
from workers import ThumbnailWorker, ExportWorker, LoadImageWorker, PreviewRenderWorker, SdImportWorker, CatalogScanWorker, CatalogThumbWorker, HdrMergeWorker, BatchExportWorker, FocusStackWorker, PanoramaWorker
from widgets import HistogramWidget, SliderRow, ImageCanvas, ToneCurveWidget, ColorWheelWidget, HistoryWidget, NavigatorWidget
from catalog import Catalog
import sys
from datetime import datetime


def collapsible_group(title: str, parent_layout, checked=True):
    """DxO-style collapsible group with enable checkbox."""
    box = QGroupBox()
    box.setCheckable(True)
    box.setChecked(checked)
    box.setTitle(title)
    box.setStyleSheet("""
        QGroupBox {
            color: #ccc; border: 1px solid #3a3a3a; border-radius: 4px;
            margin-top: 8px; font-weight: 600; font-size: 12px;
            padding-top: 8px;
        }
        QGroupBox::title {
            subcontrol-origin: margin; left: 8px; padding: 0 4px;
        }
        QGroupBox::indicator {
            width: 13px; height: 13px;
            border: 1px solid #666; border-radius: 2px; background: #2a2a2a;
        }
        QGroupBox::indicator:checked {
            background: #2a6ad4; border-color: #2a6ad4;
        }
    """)
    inner = QVBoxLayout(box)
    inner.setSpacing(4)
    inner.setContentsMargins(8, 12, 8, 8)
    parent_layout.addWidget(box)
    return box, inner




class PresetBrowserDialog(QDialog):
    """Browse and apply presets from the plugin folder (and optional extra dirs)."""

    def __init__(self, parent, plugin_dir: str, extra_dirs=None):
        super().__init__(parent)
        self.setWindowTitle("Preset Browser")
        self.setMinimumSize(520, 480)
        self.resize(560, 560)
        self._plugin_dir = plugin_dir or ""
        self._extra = list(extra_dirs or [])
        self.selected_path = None
        self._settings = QSettings("PhotoLab", "PhotoLab")
        self._favorites = set(filter(None, str(self._settings.value("preset_favorites", "")).split("\n")))

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        hdr = QLabel("Presets")
        hdr.setStyleSheet("font-size:16px; font-weight:700; color:#eee;")
        root.addWidget(hdr)

        path_row = QHBoxLayout()
        self.path_lbl = QLabel(self._plugin_dir or "(no plugin folder)")
        self.path_lbl.setStyleSheet("color:#888; font-size:11px;")
        self.path_lbl.setWordWrap(True)
        path_row.addWidget(self.path_lbl, 1)
        browse_btn = QPushButton("Folder…")
        browse_btn.setToolTip("Choose another preset folder")
        browse_btn.clicked.connect(self._pick_folder)
        path_row.addWidget(browse_btn)
        root.addLayout(path_row)

        filt_row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search presets…")
        self.search.textChanged.connect(self._filter)
        filt_row.addWidget(self.search, 1)
        self.type_combo = QComboBox()
        self.type_combo.addItem("All types", "all")
        self.type_combo.addItem("JSON", "json")
        self.type_combo.addItem("XMP", "xmp")
        self.type_combo.currentIndexChanged.connect(self._filter)
        filt_row.addWidget(self.type_combo)
        self.category_combo = QComboBox()
        self.category_combo.addItem("All categories", "all")
        self.category_combo.addItem("Favorites", "favorites")
        self.category_combo.currentIndexChanged.connect(self._filter)
        filt_row.addWidget(self.category_combo)
        root.addLayout(filt_row)

        self.list = QListWidget()
        self.list.setStyleSheet(
            "QListWidget { background:#161616; border:1px solid #333; border-radius:6px; color:#ddd; font-size:13px; }"
            "QListWidget::item { padding:8px 10px; border-bottom:1px solid #222; }"
            "QListWidget::item:selected { background:#2a5080; color:#fff; }"
            "QListWidget::item:hover { background:#1e1e28; }"
        )
        self.list.itemDoubleClicked.connect(self._accept_item)
        self.list.currentItemChanged.connect(self._on_sel)
        root.addWidget(self.list, 1)

        self.detail = QLabel("Select a preset to apply to the current image.")
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("color:#aaa; font-size:12px; padding:4px;")
        root.addWidget(self.detail)

        preset_controls = QGroupBox("Preset controls")
        controls = QVBoxLayout(preset_controls)
        strength_row = QHBoxLayout()
        strength_row.addWidget(QLabel("Strength"))
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(100)
        self.strength_slider.valueChanged.connect(self._controls_changed)
        strength_row.addWidget(self.strength_slider, 1)
        self.strength_label = QLabel("100%")
        self.strength_label.setMinimumWidth(38)
        strength_row.addWidget(self.strength_label)
        controls.addLayout(strength_row)
        module_row = QHBoxLayout()
        module_row.addWidget(QLabel("Include"))
        self.module_checks = {}
        for module in PRESET_MODULE_FIELDS:
            cb = QCheckBox(module)
            cb.setChecked(True)
            cb.toggled.connect(self._controls_changed)
            self.module_checks[module] = cb
            module_row.addWidget(cb)
        controls.addLayout(module_row)
        options_row = QHBoxLayout()
        self.preview_cb = QCheckBox("Live preview")
        self.preview_cb.setChecked(True)
        self.preview_cb.toggled.connect(self._controls_changed)
        options_row.addWidget(self.preview_cb)
        self.favorite_btn = QPushButton("☆ Favorite")
        self.favorite_btn.setEnabled(False)
        self.favorite_btn.clicked.connect(self._toggle_favorite)
        options_row.addWidget(self.favorite_btn)
        options_row.addStretch(1)
        controls.addLayout(options_row)
        root.addWidget(preset_controls)

        btn_row = QHBoxLayout()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self.reload)
        btn_row.addWidget(refresh)
        btn_row.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btn_row.addWidget(cancel)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setEnabled(False)
        self.apply_btn.setStyleSheet(
            "QPushButton { background:#2a5080; color:#fff; font-weight:600; padding:6px 18px; border-radius:4px; }"
            "QPushButton:disabled { background:#333; color:#777; }"
        )
        self.apply_btn.clicked.connect(self.accept)
        btn_row.addWidget(self.apply_btn)
        root.addLayout(btn_row)

        self._all_files = []
        self.reload()

    def reload(self):
        from presets import list_preset_files
        files = []
        dirs = []
        if self._plugin_dir and os.path.isdir(self._plugin_dir):
            dirs.append(self._plugin_dir)
        for d in self._extra:
            if d and os.path.isdir(d) and d not in dirs:
                dirs.append(d)
        for d in dirs:
            try:
                files.extend(list_preset_files(d, recursive=True))
            except Exception:
                pass
        # de-dupe by basename preference for plugin dir order
        seen = set()
        unique = []
        for f in files:
            key = os.path.normcase(os.path.abspath(f))
            if key in seen:
                continue
            seen.add(key)
            unique.append(f)
        unique.sort(key=lambda p: os.path.basename(p).lower())
        self._all_files = unique
        current = self.category_combo.currentData()
        categories = sorted({os.path.basename(os.path.dirname(p)) for p in unique if os.path.dirname(p)})
        self.category_combo.blockSignals(True)
        self.category_combo.clear()
        self.category_combo.addItem("All categories", "all")
        self.category_combo.addItem("Favorites", "favorites")
        for category in categories:
            self.category_combo.addItem(category, category)
        idx = self.category_combo.findData(current)
        self.category_combo.setCurrentIndex(max(0, idx))
        self.category_combo.blockSignals(False)
        self._filter()

    def _filter(self, *_):
        q = (self.search.text() or "").strip().lower()
        kind = self.type_combo.currentData() or "all"
        category = self.category_combo.currentData() or "all"
        self.list.clear()
        for path in self._all_files:
            name = os.path.basename(path)
            ext = os.path.splitext(name)[1].lower()
            normalized = os.path.normcase(os.path.abspath(path))
            if kind == "json" and ext != ".json":
                continue
            if kind == "xmp" and ext != ".xmp":
                continue
            if category == "favorites" and normalized not in self._favorites:
                continue
            if category not in ("all", "favorites") and os.path.basename(os.path.dirname(path)) != category:
                continue
            if q and q not in name.lower():
                continue
            item = QListWidgetItem(("★ " if normalized in self._favorites else "") + name)
            item.setData(Qt.ItemDataRole.UserRole, path)
            tip = path
            item.setToolTip(tip)
            # subtle color by type
            if ext == ".xmp":
                item.setForeground(QColor("#9cf"))
            else:
                item.setForeground(QColor("#cfc"))
            self.list.addItem(item)
        self.detail.setText(f"{self.list.count()} preset(s) — select one and click Apply.")
        self.apply_btn.setEnabled(False)
        self.selected_path = None

    def _on_sel(self, cur, _prev=None):
        if cur is None:
            self.apply_btn.setEnabled(False)
            self.favorite_btn.setEnabled(False)
            self.selected_path = None
            return
        path = cur.data(Qt.ItemDataRole.UserRole)
        self.selected_path = path
        self.apply_btn.setEnabled(True)
        self.favorite_btn.setEnabled(True)
        normalized = os.path.normcase(os.path.abspath(path))
        self.favorite_btn.setText("★ Favorited" if normalized in self._favorites else "☆ Favorite")
        ext = os.path.splitext(path)[1].upper()
        description = ""
        if ext == ".JSON":
            try:
                with open(path, "r", encoding="utf-8") as preset_file:
                    metadata = json.load(preset_file)
                description = str(metadata.get("description", "")).strip()
            except (OSError, ValueError, TypeError):
                pass
        details = [os.path.basename(path)]
        if description:
            details.append(description)
        details.extend((path, f"Type: {ext}"))
        self.detail.setText("\n".join(details))
        self._preview_selected()

    @property
    def strength(self):
        return self.strength_slider.value() / 100.0

    def selected_modules(self):
        return [name for name, cb in self.module_checks.items() if cb.isChecked()]

    def _controls_changed(self, *_):
        self.strength_label.setText(f"{self.strength_slider.value()}%")
        self._preview_selected()

    def _preview_selected(self):
        if self.selected_path and self.preview_cb.isChecked() and hasattr(self.parent(), "_preview_preset"):
            self.parent()._preview_preset(self.selected_path, self.strength, self.selected_modules())

    def _toggle_favorite(self):
        if not self.selected_path:
            return
        normalized = os.path.normcase(os.path.abspath(self.selected_path))
        if normalized in self._favorites:
            self._favorites.remove(normalized)
        else:
            self._favorites.add(normalized)
        self._settings.setValue("preset_favorites", "\n".join(sorted(self._favorites)))
        self._filter()

    def _accept_item(self, item):
        self.selected_path = item.data(Qt.ItemDataRole.UserRole)
        self.accept()

    def _pick_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Preset folder", self._plugin_dir or "")
        if d:
            self._plugin_dir = d
            self.path_lbl.setText(d)
            self.reload()



class PhotoLab(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoLab")
        # Ensure widgets inherit a valid point size (Windows can report -1)
        try:
            from PyQt6.QtGui import QFont as _QF
            _f = self.font()
            if _f.pointSize() <= 0:
                _f.setPixelSize(13)
                self.setFont(_f)
        except Exception:
            pass
        self.resize(1600, 1000)
        self.setStyleSheet(self._stylesheet())

        self.folder = None
        self.image_paths: list[str] = []
        self.recipes: dict[str, Recipe] = {}
        self.meta_cache: dict[str, dict] = {}
        self.current_path=None
        self.original_bgr: np.ndarray | None = None

        self.render_timer = QTimer()
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(45)
        self.render_timer.timeout.connect(self.render_preview)
        self._preview_generation = 0
        self._preview_source_cache = {}
        self._preview_renderer = PreviewRenderWorker()
        self._preview_renderer.rendered.connect(self._on_preview_rendered)
        self._preview_renderer.failed.connect(self._on_preview_render_failed)
        self._preview_renderer.start()

        self._load_worker = None
        self._pending_load_path = None
        self.sliders: dict[str, SliderRow] = {}
        self._history_push_pending = False
        self._local_mode = False
        self._copied_recipe = None
        self.autosave_sidecars = False
        self._recent_folders = []
        self._snapshots = {}  # path -> list[{name, recipe_dict}]
        self._load_recent_folders()
        self._image_ratings = {}  # path -> 0..5 for develop filmstrip
        self.catalog = Catalog()
        self._library_mode = False
        self._scan_worker = None
        self._lib_thumb_worker = None
        self._lib_records = []

        self._build_menu_bar()
        self._build_toolbar()
        self._build_layout()
        self._build_shortcuts()
        self._build_debug_console()
        self.statusBar().showMessage(
            "Open a folder to begin  •  Ctrl+O  •  Tools ▾ for HDR / Stack / Panorama"
        )
        self._show_develop_empty_state()
        self.log("PhotoLab started")

    # ------------------------------------------------------------------
    def _stylesheet(self):
        return """
            QMainWindow, QWidget { background: #181818; color: #ddd; letter-spacing: 0px; }
            QGroupBox, QGroupBox::title, QLabel { letter-spacing: 0px; }
            QLabel { color: #ccc; }
            QPushButton, QToolButton {
                background: #2b2b2b; color: #eee; border: 1px solid #3d3d3d;
                padding: 5px 12px; border-radius: 4px;
                font-size: 11px;
            }
            QPushButton:hover, QToolButton:hover { background: #3d3d3d; border-color: #555; }
            QPushButton:pressed, QToolButton:pressed { background: #1f1f1f; }
            QPushButton:checked, QToolButton:checked {
                background: #2a6ad4; color: #fff; border-color: #2a6ad4; font-weight: 600;
            }
            
            /* Custom Sleek Scrollbar */
            QScrollBar:vertical {
                border: none; background: #141414; width: 8px; margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #3a3a3a; min-height: 20px; border-radius: 4px;
            }
            QScrollBar::handle:vertical:hover {
                background: #4a4a4a;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                height: 0px; background: none;
            }
            QScrollBar:horizontal {
                border: none; background: #141414; height: 8px; margin: 0px;
            }
            QScrollBar::handle:horizontal {
                background: #3a3a3a; min-width: 20px; border-radius: 4px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #4a4a4a;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                width: 0px; background: none;
            }
            
            /* Custom Sliders */
            QSlider::groove:horizontal {
                background: #252525; height: 4px; border-radius: 2px; border: 1px solid #333;
            }
            QSlider::handle:horizontal {
                background: #2a6ad4; width: 10px; height: 10px; margin: -3px 0; border-radius: 5px;
            }
            QSlider::handle:horizontal:hover {
                background: #488df2;
            }
            QSlider::sub-page:horizontal {
                background: #2a6ad4; border-radius: 2px;
            }
            
            QListWidget { background: #121212; border: 1px solid #2b2b2b; color: #ddd; border-radius: 3px; }
            QListWidget::item { padding: 4px 6px; }
            QListWidget::item:hover { background: #222; }
            QListWidget::item:selected { background: #2a5080; color: #fff; }
            
            QToolBar {
                background: #202020; border-bottom: 1px solid #2d2d2d; spacing: 4px; padding: 4px;
            }
            QStatusBar { color: #aaa; background: #141414; border-top: 1px solid #282828; }
            QScrollArea { border: none; background: transparent; }
            QComboBox {
                background: #252525; color: #eee; border: 1px solid #3d3d3d;
                border-radius: 3px; padding: 3px 8px; font-size: 11px;
            }
            QComboBox::drop-down {
                border: none; width: 16px;
            }
            QComboBox QAbstractItemView {
                background: #202020; color: #eee; selection-background-color: #2a6ad4; border: 1px solid #3d3d3d;
            }
            QLineEdit {
                background: #252525; color: #eee; border: 1px solid #3d3d3d;
                border-radius: 3px; padding: 4px 8px; font-size: 11px;
            }
            QLineEdit:focus { border-color: #2a6ad4; }
            QCheckBox { color: #ccc; spacing: 6px; font-size: 11px; }
            QCheckBox::indicator {
                width: 13px; height: 13px; border: 1px solid #444; border-radius: 2px; background: #222;
            }
            QCheckBox::indicator:checked {
                background: #2a6ad4; border-color: #2a6ad4;
            }
        """

    # ------------------------------------------------------------------

    def _build_menu_bar(self):
        """DxO-style menu bar: File / Edit / View / Image / Help."""
        mb = self.menuBar()
        mb.setStyleSheet("""
            QMenuBar {
                background: #2a2a2a; color: #ddd; border-bottom: 1px solid #3a3a3a;
                padding: 2px;
            }
            QMenuBar::item {
                background: transparent; color: #ddd;
                padding: 4px 10px;
            }
            QMenuBar::item:selected {
                background: #3a3a3a; color: #fff;
            }
            QMenu {
                background: #2a2a2a; color: #ddd;
                border: 1px solid #444;
                padding: 4px;
            }
            QMenu::item {
                padding: 6px 28px 6px 24px;
            }
            QMenu::item:selected {
                background: #2a6ad4; color: #fff;
            }
            QMenu::separator {
                height: 1px; background: #444; margin: 4px 8px;
            }
            QMenu::item:disabled {
                color: #666;
            }
        """)

        def add_action(menu, text, slot=None, shortcut=None, enabled=True, checkable=False):
            act = QAction(text, self)
            if shortcut:
                act.setShortcut(QKeySequence(shortcut))
            act.setEnabled(enabled)
            act.setCheckable(checkable)
            if slot:
                act.triggered.connect(slot)
            menu.addAction(act)
            return act

        # ----- File -----
        file_m = mb.addMenu("&File")
        add_action(file_m, "Open Folder for Editing…", self.open_folder, "Ctrl+O")
        add_action(file_m, "Scan Folder into Library…", self.scan_library_folder, "Ctrl+Shift+O")
        add_action(file_m, "Import from SD Card…", self.import_from_sd)
        self.recent_menu = file_m.addMenu("Recent Folders")
        self._rebuild_recent_menu()
        file_m.addSeparator()
        add_action(file_m, "Export to Disk…", self.export_current, "Ctrl+E")
        add_action(file_m, "Batch Export Selected…", self.batch_export_selected, "Ctrl+Shift+E")
        file_m.addSeparator()
        add_action(file_m, "Save Recipe Sidecar", self.save_sidecar, "Ctrl+S")
        add_action(file_m, "Reload Recipe Sidecar", self.reload_sidecar)
        self.act_autosave = add_action(file_m, "Auto-Save Sidecars", self.toggle_autosave, checkable=True)
        add_action(file_m, "Export to Nik Collection", enabled=False)
        add_action(file_m, "Export to Application", enabled=False)
        add_action(file_m, "Export to Flickr", enabled=False)
        add_action(file_m, "Export to Lightroom", enabled=False)
        file_m.addSeparator()
        add_action(file_m, "Preset Browser…", self.load_preset)
        add_action(file_m, "Load Preset File… (XMP / JSON)", self.load_preset_file_dialog)
        add_action(file_m, "Import Preset Folder…", self.load_preset_folder)
        add_action(file_m, "Save Preset… (JSON)", self.save_preset)
        file_m.addSeparator()
        add_action(file_m, "Preferences…", self.show_preferences, "Ctrl+,")
        file_m.addSeparator()
        add_action(file_m, "Print / PDF…", self.show_print_dialog, "Ctrl+P")
        file_m.addSeparator()
        add_action(file_m, "E&xit", self.close, "Ctrl+Q")

        # ----- Edit -----
        edit_m = mb.addMenu("&Edit")
        self.act_undo = add_action(edit_m, "Undo", self.undo_edit, "Ctrl+Z")
        self.act_redo = add_action(edit_m, "Redo", self.redo_edit, "Ctrl+Y")
        edit_m.addSeparator()
        add_action(edit_m, "Reset Image", self.reset_current, "Ctrl+R")
        reset_m = edit_m.addMenu("Reset Module")
        add_action(reset_m, "Reset Tone / Light", lambda: self.reset_module("tone"))
        add_action(reset_m, "Reset Color", lambda: self.reset_module("color"))
        add_action(reset_m, "Reset Detail", lambda: self.reset_module("detail"))
        add_action(reset_m, "Reset Geometry", lambda: self.reset_module("geometry"))
        add_action(reset_m, "Reset Local", lambda: self.reset_module("local"))
        add_action(reset_m, "Reset Effects", lambda: self.reset_module("effects"))
        edit_m.addSeparator()
        add_action(edit_m, "Copy Settings", self.copy_settings, "Ctrl+Shift+C")
        add_action(edit_m, "Paste Settings", self.paste_settings, "Ctrl+Shift+V")
        add_action(edit_m, "Sync Settings to Selected…", self.sync_settings_to_selected, "Ctrl+Shift+S")
        edit_m.addSeparator()
        add_action(edit_m, "Save Snapshot…", self.save_snapshot)
        add_action(edit_m, "Restore Snapshot…", self.restore_snapshot)
        edit_m.addSeparator()
        add_action(edit_m, "Preferences…", enabled=False, shortcut="Ctrl+,")

        # ----- View -----
        view_m = mb.addMenu("&View")
        add_action(view_m, "Fit to Window", lambda: self.preview.fit_to_view(), "F")
        self.act_clipping = add_action(view_m, "Clipping Warning", self.toggle_clipping, "J", checkable=True)
        self.act_zebras = add_action(view_m, "Zebra Stripes (overexposure)", self.toggle_zebras, "Z", checkable=True)
        self.act_peaking = add_action(view_m, "Focus Peaking", self.toggle_peaking, "P", checkable=True)
        add_action(view_m, "Actual Size (1:1)", lambda: self.preview.zoom_1_to_1(), "Ctrl+1")
        view_m.addSeparator()
        add_action(view_m, "Compare Off", lambda: self.set_compare_mode(ImageCanvas.MODE_NORMAL))
        add_action(view_m, "Split Compare", lambda: self.set_compare_mode(ImageCanvas.MODE_SPLIT), "C")
        add_action(view_m, "Side-by-Side Compare", lambda: self.set_compare_mode(ImageCanvas.MODE_SIDE_BY_SIDE), "B")
        add_action(view_m, "Compare with Snapshot…", self.compare_snapshot)
        add_action(view_m, "Soft-Proof Comparison", self.compare_soft_proof)
        view_m.addSeparator()
        add_action(view_m, "Previous Image", self.prev_image, "Left")
        add_action(view_m, "Next Image", self.next_image, "Right")
        view_m.addSeparator()
        self.act_library = add_action(view_m, "Library", self.show_library_mode, "Ctrl+L", checkable=True)
        self.act_develop = add_action(view_m, "Develop", self.show_develop_mode, "Ctrl+D", checkable=True)
        self.act_develop.setChecked(True)
        view_m.addSeparator()
        self.act_debug = add_action(view_m, "Debug Console", self.toggle_debug_console, "Ctrl+Shift+D", checkable=True)
        view_m.addSeparator()
        add_action(view_m, "Full Screen", self.toggle_fullscreen, "F11")
        add_action(view_m, "Image Metadata…", self.show_metadata, "I")

        # ----- Image -----
        image_m = mb.addMenu("&Image")
        add_action(image_m, "Rotate 90° Clockwise", lambda: self._rotate(1))
        add_action(image_m, "Rotate 90° Counter-clockwise", lambda: self._rotate(-1))
        image_m.addSeparator()
        add_action(image_m, "Level Horizon (draw line)", self.start_horizon_line, "L")
        add_action(image_m, "Crop Tool", lambda: self.crop_tool_btn.setChecked(True))
        add_action(image_m, "Clear Crop", self.clear_crop)
        image_m.addSeparator()
        add_action(image_m, "Reject / Unreject", self.toggle_reject_current, "X")
        add_action(image_m, "Pick / Unpick", self.toggle_pick_current, "U")
        image_m.addSeparator()
        add_action(image_m, "Local Adjustments (Control Point)", self._toggle_local_mode)
        image_m.addSeparator()
        add_action(image_m, "Auto Exposure", self.auto_exposure)
        add_action(image_m, "Auto White Balance", self.auto_wb)
        add_action(image_m, "White Balance Picker", self.toggle_wb_picker, "W")
        image_m.addSeparator()
        add_action(image_m, "Graduated Filter", self.toggle_gradient_mode, "G")
        add_action(image_m, "Adjustment Brush", self.toggle_brush_mode, "Shift+B")
        image_m.addSeparator()
        add_action(image_m, "Merge HDR…", self.merge_hdr_selected, "Ctrl+Shift+H")
        add_action(image_m, "Focus Stack…", self.focus_stack_selected, "Ctrl+Shift+F")
        add_action(image_m, "Panorama…", self.panorama_selected, "Ctrl+Shift+P")
        add_action(image_m, "Create Pan Video…", self.create_pan_video)
        add_action(image_m, "Video Editor…", self._menu_open_video_editor)
        add_action(image_m, "Audio Editor…", self.open_audio_editor)
        image_m.addSeparator()
        add_action(image_m, "Map (GPS)…", self.show_map_view, "Ctrl+Shift+M")
        add_action(image_m, "Slideshow…", self.start_slideshow, "Ctrl+Shift+S")
        add_action(image_m, "Print / PDF…", self.show_print_dialog, "Ctrl+P")
        add_action(image_m, "Run Script…", self.run_user_script)


        # ----- Help -----
        help_m = mb.addMenu("&Help")
        add_action(help_m, "User Manual", self._show_user_manual, "F1")
        add_action(help_m, "Developer Manual", self._show_developer_manual)
        add_action(help_m, "Keyboard Shortcuts", self._show_shortcuts)
        help_m.addSeparator()
        add_action(help_m, "Clear Caches…", self.clear_caches)
        add_action(help_m, "Report a Problem…", self.export_problem_report)
        add_action(help_m, "Check for Updates…", self.check_for_updates)
        help_m.addSeparator()
        add_action(help_m, "About PhotoLab", self._show_about)

    def _show_shortcuts(self):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "Keyboard Shortcuts",
            "Ctrl+O\tOpen folder\n"
            "Ctrl+E\tExport\n"
            "Ctrl+R\tReset image\n"
            "Ctrl+Q\tExit\n"
            "C\tSplit compare\n"
            "\\ or `\tHold for temporary before view\n"
            "B\tSide-by-side compare\n"
            "F\tFit to window\n"
            "1\tActual size 1:1\n"
            "Left/Right\tPrev / Next image\n"
            "Wheel\tZoom\n"
            "Space+drag\tPan",
        )


    def _manual_path(self, name: str) -> str:
        """Resolve docs/*.md via app_paths, then next to the app / cwd."""
        candidates = []
        try:
            from app_paths import docs_dir, manual_file
            mf = manual_file(name) if callable(manual_file) else None
            if mf:
                candidates.append(mf)
            candidates.append(os.path.join(docs_dir(), name))
        except Exception:
            log.debug("_manual_path: app path lookup failed", exc_info=True)
        candidates.extend([
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs", name),
            os.path.join(os.getcwd(), "docs", name),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
            os.path.join(os.getcwd(), name),
        ])
        seen = set()
        for c in candidates:
            if not c or c in seen:
                continue
            seen.add(c)
            if os.path.isfile(c):
                return c
        return candidates[0] if candidates else name

    def _show_markdown_manual(self, title: str, filename: str):
        path = self._manual_path(filename)
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(780, 640)
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                # Lightweight Markdown → HTML for readability
                html = self._markdown_to_html(text)
                browser.setHtml(html)
            except Exception as e:
                browser.setPlainText(f"Could not read {path}:\n{e}")
        else:
            browser.setPlainText(
                f"Manual not found:\n{path}\n\n"
                "Place USER_MANUAL.md and DEVELOPER_MANUAL.md in a docs/ folder next to the app."
            )
        layout.addWidget(browser)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec()

    def _markdown_to_html(self, md: str) -> str:
        """Minimal MD subset → HTML (headers, code, tables-ish, lists, bold)."""
        import html as html_mod
        import re
        lines = md.splitlines()
        out = [
            "<html><head><style>"
            "body{font-family:Segoe UI,Arial,sans-serif;font-size:13px;color:#ddd;background:#1a1a1a;padding:12px;}"
            "h1{color:#fff;font-size:22px;border-bottom:1px solid #444;padding-bottom:6px;}"
            "h2{color:#9cf;font-size:17px;margin-top:1.2em;}"
            "h3{color:#bde;font-size:14px;}"
            "code,pre{font-family:Consolas,monospace;background:#111;color:#cfc;}"
            "pre{padding:8px;border:1px solid #333;overflow-x:auto;}"
            "table{border-collapse:collapse;margin:8px 0;}"
            "th,td{border:1px solid #444;padding:4px 8px;}"
            "th{background:#2a2a2a;}"
            "a{color:#6af;}"
            "hr{border:none;border-top:1px solid #444;}"
            "li{margin:2px 0;}"
            "</style></head><body>"
        ]
        in_code = False
        in_table = False
        for line in lines:
            if line.strip().startswith("```"):
                if in_code:
                    out.append("</pre>")
                    in_code = False
                else:
                    out.append("<pre>")
                    in_code = True
                continue
            if in_code:
                out.append(html_mod.escape(line) + "\n")
                continue
            if line.startswith("# "):
                out.append(f"<h1>{html_mod.escape(line[2:])}</h1>")
            elif line.startswith("## "):
                out.append(f"<h2>{html_mod.escape(line[3:])}</h2>")
            elif line.startswith("### "):
                out.append(f"<h3>{html_mod.escape(line[4:])}</h3>")
            elif line.strip() == "---":
                out.append("<hr/>")
            elif line.startswith("|") and "|" in line[1:]:
                cells = [c.strip() for c in line.strip("|").split("|")]
                if all(set(c) <= set("-: ") for c in cells):
                    continue  # separator row
                tag = "th" if not in_table else "td"
                if not in_table:
                    out.append("<table>")
                    in_table = True
                    tag = "th"
                row = "".join(f"<{tag}>{html_mod.escape(c)}</{tag}>" for c in cells)
                out.append(f"<tr>{row}</tr>")
            else:
                if in_table:
                    out.append("</table>")
                    in_table = False
                if line.startswith("- "):
                    out.append(f"<li>{html_mod.escape(line[2:])}</li>")
                elif re.match(r"^\d+\.\s", line):
                    out.append(f"<li>{html_mod.escape(re.sub(r'^\d+\.\s', '', line))}</li>")
                elif not line.strip():
                    out.append("<br/>")
                else:
                    s = html_mod.escape(line)
                    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
                    s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
                    out.append(f"<p>{s}</p>")
        if in_table:
            out.append("</table>")
        if in_code:
            out.append("</pre>")
        out.append("</body></html>")
        return "\n".join(out)

    def _show_user_manual(self):
        self._show_markdown_manual("PhotoLab — User Manual", "USER_MANUAL.md")

    def _show_developer_manual(self):
        self._show_markdown_manual("PhotoLab — Developer Manual", "DEVELOPER_MANUAL.md")

    def _show_about(self):
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.about(
            self,
            "About PhotoLab",
            "PhotoLab\n\n"
            "A DxO PhotoLab-inspired RAW/photo editor written in Python.\n\n"
            "Non-destructive Recipe pipeline, local adjustments, HSL, history.\n\n"
            "Built with PyQt6, OpenCV, NumPy, rawpy. Brian E. Toon - 2026",
        )

    def _build_toolbar(self):
        """Primary toolbar kept short so actions stay clickable (no >> overflow).

        Long / less-frequent tools live in a Tools dropdown (InstantPopup) so
        they never hide behind Qt's hard-to-reach extension button.
        """
        tb = QToolBar("Main")
        tb.setMovable(False)
        tb.setFloatable(False)
        tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        # Keep icons/text compact; helps fit more before overflow kicks in
        try:
            tb.setIconSize(QSize(16, 16))
        except Exception:
            pass
        self.addToolBar(tb)
        self._main_toolbar = tb

        def act(text, slot, shortcut=None, checkable=False, tip=None):
            a = QAction(text, self)
            if shortcut:
                a.setShortcut(QKeySequence(shortcut))
            a.setCheckable(checkable)
            if tip:
                a.setToolTip(tip)
            a.triggered.connect(slot)
            tb.addAction(a)
            return a

        act("Open", self.open_folder, "Ctrl+O", tip="Open folder for editing (Ctrl+O)")
        act("Scan", self.scan_library_folder, tip="Scan folder into Library")
        tb.addSeparator()

        self.act_tb_library = act("Library", self.show_library_mode, "Ctrl+L", checkable=True)
        self.act_tb_develop = act("Develop", self.show_develop_mode, "Ctrl+D", checkable=True)
        self.act_tb_develop.setChecked(True)
        tb.addSeparator()

        self.act_edit = act("Edit", lambda: self.set_compare_mode(ImageCanvas.MODE_NORMAL), checkable=True, tip="Normal edit view")
        self.act_edit.setChecked(True)
        self.act_split = act("Split", lambda: self.set_compare_mode(ImageCanvas.MODE_SPLIT), "C", True, tip="Split compare (C)")
        self.act_side = act("Side-by-Side", lambda: self.set_compare_mode(ImageCanvas.MODE_SIDE_BY_SIDE), "B", True, tip="Side-by-side compare (B)")
        tb.addSeparator()

        act("Fit", lambda: self.preview.fit_to_view(), "F", tip="Fit to window (F)")
        act("1:1", lambda: self.preview.zoom_1_to_1(), "Ctrl+1", tip="Actual size (Ctrl+1)")
        self.zoom_label = QLabel(" 100% ")
        self.zoom_label.setStyleSheet("color:#ccc; min-width:44px;")
        tb.addWidget(self.zoom_label)
        tb.addSeparator()

        act("Prev", self.prev_image, "Left", tip="Previous image")
        act("Next", self.next_image, "Right", tip="Next image")
        tb.addSeparator()

        self.act_local = act("Local", self._toggle_local_mode, checkable=True, tip="Control Point local adjustments")
        self.act_grad = act("Grad", self.toggle_gradient_mode, "G", checkable=True, tip="Graduated filter (G)")
        self.act_brush = act("Brush", self.toggle_brush_mode, "Shift+B", checkable=True, tip="Adjustment brush (Shift+B)")
        self.act_wb_pick = act("WB", self.toggle_wb_picker, "W", checkable=True, tip="Open Color panel + white balance picker (W)")
        self.act_tb_zebras = act("Zebras", self.toggle_zebras, "Z", checkable=True, tip="Zebra stripes on overexposed areas (Z)")
        tb.addSeparator()
        act("Reset", self.reset_current, "Ctrl+R", tip="Reset image (Ctrl+R)")
        act("Preset", self.load_preset, tip="Open Preset Browser (plugin folder)")
        act("Save Preset", self.save_preset, tip="Save current recipe as JSON preset")
        tb.addSeparator()

        # Tools dropdown — always reachable; avoids Qt toolbar >> overflow menu
        tools_btn = QToolButton(self)
        tools_btn.setText("Tools ▾")
        tools_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        tools_btn.setToolTip("HDR, Focus Stack, Panorama, Pan Video, Audio")
        tools_btn.setStyleSheet(
            "QToolButton { padding: 5px 10px; }"
            "QToolButton::menu-indicator { image: none; width: 0; }"
        )
        tools_menu = QMenu(tools_btn)
        tools_menu.setStyleSheet(
            "QMenu { background:#2a2a2a; color:#ddd; border:1px solid #444; padding:4px; }"
            "QMenu::item { padding:8px 28px 8px 16px; }"
            "QMenu::item:selected { background:#2a6ad4; color:#fff; }"
            "QMenu::separator { height:1px; background:#444; margin:4px 8px; }"
        )
        tools_menu.addAction("Color Calibration Studio…", self.open_color_calibration_studio)
        tools_menu.addSeparator()
        tools_menu.addAction("Merge HDR…", self.merge_hdr_selected)
        tools_menu.addAction("Focus Stack…", self.focus_stack_selected)
        tools_menu.addAction("Panorama…", self.panorama_selected)
        tools_menu.addSeparator()
        tools_menu.addAction("Pan Video…", self.create_pan_video)
        tools_menu.addAction("Video Editor…", self._menu_open_video_editor)
        tools_menu.addSeparator()
        tools_menu.addAction("Map (GPS)…", self.show_map_view)
        tools_menu.addAction("Slideshow…", self.start_slideshow)
        tools_menu.addAction("Print / PDF…", self.show_print_dialog)
        tools_menu.addAction("Run Script…", self.run_user_script)
        tools_menu.addAction("Audio Editor…", self.open_audio_editor)
        tools_btn.setMenu(tools_menu)
        tb.addWidget(tools_btn)

        act("Export", self.export_current, "Ctrl+E", tip="Export current image (Ctrl+E)")

        self.setStatusBar(QStatusBar())
        self.path_label = QLabel("")
        self.path_label.setStyleSheet("color:#888;")
        self.statusBar().addWidget(self.path_label, stretch=1)
        self.count_label = QLabel("")
        self.statusBar().addPermanentWidget(self.count_label)

    def _build_shortcuts(self):
        from PyQt6.QtGui import QShortcut, QKeySequence
        for n in range(0, 6):
            sc = QShortcut(QKeySequence(str(n)), self)
            sc.activated.connect(lambda n=n: self.rate_current(n))
        scj = QShortcut(QKeySequence("J"), self)
        scj.activated.connect(lambda: self.toggle_clipping())
        scz = QShortcut(QKeySequence("Z"), self)
        scz.activated.connect(lambda: self.toggle_zebras())
        scx = QShortcut(QKeySequence("X"), self)
        scx.activated.connect(self.toggle_reject_current)
        scu = QShortcut(QKeySequence("U"), self)
        scu.activated.connect(self.toggle_pick_current)
        scp = QShortcut(QKeySequence("P"), self)
        scp.activated.connect(lambda: self.toggle_peaking())
        sci = QShortcut(QKeySequence("I"), self)
        sci.activated.connect(self.show_metadata)
        scl = QShortcut(QKeySequence("L"), self)
        scl.activated.connect(self.start_horizon_line)
        sc_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        sc_undo.activated.connect(self.undo_edit)
        sc_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        sc_redo.activated.connect(self.redo_edit)

    # Thread-safe log signal (workers must not touch QWidgets directly)
    log_message = None  # set in _build_debug_console as pyqtSignal via helper

    def _make_progress(self, title: str, maximum: int = 0) -> QProgressDialog:
        """Non-blocking modal progress dialog for long jobs (scan, stack, export)."""
        dlg = QProgressDialog(title, "Cancel", 0, maximum, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(400)
        dlg.setAutoClose(True)
        dlg.setAutoReset(True)
        dlg.setValue(0)
        return dlg

    def _show_develop_empty_state(self):
        """Friendly empty canvas when no folder/image is loaded."""
        if getattr(self, "preview", None) is None:
            return
        if self.current_path or self.original_bgr is not None:
            return
        recent = list(getattr(self, "_recent_folders", []) or [])[:5]
        lines = [
            "PhotoLab",
            "",
            "Open a folder to edit, or scan into the Library.",
            "",
            "  Ctrl+O    Open folder",
            "  Ctrl+Shift+O    Scan into Library",
            "  Ctrl+L / Ctrl+D    Library / Develop",
            "",
            "Wheel zoom  •  Space+drag pan  •  C split  •  B side-by-side",
        ]
        if recent:
            lines.append("")
            lines.append("Recent folders:")
            for p in recent:
                lines.append(f"  • {p}")
        # ImageCanvas is a QWidget, not QLabel — use status + path label
        self.statusBar().showMessage(
            "Open a folder to begin  •  Ctrl+O  •  Tools menu has HDR / Stack / Panorama"
        )
        if hasattr(self, "path_label"):
            self.path_label.setText("No image loaded — File → Open Folder, or use the Open button")


    def _build_debug_console(self):
        """Dockable log window. Does NOT hijack sys.stdout/stderr (that crashes
        on Windows when background QThreads print)."""
        from PyQt6.QtCore import pyqtSignal, QObject

        class _LogBridge(QObject):
            message = pyqtSignal(str, str)  # text, level

        self._log_bridge = _LogBridge()
        self._log_bridge.message.connect(self._append_log_line)

        self.debug_dock = QDockWidget("Debug Console", self)
        self.debug_dock.setObjectName("DebugConsoleDock")
        self.debug_dock.setAllowedAreas(
            Qt.DockWidgetArea.BottomDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(4, 4, 4, 4)
        self.debug_console = QPlainTextEdit()
        self.debug_console.setReadOnly(True)
        self.debug_console.setMaximumBlockCount(5000)
        self.debug_console.setStyleSheet(
            "QPlainTextEdit { background:#0e0e0e; color:#b0e0b0; "
            "font-family: Consolas, 'Courier New', monospace; font-size:11px; border:1px solid #333; }"
        )
        v.addWidget(self.debug_console)
        btn_row = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.debug_console.clear)
        btn_row.addWidget(clear_btn)
        copy_btn = QPushButton("Copy All")
        copy_btn.clicked.connect(self._copy_debug_log)
        btn_row.addWidget(copy_btn)
        btn_row.addStretch(1)
        hint = QLabel("View → Debug Console (Ctrl+Shift+D)  •  thread-safe app log")
        hint.setStyleSheet("color:#666; font-size:11px;")
        btn_row.addWidget(hint)
        v.addLayout(btn_row)
        self.debug_dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.debug_dock)
        self.debug_dock.hide()
        self.debug_dock.visibilityChanged.connect(
            lambda vis: self.act_debug.setChecked(vis) if hasattr(self, "act_debug") else None
        )

    def _append_log_line(self, message, level="INFO"):
        if not hasattr(self, "debug_console"):
            return
        ts = datetime.now().strftime("%H:%M:%S")
        self.debug_console.appendPlainText(f"[{ts}] [{level}] {message}")
        sb = self.debug_console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _copy_debug_log(self):
        QApplication.clipboard().setText(self.debug_console.toPlainText())

    def toggle_debug_console(self, checked=None):
        if not hasattr(self, "debug_dock"):
            return
        if checked is None:
            checked = not self.debug_dock.isVisible()
        self.debug_dock.setVisible(bool(checked))
        if hasattr(self, "act_debug"):
            self.act_debug.setChecked(bool(checked))
        if checked:
            self.log("Debug console opened")

    def log(self, message, level="INFO"):
        """Thread-safe: safe to call from workers via signal."""
        try:
            if hasattr(self, "_log_bridge") and self._log_bridge is not None:
                self._log_bridge.message.emit(str(message), str(level))
            else:
                self._append_log_line(str(message), str(level))
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _build_layout(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.mode_stack = QStackedWidget()
        root_layout.addWidget(self.mode_stack)

        # ---- Develop page (existing editor) ----
        develop_page = QWidget()
        outer = QVBoxLayout(develop_page)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        outer.addWidget(splitter, stretch=1)

        # LEFT: histogram + navigator + history (DxO-style)
        left = QWidget()
        left.setObjectName("leftPanel")
        ll = QVBoxLayout(left)
        ll.setContentsMargins(8, 8, 8, 8)
        ll.setSpacing(8)

        # Shared clean group style — avoid ALL CAPS + letter-spacing issues on Windows
        gstyle = """
            QGroupBox {
                color: #c8c8c8;
                border: 1px solid #3a3a3a;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 6px;
                font-size: 12px;
                font-weight: 600;
                letter-spacing: 0px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 10px;
                padding: 0 6px;
                color: #b0b0b0;
                letter-spacing: 0px;
            }
        """

        folder_box = QGroupBox("Folders")
        folder_box.setStyleSheet(gstyle)
        fb = QVBoxLayout(folder_box)
        fb.setContentsMargins(4, 10, 4, 4)
        self.folder_model = QFileSystemModel()
        self.folder_model.setFilter(QDir.Filter.AllDirs | QDir.Filter.NoDotAndDotDot)
        self.folder_model.setRootPath("")
        self.folder_tree = QTreeView()
        self.folder_tree.setModel(self.folder_model)
        self.folder_tree.setHeaderHidden(True)
        self.folder_tree.setMaximumHeight(150)
        for i in range(1, 4):
            self.folder_tree.hideColumn(i)
        self.folder_tree.setStyleSheet("""
            QTreeView {
                background: #141414;
                border: none;
                color: #ccc;
                font-size: 11px;
            }
            QTreeView::item:hover { background: #2a2a2a; }
            QTreeView::item:selected { background: #2a5080; color: #fff; }
        """)
        self.folder_tree.clicked.connect(self._on_folder_tree_clicked)
        fb.addWidget(self.folder_tree)
        ll.addWidget(folder_box)

        hist_box = QGroupBox("Histogram")
        hist_box.setStyleSheet(gstyle)
        hb = QVBoxLayout(hist_box)
        hb.setContentsMargins(6, 10, 6, 6)
        self.histogram = HistogramWidget()
        self.histogram.setMinimumHeight(130)
        hb.addWidget(self.histogram)
        ll.addWidget(hist_box)

        nav_box = QGroupBox("Move / Zoom")
        nav_box.setStyleSheet(gstyle)
        nb = QVBoxLayout(nav_box)
        nb.setContentsMargins(6, 10, 6, 6)
        self.navigator = NavigatorWidget()
        self.navigator.setMinimumHeight(110)
        self.navigator.setMaximumHeight(140)
        self.navigator.panRequested.connect(self._on_nav_pan)
        nb.addWidget(self.navigator)
        ll.addWidget(nav_box)

        hist_box2 = QGroupBox("History")
        hist_box2.setStyleSheet(gstyle)
        hb2 = QVBoxLayout(hist_box2)
        hb2.setContentsMargins(6, 10, 6, 6)
        self.history_widget = HistoryWidget()
        self.history_widget.restoreRequested.connect(self._on_history_restore)
        self.history_widget.setMinimumHeight(100)
        hb2.addWidget(self.history_widget)
        ll.addWidget(hist_box2, stretch=1)
        
        meta_box = QGroupBox("Metadata")
        meta_box.setStyleSheet(gstyle)
        mb_layout = QVBoxLayout(meta_box)
        mb_layout.setContentsMargins(8, 12, 8, 8)
        self.metadata_label = QLabel("No metadata loaded")
        self.metadata_label.setWordWrap(True)
        self.metadata_label.setStyleSheet("color:#aaa; font-size:11px;")
        mb_layout.addWidget(self.metadata_label)
        gps_row = QHBoxLayout()
        self.gps_status_label = QLabel()
        self.gps_status_label.setTextFormat(Qt.TextFormat.RichText)
        gps_row.addWidget(self.gps_status_label, 1)
        self.gps_map_btn = QPushButton("Map this photo")
        self.gps_map_btn.setToolTip("Show this photo's location on the map")
        self.gps_map_btn.clicked.connect(self._map_current_photo)
        gps_row.addWidget(self.gps_map_btn)
        mb_layout.addLayout(gps_row)
        self.gps_folder_btn = QPushButton("Select GPS photos && show map")
        self.gps_folder_btn.setToolTip("Find geotagged photos in the filmstrip, select them, and open the map")
        self.gps_folder_btn.clicked.connect(self.select_gps_photos_and_map)
        mb_layout.addWidget(self.gps_folder_btn)
        self._set_gps_status(None)
        ll.addWidget(meta_box)

        left.setMinimumWidth(220)
        left.setMaximumWidth(300)
        splitter.addWidget(left)

        # CENTER: preview
        self.preview = ImageCanvas()
        self.preview.crop_dragged.connect(self.on_crop_dragged)
        self.preview.keystoneChanged.connect(self._on_keystone_changed)
        self.preview.keystoneFinished.connect(self._on_keystone_finished)
        self.preview.zoom_changed.connect(lambda s: self.zoom_label.setText(f" {s*100:.0f}% "))
        self.preview.zoom_changed.connect(lambda s: self._update_navigator_viewport())
        
        # Connect new control point signals
        self.preview.controlPointSelected.connect(self._on_canvas_point_selected)
        self.preview.controlPointMoved.connect(self._on_canvas_point_moved)
        self.preview.controlPointResized.connect(self._on_canvas_point_resized)
        self.preview.controlPointAdded.connect(self._on_canvas_point_added)
        self.preview.controlPointDragFinished.connect(self._on_canvas_point_drag_finished)
        self.preview.wbPicked.connect(self._on_wb_picked)
        self.preview.gradientChanged.connect(self._on_gradient_changed)
        self.preview.gradientSelected.connect(self._on_gradient_selected)
        self.preview.brushStrokeFinished.connect(self._on_brush_stroke_finished)
        self.preview.brushMaskChanged.connect(self._on_brush_changed)
        self.preview.horizonLineFinished.connect(self._on_horizon_line)
        splitter.addWidget(self.preview)

        # RIGHT: DxO-style tool panel
        right = QWidget()
        right.setMaximumWidth(380)
        right.setMinimumWidth(340)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.setSpacing(4)

        # Top bar: Reset + Apply preset
        top_row = QHBoxLayout()
        reset_btn = QPushButton("↺ Reset")
        reset_btn.clicked.connect(self.reset_current)
        top_row.addWidget(reset_btn)
        preset_btn = QPushButton("Apply preset")
        preset_btn.clicked.connect(self.load_preset)
        top_row.addWidget(preset_btn)
        rl.addLayout(top_row)

        # Search
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search for corrections…")
        self.search_edit.textChanged.connect(self._filter_corrections)
        rl.addWidget(self.search_edit)

        # Category tabs (Light / Color / Detail / Geometry / Effects)
        # DxO-style category strip (buttons, not cramped tabs)
        cat_row = QHBoxLayout()
        cat_row.setSpacing(4)
        self._cat_group = QButtonGroup(self)
        self._cat_group.setExclusive(True)
        self.tool_stack = QStackedWidget()
        categories = [
            ("Light", self._build_light_tab),
            ("Color", self._build_color_tab),
            ("Detail", self._build_detail_tab),
            ("Geometry", self._build_geometry_tab),
            ("Effects", self._build_effects_tab),
            ("Local", self._build_local_tab),
        ]
        self._cat_buttons = []
        for i, (name, builder) in enumerate(categories):
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setMinimumWidth(50)
            btn.setStyleSheet("""
                QPushButton {
                    background: #2a2a2a; color: #bbb; border: 1px solid #3a3a3a;
                    border-radius: 4px; padding: 7px 10px; font-size: 12px;
                }
                QPushButton:checked {
                    background: #2a6ad4; color: #fff; border-color: #2a6ad4; font-weight: 600;
                }
                QPushButton:hover:!checked { background: #3a3a3a; color: #eee; }
            """)
            if i == 0:
                btn.setChecked(True)
            self._cat_group.addButton(btn, i)
            cat_row.addWidget(btn)
            self._cat_buttons.append(btn)
            page = builder()  # returns the scroll widget
            self.tool_stack.addWidget(page)
        self._cat_group.idClicked.connect(self.tool_stack.setCurrentIndex)
        cat_row.addStretch(1)
        rl.addLayout(cat_row)
        rl.addWidget(self.tool_stack, stretch=1)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        # Filmstrip
        film_box = QGroupBox("Filmstrip")
        film_box.setFixedHeight(160)
        film_box.setStyleSheet("""
            QGroupBox {
                color: #c8c8c8; border: 1px solid #3a3a3a; border-radius: 4px;
                margin-top: 10px; padding-top: 4px; font-size: 12px; font-weight: 600;
                letter-spacing: 0px;
            }
            QGroupBox::title {
                subcontrol-origin: margin; left: 10px; padding: 0 6px;
                color: #b0b0b0; letter-spacing: 0px;
            }
        """)
        fb = QVBoxLayout(film_box)
        self.filmstrip = QListWidget()
        self.filmstrip.setViewMode(QListWidget.ViewMode.IconMode)
        self.filmstrip.setFlow(QListWidget.Flow.LeftToRight)
        self.filmstrip.setWrapping(False)
        self.filmstrip.setIconSize(QSize(96, 96))
        self.filmstrip.setGridSize(QSize(108, 118))
        self.filmstrip.setUniformItemSizes(True)
        self.filmstrip.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.filmstrip.setMovement(QListWidget.Movement.Static)
        self.filmstrip.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.filmstrip.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.filmstrip.setSpacing(4)
        self.filmstrip.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.filmstrip.itemClicked.connect(self.on_thumb_clicked)

        fr = QHBoxLayout()
        fr.addWidget(QLabel("Filmstrip min ★"))
        self.film_rating_filter = QComboBox()
        self.film_rating_filter.addItem("All", 0)
        for i in range(1, 6):
            self.film_rating_filter.addItem(f"{i}+", i)
        self.film_rating_filter.currentIndexChanged.connect(self._apply_filmstrip_filter)
        fr.addWidget(self.film_rating_filter)
        fr.addStretch(1)
        fb.addLayout(fr)
        fb.addWidget(self.filmstrip)
        outer.addWidget(film_box)

        self.preview_label = self.preview  # alias for older code paths

        self.mode_stack.addWidget(develop_page)
        self.mode_stack.addWidget(self._build_library_page())
        self.mode_stack.setCurrentIndex(0)

    def _build_library_page(self):
        """Date-tree + thumbnail grid library browser."""
        page = QWidget()
        hl = QHBoxLayout(page)
        hl.setContentsMargins(8, 8, 8, 8)
        hl.setSpacing(8)

        left = QWidget()
        left.setMinimumWidth(220)
        left.setMaximumWidth(300)
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        title = QLabel("Library")
        title.setStyleSheet("color:#8af; font-weight:600; font-size:13px;")
        ll.addWidget(title)

        scan_row = QHBoxLayout()
        scan_btn = QPushButton("Scan Folder…")
        scan_btn.clicked.connect(self.scan_library_folder)
        scan_row.addWidget(scan_btn)
        refresh_btn = QPushButton("Refresh")
        refresh_btn.clicked.connect(self.refresh_library_tree)
        scan_row.addWidget(refresh_btn)
        ll.addLayout(scan_row)

        self.lib_filter_rejected = QCheckBox("Show rejected")
        self.lib_filter_rejected.toggled.connect(self.refresh_library_tree)
        ll.addWidget(self.lib_filter_rejected)
        self.lib_smart_combo = QComboBox()
        self.lib_smart_combo.addItem("All photos", "all")
        self.lib_smart_combo.addItem("Picked (✓)", "picked")
        self.lib_smart_combo.addItem("Rejected only", "rejected")
        self.lib_smart_combo.addItem("Rated 3+", "rated3")
        self.lib_smart_combo.addItem("Rated 5", "rated5")
        self.lib_smart_combo.addItem("Unrated", "unrated")
        self.lib_smart_combo.currentIndexChanged.connect(self._on_lib_filter_changed)
        ll.addWidget(self.lib_smart_combo)

        self.lib_min_rating = QComboBox()
        self.lib_min_rating.addItem("Any rating", 0)
        for i in range(1, 6):
            self.lib_min_rating.addItem(("★" * i) + "+", i)
        self.lib_min_rating.currentIndexChanged.connect(self._on_lib_filter_changed)
        ll.addWidget(self.lib_min_rating)
        self.lib_search = QLineEdit()
        self.lib_search.setPlaceholderText("Search filename, keywords, camera…")
        self.lib_search.returnPressed.connect(self._on_lib_search)
        ll.addWidget(self.lib_search)
        search_btn = QPushButton("Search")
        search_btn.clicked.connect(self._on_lib_search)
        ll.addWidget(search_btn)
        kw_row = QHBoxLayout()
        self.lib_keywords_edit = QLineEdit()
        self.lib_keywords_edit.setPlaceholderText("Keywords for selected (comma-separated)")
        kw_row.addWidget(self.lib_keywords_edit, 1)
        kw_save = QPushButton("Save KW")
        kw_save.clicked.connect(self._on_lib_save_keywords)
        kw_row.addWidget(kw_save)
        ll.addLayout(kw_row)

        self.lib_date_tree = QTreeView()
        self.lib_date_tree.setHeaderHidden(True)
        self.lib_date_tree.setStyleSheet(
            "QTreeView { background:#141414; border:1px solid #2b2b2b; color:#ccc; font-size:12px; }"
            "QTreeView::item:hover { background:#2a2a2a; }"
            "QTreeView::item:selected { background:#2a5080; color:#fff; }"
        )
        from PyQt6.QtGui import QStandardItemModel
        self._lib_tree_model = QStandardItemModel()
        self.lib_date_tree.setModel(self._lib_tree_model)
        self.lib_date_tree.clicked.connect(self._on_lib_date_clicked)
        ll.addWidget(self.lib_date_tree, stretch=1)

        self.lib_status = QLabel("Scan a folder to build the catalog.")
        self.lib_status.setWordWrap(True)
        self.lib_status.setStyleSheet("color:#888; font-size:11px;")
        ll.addWidget(self.lib_status)
        hl.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        bar = QHBoxLayout()
        self.lib_heading = QLabel("All dates")
        self.lib_heading.setStyleSheet("color:#ddd; font-weight:600; font-size:13px;")
        bar.addWidget(self.lib_heading, stretch=1)
        rate_lbl = QLabel("Rating:")
        rate_lbl.setStyleSheet("color:#aaa;")
        bar.addWidget(rate_lbl)
        for i in range(0, 6):
            b = QToolButton()
            b.setText("0" if i == 0 else ("★" * i))
            b.setToolTip("Clear rating" if i == 0 else f"Set rating {i}")
            b.clicked.connect(lambda checked=False, r=i: self._lib_set_rating(r))
            bar.addWidget(b)
        rej = QPushButton("Reject")
        rej.clicked.connect(self._lib_toggle_reject)
        bar.addWidget(rej)
        open_btn = QPushButton("Open in Develop")
        open_btn.clicked.connect(self._lib_open_selected)
        bar.addWidget(open_btn)
        trash_btn = QPushButton("Move to Trash")
        trash_btn.setToolTip("Move selected files to the system trash and remove from catalog.")
        trash_btn.clicked.connect(self._lib_move_to_trash)
        bar.addWidget(trash_btn)
        rm_btn = QPushButton("Remove from Library")
        rm_btn.setToolTip("Remove from catalog only — files stay on disk.")
        rm_btn.clicked.connect(self._lib_remove_from_catalog)
        bar.addWidget(rm_btn)
        exp_btn = QPushButton("Export Selected…")
        exp_btn.clicked.connect(self._lib_export_selected)
        bar.addWidget(exp_btn)
        stack_btn = QPushButton("Focus Stack…")
        stack_btn.setToolTip("Focus-stack the selected library photos")
        stack_btn.clicked.connect(self._lib_focus_stack)
        bar.addWidget(stack_btn)
        pano_btn = QPushButton("Panorama…")
        pano_btn.setToolTip("Stitch selected library photos (OpenCV)")
        pano_btn.clicked.connect(self._lib_panorama)
        bar.addWidget(pano_btn)
        rl.addLayout(bar)

        self.lib_grid = QListWidget()
        self.lib_grid.setViewMode(QListWidget.ViewMode.IconMode)
        self.lib_grid.setFlow(QListWidget.Flow.LeftToRight)
        self.lib_grid.setWrapping(True)
        self.lib_grid.setIconSize(QSize(128, 128))
        self.lib_grid.setGridSize(QSize(150, 170))  # room for icon + filename
        self.lib_grid.setUniformItemSizes(True)
        self.lib_grid.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.lib_grid.setMovement(QListWidget.Movement.Static)
        self.lib_grid.setSpacing(10)
        self.lib_grid.setWordWrap(True)
        self.lib_grid.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.lib_grid.setStyleSheet(
            "QListWidget { background:#121212; border:1px solid #2b2b2b; color:#ddd; }"
            "QListWidget::item { padding:4px; }"
            "QListWidget::item:selected { background:#2a5080; }"
        )
        self.lib_grid.itemDoubleClicked.connect(self._lib_open_item)
        self.lib_grid.itemSelectionChanged.connect(self._lib_selection_changed)
        rl.addWidget(self.lib_grid, stretch=1)
        hl.addWidget(right, stretch=1)
        return page

    def _add_slider(self, layout, key, label, lo, hi, step, decimals, default):
        row = SliderRow(label, lo, hi, default, step,
                        lambda val, k=key: self.on_slider(k, val), decimals)
        self.sliders[key] = row
        layout.addWidget(row)
        return row

    # ===== LIGHT TAB =====
    def _build_light_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(4)

        # Exposure Compensation
        box, v = collapsible_group("Exposure Compensation", layout)
        self._add_slider(v, "exposure", "Exposure", -5.0, 5.0, 0.05, 2, 0.0)

        box, v = collapsible_group("Zebra Highlight Adjustment", layout)
        zebra_hint = QLabel(
            "Changes exposure only in the striped luminance range. Lower Exposure to recover "
            "bright areas; Feather softens the boundary."
        )
        zebra_hint.setWordWrap(True)
        zebra_hint.setStyleSheet("color:#888; font-size:11px;")
        v.addWidget(zebra_hint)
        self._add_slider(v, "zebra_threshold", "Zebra threshold %", 50.0, 100.0, 1, 0, 95.0)
        self._add_slider(v, "zebra_exposure", "Exposure (EV)", -5.0, 2.0, 0.05, 2, 0.0)
        self._add_slider(v, "zebra_feather", "Feather %", 0.0, 25.0, 1, 0, 5.0)
        zebra_btn = QPushButton("Show / hide zebra overlay (Z)")
        zebra_btn.clicked.connect(lambda: self.toggle_zebras(not getattr(self.preview, "show_zebras", False)))
        v.addWidget(zebra_btn)

        # Smart Lighting
        box, v = collapsible_group("Smart Lighting", layout)
        self._add_slider(v, "smart_light", "Intensity", 0.0, 100.0, 1, 0, 0.0)

        # Selective Tone
        box, v = collapsible_group("Selective Tone", layout)
        self._add_slider(v, "highlights", "Highlights", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "whites", "Whites", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "shadows", "Shadows", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "blacks", "Blacks", -100.0, 100.0, 1, 0, 0.0)

        # ClearView Plus
        box, v = collapsible_group("ClearView Plus", layout, checked=False)
        self._add_slider(v, "clearview", "Intensity", 0.0, 100.0, 1, 0, 0.0)

        # Contrast
        box, v = collapsible_group("Contrast", layout)
        self._add_slider(v, "contrast", "Contrast", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "microcontrast", "Microcontrast", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "clarity", "Clarity", -100.0, 100.0, 1, 0, 0.0)

        # Tone Curve (graph + Lightroom-style region sliders)
        box, v = collapsible_group("Tone Curve", layout)
        self.tone_curve = ToneCurveWidget()
        self.tone_curve.curveChanged.connect(self.on_curve_changed)
        v.addWidget(self.tone_curve)
        region_lbl = QLabel("Region")
        region_lbl.setStyleSheet("color:#9cf; font-weight:600; font-size:12px; margin-top:6px;")
        v.addWidget(region_lbl)
        self._add_slider(v, "curve_highlights", "Highlights", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "curve_lights", "Lights", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "curve_darks", "Darks", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "curve_shadows", "Shadows", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "gamma", "Gamma", 0.3, 2.5, 0.05, 2, 1.0)

        # Vignetting
        box, v = collapsible_group("HDR Look", layout, checked=False)
        self._add_slider(v, "hdr_look", "Amount", 0.0, 100.0, 1, 0, 0.0)
        hdr_hint = QLabel("Single-image HDR-style tone mapping (local contrast + shadow lift).")
        hdr_hint.setWordWrap(True)
        hdr_hint.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(hdr_hint)

        box, v = collapsible_group("Vignetting", layout, checked=False)
        self._add_slider(v, "vignette", "Intensity", 0.0, 100.0, 1, 0, 0.0)

        layout.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    # ===== COLOR TAB =====
    def _build_color_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        layout = QVBoxLayout(inner)

        # White Balance
        box, v = collapsible_group("White Balance", layout)
        self.wb_as_shot_cb = QCheckBox("As Shot (camera WB)")
        self.wb_as_shot_cb.setChecked(True)
        self.wb_as_shot_cb.toggled.connect(self.on_wb_as_shot)
        v.addWidget(self.wb_as_shot_cb)
        self._add_slider(v, "temperature", "Temperature (K)", 2000, 12000, 50, 0, 5500)
        self._add_slider(v, "tint", "Tint", -150, 150, 1, 0, 0)
        creative_hint = QLabel("Creative adjustment (relative; presets use these controls)")
        creative_hint.setWordWrap(True)
        creative_hint.setStyleSheet("color:#888; font-size:11px; margin-top:5px;")
        v.addWidget(creative_hint)
        self._add_slider(v, "creative_temperature", "Creative warmth", -100, 100, 1, 0, 0)
        self._add_slider(v, "creative_tint", "Creative tint", -100, 100, 1, 0, 0)

        # Color Accentuation
        box, v = collapsible_group("Color Accentuation", layout)
        self._add_slider(v, "vibrance", "Vibrancy", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "saturation", "Saturation", -100.0, 100.0, 1, 0, 0.0)

        # HSL / Color — full 8-channel panel (Lightroom-style)
        box, v = collapsible_group("HSL / Color", layout)
        self._build_hsl_panel(v)

        # Soft Proofing
        box, v = collapsible_group("Soft Proofing", layout, checked=False)
        self.soft_proof_cb = QCheckBox("Enable soft proof")
        self.soft_proof_cb.setToolTip("Simulate how the image may look in another color space / print.")
        self.soft_proof_cb.toggled.connect(self._on_soft_proof)
        v.addWidget(self.soft_proof_cb)
        row = QHBoxLayout()
        row.addWidget(QLabel("Profile"))
        self.proof_combo = QComboBox()
        self.proof_combo.addItems(["sRGB", "DisplayP3", "AdobeRGB", "CMYK", "Gray"])
        self.proof_combo.currentTextChanged.connect(self._on_proof_profile)
        row.addWidget(self.proof_combo, 1)
        v.addLayout(row)
        self.gamut_warn_cb = QCheckBox("Gamut warning (magenta)")
        self.gamut_warn_cb.setToolTip("Highlight colors that change a lot under the proof simulation.")
        self.gamut_warn_cb.toggled.connect(self._on_gamut_warning)
        v.addWidget(self.gamut_warn_cb)
        tip = QLabel("Approximate soft-proof (no ICC file required). Use for a relative check, not press certification.")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(tip)

        layout.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def _build_detail_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        layout = QVBoxLayout(inner)

        box, v = collapsible_group("Noise Reduction", layout)
        hint = QLabel(
            "Luminance softens grain; Chrominance removes color noise. "
            "Strength switches to NLM (slower, stronger). Detail Recovery "
            "brings fine structure back after NR."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:11px;")
        v.addWidget(hint)
        self._add_slider(v, "denoise_luminance", "Luminance", 0.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "denoise_chroma", "Chrominance", 0.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "denoise_strength", "Strength (NLM bias)", 0.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "denoise_detail", "Detail Recovery", 0.0, 100.0, 1, 0, 50.0)
        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method"))
        self.denoise_method_combo = QComboBox()
        self.denoise_method_combo.addItem("Auto (bilateral → NLM)", "auto")
        self.denoise_method_combo.addItem("Bilateral (fast, edge-aware)", "bilateral")
        self.denoise_method_combo.addItem("NLM (strong, slower)", "nlm")
        self.denoise_method_combo.currentIndexChanged.connect(self._on_denoise_method)
        method_row.addWidget(self.denoise_method_combo, 1)
        v.addLayout(method_row)

        box, v = collapsible_group("Capture Sharpening", layout)
        hint2 = QLabel(
            "Edge-masked unsharp mask. Raise Threshold to avoid sharpening smooth areas. "
            "Detail boosts fine structure at a small radius."
        )
        hint2.setWordWrap(True)
        hint2.setStyleSheet("color:#888; font-size:11px;")
        v.addWidget(hint2)
        self._add_slider(v, "sharpen_intensity", "Amount", 0.0, 200.0, 1, 0, 0.0)
        self._add_slider(v, "sharpen_radius", "Radius", 0.1, 5.0, 0.1, 1, 1.0)
        self._add_slider(v, "sharpen_threshold", "Masking / Threshold", 0.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "sharpen_detail", "Detail", 0.0, 100.0, 1, 0, 0.0)

        box, v = collapsible_group("Output Sharpening", layout)
        hint3 = QLabel("Applied last (before grain). Use for screen or print output.")
        hint3.setWordWrap(True)
        hint3.setStyleSheet("color:#888; font-size:11px;")
        v.addWidget(hint3)
        self._add_slider(v, "output_sharpen", "Output amount", 0.0, 100.0, 1, 0, 0.0)

        box, v = collapsible_group("Presets", layout)
        prow = QHBoxLayout()
        for label, fn in (
            ("Light NR", self._detail_preset_light_nr),
            ("Strong NR", self._detail_preset_strong_nr),
            ("Portrait", self._detail_preset_portrait),
            ("Landscape", self._detail_preset_landscape),
        ):
            b = QPushButton(label)
            b.clicked.connect(fn)
            prow.addWidget(b)
        v.addLayout(prow)

        layout.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    # ===== GEOMETRY TAB =====
    def _build_geometry_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        layout = QVBoxLayout(inner)

        box, v = collapsible_group("Horizon", layout)
        self._add_slider(v, "horizon", "Angle (°)", -45.0, 45.0, 0.1, 1, 0.0)
        self.level_horizon_btn = QPushButton("Level: draw line on image")
        self.level_horizon_btn.setToolTip("Drag a line along the horizon; angle is applied automatically.")
        self.level_horizon_btn.clicked.connect(self.start_horizon_line)
        v.addWidget(self.level_horizon_btn)
        self.show_grid_cb = QCheckBox("Show grid (rule of thirds + center)")
        self.show_grid_cb.setToolTip("Overlay grid and center crosshair to help straighten the horizon.")
        self.show_grid_cb.toggled.connect(self._on_show_grid_toggled)
        v.addWidget(self.show_grid_cb)
        self.show_spiral_cb = QCheckBox("Show Fibonacci / golden spiral")
        self.show_spiral_cb.setToolTip(
            "Golden-ratio spiral locked to the image. Drag the yellow center to move, "
            "blue corner to resize. Use orientation to flip/rotate."
        )
        self.show_spiral_cb.toggled.connect(self._on_show_spiral_toggled)
        v.addWidget(self.show_spiral_cb)
        gc_row = QHBoxLayout()
        gc_row.addWidget(QLabel("Guide color"))
        self.guide_color_combo = QComboBox()
        self.guide_color_combo.addItem("Yellow (default)", "yellow")
        self.guide_color_combo.addItem("White", "white")
        self.guide_color_combo.addItem("Cyan", "cyan")
        self.guide_color_combo.addItem("Black", "black")
        self.guide_color_combo.currentIndexChanged.connect(self._on_guide_color)
        gc_row.addWidget(self.guide_color_combo, 1)
        v.addLayout(gc_row)
        spiral_row = QHBoxLayout()
        spiral_row.addWidget(QLabel("Spiral size"))
        self.spiral_scale_slider = SliderRow(
            "", 15.0, 150.0, 85.0, 1,
            lambda val: self._on_spiral_scale(val), 0,
        )
        # compact: hide the empty name by using the row directly
        spiral_row.addWidget(self.spiral_scale_slider, 1)
        v.addLayout(spiral_row)
        orient_row = QHBoxLayout()
        orient_row.addWidget(QLabel("Orientation"))
        self.spiral_orient_combo = QComboBox()
        self.spiral_orient_combo.addItems([
            "A (default)", "B", "C", "D",
            "A mirrored", "B mirrored", "C mirrored", "D mirrored",
        ])
        self.spiral_orient_combo.currentIndexChanged.connect(self._on_spiral_orient)
        orient_row.addWidget(self.spiral_orient_combo, 1)
        v.addLayout(orient_row)
        tip = QLabel("Drag yellow handle to move • blue corner to resize (on the image).")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(tip)

        box, v = collapsible_group("Optics / Lens", layout)
        self.lens_auto_cb = QCheckBox("Try Lensfun auto-correction (if installed)")
        self.lens_auto_cb.setToolTip("Requires: pip install lensfunpy. Uses EXIF camera/lens when available.")
        self.lens_auto_cb.toggled.connect(self._on_lens_auto)
        v.addWidget(self.lens_auto_cb)
        self._add_slider(v, "ca_amount", "Chromatic aberration", -100.0, 100.0, 1, 0, 0.0)
        tip_o = QLabel("CA is a simple radial R/B shift. Lensfun needs lensfunpy + DB.")
        tip_o.setWordWrap(True)
        tip_o.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(tip_o)

        box, v = collapsible_group("Crop", layout)
        row = QHBoxLayout()
        self.crop_tool_btn = QPushButton("Crop Tool")
        self.crop_tool_btn.setCheckable(True)
        self.crop_tool_btn.toggled.connect(self.toggle_crop_mode)
        row.addWidget(self.crop_tool_btn)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.clear_crop)
        row.addWidget(clear_btn)
        v.addLayout(row)
        ar_row = QHBoxLayout()
        ar_row.addWidget(QLabel("Aspect"))
        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems([
            "Free", "Original", "1:1", "5:4", "4:3", "3:2", "16:9", "16:10",
            "2:3 (portrait)", "3:4 (portrait)", "9:16 (portrait)",
            "A4", "Square 5×5",
        ])
        ar_row.addWidget(self.aspect_combo, 1)
        v.addLayout(ar_row)

        box, v = collapsible_group("Distortion", layout, checked=False)
        self._add_slider(v, "distortion", "Barrel / pincushion", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "wide_angle", "Wide-angle edge stretch", -100.0, 100.0, 1, 0, 0.0)
        distortion_tip = QLabel("Positive edge stretch opens the sides; negative values compress exaggerated edges.")
        distortion_tip.setWordWrap(True)
        distortion_tip.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(distortion_tip)

        box, v = collapsible_group("Perspective", layout, checked=False)
        self._add_slider(v, "perspective", "Vertical", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "perspective_horizontal", "Horizontal", -100.0, 100.0, 1, 0, 0.0)
        upright_row = QHBoxLayout()
        self.auto_upright_btn = QPushButton("Auto Upright")
        self.auto_upright_btn.setToolTip("Analyze strong architectural lines and correct level and vertical convergence.")
        self.auto_upright_btn.clicked.connect(self.auto_architectural_upright)
        upright_row.addWidget(self.auto_upright_btn)
        self.keystone_tool_btn = QPushButton("4-Corner Tool")
        self.keystone_tool_btn.setCheckable(True)
        self.keystone_tool_btn.setToolTip("Drag TL, TR, BR, and BL handles around a photographed rectangle to straighten it.")
        self.keystone_tool_btn.toggled.connect(self.toggle_keystone_mode)
        upright_row.addWidget(self.keystone_tool_btn)
        v.addLayout(upright_row)
        clear_keystone = QPushButton("Clear 4-Corner Correction")
        clear_keystone.clicked.connect(self.clear_keystone)
        v.addWidget(clear_keystone)

        box, v = collapsible_group("Edge Warp", layout, checked=False)
        self._add_slider(v, "warp_top", "Top edge", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "warp_bottom", "Bottom edge", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "warp_left", "Left edge", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "warp_right", "Right edge", -100.0, 100.0, 1, 0, 0.0)
        warp_tip = QLabel("Fine-tune converging buildings, signs, frames, and off-axis subjects. Reflected edges prevent empty wedges.")
        warp_tip.setWordWrap(True)
        warp_tip.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(warp_tip)

        box, v = collapsible_group("Tilt-Shift / Diorama", layout, checked=False)
        self._add_slider(v, "diorama_strength", "Blur strength", 0.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "diorama_position", "Focus position", 0.0, 100.0, 1, 0, 50.0)
        self._add_slider(v, "diorama_width", "Focus band width", 5.0, 90.0, 1, 0, 30.0)
        self._add_slider(v, "diorama_angle", "Focus band angle", -45.0, 45.0, 1, 0, 0.0)
        diorama_tip = QLabel("Places a rotatable sharp band through the scene and progressively blurs both sides.")
        diorama_tip.setWordWrap(True)
        diorama_tip.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(diorama_tip)

        box, v = collapsible_group("Geometry Output", layout, checked=False)
        self.geometry_auto_crop_cb = QCheckBox("Auto-crop transformed edge margins")
        self.geometry_auto_crop_cb.setToolTip("Trim reflected safety margins after strong perspective and distortion corrections.")
        self.geometry_auto_crop_cb.toggled.connect(self._on_geometry_auto_crop)
        v.addWidget(self.geometry_auto_crop_cb)

        layout.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    # ===== EFFECTS TAB =====
    def _build_effects_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        layout = QVBoxLayout(inner)

        # ----- Infrared -----
        box, v = collapsible_group("Infrared", layout, checked=True)
        ir_hint = QLabel(
            "For IR-converted cameras or IR filters. Channel swap is the classic false-color start."
        )
        ir_hint.setWordWrap(True)
        ir_hint.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(ir_hint)
        swap_row = QHBoxLayout()
        swap_row.addWidget(QLabel("Channel swap"))
        self.ir_swap_combo = QComboBox()
        for name, key in (
            ("None", "none"),
            ("R ↔ B (classic IR)", "rb"),
        ):
            self.ir_swap_combo.addItem(name, key)
        self.ir_swap_combo.currentIndexChanged.connect(self._on_ir_swap)
        swap_row.addWidget(self.ir_swap_combo, 1)
        v.addLayout(swap_row)
        self._add_slider(v, "ir_false_color", "False color", 0.0, 100.0, 1, 0, 0.0)
        self.ir_mono_cb = QCheckBox("Mono IR (red-weighted)")
        self.ir_mono_cb.toggled.connect(self._on_ir_mono)
        v.addWidget(self.ir_mono_cb)
        ir_btn_row = QHBoxLayout()
        for label, fn in (
            ("Wood effect", self._ir_preset_wood),
            ("Gold/Blue", self._ir_preset_gold_blue),
            ("Mono IR", self._ir_preset_mono),
            ("Reset IR", self._ir_preset_reset),
        ):
            b = QPushButton(label)
            b.clicked.connect(fn)
            ir_btn_row.addWidget(b)
        v.addLayout(ir_btn_row)

        # ----- Astro -----
        box, v = collapsible_group("Astro", layout, checked=True)
        astro_hint = QLabel(
            "For night-sky / linear-ish data. Stretch lifts faint detail; "
            "background removal softens gradients and light pollution."
        )
        astro_hint.setWordWrap(True)
        astro_hint.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(astro_hint)
        self._add_slider(v, "astro_stretch", "Stretch (asinh)", 0.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "astro_bg_remove", "Background / gradient remove", 0.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "astro_star_emphasis", "Star emphasis", 0.0, 100.0, 1, 0, 0.0)
        astro_btn_row = QHBoxLayout()
        for label, fn in (
            ("Milky Way", self._astro_preset_milkyway),
            ("DSO soft", self._astro_preset_dso),
            ("Reset Astro", self._astro_preset_reset),
        ):
            b = QPushButton(label)
            b.clicked.connect(fn)
            astro_btn_row.addWidget(b)
        v.addLayout(astro_btn_row)

        box, v = collapsible_group("Vignetting", layout, checked=False)
        self._add_slider(v, "vignette", "Intensity", 0.0, 100.0, 1, 0, 0.0)

        box, v = collapsible_group("Film Grain", layout, checked=False)
        self._add_slider(v, "film_grain", "Amount", 0.0, 100.0, 1, 0, 0.0)

        box, v = collapsible_group("Black & White", layout, checked=False)
        self.bw_cb = QCheckBox("Convert to black & white")
        self.bw_cb.toggled.connect(self._on_bw)
        v.addWidget(self.bw_cb)

        zone_title = QLabel("Zone System (Ansel Adams)")
        zone_title.setStyleSheet("color:#8af; font-weight:600; font-size:12px;")
        v.addWidget(zone_title)
        self.zone_enabled_cb = QCheckBox("Enable zone mapping")
        self.zone_enabled_cb.setToolTip(
            "Map tones onto the classic 0–X zone scale (Zone V ≈ middle gray)."
        )
        self.zone_enabled_cb.toggled.connect(self._on_zone_enabled)
        v.addWidget(self.zone_enabled_cb)
        self._add_slider(v, "zone_placement", "Place midtones on zone", 0.0, 10.0, 0.1, 1, 5.0)
        self._add_slider(v, "zone_expansion", "Expansion (N− … N+)", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "zone_snap", "Snap to zone centers", 0.0, 100.0, 1, 0, 0.0)
        filt_row = QHBoxLayout()
        filt_row.addWidget(QLabel("B&W filter"))
        self.zone_filter_combo = QComboBox()
        for name, key in (
            ("None (panchromatic)", "none"),
            ("Yellow", "yellow"),
            ("Orange", "orange"),
            ("Red", "red"),
            ("Green", "green"),
            ("Blue", "blue"),
        ):
            self.zone_filter_combo.addItem(name, key)
        self.zone_filter_combo.currentIndexChanged.connect(self._on_zone_filter)
        filt_row.addWidget(self.zone_filter_combo, 1)
        v.addLayout(filt_row)
        self.zone_overlay_cb = QCheckBox("False-color zone overlay (preview)")
        self.zone_overlay_cb.toggled.connect(self._on_zone_overlay)
        v.addWidget(self.zone_overlay_cb)
        preset_row = QHBoxLayout()
        for label, place, exp in (
            ("Zone V", 5.0, 0.0),
            ("N−", 5.0, -40.0),
            ("N+", 5.0, 40.0),
            ("High key", 6.5, 20.0),
            ("Low key", 3.5, 20.0),
        ):
            btn = QPushButton(label)
            btn.clicked.connect(lambda _=False, p=place, e=exp: self._apply_zone_preset(p, e))
            preset_row.addWidget(btn)
        v.addLayout(preset_row)
        zone_hint = QLabel(
            "Zones 0–X: pure black → pure white. Place midtones on a zone, "
            "expand/compress like N+/N− development."
        )
        zone_hint.setWordWrap(True)
        zone_hint.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(zone_hint)

        box, v = collapsible_group("Rotate", layout, checked=False)
        row = QHBoxLayout()
        rot_l = QPushButton("⟲ 90° CCW")
        rot_l.clicked.connect(lambda: self._rotate(-1))
        rot_r = QPushButton("90° CW ⟳")
        rot_r.clicked.connect(lambda: self._rotate(1))
        row.addWidget(rot_l)
        row.addWidget(rot_r)
        v.addLayout(row)

        layout.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    # ===== LOCAL TAB =====
    def _build_local_tab(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(6)

        box, v = collapsible_group("Local Adjustments", layout)
        self.local_active_cb = QCheckBox("Enable Control Points")
        self.local_active_cb.toggled.connect(self._on_local_active_toggled)
        v.addWidget(self.local_active_cb)
        
        self.local_points_list = QListWidget()
        self.local_points_list.setMaximumHeight(120)
        self.local_points_list.setStyleSheet("""
            QListWidget {
                background: #141414; border: 1px solid #333; border-radius: 3px;
                color: #ccc; font-size: 11px;
            }
            QListWidget::item { padding: 4px; }
            QListWidget::item:selected { background: #2a5080; color: #fff; }
        """)
        self.local_points_list.currentRowChanged.connect(self._on_local_list_selection)
        v.addWidget(self.local_points_list)

        btn_row = QHBoxLayout()
        self.add_pt_btn = QPushButton("+ Add CP")
        self.add_pt_btn.clicked.connect(self._on_add_local_clicked)
        btn_row.addWidget(self.add_pt_btn)
        
        self.del_pt_btn = QPushButton("Delete")
        self.del_pt_btn.clicked.connect(self._on_delete_local_clicked)
        btn_row.addWidget(self.del_pt_btn)
        
        self.clear_pt_btn = QPushButton("Clear All")
        self.clear_pt_btn.clicked.connect(self._on_clear_local_clicked)
        btn_row.addWidget(self.clear_pt_btn)
        v.addLayout(btn_row)

        self.local_sliders_box = QGroupBox("Selected Point Adjustment")
        self.local_sliders_box.setStyleSheet("""
            QGroupBox {
                color: #ccc; border: 1px solid #3a3a3a; border-radius: 4px;
                margin-top: 8px; font-weight: 600; font-size: 12px;
                padding-top: 8px;
            }
        """)
        lsv = QVBoxLayout(self.local_sliders_box)
        
        self.local_sliders = {}
        
        def add_local_slider(name, label, lo, hi, step, decimals, default):
            row = SliderRow(label, lo, hi, default, step,
                            lambda val, n=name: self._on_local_slider_changed(n, val), decimals)
            self.local_sliders[name] = row
            lsv.addWidget(row)

        add_local_slider("local_radius", "Size (Radius %)", 1.0, 100.0, 1.0, 0, 15.0)
        add_local_slider("local_feather", "Feather %", 0.0, 100.0, 1.0, 0, 50.0)
        add_local_slider("local_exposure", "Exposure (EV)", -3.0, 3.0, 0.05, 2, 0.0)
        add_local_slider("local_contrast", "Contrast", -100.0, 100.0, 1, 0, 0.0)
        add_local_slider("local_saturation", "Saturation", -100.0, 100.0, 1, 0, 0.0)
        add_local_slider("local_clarity", "Clarity", -100.0, 100.0, 1, 0, 0.0)
        
        v.addWidget(self.local_sliders_box)
        layout.addStretch(1)
        
        
        box, v = collapsible_group("Adjustment Brush", layout)
        tipb = QLabel("Shift+B, then paint on the image. Wheel adjusts size while brushing.")
        tipb.setWordWrap(True)
        tipb.setStyleSheet("color:#888; font-size:11px;")
        v.addWidget(tipb)
        self.brush_list = QListWidget()
        self.brush_list.setMaximumHeight(90)
        self.brush_list.currentRowChanged.connect(self._on_brush_list_selection)
        v.addWidget(self.brush_list)
        brow = QHBoxLayout()
        b_en = QPushButton("Enable brush")
        b_en.clicked.connect(lambda: self.toggle_brush_mode(True))
        b_del = QPushButton("Delete")
        b_del.clicked.connect(self._on_delete_brush)
        b_clr = QPushButton("Clear")
        b_clr.clicked.connect(self._on_clear_brushes)
        brow.addWidget(b_en)
        brow.addWidget(b_del)
        brow.addWidget(b_clr)
        v.addLayout(brow)
        self.brush_erase_cb = QCheckBox("Eraser mode (or right-drag)")
        self.brush_erase_cb.toggled.connect(self._on_brush_erase)
        v.addWidget(self.brush_erase_cb)
        self.brush_mask_only_cb = QCheckBox("Show mask only")
        self.brush_mask_only_cb.toggled.connect(self._on_brush_mask_only)
        v.addWidget(self.brush_mask_only_cb)
        inv_row = QHBoxLayout()
        self.brush_invert_btn = QPushButton("Invert selected mask")
        self.brush_invert_btn.setToolTip("Apply brush adjustments outside the painted area instead.")
        self.brush_invert_btn.clicked.connect(self._on_brush_invert)
        inv_row.addWidget(self.brush_invert_btn)
        self.brush_color_range_cb = QCheckBox("Color-range mask (sample painted subject)")
        self.brush_color_range_cb.toggled.connect(
            lambda checked: self._on_brush_flag("color_range", checked)
        )
        inv_row.addWidget(self.brush_color_range_cb)
        v.addLayout(inv_row)
        intersect_row = QHBoxLayout()
        self.brush_intersect_btn = QPushButton("Intersect with previous mask")
        self.brush_intersect_btn.setToolTip("Reuse the preceding mask as a boundary for this mask.")
        self.brush_intersect_btn.clicked.connect(self._on_brush_intersect_previous)
        intersect_row.addWidget(self.brush_intersect_btn)
        v.addLayout(intersect_row)
        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Size"))
        self.brush_size_slider = SliderRow("", 1.0, 30.0, 5.0, 0.5,
            lambda val: self._on_brush_size(val), 1)
        size_row.addWidget(self.brush_size_slider, 1)
        v.addLayout(size_row)
        hard_row = QHBoxLayout()
        hard_row.addWidget(QLabel("Hardness"))
        self.brush_hard_slider = SliderRow("", 0.0, 100.0, 70.0, 1,
            lambda val: self._on_brush_hard(val), 0)
        hard_row.addWidget(self.brush_hard_slider, 1)
        v.addLayout(hard_row)
        self.brush_sliders = {}
        self.brush_sliders_box = QWidget()
        bsl = QVBoxLayout(self.brush_sliders_box)
        bsl.setContentsMargins(0, 0, 0, 0)
        for key, label, lo, hi, step, dec in (
            ("feather", "Mask feather", 0.0, 100.0, 1, 0),
            ("edge_refine", "Edge-aware refine", 0.0, 100.0, 1, 0),
            ("luminance_min", "Luminance minimum", 0.0, 100.0, 1, 0),
            ("luminance_max", "Luminance maximum", 0.0, 100.0, 1, 0),
            ("color_tolerance", "Color tolerance", 1.0, 100.0, 1, 0),
            ("exposure", "Exposure", -2.0, 2.0, 0.05, 2),
            ("contrast", "Contrast", -100.0, 100.0, 1, 0),
            ("saturation", "Saturation", -100.0, 100.0, 1, 0),
            ("clarity", "Clarity", -100.0, 100.0, 1, 0),
            ("temperature", "Temp shift", -100.0, 100.0, 1, 0),
        ):
            row = SliderRow(label, lo, hi, 0.0, step,
                            lambda val, k=key: self._on_brush_adj(k, val), dec)
            self.brush_sliders[key] = row
            bsl.addWidget(row)
        self.brush_sliders_box.setEnabled(False)
        v.addWidget(self.brush_sliders_box)
        preset_box = QGroupBox("Preset on selected mask")
        preset_layout = QVBoxLayout(preset_box)
        self.brush_preset_label = QLabel("No local preset")
        self.brush_preset_label.setWordWrap(True)
        self.brush_preset_label.setStyleSheet("color:#999; font-size:11px;")
        preset_layout.addWidget(self.brush_preset_label)
        preset_buttons = QHBoxLayout()
        self.brush_preset_apply_btn = QPushButton("Apply preset…")
        self.brush_preset_apply_btn.clicked.connect(self._apply_preset_to_brush_mask)
        self.brush_preset_clear_btn = QPushButton("Clear preset")
        self.brush_preset_clear_btn.clicked.connect(self._clear_brush_mask_preset)
        preset_buttons.addWidget(self.brush_preset_apply_btn)
        preset_buttons.addWidget(self.brush_preset_clear_btn)
        preset_layout.addLayout(preset_buttons)
        self.brush_preset_strength = SliderRow("Strength", 0.0, 100.0, 100.0, 1.0,
            self._on_brush_preset_strength, 0)
        preset_layout.addWidget(self.brush_preset_strength)
        self.brush_preset_box = preset_box
        self.brush_preset_box.setEnabled(False)
        v.addWidget(preset_box)

        box, v = collapsible_group("Graduated Filters", layout)
        note = QLabel("Press G or enable Graduated, then drag on the image.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#888; font-size:11px;")
        v.addWidget(note)
        self.grad_list = QListWidget()
        self.grad_list.setMaximumHeight(100)
        self.grad_list.currentRowChanged.connect(self._on_grad_list_selection)
        v.addWidget(self.grad_list)
        grow = QHBoxLayout()
        gadd = QPushButton("Add (drag on image)")
        gadd.clicked.connect(lambda: self.toggle_gradient_mode(True))
        gdel = QPushButton("Delete")
        gdel.clicked.connect(self._on_delete_gradient)
        gclr = QPushButton("Clear")
        gclr.clicked.connect(self._on_clear_gradients)
        grow.addWidget(gadd)
        grow.addWidget(gdel)
        grow.addWidget(gclr)
        v.addLayout(grow)
        self.grad_sliders = {}
        self.grad_sliders_box = QWidget()
        gsl = QVBoxLayout(self.grad_sliders_box)
        gsl.setContentsMargins(0, 0, 0, 0)
        for key, label, lo, hi, step, dec in (
            ("exposure", "Exposure", -2.0, 2.0, 0.05, 2),
            ("contrast", "Contrast", -100.0, 100.0, 1, 0),
            ("saturation", "Saturation", -100.0, 100.0, 1, 0),
            ("clarity", "Clarity", -100.0, 100.0, 1, 0),
            ("temperature", "Temp shift", -100.0, 100.0, 1, 0),
            ("feather", "Feather", 5.0, 100.0, 1, 0),
        ):
            row = SliderRow(label, lo, hi, 0.0 if key != "feather" else 50.0, step,
                            lambda val, k=key: self._on_grad_slider(k, val), dec)
            self.grad_sliders[key] = row
            gsl.addWidget(row)
        self.grad_sliders_box.setEnabled(False)
        v.addWidget(self.grad_sliders_box)

        layout.addStretch(1)
        scroll.setWidget(inner)
        return scroll

    def _on_bw(self, checked):
        if self.current_path is None:
            return
        self.recipes[self.current_path].black_and_white = checked
        self._schedule_history("B&W")
        self.render_timer.start()

    def _rotate(self, direction):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.rotate_90 = (int(r.rotate_90) + direction) % 4
        self._push_history("Rotate")
        self.render_preview()

    def _filter_corrections(self, text: str):
        # Simple: could hide groups whose title doesn't match
        pass

    # ------------------------------------------------------------------
    # Folder / load
    # ------------------------------------------------------------------

    def prev_image(self):
        if not self.image_paths or self.current_path is None:
            return
        try:
            idx = self.image_paths.index(self.current_path)
        except ValueError:
            return
        if idx > 0:
            self.load_image(self.image_paths[idx - 1])
            self.filmstrip.setCurrentRow(idx - 1)

    def next_image(self):
        if not self.image_paths or self.current_path is None:
            return
        try:
            idx = self.image_paths.index(self.current_path)
        except ValueError:
            return
        if idx < len(self.image_paths) - 1:
            self.load_image(self.image_paths[idx + 1])
            self.filmstrip.setCurrentRow(idx + 1)

    def _update_navigator_viewport(self):
        """Reflect main canvas pan/zoom in the navigator rectangle."""
        if not hasattr(self, "navigator") or not hasattr(self, "preview"):
            return
        canvas = self.preview
        pm = getattr(canvas, "_pixmap", None)
        if pm is None or pm.isNull():
            return
        scale = canvas.current_scale() if hasattr(canvas, "current_scale") else canvas._scale
        sw, sh = pm.width() * scale, pm.height() * scale
        ox = (canvas.width() - sw) / 2.0 + canvas._offset.x()
        oy = (canvas.height() - sh) / 2.0 + canvas._offset.y()
        # visible region in image normalized coords
        x0 = max(0.0, (0 - ox) / max(sw, 1))
        y0 = max(0.0, (0 - oy) / max(sh, 1))
        x1 = min(1.0, (canvas.width() - ox) / max(sw, 1))
        y1 = min(1.0, (canvas.height() - oy) / max(sh, 1))
        self.navigator.set_viewport(x0, y0, x1, y1)

    def _on_nav_pan(self, nx, ny):
        """Center the main view on normalized navigator coordinates."""
        from PyQt6.QtCore import QPoint
        canvas = self.preview
        pm = getattr(canvas, "_pixmap", None)
        if pm is None:
            return
        scale = canvas.current_scale()
        img_x = nx * pm.width()
        img_y = ny * pm.height()
        canvas._fit_mode = False
        # Offset so that image point (img_x, img_y) lands at widget center
        canvas._offset = QPoint(
            int(-(img_x * scale - canvas.width() / 2) - (canvas.width() - pm.width() * scale) / 2),
            int(-(img_y * scale - canvas.height() / 2) - (canvas.height() - pm.height() * scale) / 2),
        )
        canvas.update()
        self._update_navigator_viewport()


    def show_library_mode(self):
        self._library_mode = True
        if hasattr(self, "mode_stack"):
            self.mode_stack.setCurrentIndex(1)
        for act in (getattr(self, "act_library", None), getattr(self, "act_tb_library", None)):
            if act:
                act.setChecked(True)
        for act in (getattr(self, "act_develop", None), getattr(self, "act_tb_develop", None)):
            if act:
                act.setChecked(False)
        self.refresh_library_tree()
        self.statusBar().showMessage("Library — double-click a photo to open in Develop")

    def show_develop_mode(self):
        self._library_mode = False
        if hasattr(self, "mode_stack"):
            self.mode_stack.setCurrentIndex(0)
        for act in (getattr(self, "act_develop", None), getattr(self, "act_tb_develop", None)):
            if act:
                act.setChecked(True)
        for act in (getattr(self, "act_library", None), getattr(self, "act_tb_library", None)):
            if act:
                act.setChecked(False)
        self.statusBar().showMessage("Develop")

    def scan_library_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Scan folder into Library (recursive)")
        if not folder:
            return
        self.show_library_mode()
        self.lib_status.setText(f"Scanning {folder}…")
        self.statusBar().showMessage(f"Scanning library: {folder}")
        if self._scan_worker and self._scan_worker.isRunning():
            self._scan_worker.cancel()
        self._scan_progress = self._make_progress("Scanning library…", maximum=0)
        self._scan_progress.canceled.connect(lambda: self._scan_worker and self._scan_worker.cancel())
        self._scan_worker = CatalogScanWorker(folder, recursive=True)
        self._scan_worker.progress.connect(self._on_scan_progress)
        self._scan_worker.finished_ok.connect(self._on_scan_finished)
        self._scan_worker.failed.connect(self._on_scan_failed)
        self._scan_worker.start()

    def _on_scan_progress(self, stats, path):
        msg = (
            f"Seen {stats.get('seen', 0)}  •  added {stats.get('added', 0)}  •  "
            f"updated {stats.get('updated', 0)}  •  skipped {stats.get('skipped', 0)}"
        )
        self.lib_status.setText(f"{msg}\n{path}")
        dlg = getattr(self, "_scan_progress", None)
        if dlg is not None:
            dlg.setLabelText(f"{msg}\n{os.path.basename(path)}")

    def _on_scan_failed(self, e):
        self.statusBar().showMessage(f"Scan failed: {e}")
        if hasattr(self, "lib_status"):
            self.lib_status.setText(f"Scan failed: {e}")
        dlg = getattr(self, "_scan_progress", None)
        if dlg is not None:
            dlg.close()
            self._scan_progress = None

    def _on_scan_finished(self, stats):
        dlg = getattr(self, "_scan_progress", None)
        if dlg is not None:
            dlg.close()
            self._scan_progress = None
        self.statusBar().showMessage(
            f"Library scan done — {stats.get('added', 0)} added, "
            f"{stats.get('updated', 0)} updated, {stats.get('skipped', 0)} unchanged"
        )
        self.lib_status.setText(
            f"Catalog: {self.catalog.count()} photos  •  "
            f"last scan +{stats.get('added', 0)} / ~{stats.get('updated', 0)}"
        )
        self.refresh_library_tree()

    def refresh_library_tree(self):
        if not hasattr(self, "_lib_tree_model"):
            return
        from PyQt6.QtGui import QStandardItem
        self._lib_tree_model.clear()
        include_rej = bool(getattr(self, "lib_filter_rejected", None) and self.lib_filter_rejected.isChecked())
        tree = self.catalog.date_tree(include_rejected=include_rej)
        root = self._lib_tree_model.invisibleRootItem()
        all_item = QStandardItem(f"All dates ({self.catalog.count(include_rejected=include_rej)})")
        all_item.setData({"type": "all"}, Qt.ItemDataRole.UserRole)
        root.appendRow(all_item)
        month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        for yd in tree:
            y_item = QStandardItem(f"{yd['year']} ({yd['count']})")
            y_item.setData({"type": "year", "year": yd["year"]}, Qt.ItemDataRole.UserRole)
            for md in yd["months"]:
                m_item = QStandardItem(f"{month_names[md['month']]} ({md['count']})")
                m_item.setData(
                    {"type": "month", "year": yd["year"], "month": md["month"]},
                    Qt.ItemDataRole.UserRole,
                )
                for dd in md["days"]:
                    d_item = QStandardItem(f"{dd['day']:02d}  —  {dd['count']} photo(s)")
                    d_item.setData(
                        {"type": "day", "date_key": dd["date_key"]},
                        Qt.ItemDataRole.UserRole,
                    )
                    m_item.appendRow(d_item)
                y_item.appendRow(m_item)
            root.appendRow(y_item)
        self.lib_date_tree.expandToDepth(0)
        self._lib_load_grid({"type": "all"})

    def _on_lib_date_clicked(self, index):
        item = self._lib_tree_model.itemFromIndex(index)
        if item is None:
            return
        data = item.data(Qt.ItemDataRole.UserRole) or {}
        self._lib_load_grid(data)

    def _lib_smart_match(self, rec: dict) -> bool:
        """Apply smart-collection combo to a catalog record."""
        if not hasattr(self, "lib_smart_combo"):
            return True
        mode = self.lib_smart_combo.currentData() or "all"
        path = rec.get("path") or ""
        rating = int(rec.get("rating") or 0)
        rejected = bool(rec.get("reject"))
        picked = bool(getattr(self, "_pick_flags", {}).get(path, False))
        if mode == "all":
            return True
        if mode == "picked":
            return picked
        if mode == "rejected":
            return rejected
        if mode == "rated3":
            return rating >= 3 and not rejected
        if mode == "rated5":
            return rating >= 5
        if mode == "unrated":
            return rating == 0 and not rejected
        return True

    def _on_lib_filter_changed(self, *_args):
        indexes = self.lib_date_tree.selectedIndexes()
        if indexes:
            item = self._lib_tree_model.itemFromIndex(indexes[0])
            data = (item.data(Qt.ItemDataRole.UserRole) if item else None) or {"type": "all"}
        else:
            data = {"type": "all"}
        self._lib_load_grid(data)

    def _lib_load_grid(self, data: dict):
        include_rej = bool(self.lib_filter_rejected.isChecked())
        min_rating = int(self.lib_min_rating.currentData() or 0)
        t = data.get("type", "all")
        if t == "day":
            recs = self.catalog.images_for_date(
                date_key=data.get("date_key"),
                include_rejected=include_rej,
                min_rating=min_rating,
            )
            self.lib_heading.setText(f"{data.get('date_key')}  —  {len(recs)} photo(s)")
        elif t == "month":
            recs = self.catalog.images_for_date(
                year=data.get("year"), month=data.get("month"),
                include_rejected=include_rej, min_rating=min_rating,
            )
            self.lib_heading.setText(
                f"{data.get('year')}-{int(data.get('month')):02d}  —  {len(recs)} photo(s)"
            )
        elif t == "year":
            recs = self.catalog.images_for_date(
                year=data.get("year"),
                include_rejected=include_rej, min_rating=min_rating,
            )
            self.lib_heading.setText(f"{data.get('year')}  —  {len(recs)} photo(s)")
        else:
            recs = self.catalog.images_for_date(
                include_rejected=include_rej, min_rating=min_rating,
            )
            self.lib_heading.setText(f"All dates  —  {len(recs)} photo(s)")

        recs = [r for r in recs if self._lib_smart_match(r)]
        self._lib_records = recs
        self.lib_grid.clear()
        for rec in recs:
            label = rec.get("filename") or os.path.basename(rec["path"])
            stars = int(rec.get("rating") or 0)
            if stars:
                label = f"{'★' * stars} {label}"
            if rec.get("reject"):
                label = f"[R] {label}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, rec["path"])
            item.setToolTip(rec["path"])
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)
            item.setSizeHint(QSize(150, 170))
            self.lib_grid.addItem(item)

        if self._lib_thumb_worker and self._lib_thumb_worker.isRunning():
            self._lib_thumb_worker.cancel()
        self._lib_thumb_worker = CatalogThumbWorker(recs, size=140)
        self._lib_thumb_worker.thumb_ready.connect(self._on_lib_thumb_ready)
        self._lib_thumb_worker.start()

    def _on_lib_thumb_ready(self, path, pixmap):
        for i in range(self.lib_grid.count()):
            item = self.lib_grid.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == path:
                item.setIcon(QIcon(pixmap))
                break

    def _lib_selected_paths(self):
        return [
            it.data(Qt.ItemDataRole.UserRole)
            for it in self.lib_grid.selectedItems()
            if it.data(Qt.ItemDataRole.UserRole)
        ]

    def _lib_selection_changed(self):
        paths = self._lib_selected_paths()
        if len(paths) == 1:
            rec = self.catalog.get_image(paths[0])
            if rec:
                self.statusBar().showMessage(
                    f"{rec.get('filename')}  •  {rec.get('date_key')}  •  "
                    f"rating {rec.get('rating', 0)}  •  "
                    f"{'REJECT' if rec.get('reject') else 'ok'}"
                )

    def _lib_set_rating(self, rating: int):
        paths = self._lib_selected_paths()
        if not paths:
            return
        for path in paths:
            self.catalog.set_rating(path, rating)
        self._on_lib_filter_changed()


    def _on_guide_color(self, _idx=None):
        if not hasattr(self, "guide_color_combo"):
            return
        name = self.guide_color_combo.currentData() or "yellow"
        if hasattr(self, "preview"):
            self.preview.set_guide_color(name)
            self.log(f"Guide color: {name}")

    def _lib_remove_from_catalog(self):
        paths = self._lib_selected_paths()
        if not paths:
            self.statusBar().showMessage("Select library photo(s) first")
            return
        r = QMessageBox.question(
            self, "Remove from Library",
            f"Remove {len(paths)} item(s) from the catalog?\n\nFiles will remain on disk.",
        )
        if r != QMessageBox.StandardButton.Yes:
            return
        for path in paths:
            try:
                self.catalog.remove_image(path)
            except Exception as e:
                self.log(f"Catalog remove failed: {e}", level="ERR")
        self.refresh_library_tree()
        self.statusBar().showMessage(f"Removed {len(paths)} from library")

    def _lib_move_to_trash(self):
        paths = self._lib_selected_paths()
        if not paths:
            self.statusBar().showMessage("Select library photo(s) first")
            return
        r = QMessageBox.warning(
            self, "Move to Trash",
            f"Move {len(paths)} file(s) to the system trash and remove from the library?\n\n"
            "You may be able to restore them from the OS trash/recycle bin.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if r != QMessageBox.StandardButton.Yes:
            return
        trashed = 0
        for path in paths:
            try:
                self._trash_file(path)
                try:
                    self.catalog.remove_image(path)
                except Exception:
                    pass
                # drop from develop state if open
                if path in self.recipes:
                    del self.recipes[path]
                if path in self.meta_cache:
                    del self.meta_cache[path]
                if self.current_path == path:
                    self.current_path = None
                trashed += 1
            except Exception as e:
                self.log(f"Trash failed for {path}: {e}", level="ERR")
                QMessageBox.warning(self, "Trash", f"Could not trash:\n{path}\n\n{e}")
        self.refresh_library_tree()
        # refresh filmstrip if needed
        if self.folder and any(os.path.normpath(p).startswith(os.path.normpath(self.folder)) for p in paths):
            remaining = [p for p in self.image_paths if p not in paths and os.path.isfile(p)]
            self.image_paths = remaining
            # rebuild filmstrip labels simply
            self.filmstrip.clear()
            for p in self.image_paths:
                item = QListWidgetItem(os.path.basename(p))
                item.setData(Qt.ItemDataRole.UserRole, p)
                item.setSizeHint(QSize(108, 118))
                self.filmstrip.addItem(item)
        self.statusBar().showMessage(f"Moved {trashed} file(s) to trash")

    def _trash_file(self, path: str):
        """Send file to OS trash; fall back to rename into a .photolab_trash folder."""
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        # Prefer send2trash if installed
        try:
            from send2trash import send2trash
            send2trash(path)
            return
        except ImportError:
            pass
        except Exception:
            pass
        # Qt6 may not have a portable trash API; Windows recycle via PowerShell is heavy.
        # Fallback: move into sibling .photolab_trash directory
        trash_dir = os.path.join(os.path.dirname(path), ".photolab_trash")
        os.makedirs(trash_dir, exist_ok=True)
        base = os.path.basename(path)
        dest = os.path.join(trash_dir, base)
        if os.path.exists(dest):
            stem, ext = os.path.splitext(base)
            i = 1
            while os.path.exists(dest):
                dest = os.path.join(trash_dir, f"{stem}_{i}{ext}")
                i += 1
        os.rename(path, dest)
        # also move sidecar if present
        side = path + ".photolab.json"
        if os.path.isfile(side):
            try:
                os.rename(side, dest + ".photolab.json")
            except Exception:
                pass

    def _lib_toggle_reject(self):
        paths = self._lib_selected_paths()
        if not paths:
            return
        for path in paths:
            rec = self.catalog.get_image(path)
            if rec is not None:
                self.catalog.set_reject(path, not bool(rec.get("reject")))
        self._on_lib_filter_changed()

    def _lib_open_item(self, item):
        path = item.data(Qt.ItemDataRole.UserRole)
        if path:
            self._open_from_library(path)

    def _lib_open_selected(self):
        paths = self._lib_selected_paths()
        if not paths:
            return
        self._open_from_library(paths[0])

    def _open_from_library(self, path: str):
        folder = os.path.dirname(path)
        if self.folder != folder:
            self.open_folder_path(folder)
        self.show_develop_mode()
        self.load_image(path)


    def _recent_path_file(self) -> str:
        return os.path.join(os.path.expanduser("~"), ".photolab_recent.json")

    def _load_recent_folders(self):
        try:
            path = self._recent_path_file()
            if os.path.isfile(path):
                import json
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._recent_folders = [d for d in data if isinstance(d, str) and os.path.isdir(d)][:12]
        except Exception:
            self._recent_folders = []

    def _save_recent_folders(self):
        try:
            import json
            with open(self._recent_path_file(), "w", encoding="utf-8") as f:
                json.dump(self._recent_folders[:12], f)
        except Exception:
            pass

    def _add_recent_folder(self, folder: str):
        folder = os.path.normpath(folder)
        self._recent_folders = [folder] + [f for f in self._recent_folders if f != folder]
        self._recent_folders = self._recent_folders[:12]
        self._save_recent_folders()
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        if not hasattr(self, "recent_menu"):
            return
        self.recent_menu.clear()
        if not self._recent_folders:
            act = self.recent_menu.addAction("(empty)")
            act.setEnabled(False)
            return
        for folder in self._recent_folders:
            act = self.recent_menu.addAction(folder)
            act.triggered.connect(lambda checked=False, f=folder: self.open_folder_path(f))
        self.recent_menu.addSeparator()
        clr = self.recent_menu.addAction("Clear recent")
        clr.triggered.connect(self._clear_recent_folders)

    def _clear_recent_folders(self):
        self._recent_folders = []
        self._save_recent_folders()
        self._rebuild_recent_menu()

    def toggle_peaking(self, checked=False):
        on = checked if isinstance(checked, bool) else (not self.preview.show_peaking)
        if not isinstance(checked, bool):
            on = not self.preview.show_peaking
        if hasattr(self, "act_peaking"):
            self.act_peaking.setChecked(on)
        self.preview.set_show_peaking(on)
        self.statusBar().showMessage("Focus peaking ON (green edges)" if on else "Focus peaking off")

    def sync_settings_to_selected(self):
        if self.current_path is None:
            return
        paths = []
        for it in self.filmstrip.selectedItems():
            pth = it.data(Qt.ItemDataRole.UserRole)
            if pth and pth != self.current_path:
                paths.append(pth)
        if not paths:
            QMessageBox.information(
                self, "Sync Settings",
                "Select other filmstrip images (Ctrl/Shift+click), then sync the current recipe onto them.",
            )
            return
        src = self.recipes[self.current_path].to_dict()
        # Exclude geometry crop by default? Include all for simplicity; user already has selective paste for fine control
        for pth in paths:
            self.recipes[pth] = Recipe.from_dict(src)
        self.statusBar().showMessage(f"Synced settings to {len(paths)} image(s)")
        self.log(f"Synced recipe to {len(paths)} selected images")

    def save_snapshot(self):
        if self.current_path is None:
            return
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save Snapshot", "Snapshot name:", text="Snapshot")
        if not ok or not name.strip():
            return
        name = name.strip()
        lst = self._snapshots.setdefault(self.current_path, [])
        lst.append({"name": name, "recipe": self.recipes[self.current_path].to_dict()})
        self.statusBar().showMessage(f"Snapshot saved: {name}")
        self.log(f"Snapshot '{name}' for {os.path.basename(self.current_path)}")

    def restore_snapshot(self):
        if self.current_path is None:
            return
        lst = self._snapshots.get(self.current_path) or []
        if not lst:
            QMessageBox.information(self, "Snapshots", "No snapshots for this image yet.\nUse Edit → Save Snapshot… first.")
            return
        from PyQt6.QtWidgets import QInputDialog
        names = [s["name"] for s in lst]
        name, ok = QInputDialog.getItem(self, "Restore Snapshot", "Choose snapshot:", names, 0, False)
        if not ok:
            return
        for s in lst:
            if s["name"] == name:
                self.recipes[self.current_path] = Recipe.from_dict(s["recipe"])
                self.sync_sliders_to_recipe()
                self._push_history(f"Snapshot: {name}")
                self.render_preview()
                self.statusBar().showMessage(f"Restored snapshot: {name}")
                break

    def compare_snapshot(self):
        if self.current_path is None or self.original_bgr is None:
            return
        snapshots = self._snapshots.get(self.current_path) or []
        if not snapshots:
            QMessageBox.information(self, "Snapshots", "Save a snapshot first.")
            return
        names = [s["name"] for s in snapshots]
        name, ok = QInputDialog.getItem(self, "Compare Snapshot", "Snapshot:", names, 0, False)
        if not ok:
            return
        selected = next(s for s in snapshots if s["name"] == name)
        src = self._preview_source_cache.get((self.current_path, id(self.original_bgr), 1600))
        if src is None:
            h, w = self.original_bgr.shape[:2]
            scale = min(1.0, 1600/max(h, w))
            src = cv2.resize(self.original_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA) if scale < 1 else self.original_bgr
        meta = self.meta_cache.get(self.current_path, {})
        comparison = apply_recipe(src, Recipe.from_dict(selected["recipe"]),
                                  wb_multipliers=meta.get("wb_multipliers"), meta=meta)
        self.preview.set_comparison_image(cv_to_qpixmap(comparison))
        self.set_compare_mode(ImageCanvas.MODE_SIDE_BY_SIDE)
        self.statusBar().showMessage(f"Comparing current edit with snapshot: {name}")

    def compare_soft_proof(self):
        if self.current_path is None or self.original_bgr is None:
            return
        current = self.recipes[self.current_path]
        unproofed = Recipe.from_dict(current.to_dict())
        unproofed.soft_proof = False
        src = self._preview_source_cache.get((self.current_path, id(self.original_bgr), 1600))
        if src is None:
            src = self.original_bgr
        meta = self.meta_cache.get(self.current_path, {})
        result = apply_recipe(src, unproofed, wb_multipliers=meta.get("wb_multipliers"), meta=meta)
        self.preview.set_comparison_image(cv_to_qpixmap(result))
        if not current.soft_proof:
            current.soft_proof = True
            self.soft_proof_cb.setChecked(True)
            self.render_preview()
        self.set_compare_mode(ImageCanvas.MODE_SIDE_BY_SIDE)
        self.statusBar().showMessage("Soft proof comparison: unproofed (left), proofed (right)")

    def open_folder(self):
        """Open a single folder in Develop (filmstrip). Does not touch the Library catalog."""
        folder = QFileDialog.getExistingDirectory(self, "Open folder for editing (Develop)")
        if folder:
            self.show_develop_mode()
            self.open_folder_path(folder)

    def import_from_sd(self):
        """Copy camera media from a card/device folder without deleting originals."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Import from SD Card")
        dlg.setMinimumWidth(620)
        form = QFormLayout(dlg)
        settings = QSettings("PhotoLab", "PhotoLab")
        source_edit = QLineEdit(str(settings.value("sd_import_source", "")))
        destination_edit = QLineEdit(str(settings.value("sd_import_destination", "")))

        def path_row(edit, title):
            holder = QWidget()
            row = QHBoxLayout(holder)
            row.setContentsMargins(0, 0, 0, 0)
            row.addWidget(edit, 1)
            button = QPushButton("Browse…")
            button.clicked.connect(lambda: self._choose_import_folder(edit, title))
            row.addWidget(button)
            return holder

        form.addRow("Source card/folder", path_row(source_edit, "Choose SD card or camera folder"))
        form.addRow("Destination", path_row(destination_edit, "Choose import destination"))
        preserve = QCheckBox("Preserve folders from the card (for example DCIM/100NIKON)")
        preserve.setChecked(bool(settings.value("sd_import_preserve", False, type=bool)))
        form.addRow("", preserve)
        note = QLabel(
            "PhotoLab copies supported photos and videos. It never deletes files from the card. "
            "Exact duplicates are skipped; filename collisions are safely renamed."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#999; font-size:11px;")
        form.addRow(note)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Import")
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        source = os.path.abspath(source_edit.text().strip())
        destination = os.path.abspath(destination_edit.text().strip())
        if not os.path.isdir(source):
            QMessageBox.warning(self, "Import from SD Card", "Choose an existing source card or folder.")
            return
        if not destination:
            QMessageBox.warning(self, "Import from SD Card", "Choose a destination folder.")
            return
        try:
            if os.path.commonpath([source, destination]) == source:
                QMessageBox.warning(self, "Import from SD Card", "The destination cannot be inside the source card.")
                return
        except ValueError:
            pass  # Different Windows drives are expected.
        settings.setValue("sd_import_source", source)
        settings.setValue("sd_import_destination", destination)
        settings.setValue("sd_import_preserve", preserve.isChecked())

        progress = QProgressDialog("Scanning card…", "Cancel", 0, 0, self)
        progress.setWindowTitle("Import from SD Card")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        worker = SdImportWorker(source, destination, preserve.isChecked(), VIDEO_EXTENSIONS)
        self._sd_import_worker = worker
        self._sd_import_progress = progress
        progress.canceled.connect(worker.cancel)
        worker.progress.connect(self._on_sd_import_progress)
        worker.completed.connect(self._on_sd_import_completed)
        worker.failed.connect(self._on_sd_import_failed)
        worker.start()

    def _choose_import_folder(self, edit, title):
        folder = QFileDialog.getExistingDirectory(self, title, edit.text().strip())
        if folder:
            edit.setText(folder)

    def _on_sd_import_progress(self, current, total, name):
        progress = getattr(self, "_sd_import_progress", None)
        if progress is not None:
            progress.setRange(0, max(1, total))
            progress.setValue(current)
            progress.setLabelText(f"Copying {name}\n{current} of {total}")

    def _on_sd_import_completed(self, summary):
        progress = getattr(self, "_sd_import_progress", None)
        if progress is not None:
            progress.close()
        errors = summary.get("errors") or []
        message = (
            f"Found: {summary['found']}\nCopied and verified: {summary['copied']}\n"
            f"Duplicates skipped: {summary['skipped']}\nRenamed collisions: {summary['renamed']}"
        )
        if summary.get("cancelled"):
            message += "\n\nImport was cancelled; completed copies were kept."
        if errors:
            message += f"\n\nErrors: {len(errors)}\n" + "\n".join(errors[:5])
        result = QMessageBox.question(
            self, "SD Import Complete", message + "\n\nOpen the destination for editing?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        destination = summary["destination"]
        self._sd_import_worker = None
        self._sd_import_progress = None
        if result == QMessageBox.StandardButton.Yes:
            self.open_folder_path(destination)

    def _on_sd_import_failed(self, error):
        progress = getattr(self, "_sd_import_progress", None)
        if progress is not None:
            progress.close()
        self._sd_import_worker = None
        self._sd_import_progress = None
        QMessageBox.warning(self, "SD Import Failed", error)

    def open_folder_path(self, folder: str):
        """Load images from one folder into the Develop filmstrip only."""
        self.folder = folder
        self._add_recent_folder(folder)
        self.show_develop_mode()
        if getattr(self, "_library_mode", False):
            self.show_develop_mode()
        self.image_paths = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTS) or f.lower().endswith(tuple(VIDEO_EXTENSIONS))
        )
        self.filmstrip.clear()
        self.recipes = {}
        self.meta_cache = {}
        
        # Sync Folder Tree index
        if hasattr(self, "folder_tree"):
            self.folder_tree.blockSignals(True)
            idx = self.folder_model.index(folder)
            self.folder_tree.setCurrentIndex(idx)
            self.folder_tree.scrollTo(idx)
            self.folder_tree.blockSignals(False)
            
        if not self.image_paths:
            self.statusBar().showMessage("No supported images found.")
            return
        for p in self.image_paths:
            item = QListWidgetItem(os.path.basename(p))
            item.setData(Qt.ItemDataRole.UserRole, p)
            item.setTextAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignBottom)
            item.setSizeHint(QSize(108, 118))
            self.filmstrip.addItem(item)
        self.thumb_worker = ThumbnailWorker(self.image_paths)
        self.thumb_worker.thumb_ready.connect(self.on_thumb_ready)
        self.thumb_worker.start()
        n_raw = sum(1 for p in self.image_paths if is_raw(p))
        self.statusBar().showMessage(f"Loaded {len(self.image_paths)} images ({n_raw} RAW)")
        self.log(f"Opened folder: {folder} — {len(self.image_paths)} images ({n_raw} RAW)")
        self.load_image(self.image_paths[0])

    def on_thumb_ready(self, path, pixmap):
        for i in range(self.filmstrip.count()):
            item = self.filmstrip.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                item.setIcon(QIcon(pixmap))
                break

    def on_thumb_clicked(self, item):
        self.load_image(item.data(Qt.ItemDataRole.UserRole))

    def load_image(self, path: str):
        if path and self._is_video_path(path):
            self.open_video_editor(path)
            return
        self.current_path = path
        self.statusBar().showMessage(f"Loading {os.path.basename(path)}…")
        if self._load_worker is not None and self._load_worker.isRunning():
            # Never terminate LibRaw mid-decode. Keep only the latest selection
            # queued and start it as soon as the active decode exits cleanly.
            self._pending_load_path = path
            self.statusBar().showMessage(f"Queued {os.path.basename(path)}…")
            return
        self._start_image_load(path)

    def _start_image_load(self, path: str):
        try:
            from config import get_config
            working_bps = 16 if get_config().get_bool("performance", "use_16bit_pipeline", False) else 8
        except Exception:
            working_bps = 8
        self._load_worker = LoadImageWorker(path, output_bps=working_bps)
        self._load_worker.loaded.connect(self._on_image_loaded)
        self._load_worker.failed.connect(self._on_image_failed)
        self._load_worker.finished.connect(self._on_load_worker_finished)
        self._load_worker.start()

    def _on_load_worker_finished(self):
        worker = self.sender()
        if worker is self._load_worker:
            self._load_worker = None
        if worker is not None:
            worker.deleteLater()
        pending = self._pending_load_path
        self._pending_load_path = None
        if pending and pending == self.current_path:
            self.statusBar().showMessage(f"Loading {os.path.basename(pending)}…")
            self._start_image_load(pending)

    def _on_image_loaded(self, path, img, meta):
        if path != self.current_path:
            return
        self.original_bgr = img
        self.meta_cache[path] = meta
        if path not in self.recipes:
            side = load_recipe_sidecar(path)
            self.recipes[path] = side if side is not None else Recipe()
            if side is not None:
                self.log(f"Loaded sidecar for {os.path.basename(path)}")
            if meta.get("is_raw"):
                self.recipes[path].wb_as_shot = True
        self.sync_sliders_to_recipe()
        self.history_widget.clear()
        self._push_history("Original")
        self.render_preview()
        if path in self.image_paths:
            self.filmstrip.setCurrentRow(self.image_paths.index(path))
        kind = "RAW" if meta.get("is_raw") else "RGB"
        if meta.get("raw_fallback") == "embedded_preview":
            kind = "RAW embedded preview"
        self.statusBar().showMessage(
            f"{os.path.basename(path)}  •  {img.shape[1]}×{img.shape[0]}  •  {kind}"
        )
        if meta.get("raw_fallback") == "embedded_preview":
            QMessageBox.information(
                self, "RAW opened using embedded preview",
                "The sensor RAW compression is not supported by the installed LibRaw decoder.\n\n"
                "PhotoLab opened the camera's embedded JPEG so you can edit and export it. "
                "Highlight recovery and true 16-bit RAW latitude are not available for this file.\n\n"
                f"Decoder detail: {meta.get('raw_decode_error', 'unsupported format')}",
            )
        if hasattr(self, "path_label"):
            self.path_label.setText(path)
        if hasattr(self, "count_label") and self.image_paths:
            try:
                i = self.image_paths.index(path) + 1
            except ValueError:
                i = 0
            self.count_label.setText(f"  {i} / {len(self.image_paths)} images  ")
        
        # Update metadata display
        if hasattr(self, "metadata_label"):
            self._update_metadata_display(meta)

    def _on_image_failed(self, path, err):
        if path == self.current_path:
            self.statusBar().showMessage(f"Failed: {err}")
            QMessageBox.warning(self, "Could not open RAW image", err)

    # ------------------------------------------------------------------
    def sync_sliders_to_recipe(self):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        for key, row in self.sliders.items():
            if key.startswith("_hsl_"):
                continue
            if hasattr(r, key):
                row.set_value(getattr(r, key))
        if hasattr(self.preview, "set_zebra_threshold"):
            self.preview.set_zebra_threshold(float(getattr(r, "zebra_threshold", 95.0)) / 100.0)
        self.wb_as_shot_cb.blockSignals(True)
        self.wb_as_shot_cb.setChecked(r.wb_as_shot)
        self.wb_as_shot_cb.blockSignals(False)
        self.sliders["temperature"].setEnabled(not r.wb_as_shot)
        self.sliders["tint"].setEnabled(not r.wb_as_shot)
        self.tone_curve.set_values(
            r.curve_shadows, r.curve_darks, r.curve_mids,
            r.curve_lights, r.curve_highlights,
        )
        self.tone_curve.set_point_curve("luma", getattr(r, "curve_points", None) or [])
        self.tone_curve.set_point_curve("r", getattr(r, "curve_r_points", None) or [])
        self.tone_curve.set_point_curve("g", getattr(r, "curve_g_points", None) or [])
        self.tone_curve.set_point_curve("b", getattr(r, "curve_b_points", None) or [])
        # HSL channel sliders
        if hasattr(self, "_sync_all_hsl_sliders_from_recipe"):
            self._sync_all_hsl_sliders_from_recipe(r)
        idx = int(getattr(r, "hsl_active_channel", 0) or 0)
        if "_hsl_hue" in self.sliders:
            try:
                self.sliders["_hsl_hue"].set_value(r.hsl_hue[idx] if r.hsl_hue else 0)
                self.sliders["_hsl_sat"].set_value(r.hsl_sat[idx] if r.hsl_sat else 0)
                self.sliders["_hsl_lum"].set_value(r.hsl_lum[idx] if r.hsl_lum else 0)
            except Exception:
                pass
        if hasattr(self, "soft_proof_cb"):
            self.soft_proof_cb.blockSignals(True)
            self.soft_proof_cb.setChecked(r.soft_proof)
            self.soft_proof_cb.blockSignals(False)
        if hasattr(self, "gamut_warn_cb"):
            self.gamut_warn_cb.blockSignals(True)
            self.gamut_warn_cb.setChecked(bool(getattr(r, "soft_proof_gamut", False)))
            self.gamut_warn_cb.blockSignals(False)
            self.proof_combo.blockSignals(True)
            self.proof_combo.setCurrentText(r.soft_proof_profile)
            self.proof_combo.blockSignals(False)
        if hasattr(self, "bw_cb"):
            self.bw_cb.blockSignals(True)
            self.bw_cb.setChecked(bool(r.black_and_white))
            self.bw_cb.blockSignals(False)
        if hasattr(self, "_sync_ir_astro_widgets"):
            self._sync_ir_astro_widgets(r)
        if hasattr(self, "zone_enabled_cb"):
            self.zone_enabled_cb.blockSignals(True)
            self.zone_enabled_cb.setChecked(bool(getattr(r, "zone_enabled", False)))
            self.zone_enabled_cb.blockSignals(False)
            self.zone_filter_combo.blockSignals(True)
            zone_key = getattr(r, "zone_filter", "none") or "none"
            zone_idx = self.zone_filter_combo.findData(zone_key)
            self.zone_filter_combo.setCurrentIndex(max(0, zone_idx))
            self.zone_filter_combo.blockSignals(False)
            self.zone_overlay_cb.blockSignals(True)
            self.zone_overlay_cb.setChecked(bool(getattr(r, "zone_overlay", False)))
            self.zone_overlay_cb.blockSignals(False)
        # Control points sync
        self.selected_local_index = -1
        self._update_local_points_list()
        self._sync_local_sliders()
        if hasattr(self, "local_active_cb"):
            self.local_active_cb.blockSignals(True)
            self.local_active_cb.setChecked(self._local_mode)
            self.local_active_cb.blockSignals(False)
        self.preview.set_control_points(r.local_points, -1)
        
        self.crop_tool_btn.setChecked(False)
        self.preview.set_crop_mode(False)
        if hasattr(self, "keystone_tool_btn"):
            self.keystone_tool_btn.blockSignals(True)
            self.keystone_tool_btn.setChecked(False)
            self.keystone_tool_btn.blockSignals(False)
            self.preview.set_keystone_mode(False, getattr(r, "keystone_points", []))
        if hasattr(self, "geometry_auto_crop_cb"):
            self.geometry_auto_crop_cb.blockSignals(True)
            self.geometry_auto_crop_cb.setChecked(bool(getattr(r, "geometry_auto_crop", False)))
            self.geometry_auto_crop_cb.blockSignals(False)

    def on_slider(self, key, value):
        if self.current_path is None:
            return
        if key.startswith("_hsl_"):
            return  # handled by _on_hsl_slider
        if hasattr(self.recipes[self.current_path], key):
            setattr(self.recipes[self.current_path], key, value)
            self._schedule_history(key)
        if key == "zebra_threshold" and hasattr(self.preview, "set_zebra_threshold"):
            self.preview.set_zebra_threshold(float(value) / 100.0)
        # Keep ToneCurveWidget graph in sync with region sliders
        if key in ("curve_shadows", "curve_darks", "curve_mids", "curve_lights", "curve_highlights"):
            if hasattr(self, "tone_curve"):
                r = self.recipes[self.current_path]
                self.tone_curve.set_values(
                    float(getattr(r, "curve_shadows", 0) or 0),
                    float(getattr(r, "curve_darks", 0) or 0),
                    float(getattr(r, "curve_mids", 0) or 0),
                    float(getattr(r, "curve_lights", 0) or 0),
                    float(getattr(r, "curve_highlights", 0) or 0),
                )
        self.render_timer.start()

    def _schedule_history(self, label: str):
        # Debounced history push via render timer end
        self._pending_history_label = label

    def _push_history(self, label: str):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        self.history_widget.push(label, r.to_dict())
        self._maybe_autosave()

    def undo_edit(self):
        idx = self.history_widget.undo_index()
        if idx is None:
            self.statusBar().showMessage("Nothing to undo")
            return
        self.history_widget.list.setCurrentRow(idx)
        self._on_history_restore(idx)
        self.statusBar().showMessage("Undo")

    def redo_edit(self):
        idx = self.history_widget.redo_index()
        if idx is None:
            self.statusBar().showMessage("Nothing to redo")
            return
        self.history_widget.list.setCurrentRow(idx)
        self._on_history_restore(idx)
        self.statusBar().showMessage("Redo")

    def toggle_reject_current(self):
        if self.current_path is None:
            return
        path = self.current_path
        rec = None
        try:
            rec = self.catalog.get(path) if hasattr(self.catalog, "get") else None
        except Exception:
            rec = None
        currently = False
        if isinstance(rec, dict):
            currently = bool(rec.get("reject"))
        elif path in getattr(self, "_reject_flags", {}):
            currently = self._reject_flags[path]
        new_val = not currently
        if not hasattr(self, "_reject_flags"):
            self._reject_flags = {}
        self._reject_flags[path] = new_val
        try:
            self.catalog.set_reject(path, new_val)
        except Exception:
            pass
        # filmstrip visual
        for i in range(self.filmstrip.count()):
            item = self.filmstrip.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == path:
                base = os.path.basename(path)
                stars = self._image_ratings.get(path, 0)
                prefix = "⛔ " if new_val else ""
                star = (("★" * stars + "☆" * (5 - stars) + "  ") if stars else "")
                item.setText(f"{prefix}{star}{base}")
                break
        self.statusBar().showMessage("Rejected" if new_val else "Un-rejected")

    def _on_history_restore(self, index: int):
        d = self.history_widget.get_recipe_dict(index)
        if d is None or self.current_path is None:
            return
        self.recipes[self.current_path] = Recipe.from_dict(d)
        self.sync_sliders_to_recipe()
        self.render_preview()
        self.statusBar().showMessage(f"Restored history #{index + 1}")

    def _build_hsl_panel(self, parent_layout):
        """Lightroom-style HSL: Hue / Saturation / Luminance / All for 8 channels."""
        from PyQt6.QtWidgets import QTabWidget, QWidget, QVBoxLayout, QLabel, QSizePolicy, QScrollArea
        from PyQt6.QtCore import Qt

        self._HSL_NAMES = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
        self._HSL_COLORS = [
            "#e74c3c", "#e67e22", "#f1c40f", "#27ae60",
            "#1abc9c", "#3498db", "#9b59b6", "#e84393",
        ]
        self.hsl_sliders = {"hue": [], "sat": [], "lum": []}

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        tabs.setStyleSheet(
            "QTabWidget::pane { border: 1px solid #333; background:#1a1a1a; }"
            "QTabBar::tab { background:#222; color:#ccc; padding:5px 10px; margin-right:1px; }"
            "QTabBar::tab:selected { background:#2a5080; color:#fff; }"
        )

        def _make_channel_page(which: str) -> QWidget:
            page = QWidget()
            page.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
            lay = QVBoxLayout(page)
            lay.setContentsMargins(2, 2, 2, 2)
            lay.setSpacing(0)
            for i, name in enumerate(self._HSL_NAMES):
                row = SliderRow(name, -100.0, 100.0, 0.0, 1.0, decimals=0)
                row.setMaximumHeight(28)
                for child in row.findChildren(QLabel):
                    if child.text() == name:
                        child.setStyleSheet(
                            f"color: {self._HSL_COLORS[i]}; font-size: 11px; font-weight: 600;"
                        )
                row.valueChanged.connect(
                    lambda val, w=which, idx=i: self._on_hsl_channel_value(w, idx, val)
                )
                self.hsl_sliders[which].append(row)
                lay.addWidget(row)
            # No stretch — keeps panel tight under last slider
            return page

        tabs.addTab(_make_channel_page("hue"), "Hue")
        tabs.addTab(_make_channel_page("sat"), "Saturation")
        tabs.addTab(_make_channel_page("lum"), "Luminance")

        # All: scrollable but still compact row spacing
        all_inner = QWidget()
        all_lay = QVBoxLayout(all_inner)
        all_lay.setContentsMargins(2, 2, 2, 2)
        all_lay.setSpacing(1)
        self.hsl_all_sliders = []
        for i, name in enumerate(self._HSL_NAMES):
            hdr = QLabel(name)
            hdr.setStyleSheet(
                f"color: {self._HSL_COLORS[i]}; font-weight: 600; font-size: 11px; margin-top: 2px;"
            )
            all_lay.addWidget(hdr)
            trio = []
            for which, label in (("hue", "H"), ("sat", "S"), ("lum", "L")):
                row = SliderRow(label, -100.0, 100.0, 0.0, 1.0, decimals=0)
                row.setMaximumHeight(26)
                row.valueChanged.connect(
                    lambda val, w=which, idx=i: self._on_hsl_channel_value(w, idx, val)
                )
                trio.append(row)
                all_lay.addWidget(row)
            self.hsl_all_sliders.append(tuple(trio))
        all_scroll = QScrollArea()
        all_scroll.setWidgetResizable(True)
        all_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        all_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        all_scroll.setWidget(all_inner)
        all_scroll.setMinimumHeight(220)
        all_scroll.setMaximumHeight(320)
        tabs.addTab(all_scroll, "All")

        # Cap tab height so Hue/Sat/Lum don't grow into empty void
        tabs.setMinimumHeight(0)
        tabs.setMaximumHeight(280)
        parent_layout.addWidget(tabs)
        self.hsl_tabs = tabs
        self.color_wheel = None

    def _on_hsl_channel_value(self, which: str, idx: int, value: float):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]

        def replace(tup, i, v):
            lst = list(tup if tup is not None else (0.0,) * 8)
            while len(lst) < 8:
                lst.append(0.0)
            lst[i] = float(value)
            return tuple(lst)

        if which == "hue":
            r.hsl_hue = replace(r.hsl_hue, idx, value)
        elif which == "sat":
            r.hsl_sat = replace(r.hsl_sat, idx, value)
        else:
            r.hsl_lum = replace(r.hsl_lum, idx, value)
        r.hsl_active_channel = idx
        self._sync_hsl_slider_group(which, idx, value)
        self._schedule_history(f"HSL {which} {self._HSL_NAMES[idx]}")
        self.render_timer.start()

    def _sync_hsl_slider_group(self, which: str, idx: int, value: float):
        """Keep tab sliders and All-tab trio in sync without feedback loops."""
        rows = getattr(self, "hsl_sliders", {}).get(which) or []
        if idx < len(rows):
            row = rows[idx]
            if abs(row.spin.value() - value) > 1e-6:
                row.blockSignals(True)
                row.set_value(value)
                row.blockSignals(False)
        if hasattr(self, "hsl_all_sliders") and idx < len(self.hsl_all_sliders):
            map_i = {"hue": 0, "sat": 1, "lum": 2}[which]
            row = self.hsl_all_sliders[idx][map_i]
            if abs(row.spin.value() - value) > 1e-6:
                row.blockSignals(True)
                row.set_value(value)
                row.blockSignals(False)

    def _sync_all_hsl_sliders_from_recipe(self, r):
        if not hasattr(self, "hsl_sliders"):
            return
        for which, attr in (("hue", "hsl_hue"), ("sat", "hsl_sat"), ("lum", "hsl_lum")):
            vals = list(getattr(r, attr, (0.0,) * 8) or (0.0,) * 8)
            while len(vals) < 8:
                vals.append(0.0)
            for i, row in enumerate(self.hsl_sliders.get(which) or []):
                row.blockSignals(True)
                row.set_value(float(vals[i]))
                row.blockSignals(False)
            if hasattr(self, "hsl_all_sliders"):
                map_i = {"hue": 0, "sat": 1, "lum": 2}[which]
                for i, trio in enumerate(self.hsl_all_sliders):
                    trio[map_i].blockSignals(True)
                    trio[map_i].set_value(float(vals[i]))
                    trio[map_i].blockSignals(False)

    def _on_hsl_channel(self, idx: int):
        """Color wheel selected a channel — jump All tab focus / status."""
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.hsl_active_channel = idx
        names = getattr(self, "_HSL_NAMES", None) or [
            "Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"
        ]
        if hasattr(self, "hsl_channel_label"):
            self.hsl_channel_label.setText(f"Channel: {names[idx]}")
        if hasattr(self, "hsl_tabs"):
            self.hsl_tabs.setCurrentIndex(3)  # All
        self.statusBar().showMessage(f"HSL channel: {names[idx]}")
    def _on_hsl_channel(self, idx: int):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.hsl_active_channel = idx
        names = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
        self.hsl_channel_label.setText(f"Channel: {names[idx]}")
        # Load channel values into sliders
        self.sliders["_hsl_hue"].set_value(r.hsl_hue[idx])
        self.sliders["_hsl_sat"].set_value(r.hsl_sat[idx])
        self.sliders["_hsl_lum"].set_value(r.hsl_lum[idx])

    def _on_hsl_slider(self, which: str, value: float):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        idx = r.hsl_active_channel
        def replace(tup, i, v):
            lst = list(tup)
            lst[i] = v
            return tuple(lst)
        if which == "hue":
            r.hsl_hue = replace(r.hsl_hue, idx, value)
        elif which == "sat":
            r.hsl_sat = replace(r.hsl_sat, idx, value)
        else:
            r.hsl_lum = replace(r.hsl_lum, idx, value)
        r.hsl_active_channel = idx
        self._sync_hsl_slider_group(which, idx, value)
        self._schedule_history(f"HSL {which} {self._HSL_NAMES[idx]}")
        self.render_timer.start()

    def _sync_hsl_slider_group(self, which: str, idx: int, value: float):
        """Keep tab sliders and All-tab trio in sync without feedback loops."""
        rows = getattr(self, "hsl_sliders", {}).get(which) or []
        if idx < len(rows):
            row = rows[idx]
            if abs(row.spin.value() - value) > 1e-6:
                row.blockSignals(True)
                row.set_value(value)
                row.blockSignals(False)
        if hasattr(self, "hsl_all_sliders") and idx < len(self.hsl_all_sliders):
            map_i = {"hue": 0, "sat": 1, "lum": 2}[which]
            row = self.hsl_all_sliders[idx][map_i]
            if abs(row.spin.value() - value) > 1e-6:
                row.blockSignals(True)
                row.set_value(value)
                row.blockSignals(False)

    def _sync_all_hsl_sliders_from_recipe(self, r):
        if not hasattr(self, "hsl_sliders"):
            return
        for which, attr in (("hue", "hsl_hue"), ("sat", "hsl_sat"), ("lum", "hsl_lum")):
            vals = list(getattr(r, attr, (0.0,) * 8) or (0.0,) * 8)
            while len(vals) < 8:
                vals.append(0.0)
            for i, row in enumerate(self.hsl_sliders.get(which) or []):
                row.blockSignals(True)
                row.set_value(float(vals[i]))
                row.blockSignals(False)
            if hasattr(self, "hsl_all_sliders"):
                map_i = {"hue": 0, "sat": 1, "lum": 2}[which]
                for i, trio in enumerate(self.hsl_all_sliders):
                    trio[map_i].blockSignals(True)
                    trio[map_i].set_value(float(vals[i]))
                    trio[map_i].blockSignals(False)

    def _on_hsl_channel(self, idx: int):
        """Color wheel selected a channel — jump All tab focus / status."""
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.hsl_active_channel = idx
        names = getattr(self, "_HSL_NAMES", None) or [
            "Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"
        ]
        if hasattr(self, "hsl_channel_label"):
            self.hsl_channel_label.setText(f"Channel: {names[idx]}")
        if hasattr(self, "hsl_tabs"):
            self.hsl_tabs.setCurrentIndex(3)  # All
        self.statusBar().showMessage(f"HSL channel: {names[idx]}")
    def _on_hsl_channel(self, idx: int):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.hsl_active_channel = idx
        names = ["Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta"]
        self.hsl_channel_label.setText(f"Channel: {names[idx]}")
        # Load channel values into sliders
        self.sliders["_hsl_hue"].set_value(r.hsl_hue[idx])
        self.sliders["_hsl_sat"].set_value(r.hsl_sat[idx])
        self.sliders["_hsl_lum"].set_value(r.hsl_lum[idx])

    def _on_hsl_slider(self, which: str, value: float):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        idx = r.hsl_active_channel
        def replace(tup, i, v):
            lst = list(tup)
            lst[i] = v
            return tuple(lst)
        if which == "hue":
            r.hsl_hue = replace(r.hsl_hue, idx, value)
        elif which == "sat":
            r.hsl_sat = replace(r.hsl_sat, idx, value)
        else:
            r.hsl_lum = replace(r.hsl_lum, idx, value)
        self._schedule_history(f"HSL {which}")
        self.render_timer.start()
    def _on_soft_proof(self, checked: bool):
        if self.current_path is None:
            return
        self.recipes[self.current_path].soft_proof = checked
        self.render_timer.start()

    def _on_proof_profile(self, name: str):
        if self.current_path is None:
            return
        self.recipes[self.current_path].soft_proof_profile = name
        self.render_timer.start()

    def on_curve_changed(self, shadows, darks, mids, lights, highlights):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.curve_shadows = shadows
        r.curve_darks = darks
        r.curve_mids = mids
        r.curve_lights = lights
        r.curve_highlights = highlights
        # Sync region sliders under the curve
        for key, val in (
            ("curve_shadows", shadows),
            ("curve_darks", darks),
            ("curve_mids", mids),
            ("curve_lights", lights),
            ("curve_highlights", highlights),
        ):
            if key in self.sliders:
                self.sliders[key].blockSignals(True)
                self.sliders[key].set_value(val)
                self.sliders[key].blockSignals(False)
        self._schedule_history("Tone curve")
        self.render_timer.start()

    def on_wb_as_shot(self, checked):
        if self.current_path is None:
            return
        self.recipes[self.current_path].wb_as_shot = checked
        self.sliders["temperature"].setEnabled(not checked)
        self.sliders["tint"].setEnabled(not checked)
        self.render_timer.start()

    def render_preview(self):
        if self.original_bgr is None or self.current_path is None:
            return
        recipe = self.recipes[self.current_path]
        meta = self.meta_cache.get(self.current_path, {})
        h, w = self.original_bgr.shape[:2]
        max_dim = 1600
        cache_key = (self.current_path, id(self.original_bgr), max_dim)
        preview_src = self._preview_source_cache.get(cache_key)
        if preview_src is None and max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            preview_src = cv2.resize(self.original_bgr, (int(w * scale), int(h * scale)),
                                     interpolation=cv2.INTER_AREA)
        elif preview_src is None:
            preview_src = self.original_bgr
        self._preview_source_cache = {cache_key: preview_src}
        self._preview_generation += 1
        self._preview_renderer.submit(
            self._preview_generation, self.current_path, preview_src, recipe, meta
        )

    def _on_preview_rendered(self, generation, path, result, preview_src):
        if generation != self._preview_generation or path != self.current_path:
            return
        self.histogram.set_image(result)
        pix = cv_to_qpixmap(result)
        orig_pix = cv_to_qpixmap(preview_src)
        self.preview.set_image(pix, original=orig_pix)
        # Update navigator with a small thumb of the result
        if hasattr(self, "navigator"):
            self.navigator.set_image(pix)
            # Approximate full-frame viewport when fitted
            self.navigator.set_viewport(0.0, 0.0, 1.0, 1.0)
        if getattr(self, "_pending_history_label", None):
            self._push_history(self._pending_history_label)
            self._pending_history_label = None

    def _on_preview_render_failed(self, generation, path, message):
        if generation == self._preview_generation and path == self.current_path:
            self.statusBar().showMessage(f"Preview render failed: {message}")

    def set_compare_mode(self, mode):
        self.act_edit.setChecked(mode == ImageCanvas.MODE_NORMAL)
        self.act_split.setChecked(mode == ImageCanvas.MODE_SPLIT)
        self.act_side.setChecked(mode == ImageCanvas.MODE_SIDE_BY_SIDE)
        self.preview.set_compare_mode(mode)

    def toggle_crop_mode(self, checked):
        if self.original_bgr is None:
            self.crop_tool_btn.setChecked(False)
            return
        if checked and self.recipes[self.current_path].crop is not None:
            self.recipes[self.current_path].crop = None
            self.render_preview()
        self.preview.set_crop_mode(checked)

    def _aspect_ratio_value(self):
        text = self.aspect_combo.currentText()
        if text == "Free":
            return None
        if text == "Original":
            if self.original_bgr is None:
                return None
            h, w = self.original_bgr.shape[:2]
            return w / h
        mapping = {
            "1:1": 1.0, "Square 5×5": 1.0,
            "5:4": 5/4, "4:3": 4/3, "3:2": 3/2, "16:9": 16/9, "16:10": 16/10,
            "2:3 (portrait)": 2/3, "3:4 (portrait)": 3/4, "9:16 (portrait)": 9/16,
            "A4": 210/297,  # portrait A4
        }
        return mapping.get(text)

    def on_crop_dragged(self, rect: QRect):
        if self.current_path is None:
            return
        canvas = self.preview
        pm = getattr(canvas, "_pixmap", None)
        if pm is None:
            return
        scale = canvas.current_scale()
        off = getattr(canvas, "_offset", None)
        ox = (canvas.width() - pm.width() * scale) / 2.0 + (off.x() if off else 0)
        oy = (canvas.height() - pm.height() * scale) / 2.0 + (off.y() if off else 0)

        def to_norm(x, y):
            ix = (x - ox) / (scale * max(pm.width(), 1))
            iy = (y - oy) / (scale * max(pm.height(), 1))
            return max(0.0, min(1.0, ix)), max(0.0, min(1.0, iy))

        x0, y0 = to_norm(rect.left(), rect.top())
        x1, y1 = to_norm(rect.right(), rect.bottom())
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0
        if x1 - x0 < 0.01 or y1 - y0 < 0.01:
            return
        ratio = self._aspect_ratio_value()
        if ratio is not None:
            w = x1 - x0
            h = w / ratio
            if y0 + h > 1.0:
                h = 1.0 - y0
                w = h * ratio
            x1, y1 = x0 + w, y0 + h
        self.recipes[self.current_path].crop = (x0, y0, x1, y1)
        self.crop_tool_btn.setChecked(False)
        self.preview.set_crop_mode(False)
        self.render_preview()

    def clear_crop(self):
        if self.current_path is None:
            return
        self.recipes[self.current_path].crop = None
        self.crop_tool_btn.setChecked(False)
        self.preview.set_crop_mode(False)
        self.render_preview()

    def _on_show_grid_toggled(self, checked):
        if hasattr(self, "preview"):
            self.preview.set_show_grid(checked)
            self.log(f"Grid overlay {'ON' if checked else 'OFF'}")

    def _on_show_spiral_toggled(self, checked):
        if hasattr(self, "preview"):
            self.preview.set_show_spiral(checked)
            self.log(f"Fibonacci spiral {'ON' if checked else 'OFF'}")

    def _on_spiral_scale(self, val):
        if hasattr(self, "preview"):
            self.preview.set_spiral_params(scale=float(val) / 100.0)

    def _on_spiral_orient(self, idx):
        if hasattr(self, "preview"):
            self.preview.set_spiral_params(orient=int(idx))

    def _toggle_local_mode(self, checked=False):
        self._local_mode = checked if isinstance(checked, bool) else (not self._local_mode)
        self.act_local.setChecked(self._local_mode)
        if hasattr(self, "local_active_cb"):
            self.local_active_cb.blockSignals(True)
            self.local_active_cb.setChecked(self._local_mode)
            self.local_active_cb.blockSignals(False)
        self.preview.local_mode = self._local_mode
        if self._local_mode:
            # Exclusive vs other canvas tools
            self.preview.gradient_mode = False
            self.preview.brush_mode = False
            self.preview.wb_picker_mode = False
            for act_name in ("act_grad", "act_brush", "act_wb_pick"):
                a = getattr(self, act_name, None)
                if a is not None:
                    a.setChecked(False)
            self.statusBar().showMessage("Control Point mode: click on image to place a local adjustment")
            self.preview.setCursor(Qt.CursorShape.CrossCursor)
            if hasattr(self, "_cat_buttons") and len(self._cat_buttons) > 5:
                self._cat_buttons[5].setChecked(True)
                self.tool_stack.setCurrentIndex(5)
        else:
            self.preview.setCursor(Qt.CursorShape.ArrowCursor)
            self.statusBar().showMessage("Control Point mode off")





    def keyPressEvent(self, e):
        from PyQt6.QtCore import Qt as _Qt
        # Hold \\ or ` for temporary before view
        if e.key() in (_Qt.Key.Key_Backslash, _Qt.Key.Key_QuoteLeft) and not e.isAutoRepeat():
            if not getattr(self, "_temp_before", False):
                self._temp_before = True
                self.preview.set_hold_original(True)
                self.statusBar().showMessage("Original (release key to restore)")
            e.accept()
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        from PyQt6.QtCore import Qt as _Qt
        if e.key() in (_Qt.Key.Key_Backslash, _Qt.Key.Key_QuoteLeft) and not e.isAutoRepeat():
            if getattr(self, "_temp_before", False):
                self._temp_before = False
                self.preview.set_hold_original(False)
                self.statusBar().showMessage("Compare restored")
            e.accept()
            return
        super().keyReleaseEvent(e)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
            self.statusBar().showMessage("Exited full screen")
        else:
            self.showFullScreen()
            self.statusBar().showMessage("Full screen — press F11 to exit")

    def show_metadata(self):
        if self.current_path is None:
            QMessageBox.information(self, "Metadata", "Open an image first.")
            return
        meta = self.meta_cache.get(self.current_path) or {}
        try:
            from imaging import extract_exif
            more = extract_exif(self.current_path)
            for k, v in more.items():
                meta.setdefault(k, v)
        except Exception:
            pass
        lines = [f"Path: {self.current_path}", ""]
        for key in (
            "camera", "make", "lens", "focal", "aperture", "iso", "shutter",
            "datetime", "datetime_original", "is_raw", "wb_multipliers",
        ):
            if key in meta and meta[key] not in (None, "", []):
                lines.append(f"{key}: {meta[key]}")
        # any other keys
        for k, v in sorted(meta.items()):
            if k in ("camera", "make", "lens", "focal", "aperture", "iso", "shutter",
                     "datetime", "datetime_original", "is_raw", "wb_multipliers"):
                continue
            if v not in (None, "", []):
                lines.append(f"{k}: {v}")
        if self.original_bgr is not None:
            h, w = self.original_bgr.shape[:2]
            lines.insert(1, f"Size: {w} × {h}")
        dlg = QDialog(self)
        dlg.setWindowTitle("Image Metadata")
        dlg.resize(520, 420)
        layout = QVBoxLayout(dlg)
        browser = QTextBrowser()
        browser.setPlainText("\n".join(lines) if len(lines) > 2 else "No metadata found.")
        layout.addWidget(browser)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec()

    def start_horizon_line(self):
        if self.current_path is None:
            return
        self.preview.horizon_line_mode = True
        self.preview.gradient_mode = False
        self.preview.brush_mode = False
        self.preview.wb_picker_mode = False
        self.preview.setCursor(Qt.CursorShape.CrossCursor)
        self.statusBar().showMessage("Draw a line along the horizon, then release")

    def _on_horizon_line(self, angle: float):
        if self.current_path is None:
            return
        angle = float(max(-45.0, min(45.0, angle)))
        self.recipes[self.current_path].horizon = round(angle, 2)
        self.sync_sliders_to_recipe()
        self._push_history("Level horizon")
        self.render_preview()
        self.statusBar().showMessage(f"Horizon set to {angle:.2f}°")
        self.log(f"Horizon from line: {angle:.2f}°")

    def _apply_filmstrip_filter(self, _idx=None):
        min_r = 0
        if hasattr(self, "film_rating_filter"):
            min_r = int(self.film_rating_filter.currentData() or 0)
        for i in range(self.filmstrip.count()):
            item = self.filmstrip.item(i)
            if not item:
                continue
            path = item.data(Qt.ItemDataRole.UserRole)
            stars = self._image_ratings.get(path, 0)
            try:
                if hasattr(self, "catalog") and path:
                    rec = self.catalog.get(path)
                    if rec and rec.get("rating"):
                        stars = max(stars, int(rec.get("rating") or 0))
            except Exception:
                pass
            item.setHidden(min_r > 0 and stars < min_r)

    def toggle_clipping(self, checked=False):
        on = checked if isinstance(checked, bool) else (not self.preview.show_clipping)
        if not isinstance(checked, bool):
            on = not self.preview.show_clipping
        if hasattr(self, "act_clipping"):
            self.act_clipping.setChecked(on)
        self.preview.set_show_clipping(on)
        self.statusBar().showMessage(
            "Clipping: blue=shadows  red=highlights" if on else "Clipping off"
        )

    def toggle_autosave(self, checked=False):
        on = bool(checked) if isinstance(checked, bool) else (not self.autosave_sidecars)
        if not isinstance(checked, bool):
            on = not self.autosave_sidecars
        self.autosave_sidecars = on
        if hasattr(self, "act_autosave"):
            self.act_autosave.setChecked(on)
        self.statusBar().showMessage(
            "Auto-save sidecars ON" if on else "Auto-save sidecars OFF"
        )

    def _maybe_autosave(self):
        if self.autosave_sidecars and self.current_path:
            try:
                save_recipe_sidecar(self.current_path, self.recipes[self.current_path])
            except Exception as e:
                self.log(f"Autosave failed: {e}", level="ERR")

    def rate_current(self, stars: int):
        """Rate current image 0–5 (stored in memory + library catalog if present)."""
        if self.current_path is None:
            return
        stars = int(max(0, min(5, stars)))
        self._image_ratings[self.current_path] = stars
        try:
            if hasattr(self, "catalog") and self.catalog is not None:
                self.catalog.set_rating(self.current_path, stars)
        except Exception:
            pass
        # Update filmstrip label
        for i in range(self.filmstrip.count()):
            item = self.filmstrip.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == self.current_path:
                base = os.path.basename(self.current_path)
                item.setText(("★" * stars + "☆" * (5 - stars) + "  " + base) if stars else base)
                break
        self.statusBar().showMessage(f"Rating: {stars} star(s)" if stars else "Rating cleared")

    def toggle_zebras(self, checked=False):
        """Diagonal zebra stripes on near-clipped highlights (video-style exposure assist)."""
        on = checked if isinstance(checked, bool) else (not getattr(self.preview, "show_zebras", False))
        if not isinstance(checked, bool):
            on = not getattr(self.preview, "show_zebras", False)
        if hasattr(self, "act_zebras"):
            self.act_zebras.setChecked(on)
        if hasattr(self, "act_tb_zebras"):
            self.act_tb_zebras.setChecked(on)
        if hasattr(self.preview, "set_show_zebras"):
            self.preview.set_show_zebras(on)
        else:
            self.preview.show_zebras = on
            self.preview.update()
        self.statusBar().showMessage(
            "Zebras ON — striped areas are near/fully overexposed (toggle Z)"
            if on else "Zebras off"
        )

        self._apply_filmstrip_filter()

    def copy_settings(self):
        if self.current_path is None:
            return
        self._copied_recipe = self.recipes[self.current_path].to_dict()
        self.statusBar().showMessage("Settings copied")

    def paste_settings(self):
        if self.current_path is None or not self._copied_recipe:
            self.statusBar().showMessage("Nothing to paste")
            return
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout
        dlg = QDialog(self)
        dlg.setWindowTitle("Paste Settings")
        form = QFormLayout(dlg)
        form.addRow(QLabel("Choose what to paste from the copied recipe:"))
        cb_all = QCheckBox("Everything")
        cb_all.setChecked(True)
        cb_tone = QCheckBox("Tone / Light")
        cb_color = QCheckBox("Color / WB / HSL")
        cb_detail = QCheckBox("Detail (NR / sharpen)")
        cb_geo = QCheckBox("Geometry / crop")
        cb_local = QCheckBox("Local (points / gradients / brushes)")
        cb_fx = QCheckBox("Effects")
        for cb in (cb_all, cb_tone, cb_color, cb_detail, cb_geo, cb_local, cb_fx):
            form.addRow(cb)

        def _on_all(v):
            for cb in (cb_tone, cb_color, cb_detail, cb_geo, cb_local, cb_fx):
                cb.setEnabled(not v)
        cb_all.toggled.connect(_on_all)
        _on_all(True)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        src = Recipe.from_dict(self._copied_recipe)
        dst = self.recipes[self.current_path]
        if cb_all.isChecked():
            self.recipes[self.current_path] = Recipe.from_dict(self._copied_recipe)
        else:
            tone_keys = ["exposure", "smart_light", "contrast", "highlights", "shadows",
                         "whites", "blacks", "clarity", "gamma", "curve_shadows", "curve_darks",
                         "curve_mids", "curve_lights", "curve_highlights", "zebra_threshold",
                         "zebra_exposure", "zebra_feather"]
            color_keys = ["temperature", "tint", "wb_as_shot", "creative_temperature", "creative_tint", "vibrance", "saturation",
                          "hsl_hue", "hsl_sat", "hsl_lum"]
            detail_keys = ["denoise_luminance", "denoise_chroma", "denoise_strength",
                           "denoise_detail", "denoise_method", "sharpen_intensity",
                           "sharpen_radius", "sharpen_threshold", "sharpen_detail", "output_sharpen"]
            geo_keys = [
                "horizon", "distortion", "perspective", "perspective_horizontal",
                "warp_top", "warp_bottom", "warp_left", "warp_right", "wide_angle",
                "diorama_strength", "diorama_position", "diorama_width", "diorama_angle",
                "keystone_points", "geometry_auto_crop",
                "crop", "ca_amount", "lens_auto",
            ]
            local_keys = ["local_points", "gradients", "brush_masks"]
            fx_keys = ["clearview", "microcontrast", "vignette", "film_grain", "black_and_white",
                       "rotate_90", "hdr_look"]
            groups = []
            if cb_tone.isChecked():
                groups += tone_keys
            if cb_color.isChecked():
                groups += color_keys
            if cb_detail.isChecked():
                groups += detail_keys
            if cb_geo.isChecked():
                groups += geo_keys
            if cb_local.isChecked():
                groups += local_keys
            if cb_fx.isChecked():
                groups += fx_keys
            for k in groups:
                if hasattr(src, k) and hasattr(dst, k):
                    setattr(dst, k, getattr(src, k))
        self.sync_sliders_to_recipe()
        self._push_history("Paste Settings")
        self.render_preview()
        self.statusBar().showMessage("Settings pasted")


    def auto_exposure(self):
        if self.original_bgr is None or self.current_path is None:
            return
        import numpy as np
        img = self.original_bgr
        # Simple mid-gray target
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32") / 255.0
        med = float(np.median(gray))
        if med < 1e-4:
            return
        # target ~0.36 mid
        stops = float(np.log2(0.36 / med))
        stops = max(-2.0, min(2.0, stops))
        self.recipes[self.current_path].exposure = round(stops, 2)
        self.sync_sliders_to_recipe()
        self._push_history("Auto Exposure")
        self.render_preview()

    def auto_wb(self):
        if self.original_bgr is None or self.current_path is None:
            return
        import numpy as np
        img = self.original_bgr.astype("float32")
        b, g, r = img[:,:,0].mean(), img[:,:,1].mean(), img[:,:,2].mean()
        avg = (r + g + b) / 3.0 + 1e-6
        # crude temp from R/B ratio
        rb = r / (b + 1e-6)
        # map ratio to temp-ish
        temp = 5500 + (rb - 1.0) * 1500
        temp = max(2500, min(9000, temp))
        tint = (g / avg - 1.0) * 80
        rcp = self.recipes[self.current_path]
        rcp.wb_as_shot = False
        rcp.temperature = round(temp, 0)
        rcp.tint = round(max(-150, min(150, tint)), 0)
        self.sync_sliders_to_recipe()
        self._push_history("Auto WB")
        self.render_preview()


    def reset_module(self, which: str):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        fresh = Recipe()
        groups = {
            "tone": ["exposure", "smart_light", "contrast", "highlights", "shadows",
                     "whites", "blacks", "clarity", "gamma", "curve_shadows", "curve_darks",
                     "curve_mids", "curve_lights", "curve_highlights", "zebra_threshold",
                     "zebra_exposure", "zebra_feather"],
            "color": ["temperature", "tint", "wb_as_shot", "vibrance", "saturation",
                      "hsl_hue", "hsl_sat", "hsl_lum", "soft_proof", "soft_proof_profile",
                      "soft_proof_gamut"],
            "detail": ["denoise_luminance", "denoise_chroma", "denoise_strength",
                       "denoise_detail", "denoise_method", "sharpen_intensity",
                       "sharpen_radius", "sharpen_threshold", "sharpen_detail", "output_sharpen"],
            "geometry": [
                "horizon", "distortion", "perspective", "perspective_horizontal",
                "warp_top", "warp_bottom", "warp_left", "warp_right", "wide_angle",
                "diorama_strength", "diorama_position", "diorama_width", "diorama_angle",
                "keystone_points", "geometry_auto_crop",
                "crop", "ca_amount", "lens_auto",
            ],
            "local": ["local_points", "gradients", "brush_masks"],
            "effects": ["clearview", "microcontrast", "vignette", "film_grain",
                        "black_and_white", "rotate_90", "hdr_look"],
        }
        for k in groups.get(which, []):
            if hasattr(r, k) and hasattr(fresh, k):
                setattr(r, k, getattr(fresh, k))
        self.sync_sliders_to_recipe()
        self._push_history(f"Reset {which}")
        self.render_preview()
        self.statusBar().showMessage(f"Reset {which}")

    def toggle_pick_current(self):
        if self.current_path is None:
            return
        path = self.current_path
        if not hasattr(self, "_pick_flags"):
            self._pick_flags = {}
        new_val = not self._pick_flags.get(path, False)
        self._pick_flags[path] = new_val
        # visual on filmstrip
        for i in range(self.filmstrip.count()):
            item = self.filmstrip.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == path:
                base = os.path.basename(path)
                stars = self._image_ratings.get(path, 0)
                rej = getattr(self, "_reject_flags", {}).get(path, False)
                prefix = ""
                if new_val:
                    prefix += "✓ "
                if rej:
                    prefix += "⛔ "
                star = (("★" * stars + "☆" * (5 - stars) + "  ") if stars else "")
                item.setText(f"{prefix}{star}{base}")
                break
        self.statusBar().showMessage("Picked" if new_val else "Unpicked")

    def _lib_focus_stack(self):
        paths = self._lib_selected_paths()
        if len(paths) < 2:
            QMessageBox.information(
                self, "Focus Stack",
                "Select at least 2 library photos (focus brackets), then try again.",
            )
            return
        # Reuse filmstrip stack dialog by temporarily pointing selection flow
        # Sort paths for consistent near-far order (natural-ish)
        paths = sorted(paths)
        # Inject into a lightweight call: set filmstrip selection analog via attribute
        self._stack_paths_override = paths
        try:
            self.focus_stack_selected()
        finally:
            self._stack_paths_override = None

    def _lib_export_selected(self):

        paths = self._lib_selected_paths()
        if not paths:
            QMessageBox.information(self, "Export", "Select library photo(s) first.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Export library selection to folder")
        if not out_dir:
            return
        jobs = []
        for path in paths:
            base = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(out_dir, f"{base}_edited.jpg")
            recipe = self.recipes.get(path) or Recipe()
            # try load sidecar if no in-memory recipe edits
            if path not in self.recipes:
                try:
                    side = load_recipe_sidecar(path)
                    if side is not None:
                        recipe = side
                except Exception:
                    pass
            meta = self.meta_cache.get(path, {})
            jobs.append({
                "path": path,
                "recipe": recipe,
                "out_path": out_path,
                "wb_multipliers": meta.get("wb_multipliers"),
            })
        self.statusBar().showMessage(f"Exporting {len(jobs)} from library…")
        self._batch_worker = BatchExportWorker(jobs, max_dim=0)
        self._batch_worker.progress.connect(
            lambda i, n, p: self.statusBar().showMessage(f"Export {i+1}/{n}: {os.path.basename(p)}")
        )
        self._batch_worker.finished_ok.connect(
            lambda n: self.statusBar().showMessage(f"Library export done — {n} file(s)")
        )
        self._batch_worker.failed.connect(
            lambda e: self.statusBar().showMessage(f"Export error: {e}")
        )
        self._batch_worker.start()

    def reset_current(self):
        if self.current_path is None:
            return
        self.recipes[self.current_path].reset()
        self.sync_sliders_to_recipe()
        self.render_preview()


    def _selected_filmstrip_paths(self):
        paths = []
        for item in self.filmstrip.selectedItems():
            p = item.data(Qt.ItemDataRole.UserRole)
            if p:
                paths.append(p)
        return paths





    def open_audio_editor(self):
        """Launch the companion audio editor (optional soundtrack prep)."""
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audio_editor.py")
        if not os.path.isfile(script):
            script = os.path.join(os.getcwd(), "audio_editor.py")
        if not os.path.isfile(script):
            QMessageBox.warning(self, "Audio Editor", "audio_editor.py not found next to the app.")
            return
        import sys, subprocess
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        else:
            kwargs["start_new_session"] = True
        try:
            subprocess.Popen([sys.executable, script], **kwargs)
            self.statusBar().showMessage("Audio Editor opened")
        except Exception as e:
            QMessageBox.warning(self, "Audio Editor", str(e))

    def open_color_calibration_studio(self):
        """Open measured display/camera profiling backed by ArgyllCMS."""
        from color_calibration import ColorCalibrationDialog
        dlg = ColorCalibrationDialog(self)
        dlg.exec()



    def _on_ir_swap(self, _idx=0):
        if self.current_path is None or not hasattr(self, "ir_swap_combo"):
            return
        key = self.ir_swap_combo.currentData() or "none"
        self.recipes[self.current_path].ir_channel_swap = key
        self._schedule_history("IR swap")
        self.render_timer.start()

    def _on_ir_mono(self, checked):
        if self.current_path is None:
            return
        self.recipes[self.current_path].ir_mono = bool(checked)
        self._schedule_history("IR mono")
        self.render_timer.start()

    def _ir_preset_wood(self):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.ir_channel_swap = "rb"
        r.ir_false_color = 55.0
        r.ir_mono = False
        self._sync_ir_astro_widgets(r)
        self._push_history("IR Wood effect")
        self.render_preview()

    def _ir_preset_gold_blue(self):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.ir_channel_swap = "rb"
        r.ir_false_color = 80.0
        r.ir_mono = False
        self._sync_ir_astro_widgets(r)
        self._push_history("IR Gold/Blue")
        self.render_preview()

    def _ir_preset_mono(self):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.ir_channel_swap = "none"
        r.ir_false_color = 0.0
        r.ir_mono = True
        self._sync_ir_astro_widgets(r)
        self._push_history("IR Mono")
        self.render_preview()

    def _ir_preset_reset(self):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.ir_channel_swap = "none"
        r.ir_false_color = 0.0
        r.ir_mono = False
        self._sync_ir_astro_widgets(r)
        self._push_history("IR Reset")
        self.render_preview()

    def _astro_preset_milkyway(self):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.astro_stretch = 55.0
        r.astro_bg_remove = 40.0
        r.astro_star_emphasis = 25.0
        self._sync_ir_astro_widgets(r)
        self._push_history("Astro Milky Way")
        self.render_preview()

    def _astro_preset_dso(self):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.astro_stretch = 70.0
        r.astro_bg_remove = 55.0
        r.astro_star_emphasis = 15.0
        self._sync_ir_astro_widgets(r)
        self._push_history("Astro DSO")
        self.render_preview()

    def _astro_preset_reset(self):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.astro_stretch = 0.0
        r.astro_bg_remove = 0.0
        r.astro_star_emphasis = 0.0
        self._sync_ir_astro_widgets(r)
        self._push_history("Astro Reset")
        self.render_preview()

    def _sync_ir_astro_widgets(self, r):
        if hasattr(self, "ir_swap_combo"):
            key = str(getattr(r, "ir_channel_swap", "none") or "none")
            idx = self.ir_swap_combo.findData(key)
            if idx < 0:
                idx = 0
            self.ir_swap_combo.blockSignals(True)
            self.ir_swap_combo.setCurrentIndex(idx)
            self.ir_swap_combo.blockSignals(False)
        if hasattr(self, "ir_mono_cb"):
            self.ir_mono_cb.blockSignals(True)
            self.ir_mono_cb.setChecked(bool(getattr(r, "ir_mono", False)))
            self.ir_mono_cb.blockSignals(False)
        for key in ("ir_false_color", "astro_stretch", "astro_bg_remove", "astro_star_emphasis"):
            if key in getattr(self, "sliders", {}):
                self.sliders[key].blockSignals(True)
                self.sliders[key].set_value(float(getattr(r, key, 0) or 0))
                self.sliders[key].blockSignals(False)

    def _on_zone_enabled(self, checked: bool):
        if self.current_path is None:
            return
        self.recipes[self.current_path].zone_enabled = bool(checked)
        self._schedule_history("Zone enable")
        self.render_timer.start()

    def _on_zone_filter(self, _idx: int = 0):
        if self.current_path is None or not hasattr(self, "zone_filter_combo"):
            return
        key = self.zone_filter_combo.currentData()
        self.recipes[self.current_path].zone_filter = key or "none"
        self._schedule_history("Zone filter")
        self.render_timer.start()

    def _on_zone_overlay(self, checked: bool):
        if self.current_path is None:
            return
        self.recipes[self.current_path].zone_overlay = bool(checked)
        self.render_timer.start()

    def _apply_zone_preset(self, placement: float, expansion: float):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.zone_enabled = True
        r.zone_placement = float(placement)
        r.zone_expansion = float(expansion)
        r.black_and_white = True
        if hasattr(self, "zone_enabled_cb"):
            self.zone_enabled_cb.blockSignals(True)
            self.zone_enabled_cb.setChecked(True)
            self.zone_enabled_cb.blockSignals(False)
        if hasattr(self, "bw_cb"):
            self.bw_cb.blockSignals(True)
            self.bw_cb.setChecked(True)
            self.bw_cb.blockSignals(False)
        self.sync_sliders_to_recipe()
        self._push_history("Zone preset")
        self.render_preview()

    def show_map_view(self):
        paths = []
        try:
            paths = self._selected_filmstrip_paths()
        except Exception:
            paths = []
        if len(paths) < 1:
            paths = list(getattr(self, "image_paths", []) or [])
        if not paths:
            QMessageBox.information(self, "Map", "Open a folder or select images first.")
            return
        from map_view import MapDialog
        dlg = MapDialog(paths, parent=self, on_open_path=self.load_image)
        dlg.exec()

    def _map_current_photo(self):
        """Open the map list for only the currently displayed geotagged photo."""
        if not self.current_path:
            return
        from imaging import extract_gps
        if not extract_gps(self.current_path):
            QMessageBox.information(self, "Map", "This photo does not contain GPS coordinates.")
            return
        from map_view import MapDialog
        MapDialog([self.current_path], parent=self, on_open_path=self.load_image).exec()

    def select_gps_photos_and_map(self):
        """Select geotagged filmstrip items, then display that selection on the map."""
        if not getattr(self, "image_paths", None):
            QMessageBox.information(self, "Map", "Open a folder first.")
            return
        from imaging import extract_gps
        progress = QProgressDialog("Checking photos for GPS coordinates…", "Cancel", 0,
                                   self.filmstrip.count(), self)
        progress.setWindowTitle("Find GPS photos")
        progress.setMinimumDuration(300)
        found = []
        self.filmstrip.clearSelection()
        for i in range(self.filmstrip.count()):
            if progress.wasCanceled():
                break
            item = self.filmstrip.item(i)
            path = item.data(Qt.ItemDataRole.UserRole)
            if path and extract_gps(path):
                item.setSelected(True)
                found.append(path)
            progress.setValue(i + 1)
            QApplication.processEvents()
        progress.close()
        if not found:
            QMessageBox.information(self, "Map", "No photos with GPS coordinates were found.")
            return
        self.statusBar().showMessage(f"Selected {len(found)} GPS-tagged photo(s)")
        from map_view import MapDialog
        MapDialog(found, parent=self, on_open_path=self.load_image).exec()

    def start_slideshow(self):
        paths = []
        try:
            paths = self._selected_filmstrip_paths()
        except Exception:
            paths = []
        if len(paths) < 1:
            paths = list(getattr(self, "image_paths", []) or [])
        if not paths:
            QMessageBox.information(self, "Slideshow", "Open a folder or select images first.")
            return
        def _load(path):
            from imaging import load_image, apply_recipe
            img, meta = load_image(path)
            r = self.recipes.get(path)
            if r is not None:
                img = apply_recipe(img, r, wb_multipliers=meta.get("wb_multipliers"), meta=meta)
            return img
        from slideshow import SlideshowWindow
        self._slideshow = SlideshowWindow(paths, _load, interval_ms=4000, ken_burns=True)
        self._slideshow.show()
        self.statusBar().showMessage("Slideshow — Esc exit, Space pause, K Ken Burns")

    def show_print_dialog(self):
        if getattr(self, "original_bgr", None) is None or self.current_path is None:
            QMessageBox.information(self, "Print", "Open an image first.")
            return
        from print_dialog import PrintDialog
        r = self.recipes.get(self.current_path)
        suggested = os.path.splitext(self.current_path)[0] + "_print.pdf"
        dlg = PrintDialog(self, self.original_bgr, r, default_path=suggested)
        dlg.exec()

    def run_user_script(self):
        from PyQt6.QtWidgets import QInputDialog, QMessageBox

        if self.current_path is None:
            QMessageBox.information(self, "Run Script", "Open an image first.")
            return
        from script_runner import list_scripts, run_script, scripts_dir
        scripts = list_scripts()
        if not scripts:
            QMessageBox.information(
                self, "Run Script",
                f"No scripts in:\n{scripts_dir()}\n\nAdd a .py file that accepts --path and --recipe.",
            )
            return
        names = [os.path.basename(s) for s in scripts]
        name, ok = QInputDialog.getItem(self, "Run Script", "Script:", names, 0, False)
        if not ok:
            return
        script = scripts[names.index(name)]
        recipe = self.recipes.get(self.current_path)
        code, out, err = run_script(script, self.current_path, recipe)
        msg = (out or "") + (("\n" + err) if err else "")
        self.log(f"Script {name} exit={code}")
        QMessageBox.information(self, "Run Script", f"{name} finished (code {code})\n\n{(msg or '(no output)')[:1500]}")

    def check_for_updates(self):
        from PyQt6.QtGui import QDesktopServices
        from PyQt6.QtCore import QUrl
        try:
            from config import get_config
            url = get_config().get("ui", "check_for_updates_url") or "https://github.com/betoon/photo_lab/releases"
        except Exception:
            url = "https://github.com/betoon/photo_lab/releases"
        QDesktopServices.openUrl(QUrl(url))
        self.statusBar().showMessage(f"Opened: {url}")

    def export_problem_report(self):
        """Write a diagnostics text file: system info + recent log lines."""
        import platform, traceback
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Problem Report", "photolab_problem_report.txt", "Text (*.txt)"
        )
        if not path:
            return
        lines = [
            "PhotoLab problem report",
            f"Platform: {platform.platform()}",
            f"Python: {platform.python_version()}",
            f"Current image: {self.current_path or '(none)'}",
        ]
        try:
            import cv2, numpy
            lines.append(f"OpenCV: {cv2.__version__}")
            lines.append(f"NumPy: {numpy.__version__}")
        except Exception as e:
            lines.append(f"Import check: {e}")
        # recent debug log if available
        log_buf = getattr(self, "_debug_log_lines", None) or getattr(self, "debug_lines", None)
        if log_buf:
            lines.append("--- debug log (tail) ---")
            lines.extend(list(log_buf)[-50:])
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            self.statusBar().showMessage(f"Report saved → {path}")
            QMessageBox.information(self, "Report a Problem", f"Saved:\n{path}")
        except Exception as e:
            QMessageBox.warning(self, "Report a Problem", str(e))

    def clear_caches(self):
        """Clear proxy/preview caches and optional thumb cache dir."""
        cleared = []
        for attr in ("_preview_cache", "_proxy_cache", "proxy_cache", "thumb_cache"):
            obj = getattr(self, attr, None)
            if isinstance(obj, dict):
                obj.clear()
                cleared.append(attr)
        try:
            from catalog import default_thumb_dir
            import shutil
            td = default_thumb_dir()
            if os.path.isdir(td):
                # only clear files, keep dir
                n = 0
                for name in os.listdir(td):
                    fp = os.path.join(td, name)
                    if os.path.isfile(fp):
                        try:
                            os.remove(fp)
                            n += 1
                        except OSError:
                            pass
                cleared.append(f"thumbs:{n}")
        except Exception as e:
            cleared.append(f"thumbs-error:{e}")
        self.statusBar().showMessage(f"Caches cleared ({', '.join(cleared) or 'nothing'})")
        QMessageBox.information(self, "Clear Caches", "Preview/proxy caches cleared.\nThumb cache cleaned when available.")



    def _menu_open_video_editor(self):
        path = getattr(self, "current_path", None)
        if path and self._is_video_path(path):
            self.open_video_editor(path)
        else:
            self.open_video_editor(None)

    def open_video_editor(self, path=None):
        """Launch VeloCut Studio (video_editor.py), optionally with a clip preloaded."""
        import sys
        import subprocess
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "video_editor.py")
        if not os.path.isfile(script):
            alt = os.path.join(os.getcwd(), "video_editor.py")
            script = alt if os.path.isfile(alt) else script
        if not os.path.isfile(script):
            QMessageBox.warning(
                self, "Video Editor",
                f"Could not find video_editor.py next to the app.\n\n{script}",
            )
            return
        cmd = [sys.executable, script]
        if path and os.path.isfile(path):
            cmd.append(path)
        self.log(f"Launching Video Editor: {' '.join(cmd)}")
        try:
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            else:
                kwargs["start_new_session"] = True
            subprocess.Popen(cmd, **kwargs)
            self.statusBar().showMessage(
                f"Video Editor opened" + (f" — {os.path.basename(path)}" if path else "")
            )
        except Exception as e:
            self.log(f"Video Editor launch failed: {e}", level="ERR")
            QMessageBox.warning(self, "Video Editor", f"Could not launch:\n{e}")

    def _is_video_path(self, path: str) -> bool:
        if not path:
            return False
        return os.path.splitext(path)[1].lower() in VIDEO_EXTENSIONS

    def create_pan_video(self):
        """Launch the full Panorama-to-Video tool with the current image preloaded.

        Keeps every feature of pano_video.py (Ken Burns, LUTs, ffmpeg, audio, etc.)
        by running it as a separate process — no feature loss.
        """
        path = self.current_path
        if not path or not os.path.isfile(path):
            QMessageBox.information(
                self,
                "Create Pan Video",
                "Open a panorama or wide image in Develop first.\n\n"
                "Tip: stitch with Image → Panorama… then send the result here.",
            )
            return

        # Prefer exporting a current recipe preview if heavy edits exist?
        # v1: pass the source file path; user can export a flat TIFF first if needed.
        script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pano_video.py")
        if not os.path.isfile(script):
            # also try cwd
            alt = os.path.join(os.getcwd(), "pano_video.py")
            script = alt if os.path.isfile(alt) else script
        if not os.path.isfile(script):
            QMessageBox.warning(
                self,
                "Create Pan Video",
                f"Could not find pano_video.py next to the application.\n\nLooked for:\n{script}",
            )
            return

        out_folder = self.folder or os.path.dirname(path)
        import sys
        import subprocess
        cmd = [
            sys.executable,
            script,
            "--image", path,
            "--output-folder", out_folder,
            "--title", "Panorama to Video (from PhotoLab)",
        ]
        self.log(f"Launching Pan Video: {' '.join(cmd)}")
        self.statusBar().showMessage("Launching Panorama to Video…")
        try:
            # Detached so PhotoLab stays responsive; on Windows use CREATE_NEW_PROCESS_GROUP
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            else:
                kwargs["start_new_session"] = True
            subprocess.Popen(cmd, **kwargs)
            self.statusBar().showMessage("Panorama to Video opened — finish render there")
        except Exception as e:
            self.log(f"Pan Video launch failed: {e}", level="ERR")
            QMessageBox.warning(self, "Create Pan Video", f"Could not launch:\n{e}")

    def panorama_selected(self):
        """OpenCV Stitcher v1 — automatic panorama from selected filmstrip/library frames."""
        paths = getattr(self, "_pano_paths_override", None)
        if not paths:
            paths = self._selected_filmstrip_paths()
        if len(paths) < 2:
            paths = list(getattr(self, "image_paths", []) or [])
        if len(paths) < 2:
            QMessageBox.information(
                self,
                "Panorama",
                "Select at least 2 overlapping images in the filmstrip "
                "(left→right or ordered around the scene),\n"
                "then Image → Panorama…\n\n"
                "Best results: ~30% overlap, consistent exposure, minimal parallax.",
            )
            return

        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Panorama (OpenCV) — {len(paths)} frames")
        form = QFormLayout(dlg)

        mode_combo = QComboBox()
        mode_combo.addItem("Panorama (default)", "panoramas")
        mode_combo.addItem("Scans / flat copy", "scans")
        mode_combo.addItem("Auto", "auto")
        form.addRow("Mode", mode_combo)

        size_combo = QComboBox()
        size_combo.addItem("Full resolution", 0)
        size_combo.addItem("Long edge 3000 px", 3000)
        size_combo.addItem("Long edge 2000 px", 2000)
        size_combo.addItem("Long edge 1200 px (preview)", 1200)
        form.addRow("Working size", size_combo)

        tip = QLabel(
            "OpenCV automatic stitch — good for clean rows with solid overlap.\n"
            "Not ideal for strong parallax, moving subjects, or huge exposure gaps.\n"
            "Order frames left→right when possible. Result opens in Develop."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#888; font-size:11px;")
        form.addRow(tip)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        folder = self.folder or os.path.dirname(paths[0])
        suggested = os.path.join(folder, "panorama_result.tif")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save panorama", suggested,
            "TIFF (*.tif);;JPEG (*.jpg);;PNG (*.png);;All (*.*)",
        )
        if not out_path:
            return

        self.statusBar().showMessage("Stitching panorama…")
        self.log(f"Panorama: {len(paths)} frames → {out_path}")
        self._pano_worker = PanoramaWorker(
            paths,
            out_path,
            mode=mode_combo.currentData() or "panoramas",
            max_dim=int(size_combo.currentData() or 0),
        )
        self._pano_worker.progress.connect(
            lambda m: self.statusBar().showMessage(f"Panorama: {m}")
        )
        self._pano_worker.finished_ok.connect(self._on_panorama_done)
        self._pano_worker.failed.connect(
            lambda e: (
                self.statusBar().showMessage(f"Panorama failed: {e}"),
                self.log(f"Panorama failed: {e}", level="ERR"),
                QMessageBox.warning(self, "Panorama", f"Failed:\n{e}"),
            )
        )
        self._pano_worker.start()

    def _on_panorama_done(self, out_path: str, report: object):
        self.statusBar().showMessage(f"Panorama saved → {out_path}")
        self.log(f"Panorama done: {out_path}")
        if isinstance(report, dict) and report.get("result_size"):
            w, h = report["result_size"]
            self.log(f"Panorama size: {w}×{h}")
        folder = os.path.dirname(out_path)
        try:
            if self.folder != folder:
                self.open_folder_path(folder)
            else:
                if out_path not in self.image_paths:
                    self.image_paths.append(out_path)
                    item = QListWidgetItem(os.path.basename(out_path))
                    item.setData(Qt.ItemDataRole.UserRole, out_path)
                    item.setSizeHint(QSize(108, 118))
                    self.filmstrip.addItem(item)
            self.load_image(out_path)
            self.show_develop_mode()
        except Exception as e:
            self.log(f"Could not open panorama: {e}", level="ERR")
            QMessageBox.information(
                self, "Panorama",
                f"Saved:\n{out_path}\n\nOpen it from the filmstrip when ready.",
            )

    def _lib_panorama(self):
        paths = self._lib_selected_paths()
        if len(paths) < 2:
            QMessageBox.information(
                self, "Panorama",
                "Select at least 2 overlapping library photos, then try again.",
            )
            return
        self._pano_paths_override = sorted(paths)
        try:
            self.panorama_selected()
        finally:
            self._pano_paths_override = None


    def open_focus_stacker_pro(self):
        """Launch the full Focus Stacker Pro GUI (microscope, retouch, 16-bit, etc.)."""
        import sys
        import subprocess
        base = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.join(base, "run_focus_stacker_pro.py"),
            os.path.join(base, "focus_stacker_pro", "run.py"),
            os.path.expanduser(r"~/Documents/GitHub/focus_stacker/run.py"),
            r"C:\Users\brian\Documents\GitHub\focus_stacker\run.py",
        ]
        script = next((c for c in candidates if os.path.isfile(c)), None)
        if not script:
            QMessageBox.warning(
                self, "Focus Stacker Pro",
                "Could not find Focus Stacker Pro.\n\n"
                "Expected run_focus_stacker_pro.py next to PhotoLab, or\n"
                "C:\\Users\\brian\\Documents\\GitHub\\focus_stacker\\run.py\n\n"
                "Install PySide6 in that environment: pip install PySide6",
            )
            return
        cmd = [sys.executable, script]
        self.log(f"Launching Focus Stacker Pro: {' '.join(cmd)}")
        try:
            kwargs = {}
            if os.name == "nt":
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
            else:
                kwargs["start_new_session"] = True
            # cwd so relative imports work for focus_stacker_pro/run.py
            kwargs["cwd"] = os.path.dirname(script) if script.endswith("run.py") else base
            subprocess.Popen(cmd, **kwargs)
            self.statusBar().showMessage("Focus Stacker Pro opened")
        except Exception as e:
            QMessageBox.warning(self, "Focus Stacker Pro", f"Could not launch:\n{e}")

    def focus_stack_selected(self):
        """Thin focus-stack UI: select 2+ filmstrip frames → align → fuse → open result."""
        paths = getattr(self, "_stack_paths_override", None)
        if not paths:
            paths = self._selected_filmstrip_paths()
        if len(paths) < 2:
            # fall back to all filmstrip if nothing selected
            paths = list(getattr(self, "image_paths", []) or [])
        if len(paths) < 2:
            QMessageBox.information(
                self,
                "Focus Stack",
                "Select at least 2 images in the filmstrip (or open a folder of focus brackets),\n"
                "then run Image → Focus Stack…",
            )
            return

        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QSpinBox, QDoubleSpinBox
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Focus Stack — {len(paths)} frames")
        form = QFormLayout(dlg)

        align_combo = QComboBox()
        align_combo.addItem("ECC Affine (recommended)", "ecc_affine")
        align_combo.addItem("ECC Translation", "ecc_translation")
        align_combo.addItem("ECC Rigid", "ecc_rigid")
        align_combo.addItem("ECC Homography", "ecc_homography")
        align_combo.addItem("ORB Affine", "orb_affine")
        align_combo.addItem("ORB Homography", "orb_homography")
        form.addRow("Alignment", align_combo)

        fusion_combo = QComboBox()
        fusion_combo.addItem("Depth map", "depth")
        fusion_combo.addItem("Weighted", "weighted")
        fusion_combo.addItem("Pyramid (Laplacian)", "pyramid")
        fusion_combo.addItem("Average", "average")
        form.addRow("Fusion", fusion_combo)

        ref_combo = QComboBox()
        ref_combo.addItem("Middle frame", "middle")
        ref_combo.addItem("First frame", "first")
        ref_combo.addItem("Last frame", "last")
        form.addRow("Reference", ref_combo)

        size_combo = QComboBox()
        size_combo.addItem("Full resolution", 0)
        size_combo.addItem("Long edge 3000 px", 3000)
        size_combo.addItem("Long edge 2000 px", 2000)
        size_combo.addItem("Long edge 1200 px (preview)", 1200)
        form.addRow("Working size", size_combo)

        radius_spin = QSpinBox()
        radius_spin.setRange(1, 15)
        radius_spin.setValue(5)
        radius_spin.setToolTip("Focus measure radius (Focus Stacker Pro default 5)")
        form.addRow("Focus radius", radius_spin)

        smooth_spin = QSpinBox()
        smooth_spin.setRange(0, 31)
        smooth_spin.setValue(7)
        form.addRow("Boundary smooth", smooth_spin)

        levels_spin = QSpinBox()
        levels_spin.setRange(2, 8)
        levels_spin.setValue(5)
        form.addRow("Pyramid levels", levels_spin)

        min_score = QDoubleSpinBox()
        min_score.setRange(0.0, 1.0)
        min_score.setSingleStep(0.05)
        min_score.setValue(0.0)
        min_score.setToolTip("Exclude non-reference frames with weaker alignment (0 = keep all)")
        form.addRow("Min align score", min_score)

        crop_cb = QCheckBox("Crop common area")
        crop_cb.setChecked(True)
        form.addRow(crop_cb)
        depth_cb = QCheckBox("Also save depth map PNG")
        form.addRow(depth_cb)
        norm_cb = QCheckBox("Normalize exposure across stack")
        form.addRow(norm_cb)

        tip = QLabel(
            "Focus brackets near→far on a stable camera.\n"
            "For microscope tools, alignment inspector, retouch, and 16-bit archival export,\n"
            "use Tools → Open Focus Stacker Pro…"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#888; font-size:11px;")
        form.addRow(tip)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        folder = self.folder or os.path.dirname(paths[0])
        suggested = os.path.join(folder, "focus_stack_result.tif")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save focus stack result", suggested,
            "TIFF (*.tif);;JPEG (*.jpg);;PNG (*.png);;All (*.*)",
        )
        if not out_path:
            return

        self.statusBar().showMessage("Focus stacking…")
        self.log(f"Focus stack: {len(paths)} frames → {out_path}")
        self._stack_worker = FocusStackWorker(
            paths,
            out_path,
            align_mode=align_combo.currentData(),
            fusion_mode=fusion_combo.currentData(),
            reference=ref_combo.currentData(),
            max_dim=int(size_combo.currentData() or 0),
            focus_radius=int(radius_spin.value()),
            boundary_smooth=int(smooth_spin.value()),
            pyramid_levels=int(levels_spin.value()),
            crop_common=crop_cb.isChecked(),
            save_depth=depth_cb.isChecked(),
            min_align_score=float(min_score.value()),
            normalize_exposure=bool(norm_cb.isChecked()),
        )
        self._stack_worker.progress.connect(
            lambda m: self.statusBar().showMessage(f"Focus stack: {m}")
        )
        self._stack_worker.finished_ok.connect(self._on_focus_stack_done)
        self._stack_worker.failed.connect(
            lambda e: (
                self.statusBar().showMessage(f"Focus stack failed: {e}"),
                self.log(f"Focus stack failed: {e}", level="ERR"),
                QMessageBox.warning(self, "Focus Stack", f"Failed:\n{e}"),
            )
        )
        self._stack_worker.start()

    def _on_focus_stack_done(self, out_path: str, report: object):
        self.statusBar().showMessage(f"Focus stack saved → {out_path}")
        self.log(f"Focus stack done: {out_path} ({getattr(report, 'get', lambda k, d=None: d)('frames', '?') if isinstance(report, dict) else ''} frames)")
        if isinstance(report, dict):
            scores = report.get("scores") or []
            low = [s for s in scores if float(s.get("score") or 0) < 0.15 and s.get("index") != report.get("reference")]
            if low:
                self.log(f"Warning: {len(low)} frame(s) had weak alignment scores")
        # Open folder containing result and load it
        folder = os.path.dirname(out_path)
        try:
            if self.folder != folder:
                self.open_folder_path(folder)
            else:
                # ensure path in filmstrip
                if out_path not in self.image_paths:
                    self.image_paths.append(out_path)
                    item = QListWidgetItem(os.path.basename(out_path))
                    item.setData(Qt.ItemDataRole.UserRole, out_path)
                    item.setSizeHint(QSize(108, 118))
                    self.filmstrip.addItem(item)
            self.load_image(out_path)
            self.show_develop_mode()
        except Exception as e:
            self.log(f"Could not open stack result: {e}", level="ERR")
            QMessageBox.information(
                self, "Focus Stack",
                f"Saved:\n{out_path}\n\nOpen it from the filmstrip when ready.",
            )

    def merge_hdr_selected(self):
        paths = self._selected_filmstrip_paths()
        if len(paths) < 2:
            QMessageBox.information(
                self,
                "Merge HDR",
                "Select 2 or more images in the filmstrip first.\n\n"
                "Tip: Ctrl+click or Shift+click to multi-select brackets,\n"
                "then Image → Merge HDR… (Ctrl+Shift+H).",
            )
            return
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QSpinBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Merge HDR")
        dlg.setMinimumWidth(420)
        form = QFormLayout(dlg)
        names = "\n".join("  • " + os.path.basename(p) for p in paths[:12])
        if len(paths) > 12:
            names += "\n  …"
        info = QLabel(f"{len(paths)} images selected:\n{names}")
        info.setWordWrap(True)
        form.addRow(info)
        align_cb = QCheckBox("Align frames (recommended for handheld)")
        align_cb.setChecked(True)
        form.addRow(align_cb)
        method_combo = QComboBox()
        method_combo.addItem("Mertens (exposure fusion)", "mertens")
        method_combo.addItem("Debevec + tonemap (true HDR)", "debevec")
        form.addRow("Method", method_combo)
        deghost_spin = QSpinBox()
        deghost_spin.setRange(0, 100)
        deghost_spin.setValue(0)
        deghost_spin.setToolTip("0 = off. Higher values reduce ghosting from movement.")
        form.addRow("Deghost strength", deghost_spin)
        size_combo = QComboBox()
        size_combo.addItem("Full resolution", 0)
        size_combo.addItem("Preview (long edge 2000px)", 2000)
        size_combo.addItem("Fast preview (long edge 1200px)", 1200)
        form.addRow("Output size", size_combo)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        align = align_cb.isChecked()
        max_dim = int(size_combo.currentData() or 0)
        method = method_combo.currentData() or "mertens"
        deghost = int(deghost_spin.value())
        base = os.path.splitext(os.path.basename(paths[0]))[0]
        suggested = os.path.join(self.folder or ".", f"{base}_HDR.jpg")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save HDR merge", suggested,
            "JPEG (*.jpg);;PNG (*.png);;TIFF (*.tif);;All (*.*)",
        )
        if not out_path:
            return
        self.statusBar().showMessage(f"HDR merge: {len(paths)} frames…")
        self.log(f"HDR merge started ({len(paths)} frames, {method}, deghost={deghost}) → {out_path}")
        self._hdr_worker = HdrMergeWorker(
            paths, out_path, align=align, max_dim=max_dim,
            method=method, deghost=deghost,
        )
        self._hdr_worker.progress.connect(lambda m: self.statusBar().showMessage(m))
        self._hdr_worker.finished_ok.connect(self._on_hdr_merge_done)
        self._hdr_worker.failed.connect(
            lambda e: (
                self.statusBar().showMessage(f"HDR merge failed: {e}"),
                self.log(f"HDR merge FAILED: {e}", level="ERR"),
                QMessageBox.warning(self, "Merge HDR", str(e)),
            )
        )
        self._hdr_worker.start()

    def _on_hdr_merge_done(self, out_path):
        self.statusBar().showMessage(f"HDR merge saved → {out_path}")
        self.log(f"HDR merge OK: {out_path}")
        try:
            folder = os.path.dirname(out_path)
            if self.folder and os.path.normpath(folder) == os.path.normpath(self.folder):
                if out_path not in self.image_paths:
                    self.image_paths.append(out_path)
                    self.image_paths.sort()
                    item = QListWidgetItem(os.path.basename(out_path))
                    item.setData(Qt.ItemDataRole.UserRole, out_path)
                    self.filmstrip.addItem(item)
                    img = cv2.imread(out_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        scale = 120 / max(h, w)
                        thumb = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))))
                        item.setIcon(QIcon(cv_to_qpixmap(thumb)))
                self.load_image(out_path)
            else:
                QMessageBox.information(self, "Merge HDR", f"Saved:\n{out_path}")
        except Exception as e:
            QMessageBox.information(self, "Merge HDR", f"Saved:\n{out_path}\n\n({e})")




    def save_sidecar(self):
        if self.current_path is None:
            return
        path = save_recipe_sidecar(self.current_path, self.recipes[self.current_path])
        self.statusBar().showMessage(f"Recipe saved → {path}")
        self.log(f"Sidecar written: {path}")

    def reload_sidecar(self):
        if self.current_path is None:
            return
        r = load_recipe_sidecar(self.current_path)
        if r is None:
            QMessageBox.information(self, "Sidecar", "No .photolab.json next to this file.")
            return
        self.recipes[self.current_path] = r
        self.sync_sliders_to_recipe()
        self.render_preview()
        self.statusBar().showMessage("Recipe reloaded from sidecar")


    def _on_gamut_warning(self, checked):
        if self.current_path is None:
            return
        self.recipes[self.current_path].soft_proof_gamut = bool(checked)
        self.render_timer.start()

    def _on_brush_invert(self):
        if self.current_path is None or getattr(self, "selected_brush_index", -1) < 0:
            return
        masks = self.recipes[self.current_path].brush_masks or []
        if self.selected_brush_index >= len(masks):
            return
        m = masks[self.selected_brush_index]
        m["invert"] = not bool(m.get("invert") or m.get("inverted"))
        m.pop("inverted", None)
        self.preview.set_brush_masks(masks, self.selected_brush_index)
        self._update_brush_list()
        self._push_history("Invert brush mask")
        self.render_preview()
        self.statusBar().showMessage(
            "Brush mask inverted" if m.get("invert") else "Brush mask normal"
        )

    def _on_brush_erase(self, checked):
        self.preview.brush_erase = bool(checked)

    def _on_brush_mask_only(self, checked):
        self.preview.show_mask_only = bool(checked)
        self.preview.update()

    def toggle_brush_mode(self, checked=False):
        on = checked if isinstance(checked, bool) else (not self.preview.brush_mode)
        if hasattr(self, "act_brush"):
            if not isinstance(checked, bool):
                on = not self.preview.brush_mode
            self.act_brush.setChecked(on)
        self.preview.brush_mode = on
        if on:
            self.preview.gradient_mode = False
            self.preview.wb_picker_mode = False
            self.preview.local_mode = False
            self._local_mode = False
            for act_name in ("act_grad", "act_wb_pick", "act_local"):
                a = getattr(self, act_name, None)
                if a:
                    a.setChecked(False)
            if hasattr(self, "_cat_buttons") and len(self._cat_buttons) > 5:
                self._cat_buttons[5].setChecked(True)
                self.tool_stack.setCurrentIndex(5)
            self.preview.setCursor(Qt.CursorShape.CrossCursor)
            self.statusBar().showMessage("Brush: paint on image • Shift+B to toggle • size slider in Local tab")
        else:
            self.preview.setCursor(Qt.CursorShape.ArrowCursor)
            self.statusBar().showMessage("Brush off")

    def _on_brush_stroke_finished(self):
        if self.current_path is None:
            return
        self.recipes[self.current_path].brush_masks = list(self.preview.brush_masks)
        self.selected_brush_index = self.preview.selected_brush
        self._update_brush_list()
        self._sync_brush_sliders()
        self._push_history("Brush stroke")
        self.render_preview()

    def _on_brush_changed(self):
        if self.current_path is None:
            return
        self.recipes[self.current_path].brush_masks = list(self.preview.brush_masks)
        self.render_timer.start()

    def _update_brush_list(self):
        if not hasattr(self, "brush_list") or self.current_path is None:
            return
        self.brush_list.blockSignals(True)
        self.brush_list.clear()
        masks = self.recipes[self.current_path].brush_masks or []
        for i, m in enumerate(masks):
            n = len(m.get("strokes") or [])
            inv = " INV" if m.get("invert") or m.get("inverted") else ""
            preset = f" · {m.get('preset_name')}" if m.get("local_preset") else ""
            self.brush_list.addItem(f"Brush {i+1}{inv}  ({n} dabs, exp {m.get('exposure', 0):+.2f}){preset}")
        sel = getattr(self, "selected_brush_index", -1)
        if 0 <= sel < len(masks):
            self.brush_list.setCurrentRow(sel)
        self.brush_list.blockSignals(False)
        self.preview.set_brush_masks(masks, sel)

    def _on_brush_list_selection(self, row):
        self.selected_brush_index = row
        self.preview.selected_brush = row
        self.preview.update()
        self._sync_brush_sliders()

    def _sync_brush_sliders(self):
        if not hasattr(self, "brush_sliders_box"):
            return
        masks = (self.recipes.get(self.current_path).brush_masks if self.current_path else None) or []
        ok = getattr(self, "selected_brush_index", -1) >= 0 and self.selected_brush_index < len(masks)
        self.brush_sliders_box.setEnabled(ok)
        if hasattr(self, "brush_preset_box"):
            self.brush_preset_box.setEnabled(ok)
        if not ok:
            if hasattr(self, "brush_preset_label"):
                self.brush_preset_label.setText("No local preset")
            return
        m = masks[self.selected_brush_index]
        for key, row in self.brush_sliders.items():
            row.blockSignals(True)
            defaults = {"luminance_max": 1.0, "color_tolerance": 0.2}
            value = float(m.get(key, defaults.get(key, 0.0)))
            if key in ("edge_refine", "luminance_min", "luminance_max", "color_tolerance"):
                value *= 100.0
            row.set_value(value)
            row.blockSignals(False)
        self.brush_color_range_cb.blockSignals(True)
        self.brush_color_range_cb.setChecked(bool(m.get("color_range", False)))
        self.brush_color_range_cb.blockSignals(False)
        if hasattr(self, "brush_preset_label"):
            self.brush_preset_label.setText(
                f"Preset: {m.get('preset_name', 'Local look')}" if m.get("local_preset") else "No local preset"
            )
            self.brush_preset_strength.blockSignals(True)
            self.brush_preset_strength.set_value(float(m.get("preset_strength", 1.0)) * 100.0)
            self.brush_preset_strength.blockSignals(False)

    def _apply_preset_to_brush_mask(self):
        idx = getattr(self, "selected_brush_index", -1)
        if self.current_path is None or idx < 0:
            QMessageBox.information(self, "Local preset", "Paint or select a brush mask first.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Apply preset to selected mask", self._preset_start_dir(),
            "All Presets (*.xmp *.json);;Lightroom XMP (*.xmp);;PhotoLab JSON (*.json)"
        )
        if not path:
            return
        try:
            # A neutral base makes the stored look independent of current global edits.
            local_recipe = load_preset_file(path, base=Recipe())
            mask = self.recipes[self.current_path].brush_masks[idx]
            mask["local_preset"] = local_recipe.to_dict()
            mask["preset_name"] = os.path.basename(path)
            mask.setdefault("preset_strength", 1.0)
            self._update_brush_list()
            self._sync_brush_sliders()
            self._push_history(f"Local preset: {os.path.basename(path)}")
            self.render_preview()
        except Exception as exc:
            QMessageBox.warning(self, "Local preset", f"Could not load preset:\n{exc}")

    def _clear_brush_mask_preset(self):
        idx = getattr(self, "selected_brush_index", -1)
        if self.current_path is None or idx < 0:
            return
        masks = self.recipes[self.current_path].brush_masks or []
        if idx >= len(masks):
            return
        masks[idx].pop("local_preset", None)
        masks[idx].pop("preset_name", None)
        masks[idx].pop("preset_strength", None)
        self._update_brush_list()
        self._sync_brush_sliders()
        self._push_history("Clear local preset")
        self.render_preview()

    def toggle_keystone_mode(self, checked):
        if self.original_bgr is None or self.current_path is None:
            self.keystone_tool_btn.blockSignals(True)
            self.keystone_tool_btn.setChecked(False)
            self.keystone_tool_btn.blockSignals(False)
            return
        if checked:
            self.crop_tool_btn.setChecked(False)
            self.preview.set_crop_mode(False)
            self.set_compare_mode(ImageCanvas.MODE_NORMAL)
        points = getattr(self.recipes[self.current_path], "keystone_points", [])
        self.preview.set_keystone_mode(checked, points)
        self.statusBar().showMessage(
            "Drag the four blue handles around the photographed rectangle; release to preview correction."
            if checked else "4-corner perspective tool off"
        )

    def _on_keystone_changed(self, points):
        if self.current_path is not None:
            self.recipes[self.current_path].keystone_points = [list(p) for p in points]

    def _on_keystone_finished(self, points):
        if self.current_path is None:
            return
        valid = normalize_keystone_points(points)
        if not valid:
            self.statusBar().showMessage("The four corners crossed or became too small; correction was not applied.")
            return
        self.recipes[self.current_path].keystone_points = valid
        self._schedule_history("4-corner perspective")
        self.render_preview()

    def clear_keystone(self):
        if self.current_path is None:
            return
        self.recipes[self.current_path].keystone_points = []
        self.preview.set_keystone_mode(self.keystone_tool_btn.isChecked(), [])
        self._schedule_history("clear 4-corner perspective")
        self.render_preview()

    def auto_architectural_upright(self):
        if self.original_bgr is None or self.current_path is None:
            return
        horizon, vertical, count = detect_architectural_upright(self.original_bgr)
        if count < 2:
            QMessageBox.information(self, "Auto Upright", "Not enough strong architectural lines were found in this image.")
            return
        recipe = self.recipes[self.current_path]
        recipe.horizon = horizon
        recipe.perspective = vertical
        for key, value in (("horizon", horizon), ("perspective", vertical)):
            if key in self.sliders:
                self.sliders[key].blockSignals(True)
                self.sliders[key].set_value(value)
                self.sliders[key].blockSignals(False)
        self._schedule_history("auto architectural upright")
        self.render_preview()
        self.statusBar().showMessage(
            f"Auto Upright used {count} lines: level {horizon:+.1f}°, vertical {vertical:+.0f}."
        )

    def _on_geometry_auto_crop(self, checked):
        if self.current_path is None:
            return
        self.recipes[self.current_path].geometry_auto_crop = bool(checked)
        self._schedule_history("geometry auto crop")
        self.render_preview()

    def _on_brush_preset_strength(self, value):
        idx = getattr(self, "selected_brush_index", -1)
        if self.current_path is None or idx < 0:
            return
        masks = self.recipes[self.current_path].brush_masks or []
        if idx < len(masks) and masks[idx].get("local_preset"):
            masks[idx]["preset_strength"] = float(value) / 100.0
            self._schedule_history("Local preset strength")
            self.render_timer.start()

    def _on_brush_adj(self, key, val):
        if self.current_path is None or getattr(self, "selected_brush_index", -1) < 0:
            return
        masks = self.recipes[self.current_path].brush_masks
        if not masks or self.selected_brush_index >= len(masks):
            return
        stored = float(val)
        if key in ("edge_refine", "luminance_min", "luminance_max", "color_tolerance"):
            stored /= 100.0
        masks[self.selected_brush_index][key] = stored
        self.preview.set_brush_masks(masks, self.selected_brush_index)
        self._update_brush_list()
        self.render_timer.start()

    def _on_brush_flag(self, key, value):
        if self.current_path is None or getattr(self, "selected_brush_index", -1) < 0:
            return
        masks = self.recipes[self.current_path].brush_masks or []
        if self.selected_brush_index < len(masks):
            masks[self.selected_brush_index][key] = bool(value)
            self.preview.set_brush_masks(masks, self.selected_brush_index)
            self.render_timer.start()

    def _on_brush_intersect_previous(self):
        idx = getattr(self, "selected_brush_index", -1)
        if self.current_path is None or idx <= 0:
            self.statusBar().showMessage("Select the second or later brush mask to intersect it.")
            return
        masks = self.recipes[self.current_path].brush_masks or []
        current, previous = masks[idx], masks[idx-1]
        previous.setdefault("id", f"mask-{idx-1}")
        refs = list(current.get("intersect_with") or [])
        ref = previous["id"]
        if ref in refs:
            refs.remove(ref)
        else:
            refs.append(ref)
        current["intersect_with"] = refs
        self._update_brush_list()
        self.render_preview()

    def _on_brush_size(self, val):
        # val is 1..30 percent of image
        self.preview.brush_radius = max(0.005, float(val) / 100.0)

    def _on_brush_hard(self, val):
        self.preview.brush_hardness = float(val) / 100.0

    def _on_delete_brush(self):
        if self.current_path is None or getattr(self, "selected_brush_index", -1) < 0:
            return
        masks = list(self.recipes[self.current_path].brush_masks or [])
        if 0 <= self.selected_brush_index < len(masks):
            masks.pop(self.selected_brush_index)
            self.recipes[self.current_path].brush_masks = masks
            self.selected_brush_index = min(self.selected_brush_index, len(masks) - 1)
            self._update_brush_list()
            self._sync_brush_sliders()
            self.render_preview()

    def _on_clear_brushes(self):
        if self.current_path is None:
            return
        self.recipes[self.current_path].brush_masks = []
        self.selected_brush_index = -1
        self._update_brush_list()
        self._sync_brush_sliders()
        self.render_preview()

    def _on_lens_auto(self, checked):
        if self.current_path is None:
            return
        self.recipes[self.current_path].lens_auto = bool(checked)
        self._schedule_history("Lensfun")
        self.render_timer.start()

    def _on_lib_search(self):
        q = self.lib_search.text().strip() if hasattr(self, "lib_search") else ""
        include_rej = bool(self.lib_filter_rejected.isChecked())
        if not q:
            self._lib_load_grid({"type": "all"})
            return
        recs = self.catalog.search(q, include_rejected=include_rej)
        min_rating = int(self.lib_min_rating.currentData() or 0)
        if min_rating > 0:
            recs = [r for r in recs if int(r.get("rating") or 0) >= min_rating]
        recs = [r for r in recs if self._lib_smart_match(r)]
        self.lib_heading.setText(f"Search “{q}” — {len(recs)} photo(s)")
        self._lib_records = recs
        self.lib_grid.clear()
        for rec in recs:
            label = rec.get("filename") or os.path.basename(rec["path"])
            kw = (rec.get("keywords") or "").strip()
            if kw:
                label = f"{label}  [{kw}]"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, rec["path"])
            item.setToolTip(rec["path"])
            item.setSizeHint(QSize(150, 170))
            self.lib_grid.addItem(item)
        if self._lib_thumb_worker and self._lib_thumb_worker.isRunning():
            self._lib_thumb_worker.cancel()
        from workers import CatalogThumbWorker
        self._lib_thumb_worker = CatalogThumbWorker(recs, size=140)
        self._lib_thumb_worker.thumb_ready.connect(self._on_lib_thumb_ready)
        self._lib_thumb_worker.start()

    def _on_lib_save_keywords(self):
        paths = self._lib_selected_paths()
        if not paths:
            self.statusBar().showMessage("Select library photo(s) to tag")
            return
        kw = self.lib_keywords_edit.text().strip()
        for path in paths:
            self.catalog.set_keywords(path, kw)
        self.statusBar().showMessage(f"Keywords saved on {len(paths)} photo(s)")
        self._on_lib_search()

    def toggle_wb_picker(self, checked=False):
        on = checked if isinstance(checked, bool) else (not getattr(self.preview, "wb_picker_mode", False))
        if hasattr(self, "act_wb_pick"):
            # if triggered from menu without check state
            if not isinstance(checked, bool):
                on = not self.preview.wb_picker_mode
            self.act_wb_pick.setChecked(on)
        self.preview.wb_picker_mode = on
        if on:
            self.preview.gradient_mode = False
            self.preview.brush_mode = False
            self.preview.local_mode = False
            self._local_mode = False
            for act_name in ("act_grad", "act_brush", "act_local"):
                a = getattr(self, act_name, None)
                if a is not None:
                    a.setChecked(False)
            if hasattr(self, "_cat_buttons") and len(self._cat_buttons) > 1:
                self._cat_buttons[1].setChecked(True)
                self.tool_stack.setCurrentIndex(1)
            try:
                self.show_develop_mode()
            except Exception:
                pass
            self.preview.setCursor(Qt.CursorShape.CrossCursor)
            self.statusBar().showMessage("Color → White Balance — click a neutral gray/white area")
        else:
            self.preview.setCursor(Qt.CursorShape.ArrowCursor)
            self.statusBar().showMessage("WB Picker off")

    def _on_wb_picked(self, b, g, r):
        if self.current_path is None:
            return
        # Estimate temperature/tint from sample (neutral target)
        avg = (r + g + b) / 3.0 + 1e-6
        # Map R/B ratio toward 5500K-ish
        rb = (r + 1e-6) / (b + 1e-6)
        temp = 5500.0 + (rb - 1.0) * 2000.0
        temp = max(2500.0, min(10000.0, temp))
        tint = ((g / avg) - 1.0) * 100.0
        tint = max(-150.0, min(150.0, tint))
        rcp = self.recipes[self.current_path]
        rcp.wb_as_shot = False
        rcp.temperature = round(temp, 0)
        rcp.tint = round(tint, 0)
        self.sync_sliders_to_recipe()
        self._push_history("WB Picker")
        self.render_preview()
        self.statusBar().showMessage(f"WB from sample → {temp:.0f}K  tint {tint:.0f}")
        self.log(f"WB picker: T={temp:.0f} tint={tint:.0f}")
        # stay in picker mode for more clicks, or turn off:
        # self.toggle_wb_picker(False)

    def toggle_gradient_mode(self, checked=False):
        on = checked if isinstance(checked, bool) else (not self.preview.gradient_mode)
        if hasattr(self, "act_grad"):
            if not isinstance(checked, bool):
                on = not self.preview.gradient_mode
            self.act_grad.setChecked(on)
        self.preview.gradient_mode = on
        if on:
            self.preview.wb_picker_mode = False
            self.preview.brush_mode = False
            self.preview.local_mode = False
            self._local_mode = False
            for act_name in ("act_wb_pick", "act_brush", "act_local"):
                a = getattr(self, act_name, None)
                if a is not None:
                    a.setChecked(False)
            # switch to Local tab if present
            if hasattr(self, "_cat_buttons") and len(self._cat_buttons) > 5:
                self._cat_buttons[5].setChecked(True)
                self.tool_stack.setCurrentIndex(5)
            self.preview.setCursor(Qt.CursorShape.CrossCursor)
            self.statusBar().showMessage("Graduated filter: drag on the image to place")
        else:
            self.preview.setCursor(Qt.CursorShape.ArrowCursor)
            self.statusBar().showMessage("Graduated filter off")

    def _on_gradient_changed(self):
        if self.current_path is None:
            return
        self.recipes[self.current_path].gradients = list(self.preview.gradients)
        self._update_grad_list()
        self.render_timer.start()

    def _on_gradient_selected(self, idx):
        self.selected_grad_index = idx
        if hasattr(self, "grad_list"):
            self.grad_list.blockSignals(True)
            self.grad_list.setCurrentRow(idx)
            self.grad_list.blockSignals(False)
        self._sync_grad_sliders()

    def _update_grad_list(self):
        if not hasattr(self, "grad_list") or self.current_path is None:
            return
        self.grad_list.blockSignals(True)
        self.grad_list.clear()
        grads = self.recipes[self.current_path].gradients or []
        for i, g in enumerate(grads):
            self.grad_list.addItem(f"Gradient {i+1}  (exp {g.get('exposure', 0):+.2f})")
        sel = getattr(self, "selected_grad_index", -1)
        if 0 <= sel < len(grads):
            self.grad_list.setCurrentRow(sel)
        self.grad_list.blockSignals(False)
        self.preview.set_gradients(grads, sel)

    def _on_grad_list_selection(self, row):
        self.selected_grad_index = row
        self.preview.selected_gradient = row
        self.preview.update()
        self._sync_grad_sliders()

    def _sync_grad_sliders(self):
        if not hasattr(self, "grad_sliders_box"):
            return
        ok = (self.current_path is not None and getattr(self, "selected_grad_index", -1) >= 0)
        grads = (self.recipes.get(self.current_path).gradients if self.current_path else None) or []
        if ok and self.selected_grad_index >= len(grads):
            ok = False
        self.grad_sliders_box.setEnabled(ok)
        if not ok:
            return
        g = grads[self.selected_grad_index]
        for key, row in self.grad_sliders.items():
            row.blockSignals(True)
            if key == "feather":
                row.set_value(float(g.get("feather", 0.5)) * 100.0)
            else:
                row.set_value(float(g.get(key, 0.0)))
            row.blockSignals(False)

    def _on_grad_slider(self, key, val):
        if self.current_path is None or getattr(self, "selected_grad_index", -1) < 0:
            return
        grads = self.recipes[self.current_path].gradients
        if not grads or self.selected_grad_index >= len(grads):
            return
        g = grads[self.selected_grad_index]
        if key == "feather":
            g["feather"] = float(val) / 100.0
        else:
            g[key] = float(val)
        self.preview.set_gradients(grads, self.selected_grad_index)
        self._update_grad_list()
        self.render_timer.start()

    def _on_delete_gradient(self):
        if self.current_path is None or getattr(self, "selected_grad_index", -1) < 0:
            return
        grads = list(self.recipes[self.current_path].gradients or [])
        if 0 <= self.selected_grad_index < len(grads):
            grads.pop(self.selected_grad_index)
            self.recipes[self.current_path].gradients = grads
            self.selected_grad_index = min(self.selected_grad_index, len(grads) - 1)
            self._update_grad_list()
            self._sync_grad_sliders()
            self.render_preview()

    def _on_clear_gradients(self):
        if self.current_path is None:
            return
        self.recipes[self.current_path].gradients = []
        self.selected_grad_index = -1
        self._update_grad_list()
        self._sync_grad_sliders()
        self.render_preview()

    def batch_export_selected(self):
        paths = []
        if hasattr(self, "filmstrip"):
            for it in self.filmstrip.selectedItems():
                pth = it.data(Qt.ItemDataRole.UserRole)
                if pth:
                    paths.append(pth)
        if not paths:
            QMessageBox.information(
                self, "Batch Export",
                "Select one or more images in the filmstrip first (Ctrl/Shift+click).",
            )
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Batch export folder")
        if not out_dir:
            return
        from PyQt6.QtWidgets import QInputDialog
        max_dim, ok = QInputDialog.getInt(
            self, "Batch Export", "Max long edge (0 = full resolution):", 0, 0, 20000, 100
        )
        if not ok:
            return
        jobs = []
        for path in paths:
            base = os.path.splitext(os.path.basename(path))[0]
            ext = ".jpg"
            out_path = os.path.join(out_dir, f"{base}_edited{ext}")
            recipe = self.recipes.get(path) or Recipe()
            meta = self.meta_cache.get(path, {})
            jobs.append({
                "path": path,
                "recipe": recipe,
                "out_path": out_path,
                "wb_multipliers": meta.get("wb_multipliers"),
            })
        self.statusBar().showMessage(f"Batch exporting {len(jobs)} images…")
        self.log(f"Batch export: {len(jobs)} files → {out_dir}")
        self._batch_worker = BatchExportWorker(jobs, max_dim=max_dim or 0)
        self._batch_worker.progress.connect(
            lambda i, n, p: self.statusBar().showMessage(f"Export {i+1}/{n}: {os.path.basename(p)}")
        )
        self._batch_worker.finished_ok.connect(
            lambda n: self.statusBar().showMessage(f"Batch export done — {n} file(s)")
        )
        self._batch_worker.failed.connect(
            lambda e: self.statusBar().showMessage(f"Batch export error: {e}")
        )
        self._batch_worker.start()


    def _on_denoise_method(self, _idx=None):
        if self.current_path is None:
            return
        method = self.denoise_method_combo.currentData()
        self.recipes[self.current_path].denoise_method = method or "auto"
        self._schedule_history("Denoise method")
        self.render_timer.start()

    def _apply_detail_preset(self, **kwargs):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        for k, v in kwargs.items():
            setattr(r, k, v)
        self.sync_sliders_to_recipe()
        if hasattr(self, "denoise_method_combo") and "denoise_method" in kwargs:
            for i in range(self.denoise_method_combo.count()):
                if self.denoise_method_combo.itemData(i) == kwargs["denoise_method"]:
                    self.denoise_method_combo.blockSignals(True)
                    self.denoise_method_combo.setCurrentIndex(i)
                    self.denoise_method_combo.blockSignals(False)
                    break
        self._push_history("Detail preset")
        self.render_preview()

    def _detail_preset_light_nr(self):
        self._apply_detail_preset(
            denoise_luminance=25, denoise_chroma=35, denoise_strength=15,
            denoise_detail=60, denoise_method="bilateral",
            sharpen_intensity=40, sharpen_radius=0.8, sharpen_threshold=15,
            sharpen_detail=20, output_sharpen=0,
        )

    def _detail_preset_strong_nr(self):
        self._apply_detail_preset(
            denoise_luminance=55, denoise_chroma=70, denoise_strength=65,
            denoise_detail=45, denoise_method="nlm",
            sharpen_intensity=55, sharpen_radius=1.0, sharpen_threshold=25,
            sharpen_detail=25, output_sharpen=10,
        )

    def _detail_preset_portrait(self):
        self._apply_detail_preset(
            denoise_luminance=40, denoise_chroma=55, denoise_strength=40,
            denoise_detail=55, denoise_method="auto",
            sharpen_intensity=30, sharpen_radius=0.7, sharpen_threshold=30,
            sharpen_detail=15, output_sharpen=5,
        )

    def _detail_preset_landscape(self):
        self._apply_detail_preset(
            denoise_luminance=20, denoise_chroma=40, denoise_strength=20,
            denoise_detail=70, denoise_method="bilateral",
            sharpen_intensity=70, sharpen_radius=1.2, sharpen_threshold=10,
            sharpen_detail=40, output_sharpen=15,
        )

    def export_current(self):
        if self.current_path is None:
            return
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout
        base, ext = os.path.splitext(os.path.basename(self.current_path))
        if is_raw(self.current_path):
            ext = ".jpg"
        suggested = os.path.join(self.folder or ".", f"{base}_edited{ext}")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Export image", suggested,
            "JPEG (*.jpg);;PNG (*.png);;TIFF (*.tif);;All (*.*)"
        )
        if not out_path:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Export options")
        form = QFormLayout(dlg)
        preset_combo = QComboBox()
        preset_combo.addItem("Custom", "custom")
        preset_combo.addItem("Web (2048 px, JPEG 85)", "web")
        preset_combo.addItem("Print (full res, JPEG 95)", "print")
        preset_combo.addItem("Archival (full res, TIFF)", "archival")
        form.addRow("Preset", preset_combo)
        wm_check = QCheckBox("Add text watermark")
        wm_edit = QLineEdit()
        wm_edit.setPlaceholderText("© Your Name")
        wm_edit.setEnabled(False)
        wm_check.toggled.connect(wm_edit.setEnabled)
        form.addRow(wm_check)
        form.addRow("Watermark text", wm_edit)
        size_combo = QComboBox()
        size_combo.addItem("Full resolution", 0)
        size_combo.addItem("Long edge 2048 px", 2048)
        size_combo.addItem("Long edge 1500 px", 1500)
        size_combo.addItem("Long edge 1024 px", 1024)
        form.addRow("Size", size_combo)
        side_check = QCheckBox("Also save recipe sidecar (.photolab.json)")
        side_check.setChecked(True)
        form.addRow(side_check)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        wm = wm_edit.text().strip() if wm_check.isChecked() else ""
        max_dim = int(size_combo.currentData() or 0)
        preset = preset_combo.currentData() or "custom"
        jpeg_q = 92
        if preset == "web":
            max_dim = 2048
            jpeg_q = 85
            if not out_path.lower().endswith((".jpg", ".jpeg")):
                out_path = os.path.splitext(out_path)[0] + ".jpg"
        elif preset == "print":
            max_dim = 0
            jpeg_q = 95
        elif preset == "archival":
            max_dim = 0
            jpeg_q = 100
            if not out_path.lower().endswith((".tif", ".tiff")):
                out_path = os.path.splitext(out_path)[0] + ".tif"
        if side_check.isChecked():
            try:
                save_recipe_sidecar(self.current_path, self.recipes[self.current_path])
            except Exception as e:
                self.log(f"Sidecar save failed: {e}", level="ERR")
        self.statusBar().showMessage("Exporting…")
        meta = self.meta_cache.get(self.current_path, {})
        self.export_worker = ExportWorker(
            self.current_path, self.recipes[self.current_path], out_path,
            wb_multipliers=meta.get("wb_multipliers"),
            watermark_text=wm,
            max_dim=max_dim,
            jpeg_quality=jpeg_q,
        )
        self.export_worker.finished_ok.connect(lambda p: self.statusBar().showMessage(f"Exported → {p}"))
        self.export_worker.failed.connect(lambda e: self.statusBar().showMessage(f"Export failed: {e}"))
        self.export_worker.start()



    def show_preferences(self):
        """Edit ~/.photolab/photolab.ini paths and options."""
        from PyQt6.QtWidgets import (
            QDialog, QFormLayout, QLineEdit, QSpinBox, QCheckBox,
            QDialogButtonBox, QFileDialog, QHBoxLayout, QLabel, QPushButton,
            QTabWidget, QWidget, QVBoxLayout, QMessageBox,
        )
        from config import get_config, reload_config, user_ini_path

        cfg = get_config()
        dlg = QDialog(self)
        dlg.setWindowTitle("Preferences")
        dlg.resize(520, 420)
        root = QVBoxLayout(dlg)
        tabs = QTabWidget()
        root.addWidget(tabs)

        def path_row(parent_form, label, key, is_file=False):
            row = QHBoxLayout()
            edit = QLineEdit(cfg.path(key))
            edit.setPlaceholderText("(auto)")
            btn = QPushButton("…")
            btn.setFixedWidth(32)

            def browse():
                if is_file:
                    p, _ = QFileDialog.getOpenFileName(dlg, label, edit.text() or "")
                else:
                    p = QFileDialog.getExistingDirectory(dlg, label, edit.text() or "")
                if p:
                    edit.setText(p)

            btn.clicked.connect(browse)
            row.addWidget(edit, 1)
            row.addWidget(btn)
            parent_form.addRow(label, row)
            return edit

        # Paths tab
        paths_w = QWidget()
        pf = QFormLayout(paths_w)
        e_plugin = path_row(pf, "Plugin / presets folder", "plugin_dir")
        e_docs = path_row(pf, "Docs folder", "docs_dir")
        e_lensfun = path_row(pf, "Lensfun folder", "lensfun_dir")
        e_ffmpeg = path_row(pf, "ffmpeg executable", "ffmpeg", is_file=True)
        e_catalog = path_row(pf, "Catalog database", "catalog_db", is_file=True)
        e_thumbs = path_row(pf, "Thumb cache folder", "thumb_cache")
        e_export = path_row(pf, "Default export folder", "export_default_dir")
        e_scripts = path_row(pf, "Scripts folder", "scripts_dir")
        tabs.addTab(paths_w, "Paths")

        # Performance
        perf_w = QWidget()
        pr = QFormLayout(perf_w)
        sp_workers = QSpinBox()
        sp_workers.setRange(1, 16)
        sp_workers.setValue(cfg.get_int("performance", "max_raw_workers", 2))
        pr.addRow("Max concurrent RAW workers", sp_workers)
        sp_proxy = QSpinBox()
        sp_proxy.setRange(400, 8000)
        sp_proxy.setValue(cfg.get_int("performance", "proxy_max_dimension", 1600))
        pr.addRow("Proxy max dimension (px)", sp_proxy)
        cb_16 = QCheckBox("Prefer 16-bit pipeline where supported")
        cb_16.setChecked(cfg.get_bool("performance", "use_16bit_pipeline", False))
        pr.addRow(cb_16)
        tabs.addTab(perf_w, "Performance")

        # UI
        ui_w = QWidget()
        ur = QFormLayout(ui_w)
        cb_remember = QCheckBox("Remember last folder")
        cb_remember.setChecked(cfg.get_bool("ui", "remember_last_folder", True))
        ur.addRow(cb_remember)
        e_last = QLineEdit(cfg.get("ui", "last_folder", ""))
        ur.addRow("Last folder", e_last)
        e_updates = QLineEdit(cfg.get("ui", "check_for_updates_url", ""))
        ur.addRow("Updates URL", e_updates)
        tabs.addTab(ui_w, "UI")

        # Licensing / integrations (masked-ish)
        sec_w = QWidget()
        sf = QFormLayout(sec_w)
        e_serial = QLineEdit(cfg.get("licensing", "serial", ""))
        e_serial.setEchoMode(QLineEdit.EchoMode.Password)
        e_email = QLineEdit(cfg.get("licensing", "customer_email", ""))
        e_api = QLineEdit(cfg.get("integrations", "api_key", ""))
        e_api.setEchoMode(QLineEdit.EchoMode.Password)
        e_endpoint = QLineEdit(cfg.get("integrations", "api_endpoint", ""))
        sf.addRow("Serial", e_serial)
        sf.addRow("Customer email", e_email)
        sf.addRow("API key", e_api)
        sf.addRow("API endpoint", e_endpoint)
        hint = QLabel(
            f"Saved to:\n{user_ini_path()}\n\n"
            "Env overrides: PHOTOLAB_API_KEY, PHOTOLAB_SERIAL, PHOTOLAB_PLUGIN_DIR, …"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#888; font-size:11px;")
        sf.addRow(hint)
        tabs.addTab(sec_w, "License / API")

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        root.addWidget(buttons)

        def on_save():
            cfg.set("paths", "plugin_dir", e_plugin.text().strip())
            cfg.set("paths", "docs_dir", e_docs.text().strip())
            cfg.set("paths", "lensfun_dir", e_lensfun.text().strip())
            cfg.set("paths", "ffmpeg", e_ffmpeg.text().strip())
            cfg.set("paths", "catalog_db", e_catalog.text().strip())
            cfg.set("paths", "thumb_cache", e_thumbs.text().strip())
            cfg.set("paths", "export_default_dir", e_export.text().strip())
            cfg.set("paths", "scripts_dir", e_scripts.text().strip())
            cfg.set("performance", "max_raw_workers", sp_workers.value())
            cfg.set("performance", "proxy_max_dimension", sp_proxy.value())
            cfg.set("performance", "use_16bit_pipeline", "true" if cb_16.isChecked() else "false")
            cfg.set("ui", "remember_last_folder", "true" if cb_remember.isChecked() else "false")
            cfg.set("ui", "last_folder", e_last.text().strip())
            cfg.set("ui", "check_for_updates_url", e_updates.text().strip())
            cfg.set("licensing", "serial", e_serial.text().strip())
            cfg.set("licensing", "customer_email", e_email.text().strip())
            cfg.set("integrations", "api_key", e_api.text().strip())
            cfg.set("integrations", "api_endpoint", e_endpoint.text().strip())
            path = cfg.save_user()
            reload_config()
            self.log(f"Preferences saved → {path}")
            self.statusBar().showMessage(f"Preferences saved → {path}")
            QMessageBox.information(
                dlg, "Preferences",
                f"Saved:\n{path}\n\nSome path changes apply on next catalog open or restart.",
            )
            dlg.accept()

        buttons.accepted.connect(on_save)
        buttons.rejected.connect(dlg.reject)
        dlg.exec()

    def _preset_start_dir(self) -> str:
        """Default folder for Load/Save Preset dialogs (plugin/)."""
        try:
            from app_paths import ensure_plugin_dir
            return ensure_plugin_dir()
        except Exception:
            d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plugin")
            try:
                os.makedirs(d, exist_ok=True)
            except Exception:
                pass
            return d if os.path.isdir(d) else ""

    def save_preset(self):
        if self.current_path is None:
            return
        start = os.path.join(self._preset_start_dir(), "preset.json")
        path, _ = QFileDialog.getSaveFileName(self, "Save Preset", start, "JSON (*.json)")
        if path:
            try:
                self.recipes[self.current_path].save_json(path)
                self.statusBar().showMessage(f"Preset saved → {path}")
            except Exception as e:
                QMessageBox.warning(self, "Save Preset", str(e))

    def load_preset(self):
        """Open the Preset Browser (plugin folder). File → Load Preset… and Apply preset use this."""
        if self.current_path is None:
            QMessageBox.information(self, "Load Preset", "Open an image first.")
            return
        self._preset_preview_base = Recipe.from_dict(self.recipes[self.current_path].to_dict())
        dlg = PresetBrowserDialog(self, self._preset_start_dir())
        accepted = dlg.exec() == QDialog.DialogCode.Accepted
        self.recipes[self.current_path] = Recipe.from_dict(self._preset_preview_base.to_dict())
        if not accepted:
            self.sync_sliders_to_recipe()
            self.render_preview()
            self._preset_preview_base = None
            return
        path = dlg.selected_path
        if not path:
            self._preset_preview_base = None
            return
        self._apply_preset_path(path, dlg.strength, dlg.selected_modules())
        self._preset_preview_base = None

    def _preview_preset(self, path: str, strength: float, modules):
        if self.current_path is None or not path or not getattr(self, "_preset_preview_base", None):
            return
        try:
            self.recipes[self.current_path] = apply_preset_file(
                path, base=self._preset_preview_base, strength=strength, modules=modules
            )
            self.sync_sliders_to_recipe()
            self.render_preview()
        except Exception as exc:
            self.statusBar().showMessage(f"Preset preview unavailable: {exc}")

    def _apply_preset_path(self, path: str, strength: float = 1.0, modules=None):
        if self.current_path is None or not path:
            return
        try:
            r = apply_preset_file(
                path, base=self.recipes.get(self.current_path), strength=strength, modules=modules
            )
            self.recipes[self.current_path] = r
            self.sync_sliders_to_recipe()
            self._push_history(f"Preset: {os.path.basename(path)}")
            self.render_preview()
            self.statusBar().showMessage(
                f"Preset loaded ← {os.path.basename(path)} ({round(strength * 100)}%)"
            )
        except Exception as e:
            QMessageBox.warning(self, "Load Preset", f"Could not load preset:\n{e}")

    def load_preset_file_dialog(self):
        """Classic file picker (also available from browser Folder…)."""
        if self.current_path is None:
            QMessageBox.information(self, "Load Preset", "Open an image first.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Preset File",
            self._preset_start_dir(),
            "All Presets (*.xmp *.json);;Lightroom XMP (*.xmp);;PhotoLab JSON (*.json);;All (*.*)",
        )
        if path:
            self._apply_preset_path(path)

    def load_preset_folder(self):
        """Import all .xmp/.json presets from a folder and apply the first one (list in status)."""
        if self.current_path is None:
            QMessageBox.information(self, "Import Presets", "Open an image first.")
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Choose preset folder (XMP/JSON)", self._preset_start_dir()
        )
        if not folder:
            return
        files = list_preset_files(folder)
        if not files:
            QMessageBox.information(self, "Import Presets", "No .xmp or .json presets found in that folder.")
            return
        # Apply first; report count
        try:
            r = load_preset_file(files[0], base=self.recipes.get(self.current_path))
            self.recipes[self.current_path] = r
            self.sync_sliders_to_recipe()
            self._push_history(f"Preset: {os.path.basename(files[0])}")
            self.render_preview()
            names = ", ".join(os.path.basename(f) for f in files[:5])
            more = f" (+{len(files)-5} more)" if len(files) > 5 else ""
            self.statusBar().showMessage(f"Found {len(files)} presets. Applied: {names}{more}")
            QMessageBox.information(
                self,
                "Presets Found",
                f"Found {len(files)} preset(s) in:\n{folder}\n\nApplied:\n{os.path.basename(files[0])}\n\n"
                "Use File → Load Preset… to apply others individually.",
            )
        except Exception as e:
            QMessageBox.warning(self, "Import Presets", str(e))


    # ----- Control Point Canvas & UI Interactions -----
    def _on_canvas_point_selected(self, idx):
        self.selected_local_index = idx
        self.local_points_list.blockSignals(True)
        self.local_points_list.setCurrentRow(idx)
        self.local_points_list.blockSignals(False)
        self._sync_local_sliders()

    def _on_canvas_point_moved(self, idx, nx, ny):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        if r.local_points and 0 <= idx < len(r.local_points):
            r.local_points[idx]["x"] = nx
            r.local_points[idx]["y"] = ny
            
            # Update coordinate text in the list widget
            self.local_points_list.blockSignals(True)
            item = self.local_points_list.item(idx)
            if item:
                item.setText(f"Control Point {idx+1} ({nx:.2f}, {ny:.2f})")
            self.local_points_list.blockSignals(False)
            
            self.render_timer.start()

    def _on_canvas_point_resized(self, idx, radius):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        if r.local_points and 0 <= idx < len(r.local_points):
            r.local_points[idx]["radius"] = radius
            if "local_radius" in self.local_sliders:
                self.local_sliders["local_radius"].blockSignals(True)
                self.local_sliders["local_radius"].set_value(radius * 100.0)
                self.local_sliders["local_radius"].blockSignals(False)
            self.render_timer.start()

    def _on_canvas_point_added(self, nx, ny):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        pt = {
            "x": nx, "y": ny, "radius": 0.15, "feather": 0.5,
            "exposure": 0.0, "contrast": 0.0, "saturation": 0.0, "clarity": 0.0,
        }
        if r.local_points is None:
            r.local_points = []
        r.local_points = list(r.local_points) + [pt]
        self.selected_local_index = len(r.local_points) - 1
        self._push_history("Add Control Point")
        self._update_local_points_list()
        self._sync_local_sliders()
        self.preview.set_control_points(r.local_points, self.selected_local_index)
        self.render_preview()

    def _on_canvas_point_drag_finished(self):
        self._push_history("Modify Control Point")

    def _on_local_active_toggled(self, checked):
        self._local_mode = checked
        self.act_local.setChecked(checked)
        self.preview.local_mode = checked
        if checked:
            self.statusBar().showMessage("Control Point mode: click on image to place a local adjustment")
            self.preview.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.preview.setCursor(Qt.CursorShape.ArrowCursor)
            self.statusBar().showMessage("Control Point mode off")

    def _on_local_list_selection(self, row):
        self.selected_local_index = row
        self.preview.selected_point_index = row
        self.preview.update()
        self._sync_local_sliders()

    def _on_add_local_clicked(self):
        self.local_active_cb.setChecked(True)
        self.statusBar().showMessage("Click on the preview image to place the control point.")

    def _on_delete_local_clicked(self):
        if self.current_path is None or self.selected_local_index < 0:
            return
        r = self.recipes[self.current_path]
        if r.local_points and 0 <= self.selected_local_index < len(r.local_points):
            r.local_points.pop(self.selected_local_index)
            self.selected_local_index = min(self.selected_local_index, len(r.local_points) - 1)
            self._push_history("Delete Control Point")
            self._update_local_points_list()
            self._sync_local_sliders()
            self.preview.set_control_points(r.local_points, self.selected_local_index)
            self.render_preview()

    def _on_clear_local_clicked(self):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        if r.local_points:
            r.local_points = []
            self.selected_local_index = -1
            self._push_history("Clear Control Points")
            self._update_local_points_list()
            self._sync_local_sliders()
            self.preview.set_control_points([], -1)
            self.render_preview()

    def _on_local_slider_changed(self, name, val):
        if self.current_path is None or self.selected_local_index < 0:
            return
        r = self.recipes[self.current_path]
        if not r.local_points or not (0 <= self.selected_local_index < len(r.local_points)):
            return
        pt = r.local_points[self.selected_local_index]
        if name == "local_radius":
            pt["radius"] = val / 100.0
        elif name == "local_feather":
            pt["feather"] = val / 100.0
        elif name == "local_exposure":
            pt["exposure"] = val
        elif name == "local_contrast":
            pt["contrast"] = val
        elif name == "local_saturation":
            pt["saturation"] = val
        elif name == "local_clarity":
            pt["clarity"] = val
        
        self._schedule_history("Modify Control Point")
        self.preview.set_control_points(r.local_points, self.selected_local_index)
        self.render_timer.start()

    def _update_local_points_list(self):
        self.local_points_list.blockSignals(True)
        self.local_points_list.clear()
        if self.current_path is not None:
            r = self.recipes[self.current_path]
            if r.local_points:
                for i, pt in enumerate(r.local_points):
                    self.local_points_list.addItem(f"Control Point {i+1} ({pt['x']:.2f}, {pt['y']:.2f})")
                if 0 <= self.selected_local_index < len(r.local_points):
                    self.local_points_list.setCurrentRow(self.selected_local_index)
        self.local_points_list.blockSignals(False)

    def _sync_local_sliders(self):
        has_sel = (self.selected_local_index >= 0 and self.current_path is not None)
        if has_sel:
            r = self.recipes[self.current_path]
            has_sel = r.local_points and (0 <= self.selected_local_index < len(r.local_points))
        
        self.local_sliders_box.setEnabled(has_sel)
        if has_sel:
            pt = self.recipes[self.current_path].local_points[self.selected_local_index]
            for name, row in self.local_sliders.items():
                row.blockSignals(True)
                
            self.local_sliders["local_radius"].set_value(pt.get("radius", 0.15) * 100.0)
            self.local_sliders["local_feather"].set_value(pt.get("feather", 0.5) * 100.0)
            self.local_sliders["local_exposure"].set_value(pt.get("exposure", 0.0))
            self.local_sliders["local_contrast"].set_value(pt.get("contrast", 0.0))
            self.local_sliders["local_saturation"].set_value(pt.get("saturation", 0.0))
            self.local_sliders["local_clarity"].set_value(pt.get("clarity", 0.0))
            
            for name, row in self.local_sliders.items():
                row.blockSignals(False)

    # ----- Folder Tree & Metadata Display -----
    def _on_folder_tree_clicked(self, index):
        path = self.folder_model.filePath(index)
        if os.path.isdir(path):
            self.open_folder_path(path)

    def _update_metadata_display(self, meta: dict):
        self._set_gps_status((meta or {}).get("gps"))
        if not meta:
            self.metadata_label.setText("No metadata available")
            return
        
        lines = []
        camera = meta.get("camera")
        lens = meta.get("lens")
        shutter = meta.get("shutter")
        aperture = meta.get("aperture")
        iso = meta.get("iso")
        focal = meta.get("focal")
        dt = meta.get("datetime")
        
        if camera:
            lines.append(f"📷 <b>Camera:</b> {camera}")
        if lens:
            lines.append(f"🔍 <b>Lens:</b> {lens}")
            
        settings = []
        if shutter: settings.append(shutter)
        if aperture: settings.append(aperture)
        if iso: settings.append(iso)
        if focal: settings.append(focal)
        
        if settings:
            lines.append(f"⚙️ <b>Settings:</b> {' • '.join(settings)}")
        if dt:
            lines.append(f"📅 <b>Taken:</b> {dt}")
            
        if not lines:
            self.metadata_label.setText("No metadata available")
        else:
            self.metadata_label.setText("<br>".join(lines))

    def _set_gps_status(self, gps):
        """Update the metadata panel's red/green GPS availability indicator."""
        available = bool(gps and len(gps) >= 2)
        if available:
            lat, lon = float(gps[0]), float(gps[1])
            self.gps_status_label.setText(
                f'<span style="color:#43c96b; font-size:16px;">●</span> '
                f'<b>GPS available</b><br><span style="color:#999;">{lat:.5f}, {lon:.5f}</span>'
            )
            self.gps_status_label.setToolTip("This photo contains valid GPS coordinates")
        else:
            self.gps_status_label.setText(
                '<span style="color:#e05252; font-size:16px;">●</span> <b>GPS not available</b>'
            )
            self.gps_status_label.setToolTip("No GPS coordinates were found in this photo")
        self.gps_map_btn.setEnabled(available)

    def closeEvent(self, event):
        renderer = getattr(self, "_preview_renderer", None)
        if renderer is not None:
            renderer.stop()
            renderer.wait(2000)
        try:
            self.catalog.close()
        except Exception:
            pass
        super().closeEvent(event)


# Backward-compatible alias
PhotoStudio = PhotoLab
