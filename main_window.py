"""main_window.py — PhotoLab main window, DxO-inspired layout.

Right panel category tabs (icons): Light | Color | Detail | Geometry | Effects
Collapsible correction groups matching DxO PhotoLab structure.
"""

from __future__ import annotations

import os
import cv2
import numpy as np

from PyQt6.QtCore import Qt, QTimer, QSize, QRect, QDir
from PyQt6.QtGui import QIcon, QAction, QKeySequence, QFont, QFileSystemModel
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QScrollArea, QListWidget, QListWidgetItem, QFileDialog, QToolBar,
    QGroupBox, QPushButton, QSplitter, QStatusBar, QComboBox, QTabWidget,
    QCheckBox, QFrame, QLineEdit, QToolButton, QSizePolicy, QMessageBox, QStackedWidget, QButtonGroup, QMenu,
    QTreeView, QTextEdit, QTextBrowser, QDockWidget, QPlainTextEdit, QApplication,
    QDialog, QDialogButtonBox, QFormLayout, QInputDialog, QProgressDialog,QDoubleSpinBox,
)

from imaging import (Recipe, apply_recipe, IMAGE_EXTS, load_image, is_raw,
                     load_recipe_sidecar, save_recipe_sidecar, apply_watermark,
                     recipe_to_dict, load_snapshots_sidecar, save_snapshots_sidecar)

import logging

log = logging.getLogger(__name__)
from presets import load_preset_file, list_preset_files
from qt_utils import cv_to_qpixmap
from workers import ThumbnailWorker, ExportWorker, LoadImageWorker, CatalogScanWorker, CatalogThumbWorker, HdrMergeWorker, BatchExportWorker, FocusStackWorker, PanoramaWorker
from widgets import HistogramWidget, SliderRow, ImageCanvas, ToneCurveWidget, HistoryWidget, NavigatorWidget, HSLPanelWidget
from catalog import Catalog
from app_paths import plugin_dir, ensure_plugin_dir, list_bundled_presets, manual_file, docs_dir
import sys
import hashlib
import json
from datetime import datetime

# Interactive preview works on a persistent proxy so slider drags stay responsive.
PROXY_MAX_DIM = 1800
PREVIEW_CACHE_MAX = 8


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


class PhotoLab(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PhotoLab")
        # Ensure widgets inherit a valid point size (Windows can report -1)
        try:
            from PyQt6.QtGui import QFont as _QF
            _f = self.font()
            if _f.pointSize() <= 0:
                _f.setPointSize(10)
                self.setFont(_f)
        except Exception:
            log.debug("__init__: non-critical failure, continuing", exc_info=True)
        self.resize(1600, 1000)
        self.setStyleSheet(self._stylesheet())

        self.folder = None
        self.image_paths: list[str] = []
        self.recipes: dict[str, Recipe] = {}
        self.meta_cache: dict[str, dict] = {}
        self.current_path: str | None = None
        self.original_bgr: np.ndarray | None = None
        self.proxy_bgr: np.ndarray | None = None
        self._proxy_scale: float = 1.0
        self._preview_cache: dict[str, tuple] = {}
        self._preview_cache_order: list[str] = []

        self.render_timer = QTimer()
        self.render_timer.setSingleShot(True)
        self.render_timer.setInterval(45)
        self.render_timer.timeout.connect(self.render_preview)

        self._load_worker = None
        self.sliders: dict[str, SliderRow] = {}
        self._history_push_pending = False
        self._local_mode = False
        self._copied_recipe = None
        self.autosave_sidecars = False
        self._recent_folders = []
        self._snapshots = {}  # path -> list[{name, recipe_dict, ts?}]
        self._load_recent_folders()
        self._image_ratings = {}  # path -> 0..5 for develop filmstrip
        self._reject_flags = {}   # path -> bool
        self._pick_flags = {}     # path -> bool
        # Color labels: None | red | yellow | green | blue | purple
        self._color_labels = {}
        self.COLOR_LABELS = {
            "red": ("🔴", "#e11d48"),
            "yellow": ("🟡", "#eab308"),
            "green": ("🟢", "#22c55e"),
            "blue": ("🔵", "#3b82f6"),
            "purple": ("🟣", "#a855f7"),
        }
        self.catalog = Catalog()
        self._library_mode = False
        self._scan_worker = None
        self._lib_thumb_worker = None
        self._lib_records = []
        ensure_plugin_dir()

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
        self.recent_menu = file_m.addMenu("Recent Folders")
        self._rebuild_recent_menu()
        file_m.addSeparator()
        add_action(file_m, "Import Photos…", self.import_photos_dialog, "Ctrl+Shift+I")
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
        add_action(file_m, "Load Preset… (XMP / JSON)", self.load_preset)
        add_action(file_m, "Import Preset Folder…", self.load_preset_folder)
        add_action(file_m, "Save Preset… (JSON)", self.save_preset)
        file_m.addSeparator()
        add_action(file_m, "Print", enabled=False, shortcut="Ctrl+P")
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
        self.act_peaking = add_action(view_m, "Focus Peaking", self.toggle_peaking, "P", checkable=True)
        add_action(view_m, "Actual Size (1:1)", lambda: self.preview.zoom_1_to_1(), "1")
        view_m.addSeparator()
        add_action(view_m, "Compare Off", lambda: self.set_compare_mode(ImageCanvas.MODE_NORMAL))
        add_action(view_m, "Split Compare", lambda: self.set_compare_mode(ImageCanvas.MODE_SPLIT), "C")
        add_action(view_m, "Side-by-Side Compare", lambda: self.set_compare_mode(ImageCanvas.MODE_SIDE_BY_SIDE), "B")
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
        add_action(view_m, "Culling Mode (full screen)", self.start_culling_mode, "F7")
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
        color_m = image_m.addMenu("Color Label")
        add_action(color_m, "None (clear)", lambda: self.set_color_label(None), "Ctrl+Shift+0")
        add_action(color_m, "Red", lambda: self.set_color_label("red"), "Ctrl+Shift+1")
        add_action(color_m, "Yellow", lambda: self.set_color_label("yellow"), "Ctrl+Shift+2")
        add_action(color_m, "Green", lambda: self.set_color_label("green"), "Ctrl+Shift+3")
        add_action(color_m, "Blue", lambda: self.set_color_label("blue"), "Ctrl+Shift+4")
        add_action(color_m, "Purple", lambda: self.set_color_label("purple"), "Ctrl+Shift+5")
        image_m.addSeparator()
        add_action(image_m, "Compare Selected…", self.compare_selected_images, "Ctrl+Shift+B")
        image_m.addSeparator()
        add_action(image_m, "Local Adjustments (Control Point)", self._toggle_local_mode)
        image_m.addSeparator()
        add_action(image_m, "Auto Exposure", self.auto_exposure)
        add_action(image_m, "Auto White Balance", self.auto_wb)
        add_action(image_m, "Match Exposure to Current", self.match_exposure_selected)
        add_action(image_m, "Match White Balance to Current", self.match_wb_selected)
        add_action(image_m, "White Balance Picker", self.toggle_wb_picker, "W")
        image_m.addSeparator()
        add_action(image_m, "Graduated Filter", self.toggle_gradient_mode, "G")
        add_action(image_m, "Adjustment Brush", self.toggle_brush_mode, "Shift+B")
        image_m.addSeparator()
        add_action(image_m, "Merge HDR…", self.merge_hdr_selected, "Ctrl+Shift+H")
        add_action(image_m, "Focus Stack…", self.focus_stack_selected, "Ctrl+Shift+F")
        add_action(image_m, "Panorama…", self.panorama_selected, "Ctrl+Shift+P")
        add_action(image_m, "Create Pan Video…", self.create_pan_video)
        add_action(image_m, "Audio Editor…", self.open_audio_editor)

        # ----- Help -----
        help_m = mb.addMenu("&Help")
        add_action(help_m, "User Manual", self._show_user_manual, "F1")
        add_action(help_m, "Developer Manual", self._show_developer_manual)
        add_action(help_m, "Keyboard Shortcuts", self._show_shortcuts)
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
            "C\tSplit compare (before/after)\n"
            "\\ or `\tHold for temporary before view\n"
            "B\tSide-by-side before/after\n"
            "Ctrl+Shift+B\tCompare selected filmstrip images\n"
            "F\tFit to window\n"
            "1\tActual size 1:1\n"
            "Left/Right\tPrev / Next (skips filtered)\n"
            "0–5\tRate selected / current\n"
            "X / U\tReject / Pick selected\n"
            "Ctrl+Shift+0–5\tColor label (clear / R Y G B Purple)\n"
            "Wheel\tZoom\n"
            "Space+drag\tPan",
        )


    def _manual_path(self, name: str) -> str:
        """Resolve docs/*.md for source runs and frozen executables."""
        found = manual_file(name)
        if found:
            return found
        return os.path.join(docs_dir(), name)

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
                "Help always loads from the docs/ folder next to the app "
                "(or inside the frozen bundle). Place USER_MANUAL.md and "
                "DEVELOPER_MANUAL.md there."
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
            log.debug("_build_toolbar: non-critical failure, continuing", exc_info=True)
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
        act("1:1", lambda: self.preview.zoom_1_to_1(), "1", tip="Actual size (1)")
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
        self.act_wb_pick = act("WB", self.toggle_wb_picker, "W", checkable=True, tip="White balance picker (W)")
        tb.addSeparator()
        act("Reset", self.reset_current, "Ctrl+R", tip="Reset image (Ctrl+R)")
        act("Preset", self.load_preset, tip="Load preset (XMP / JSON)")
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
        tools_menu.addAction("Merge HDR…", self.merge_hdr_selected)
        tools_menu.addAction("Focus Stack…", self.focus_stack_selected)
        tools_menu.addAction("Panorama…", self.panorama_selected)
        tools_menu.addSeparator()
        tools_menu.addAction("Pan Video…", self.create_pan_video)
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
            log.debug("log: non-critical failure, continuing", exc_info=True)

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
        self.history_widget.previewRequested.connect(self._on_history_preview)
        self.history_widget.copySettingsRequested.connect(self._on_history_copy_settings)
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
        ll.addWidget(meta_box)

        left.setMinimumWidth(220)
        left.setMaximumWidth(300)
        splitter.addWidget(left)

        # CENTER: preview
        self.preview = ImageCanvas()
        self.preview.crop_dragged.connect(self.on_crop_dragged)
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
        self.preview.keystoneChanged.connect(self._on_keystone_changed)
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
        fr.addSpacing(8)
        fr.addWidget(QLabel("Color"))
        self.film_color_filter = QComboBox()
        self.film_color_filter.addItem("All", "")
        self.film_color_filter.addItem("None", "none")
        for key, (emoji, _hex) in self.COLOR_LABELS.items():
            self.film_color_filter.addItem(f"{emoji} {key.title()}", key)
        self.film_color_filter.currentIndexChanged.connect(self._apply_filmstrip_filter)
        fr.addWidget(self.film_color_filter)
        fr.addSpacing(8)
        # Quick multi-select actions on the filmstrip bar
        btn_rate = QPushButton("★ Rate")
        btn_rate.setToolTip("Apply rating to selected (or current) — use keys 0–5")
        btn_rate.clicked.connect(lambda: self._prompt_rate_selected())
        fr.addWidget(btn_rate)
        btn_pick = QPushButton("✓ Pick")
        btn_pick.setToolTip("Toggle pick on selected filmstrip images (U)")
        btn_pick.clicked.connect(self.toggle_pick_current)
        fr.addWidget(btn_pick)
        btn_rej = QPushButton("⛔ Reject")
        btn_rej.setToolTip("Toggle reject on selected filmstrip images (X)")
        btn_rej.clicked.connect(self.toggle_reject_current)
        fr.addWidget(btn_rej)
        btn_cmp = QPushButton("Compare")
        btn_cmp.setToolTip("Side-by-side compare of selected images (Ctrl+Shift+B)")
        btn_cmp.clicked.connect(self.compare_selected_images)
        fr.addWidget(btn_cmp)
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
        people_row = QHBoxLayout()
        self.lib_people_edit = QLineEdit()
        self.lib_people_edit.setPlaceholderText("People / faces (comma-separated)")
        people_row.addWidget(self.lib_people_edit, 1)
        people_save = QPushButton("Save people")
        people_save.clicked.connect(self._on_lib_save_people)
        people_row.addWidget(people_save)
        ll.addLayout(people_row)

        coll_box = QLabel("Collections")
        coll_box.setStyleSheet("color:#aaa; font-size:11px; margin-top:6px;")
        ll.addWidget(coll_box)
        self.lib_collections = QComboBox()
        self.lib_collections.setToolTip("Select a collection to view its members")
        self.lib_collections.currentIndexChanged.connect(self._on_lib_collection_changed)
        ll.addWidget(self.lib_collections)
        coll_row = QHBoxLayout()
        new_coll = QPushButton("New")
        new_coll.clicked.connect(self._lib_new_collection)
        coll_row.addWidget(new_coll)
        add_coll = QPushButton("Add selected")
        add_coll.setToolTip("Add selected library thumbs to the current collection")
        add_coll.clicked.connect(self._lib_add_to_collection)
        coll_row.addWidget(add_coll)
        del_coll = QPushButton("Delete coll.")
        del_coll.clicked.connect(self._lib_delete_collection)
        coll_row.addWidget(del_coll)
        ll.addLayout(coll_row)

        tools_row = QHBoxLayout()
        dups_btn = QPushButton("Find duplicates")
        dups_btn.setToolTip("Group images that share the same content fingerprint")
        dups_btn.clicked.connect(self._lib_find_duplicates)
        tools_row.addWidget(dups_btn)
        vc_btn = QPushButton("Virtual copy")
        vc_btn.setToolTip("Create a virtual copy of the current Develop image (sidecar recipe clone)")
        vc_btn.clicked.connect(self._create_virtual_copy)
        tools_row.addWidget(vc_btn)
        ll.addLayout(tools_row)

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

        # Tone Curve (parametric + point L/RGB)
        box, v = collapsible_group("Tone Curve", layout)
        ch_row = QHBoxLayout()
        self.curve_channel_group = []
        for label, key in (
            ("Param", "param"),
            ("Luma", "luma"),
            ("R", "r"),
            ("G", "g"),
            ("B", "b"),
        ):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(key == "param")
            btn.setMaximumWidth(48)
            btn.clicked.connect(lambda checked=False, k=key: self._on_curve_channel(k))
            ch_row.addWidget(btn)
            self.curve_channel_group.append((btn, key))
        reset_curve_btn = QPushButton("Reset")
        reset_curve_btn.setMaximumWidth(52)
        reset_curve_btn.clicked.connect(lambda: self.tone_curve.reset_current())
        ch_row.addWidget(reset_curve_btn)
        v.addLayout(ch_row)
        self.tone_curve = ToneCurveWidget()
        self.tone_curve.curveChanged.connect(self.on_curve_changed)
        self.tone_curve.pointCurveChanged.connect(self.on_point_curve_changed)
        v.addWidget(self.tone_curve)
        tip_c = QLabel("Parametric: 5 region handles. Point curves: double-click to add/remove points.")
        tip_c.setWordWrap(True)
        tip_c.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(tip_c)
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
        self.wb_dual_cb = QCheckBox("Dual illuminant (blend two WBs)")
        self.wb_dual_cb.setToolTip(
            "Mix a second temperature/tint — useful for mixed lighting "
            "(e.g. tungsten + window daylight)."
        )
        self.wb_dual_cb.toggled.connect(self._on_wb_dual)
        v.addWidget(self.wb_dual_cb)
        self._add_slider(v, "temperature2", "Temp 2 (K)", 2000, 12000, 50, 0, 6500)
        self._add_slider(v, "tint2", "Tint 2", -150, 150, 1, 0, 0)
        self._add_slider(v, "wb_mix", "Mix toward Temp 2", 0.0, 100.0, 1, 0, 0.0)
        dual_tip = QLabel("0% = primary only · 100% = secondary only. Disable As Shot when blending.")
        dual_tip.setWordWrap(True)
        dual_tip.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(dual_tip)

        # Color Accentuation
        box, v = collapsible_group("Color Accentuation", layout)
        self._add_slider(v, "vibrance", "Vibrancy", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "saturation", "Saturation", -100.0, 100.0, 1, 0, 0.0)

        # HSL — Hue / Saturation / Luminance / All tabs, 8 color bands each
        box, v = collapsible_group("HSL", layout)
        self.hsl_panel = HSLPanelWidget()
        self.hsl_panel.hueChanged.connect(lambda idx, val: self._on_hsl_row("hue", idx, val))
        self.hsl_panel.satChanged.connect(lambda idx, val: self._on_hsl_row("sat", idx, val))
        self.hsl_panel.lumChanged.connect(lambda idx, val: self._on_hsl_row("lum", idx, val))
        v.addWidget(self.hsl_panel)

        # Split Toning
        box, v = collapsible_group("Split Toning", layout, checked=False)
        self._add_slider(v, "split_shadow_hue", "Shadows Hue", 0.0, 360.0, 1, 0, 0.0)
        self._add_slider(v, "split_shadow_sat", "Shadows Sat", 0.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "split_highlight_hue", "Highlights Hue", 0.0, 360.0, 1, 0, 0.0)
        self._add_slider(v, "split_highlight_sat", "Highlights Sat", 0.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "split_balance", "Balance", -100.0, 100.0, 1, 0, 0.0)
        tip_st = QLabel("Tint shadows and highlights independently. Balance shifts the crossover.")
        tip_st.setWordWrap(True)
        tip_st.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(tip_st)

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
        intent_row = QHBoxLayout()
        intent_row.addWidget(QLabel("Intent"))
        self.proof_intent_combo = QComboBox()
        self.proof_intent_combo.addItem("Relative", "relative")
        self.proof_intent_combo.addItem("Perceptual", "perceptual")
        self.proof_intent_combo.addItem("Saturation", "saturation")
        self.proof_intent_combo.addItem("Absolute", "absolute")
        self.proof_intent_combo.currentIndexChanged.connect(self._on_proof_intent)
        intent_row.addWidget(self.proof_intent_combo, 1)
        v.addLayout(intent_row)
        icc_row = QHBoxLayout()
        self.proof_icc_label = QLabel("ICC: (built-in / system)")
        self.proof_icc_label.setStyleSheet("color:#888; font-size:11px;")
        self.proof_icc_label.setWordWrap(True)
        icc_row.addWidget(self.proof_icc_label, 1)
        icc_btn = QPushButton("Load ICC…")
        icc_btn.setToolTip("Use a custom ICC/ICM profile for soft-proofing (Pillow ImageCms).")
        icc_btn.clicked.connect(self._browse_proof_icc)
        icc_row.addWidget(icc_btn)
        icc_clear = QPushButton("Clear")
        icc_clear.clicked.connect(self._clear_proof_icc)
        icc_row.addWidget(icc_clear)
        v.addLayout(icc_row)
        self.gamut_warn_cb = QCheckBox("Gamut warning (magenta)")
        self.gamut_warn_cb.setToolTip("Highlight colors that change a lot under the proof simulation.")
        self.gamut_warn_cb.toggled.connect(self._on_gamut_warning)
        v.addWidget(self.gamut_warn_cb)
        self.paper_white_cb = QCheckBox("Simulate paper white")
        self.paper_white_cb.setToolTip("Tint the proof toward a slightly warm paper white.")
        self.paper_white_cb.toggled.connect(self._on_paper_white)
        v.addWidget(self.paper_white_cb)
        self.gamut_pct_label = QLabel("Gamut shift: —")
        self.gamut_pct_label.setStyleSheet("color:#9cf; font-size:11px;")
        v.addWidget(self.gamut_pct_label)
        tip = QLabel(
            "Uses a real ICC transform when Pillow ImageCms finds a profile "
            "(custom file or system ICC). Otherwise falls back to an approximation."
        )
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
        hint3 = QLabel(
            "Applied last (before grain). Set PPI + media, then Apply suggestion, "
            "or set amount manually."
        )
        hint3.setWordWrap(True)
        hint3.setStyleSheet("color:#888; font-size:11px;")
        v.addWidget(hint3)
        ppi_row = QHBoxLayout()
        ppi_row.addWidget(QLabel("Output PPI"))
        self.output_ppi_spin = QDoubleSpinBox()
        self.output_ppi_spin.setRange(36, 600)
        self.output_ppi_spin.setValue(300)
        self.output_ppi_spin.setDecimals(0)
        self.output_ppi_spin.setSingleStep(12)
        self.output_ppi_spin.valueChanged.connect(self._on_output_ppi)
        ppi_row.addWidget(self.output_ppi_spin, 1)
        v.addLayout(ppi_row)
        media_row = QHBoxLayout()
        media_row.addWidget(QLabel("Media"))
        self.output_media_combo = QComboBox()
        self.output_media_combo.addItem("Screen", "screen")
        self.output_media_combo.addItem("Matte print", "matte")
        self.output_media_combo.addItem("Glossy print", "glossy")
        self.output_media_combo.currentIndexChanged.connect(self._on_output_media)
        media_row.addWidget(self.output_media_combo, 1)
        v.addLayout(media_row)
        sug_row = QHBoxLayout()
        self.output_suggest_label = QLabel("Suggested: —")
        self.output_suggest_label.setStyleSheet("color:#9cf; font-size:11px;")
        sug_row.addWidget(self.output_suggest_label, 1)
        apply_sug = QPushButton("Apply suggestion")
        apply_sug.clicked.connect(self._apply_output_sharpen_suggestion)
        sug_row.addWidget(apply_sug)
        v.addLayout(sug_row)
        self._add_slider(v, "output_sharpen", "Output amount", 0.0, 100.0, 1, 0, 0.0)
        quick = QHBoxLayout()
        for label, ppi, media in (
            ("Screen 96", 96, "screen"),
            ("Print 240", 240, "matte"),
            ("Print 300", 300, "glossy"),
            ("Print 360", 360, "glossy"),
        ):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, p=ppi, m=media: self._quick_output_sharpen(p, m))
            quick.addWidget(b)
        v.addLayout(quick)

        box, v = collapsible_group("Skin protection", layout, checked=False)
        self._add_slider(v, "protect_skin", "Protect skin tones", 0.0, 100.0, 1, 0, 0.0)
        sk = QLabel("Reduces capture/output sharpening and vibrance/saturation on skin-like hues.")
        sk.setWordWrap(True)
        sk.setStyleSheet("color:#888; font-size:11px;")
        v.addWidget(sk)

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
        hz_row = QHBoxLayout()
        self.level_horizon_btn = QPushButton("Draw level line")
        self.level_horizon_btn.setToolTip("Drag a line along the horizon; angle is applied automatically.")
        self.level_horizon_btn.clicked.connect(self.start_horizon_line)
        hz_row.addWidget(self.level_horizon_btn)
        auto_lvl = QPushButton("Auto level")
        auto_lvl.setToolTip("Detect dominant near-horizontal edges (Hough) and set horizon angle.")
        auto_lvl.clicked.connect(self.auto_level_horizon)
        hz_row.addWidget(auto_lvl)
        v.addLayout(hz_row)
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
        self.lens_auto_cb = QCheckBox("Apply Lensfun correction")
        self.lens_auto_cb.setToolTip(
            "Requires: pip install lensfunpy + Lensfun database.\n"
            "Matches EXIF camera/lens when available."
        )
        self.lens_auto_cb.toggled.connect(self._on_lens_auto)
        v.addWidget(self.lens_auto_cb)
        self._add_slider(v, "lens_strength", "Lensfun strength", 0.0, 100.0, 1, 0, 100.0)
        lf_row = QHBoxLayout()
        probe_btn = QPushButton("Test match…")
        probe_btn.setToolTip("Show how Lensfun resolves the current file’s camera/lens EXIF.")
        probe_btn.clicked.connect(self._probe_lensfun)
        lf_row.addWidget(probe_btn)
        v.addLayout(lf_row)
        self.lensfun_status = QLabel("Lensfun: not tested")
        self.lensfun_status.setWordWrap(True)
        self.lensfun_status.setStyleSheet("color:#9cf; font-size:11px;")
        v.addWidget(self.lensfun_status)
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
        self._add_slider(v, "distortion", "Amount", -100.0, 100.0, 1, 0, 0.0)

        box, v = collapsible_group("Perspective", layout, checked=False)
        self._add_slider(v, "perspective", "Vertical", -100.0, 100.0, 1, 0, 0.0)
        self._add_slider(v, "perspective_h", "Horizontal", -100.0, 100.0, 1, 0, 0.0)
        tip_p = QLabel("Simple trapezoid correction. For freeform control use Keystone.")
        tip_p.setWordWrap(True)
        tip_p.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(tip_p)

        box, v = collapsible_group("Keystone (4 corners)", layout, checked=False)
        ks_row = QHBoxLayout()
        self.keystone_btn = QPushButton("Edit corners on image")
        self.keystone_btn.setCheckable(True)
        self.keystone_btn.toggled.connect(self.toggle_keystone_mode)
        ks_row.addWidget(self.keystone_btn)
        ks_reset = QPushButton("Reset")
        ks_reset.clicked.connect(self.reset_keystone)
        ks_row.addWidget(ks_reset)
        v.addLayout(ks_row)
        tip_k = QLabel("Drag TL / TR / BR / BL handles. Release to apply the warp.")
        tip_k.setWordWrap(True)
        tip_k.setStyleSheet("color:#777; font-size:11px;")
        v.addWidget(tip_k)

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

        box, v = collapsible_group("Vignetting", layout, checked=False)
        self._add_slider(v, "vignette", "Intensity", 0.0, 100.0, 1, 0, 0.0)

        box, v = collapsible_group("Film Grain", layout, checked=False)
        self._add_slider(v, "film_grain", "Amount", 0.0, 100.0, 1, 0, 0.0)

        box, v = collapsible_group("Black & White", layout, checked=False)
        self.bw_cb = QCheckBox("Convert to black & white")
        self.bw_cb.toggled.connect(self._on_bw)
        v.addWidget(self.bw_cb)

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
        add_local_slider("local_chroma", "Chroma range", 0.0, 100.0, 1.0, 0, 100.0)
        add_local_slider("local_luma", "Luma similarity", 0.0, 100.0, 1.0, 0, 100.0)
        add_local_slider("local_luma_min", "Luma min", 0.0, 100.0, 1.0, 0, 0.0)
        add_local_slider("local_luma_max", "Luma max", 0.0, 100.0, 1.0, 0, 100.0)
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
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mask mode"))
        self.brush_mode_combo = QComboBox()
        self.brush_mode_combo.addItem("Add", "add")
        self.brush_mode_combo.addItem("Subtract", "subtract")
        self.brush_mode_combo.addItem("Intersect", "intersect")
        self.brush_mode_combo.currentIndexChanged.connect(self._on_brush_paint_mode)
        mode_row.addWidget(self.brush_mode_combo, 1)
        v.addLayout(mode_row)
        self.brush_mask_only_cb = QCheckBox("Show mask only")
        self.brush_mask_only_cb.toggled.connect(self._on_brush_mask_only)
        v.addWidget(self.brush_mask_only_cb)
        inv_row = QHBoxLayout()
        self.brush_invert_btn = QPushButton("Invert selected mask")
        self.brush_invert_btn.setToolTip("Apply brush adjustments outside the painted area instead.")
        self.brush_invert_btn.clicked.connect(self._on_brush_invert)
        inv_row.addWidget(self.brush_invert_btn)
        subj_btn = QPushButton("Auto subject mask")
        subj_btn.setToolTip("Offline GrabCut subject detection → new brush mask (no neural net).")
        subj_btn.clicked.connect(self._on_auto_subject_mask)
        inv_row.addWidget(subj_btn)
        v.addLayout(inv_row)
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
        flow_row = QHBoxLayout()
        flow_row.addWidget(QLabel("Flow"))
        self.brush_flow_slider = SliderRow("", 5.0, 100.0, 100.0, 1,
            lambda val: self._on_brush_flow(val), 0)
        flow_row.addWidget(self.brush_flow_slider, 1)
        v.addLayout(flow_row)
        opac_row = QHBoxLayout()
        opac_row.addWidget(QLabel("Opacity"))
        self.brush_opacity_slider = SliderRow("", 5.0, 100.0, 100.0, 1,
            lambda val: self._on_brush_opacity(val), 0)
        opac_row.addWidget(self.brush_opacity_slider, 1)
        v.addLayout(opac_row)
        self.brush_sliders = {}
        self.brush_sliders_box = QWidget()
        bsl = QVBoxLayout(self.brush_sliders_box)
        bsl.setContentsMargins(0, 0, 0, 0)
        for key, label, lo, hi, step, dec in (
            ("exposure", "Exposure", -2.0, 2.0, 0.05, 2),
            ("contrast", "Contrast", -100.0, 100.0, 1, 0),
            ("saturation", "Saturation", -100.0, 100.0, 1, 0),
            ("clarity", "Clarity", -100.0, 100.0, 1, 0),
            ("temperature", "Temp shift", -100.0, 100.0, 1, 0),
            ("flow", "Mask flow", 0.0, 100.0, 1, 0),
            ("opacity", "Mask opacity", 0.0, 100.0, 1, 0),
        ):
            row = SliderRow(label, lo, hi, 100.0 if key in ("flow", "opacity") else 0.0, step,
                            lambda val, k=key: self._on_brush_adj(k, val), dec)
            self.brush_sliders[key] = row
            bsl.addWidget(row)
        self.brush_sliders_box.setEnabled(False)
        v.addWidget(self.brush_sliders_box)

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

    def _visible_filmstrip_paths(self) -> list:
        """Paths of filmstrip items that are not hidden by the current filters."""
        out = []
        for i in range(self.filmstrip.count()):
            item = self.filmstrip.item(i)
            if item and not item.isHidden():
                p = item.data(Qt.ItemDataRole.UserRole)
                if p:
                    out.append(p)
        return out

    def prev_image(self):
        """Previous image, skipping items hidden by filmstrip filters."""
        visible = self._visible_filmstrip_paths()
        if not visible:
            return
        if self.current_path in visible:
            idx = visible.index(self.current_path)
        else:
            # Current hidden — jump to last visible before current in full list
            try:
                full_idx = self.image_paths.index(self.current_path)
            except ValueError:
                full_idx = 0
            idx = 0
            for i, p in enumerate(visible):
                try:
                    if self.image_paths.index(p) < full_idx:
                        idx = i + 1
                except ValueError:
                    pass
            idx = min(idx, len(visible) - 1)
        if idx > 0:
            path = visible[idx - 1]
            self.load_image(path)
            self._select_filmstrip_path(path)

    def next_image(self):
        """Next image, skipping items hidden by filmstrip filters."""
        visible = self._visible_filmstrip_paths()
        if not visible:
            return
        if self.current_path in visible:
            idx = visible.index(self.current_path)
        else:
            try:
                full_idx = self.image_paths.index(self.current_path)
            except ValueError:
                full_idx = -1
            idx = -1
            for i, p in enumerate(visible):
                try:
                    if self.image_paths.index(p) > full_idx:
                        idx = i - 1
                        break
                except ValueError:
                    pass
            if idx < 0:
                idx = -1
        if idx < len(visible) - 1:
            path = visible[idx + 1]
            self.load_image(path)
            self._select_filmstrip_path(path)

    def _select_filmstrip_path(self, path: str):
        for i in range(self.filmstrip.count()):
            item = self.filmstrip.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == path:
                self.filmstrip.setCurrentItem(item)
                self.filmstrip.scrollToItem(item)
                break

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
        if hasattr(self, "lib_collections"):
            try:
                self._refresh_collections_combo()
            except Exception:
                log.debug("refresh_library_tree: non-critical failure, continuing", exc_info=True)

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
                    log.debug("_lib_move_to_trash: non-critical failure, continuing", exc_info=True)
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
            log.debug("_trash_file: non-critical failure, continuing", exc_info=True)
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
                log.debug("_trash_file: non-critical failure, continuing", exc_info=True)

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
            log.debug("_save_recent_folders: non-critical failure, continuing", exc_info=True)

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
        entry = {
            "name": name,
            "recipe": self.recipes[self.current_path].to_dict(),
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        lst = self._snapshots.setdefault(self.current_path, [])
        # Replace same name if present
        lst = [s for s in lst if s.get("name") != name]
        lst.append(entry)
        self._snapshots[self.current_path] = lst
        try:
            save_snapshots_sidecar(self.current_path, lst)
            # Also keep the main recipe in the same sidecar
            save_recipe_sidecar(self.current_path, self.recipes[self.current_path], snapshots=lst)
        except Exception as e:
            self.log(f"Snapshot sidecar save failed: {e}", level="ERR")
        self.statusBar().showMessage(f"Snapshot saved: {name} (persisted to sidecar)")
        self.log(f"Snapshot '{name}' for {os.path.basename(self.current_path)}")

    def restore_snapshot(self):
        if self.current_path is None:
            return
        lst = self._snapshots.get(self.current_path) or []
        if not lst:
            # Try loading from sidecar
            try:
                lst = load_snapshots_sidecar(self.current_path)
                if lst:
                    self._snapshots[self.current_path] = lst
            except Exception:
                log.debug("restore_snapshot: non-critical failure, continuing", exc_info=True)
        if not lst:
            QMessageBox.information(
                self, "Snapshots",
                "No snapshots for this image yet.\nUse Edit → Save Snapshot… first.\n"
                "Snapshots are stored in the .photolab.json sidecar.",
            )
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

    def open_folder(self):
        """Open a single folder in Develop (filmstrip). Does not touch the Library catalog."""
        folder = QFileDialog.getExistingDirectory(self, "Open folder for editing (Develop)")
        if folder:
            self.show_develop_mode()
            self.open_folder_path(folder)

    def open_folder_path(self, folder: str):
        """Load images from one folder into the Develop filmstrip only."""
        self.folder = folder
        self._add_recent_folder(folder)
        self.show_develop_mode()
        if getattr(self, "_library_mode", False):
            self.show_develop_mode()
        self.image_paths = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTS)
        )
        self.filmstrip.clear()
        self.recipes = {}
        self.meta_cache = {}
        # Keep ratings/labels across folder re-open only for paths still present;
        # soft-reset flags for a clean filmstrip (paths from other folders remain in dicts).
        
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
        self.current_path = path
        self.statusBar().showMessage(f"Loading {os.path.basename(path)}…")
        if self._load_worker is not None and self._load_worker.isRunning():
            try:
                self._load_worker.terminate()
            except Exception:
                log.debug("load_image: non-critical failure, continuing", exc_info=True)
        self._load_worker = LoadImageWorker(path)
        self._load_worker.loaded.connect(self._on_image_loaded)
        self._load_worker.failed.connect(self._on_image_failed)
        self._load_worker.start()

    def _on_image_loaded(self, path, img, meta):
        if path != self.current_path:
            return
        self.original_bgr = img
        self.meta_cache[path] = meta
        self._build_proxy(img)
        self._clear_preview_cache()
        if path not in self.recipes:
            side = load_recipe_sidecar(path)
            self.recipes[path] = side if side is not None else Recipe()
            if side is not None:
                self.log(f"Loaded sidecar for {os.path.basename(path)}")
            if meta.get("is_raw"):
                self.recipes[path].wb_as_shot = True
        # Named snapshots from sidecar (persisted)
        try:
            snaps = load_snapshots_sidecar(path)
            if snaps:
                self._snapshots[path] = snaps
                self.log(f"Loaded {len(snaps)} snapshot(s) from sidecar")
        except Exception as e:
            self.log(f"Snapshot load failed: {e}", level="ERR")
        self.sync_sliders_to_recipe()
        self.history_widget.clear()
        self._push_history("Original")
        self.render_preview()
        if path in self.image_paths:
            self.filmstrip.setCurrentRow(self.image_paths.index(path))
        kind = "RAW" if meta.get("is_raw") else "RGB"
        ph, pw = img.shape[:2]
        proxy_note = ""
        if self.proxy_bgr is not None and self._proxy_scale < 0.99:
            pph, ppw = self.proxy_bgr.shape[:2]
            proxy_note = f"  •  preview {ppw}×{pph}"
        self.statusBar().showMessage(
            f"{os.path.basename(path)}  •  {pw}×{ph}  •  {kind}{proxy_note}"
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
        self.wb_as_shot_cb.blockSignals(True)
        self.wb_as_shot_cb.setChecked(r.wb_as_shot)
        self.wb_as_shot_cb.blockSignals(False)
        self.sliders["temperature"].setEnabled(not r.wb_as_shot or bool(getattr(r, "wb_dual", False)))
        self.sliders["tint"].setEnabled(not r.wb_as_shot or bool(getattr(r, "wb_dual", False)))
        if hasattr(self, "wb_dual_cb"):
            self.wb_dual_cb.blockSignals(True)
            self.wb_dual_cb.setChecked(bool(getattr(r, "wb_dual", False)))
            self.wb_dual_cb.blockSignals(False)
            self._set_dual_wb_enabled(bool(getattr(r, "wb_dual", False)))
        if hasattr(self, "output_ppi_spin"):
            self.output_ppi_spin.blockSignals(True)
            self.output_ppi_spin.setValue(float(getattr(r, "output_ppi", 300.0) or 300.0))
            self.output_ppi_spin.blockSignals(False)
        if hasattr(self, "output_media_combo"):
            media = getattr(r, "output_media", "screen") or "screen"
            self.output_media_combo.blockSignals(True)
            for i in range(self.output_media_combo.count()):
                if self.output_media_combo.itemData(i) == media:
                    self.output_media_combo.setCurrentIndex(i)
                    break
            self.output_media_combo.blockSignals(False)
            self._refresh_output_suggest_label()
        self.tone_curve.set_values(
            r.curve_shadows, r.curve_darks, r.curve_mids,
            r.curve_lights, r.curve_highlights,
        )
        self.tone_curve.set_point_curve("luma", getattr(r, "curve_points", None) or [])
        self.tone_curve.set_point_curve("r", getattr(r, "curve_r_points", None) or [])
        self.tone_curve.set_point_curve("g", getattr(r, "curve_g_points", None) or [])
        self.tone_curve.set_point_curve("b", getattr(r, "curve_b_points", None) or [])
        # HSL panel — all 8 bands, all three channels
        if hasattr(self, "hsl_panel"):
            self.hsl_panel.set_values(r.hsl_hue, r.hsl_sat, r.hsl_lum)
        if hasattr(self, "soft_proof_cb"):
            self.soft_proof_cb.blockSignals(True)
            self.soft_proof_cb.setChecked(r.soft_proof)
            self.soft_proof_cb.blockSignals(False)
        if hasattr(self, "gamut_warn_cb"):
            self.gamut_warn_cb.blockSignals(True)
            self.gamut_warn_cb.setChecked(bool(getattr(r, "soft_proof_gamut", False)))
            self.gamut_warn_cb.blockSignals(False)
        if hasattr(self, "proof_combo"):
            self.proof_combo.blockSignals(True)
            self.proof_combo.setCurrentText(r.soft_proof_profile)
            self.proof_combo.blockSignals(False)
        if hasattr(self, "paper_white_cb"):
            self.paper_white_cb.blockSignals(True)
            self.paper_white_cb.setChecked(bool(getattr(r, "soft_proof_paper_white", False)))
            self.paper_white_cb.blockSignals(False)
        if hasattr(self, "proof_intent_combo"):
            intent = getattr(r, "soft_proof_intent", "relative") or "relative"
            self.proof_intent_combo.blockSignals(True)
            for i in range(self.proof_intent_combo.count()):
                if self.proof_intent_combo.itemData(i) == intent:
                    self.proof_intent_combo.setCurrentIndex(i)
                    break
            self.proof_intent_combo.blockSignals(False)
        if hasattr(self, "proof_icc_label"):
            icc = getattr(r, "soft_proof_icc_path", "") or ""
            self.proof_icc_label.setText(
                f"ICC: {os.path.basename(icc)}" if icc else "ICC: (built-in / system)"
            )
        if hasattr(self, "bw_cb"):
            self.bw_cb.blockSignals(True)
            self.bw_cb.setChecked(bool(r.black_and_white))
            self.bw_cb.blockSignals(False)
            
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

    def on_slider(self, key, value):
        if self.current_path is None:
            return
        if key.startswith("_hsl_"):
            return  # handled by _on_hsl_slider
        if hasattr(self.recipes[self.current_path], key):
            setattr(self.recipes[self.current_path], key, value)
            self._schedule_history(key)
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

    def _target_filmstrip_paths(self) -> list:
        """Selected filmstrip paths, or [current] if nothing multi-selected."""
        paths = self._selected_filmstrip_paths()
        if not paths and self.current_path:
            paths = [self.current_path]
        # De-dupe preserve order
        seen = set()
        out = []
        for p in paths:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    def _filmstrip_label_text(self, path: str) -> str:
        """Unified filmstrip caption: pick / reject / color / stars / name."""
        base = os.path.basename(path)
        parts = []
        if self._pick_flags.get(path):
            parts.append("✓")
        if self._reject_flags.get(path):
            parts.append("⛔")
        color = self._color_labels.get(path)
        if color and color in self.COLOR_LABELS:
            parts.append(self.COLOR_LABELS[color][0])
        stars = int(self._image_ratings.get(path, 0) or 0)
        if stars:
            parts.append("★" * stars + "☆" * (5 - stars))
        prefix = (" ".join(parts) + "  ") if parts else ""
        return f"{prefix}{base}"

    def _refresh_filmstrip_item(self, path: str):
        for i in range(self.filmstrip.count()):
            item = self.filmstrip.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == path:
                item.setText(self._filmstrip_label_text(path))
                # Optional tint via foreground for color labels
                color = self._color_labels.get(path)
                if color and color in self.COLOR_LABELS:
                    from PyQt6.QtGui import QColor, QBrush
                    item.setForeground(QBrush(QColor(self.COLOR_LABELS[color][1])))
                else:
                    from PyQt6.QtGui import QBrush, QColor
                    item.setForeground(QBrush(QColor("#ddd")))
                break

    def toggle_reject_current(self):
        paths = self._target_filmstrip_paths()
        if not paths:
            return
        # Toggle based on the primary (current or first selected) state
        primary = self.current_path if self.current_path in paths else paths[0]
        currently = bool(self._reject_flags.get(primary, False))
        try:
            rec = self.catalog.get(primary) if hasattr(self.catalog, "get") else None
            if isinstance(rec, dict) and primary not in self._reject_flags:
                currently = bool(rec.get("reject"))
        except Exception:
            log.debug("toggle_reject_current: non-critical failure, continuing", exc_info=True)
        new_val = not currently
        for path in paths:
            self._reject_flags[path] = new_val
            try:
                self.catalog.set_reject(path, new_val)
            except Exception:
                log.debug("toggle_reject_current: non-critical failure, continuing", exc_info=True)
            self._refresh_filmstrip_item(path)
        n = len(paths)
        msg = "Rejected" if new_val else "Un-rejected"
        self.statusBar().showMessage(f"{msg} ({n} image{'s' if n != 1 else ''})")
        self._apply_filmstrip_filter()

    def _on_history_restore(self, index: int):
        d = self.history_widget.get_recipe_dict(index)
        if d is None or self.current_path is None:
            return
        self.recipes[self.current_path] = Recipe.from_dict(d)
        self.sync_sliders_to_recipe()
        self.render_preview()
        label = self.history_widget.get_label(index) or f"#{index + 1}"
        self.statusBar().showMessage(f"Restored history: {label}")

    def _on_history_copy_settings(self, index: int):
        d = self.history_widget.get_recipe_dict(index)
        if d is None:
            self.statusBar().showMessage("Nothing to copy from history")
            return
        self._copied_recipe = dict(d)
        label = self.history_widget.get_label(index) or f"#{index + 1}"
        self.statusBar().showMessage(f"Copied settings from history: {label}  (Ctrl+Shift+V to paste)")

    def _on_history_preview(self, index: int):
        """Visual before/after: history entry vs current recipe on the proxy."""
        if self.current_path is None or self.original_bgr is None:
            return
        d = self.history_widget.get_recipe_dict(index)
        if d is None:
            return
        label = self.history_widget.get_label(index) or f"#{index + 1}"
        hist_recipe = Recipe.from_dict(d)
        cur_recipe = self.recipes[self.current_path]
        meta = self.meta_cache.get(self.current_path, {})
        multipliers = meta.get("wb_multipliers")
        src = self.proxy_bgr if self.proxy_bgr is not None else self.original_bgr
        try:
            before = apply_recipe(src, hist_recipe, wb_multipliers=multipliers, meta=meta)
            after = apply_recipe(src, cur_recipe, wb_multipliers=multipliers, meta=meta)
        except Exception as e:
            QMessageBox.warning(self, "History preview", str(e))
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(f"History preview — {label}  vs  current")
        dlg.resize(960, 520)
        layout = QVBoxLayout(dlg)
        row = QHBoxLayout()
        layout.addLayout(row)
        for title, img in ((f"History: {label}", before), ("Current", after)):
            col = QVBoxLayout()
            cap = QLabel(title)
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cap.setStyleSheet("color:#ccc; font-weight:600;")
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background:#111; border:1px solid #333;")
            lbl.setPixmap(cv_to_qpixmap(img))
            col.addWidget(cap)
            col.addWidget(lbl)
            row.addLayout(col)
        btns = QHBoxLayout()
        restore_btn = QPushButton("Restore history state")
        restore_btn.clicked.connect(lambda: (self._on_history_restore(index), dlg.accept()))
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        btns.addWidget(restore_btn)
        btns.addStretch(1)
        btns.addWidget(close_btn)
        layout.addLayout(btns)
        dlg.exec()

    def _on_hsl_row(self, which: str, idx: int, value: float):
        """One HSL band slider (Red..Magenta) moved in the Hue/Saturation/
        Luminance/All panel. `idx` is the color band (0=Red..7=Magenta)."""
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]

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

    def _on_proof_intent(self, _idx=None):
        if self.current_path is None or not hasattr(self, "proof_intent_combo"):
            return
        intent = self.proof_intent_combo.currentData() or "relative"
        self.recipes[self.current_path].soft_proof_intent = intent
        self.render_timer.start()

    def _on_paper_white(self, checked: bool):
        if self.current_path is None:
            return
        self.recipes[self.current_path].soft_proof_paper_white = bool(checked)
        self.render_timer.start()

    def _browse_proof_icc(self):
        if self.current_path is None:
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Soft-proof ICC profile", "",
            "ICC profiles (*.icc *.icm);;All (*.*)",
        )
        if not path:
            return
        self.recipes[self.current_path].soft_proof_icc_path = path
        if hasattr(self, "proof_icc_label"):
            self.proof_icc_label.setText(f"ICC: {os.path.basename(path)}")
        self.recipes[self.current_path].soft_proof = True
        if hasattr(self, "soft_proof_cb"):
            self.soft_proof_cb.blockSignals(True)
            self.soft_proof_cb.setChecked(True)
            self.soft_proof_cb.blockSignals(False)
        self.render_timer.start()

    def _clear_proof_icc(self):
        if self.current_path is None:
            return
        self.recipes[self.current_path].soft_proof_icc_path = ""
        if hasattr(self, "proof_icc_label"):
            self.proof_icc_label.setText("ICC: (built-in / system)")
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
        self._schedule_history("Tone curve")
        self.render_timer.start()

    def _on_curve_channel(self, key: str):
        if hasattr(self, "curve_channel_group"):
            for btn, k in self.curve_channel_group:
                btn.blockSignals(True)
                btn.setChecked(k == key)
                btn.blockSignals(False)
        if hasattr(self, "tone_curve"):
            self.tone_curve.set_channel(key)

    def on_point_curve_changed(self, channel: str, points: list):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        key_map = {
            "luma": "curve_points",
            "r": "curve_r_points",
            "g": "curve_g_points",
            "b": "curve_b_points",
        }
        attr = key_map.get(channel)
        if not attr:
            return
        setattr(r, attr, [list(p) for p in points])
        self._schedule_history(f"Point curve {channel}")
        self.render_timer.start()

    def on_wb_as_shot(self, checked):
        if self.current_path is None:
            return
        self.recipes[self.current_path].wb_as_shot = checked
        self.sliders["temperature"].setEnabled(not checked)
        self.sliders["tint"].setEnabled(not checked)
        dual = bool(getattr(self.recipes[self.current_path], "wb_dual", False))
        self._set_dual_wb_enabled(not checked or dual)
        self.render_timer.start()

    def _set_dual_wb_enabled(self, enabled: bool):
        for k in ("temperature2", "tint2", "wb_mix"):
            if k in self.sliders:
                self.sliders[k].setEnabled(enabled)

    def _on_wb_dual(self, checked):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.wb_dual = bool(checked)
        if checked:
            r.wb_as_shot = False
            if hasattr(self, "wb_as_shot_cb"):
                self.wb_as_shot_cb.blockSignals(True)
                self.wb_as_shot_cb.setChecked(False)
                self.wb_as_shot_cb.blockSignals(False)
            self.sliders["temperature"].setEnabled(True)
            self.sliders["tint"].setEnabled(True)
        self._set_dual_wb_enabled(bool(checked))
        self._schedule_history("Dual WB")
        self.render_timer.start()

    def _on_output_ppi(self, val):
        if self.current_path is None:
            return
        self.recipes[self.current_path].output_ppi = float(val)
        self._refresh_output_suggest_label()

    def _on_output_media(self, _idx=None):
        if self.current_path is None or not hasattr(self, "output_media_combo"):
            return
        media = self.output_media_combo.currentData() or "screen"
        self.recipes[self.current_path].output_media = media
        self._refresh_output_suggest_label()

    def _refresh_output_suggest_label(self):
        if not hasattr(self, "output_suggest_label"):
            return
        try:
            from imaging import output_sharpen_params
            ppi = float(self.output_ppi_spin.value()) if hasattr(self, "output_ppi_spin") else 300.0
            media = "screen"
            if hasattr(self, "output_media_combo"):
                media = self.output_media_combo.currentData() or "screen"
            amt, rad = output_sharpen_params(ppi, media)
            self.output_suggest_label.setText(f"Suggested: amount {amt:.0f}, radius {rad:.2f}")
        except Exception:
            self.output_suggest_label.setText("Suggested: —")

    def _apply_output_sharpen_suggestion(self):
        if self.current_path is None:
            return
        from imaging import output_sharpen_params
        r = self.recipes[self.current_path]
        ppi = float(getattr(r, "output_ppi", 300.0) or 300.0)
        media = getattr(r, "output_media", "screen") or "screen"
        if hasattr(self, "output_ppi_spin"):
            ppi = float(self.output_ppi_spin.value())
            r.output_ppi = ppi
        if hasattr(self, "output_media_combo"):
            media = self.output_media_combo.currentData() or media
            r.output_media = media
        amt, _rad = output_sharpen_params(ppi, media)
        r.output_sharpen = amt
        self.sync_sliders_to_recipe()
        self._push_history("Output sharpen (PPI)")
        self.render_preview()
        self.statusBar().showMessage(f"Output sharpen → {amt:.0f} ({media}, {ppi:.0f} PPI)")

    def _quick_output_sharpen(self, ppi, media):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        r.output_ppi = float(ppi)
        r.output_media = media
        if hasattr(self, "output_ppi_spin"):
            self.output_ppi_spin.blockSignals(True)
            self.output_ppi_spin.setValue(float(ppi))
            self.output_ppi_spin.blockSignals(False)
        if hasattr(self, "output_media_combo"):
            self.output_media_combo.blockSignals(True)
            for i in range(self.output_media_combo.count()):
                if self.output_media_combo.itemData(i) == media:
                    self.output_media_combo.setCurrentIndex(i)
                    break
            self.output_media_combo.blockSignals(False)
        self._refresh_output_suggest_label()
        self._apply_output_sharpen_suggestion()

    def _build_proxy(self, img: np.ndarray):
        if img is None:
            self.proxy_bgr = None
            self._proxy_scale = 1.0
            return
        h, w = img.shape[:2]
        long_edge = max(h, w)
        if long_edge > PROXY_MAX_DIM:
            scale = PROXY_MAX_DIM / long_edge
            self.proxy_bgr = cv2.resize(
                img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
            )
            self._proxy_scale = scale
        else:
            self.proxy_bgr = img
            self._proxy_scale = 1.0

    def _clear_preview_cache(self):
        self._preview_cache.clear()
        self._preview_cache_order.clear()

    def _recipe_fingerprint(self, recipe: Recipe) -> str:
        try:
            d = recipe_to_dict(recipe)
            payload = json.dumps(d, sort_keys=True, default=str)
        except Exception:
            payload = repr(recipe)
        return hashlib.sha1(payload.encode("utf-8", errors="ignore")).hexdigest()

    def _preview_cache_get(self, key: str):
        hit = self._preview_cache.get(key)
        if hit is None:
            return None
        try:
            self._preview_cache_order.remove(key)
        except ValueError:
            pass
        self._preview_cache_order.append(key)
        return hit

    def _preview_cache_put(self, key: str, result_bgr, src_bgr):
        if key in self._preview_cache:
            try:
                self._preview_cache_order.remove(key)
            except ValueError:
                pass
        self._preview_cache[key] = (result_bgr, src_bgr)
        self._preview_cache_order.append(key)
        while len(self._preview_cache_order) > PREVIEW_CACHE_MAX:
            old = self._preview_cache_order.pop(0)
            self._preview_cache.pop(old, None)

    def render_preview(self):
        if self.original_bgr is None or self.current_path is None:
            return
        recipe = self.recipes[self.current_path]
        meta = self.meta_cache.get(self.current_path, {})
        multipliers = meta.get("wb_multipliers")

        preview_src = self.proxy_bgr
        if preview_src is None:
            h, w = self.original_bgr.shape[:2]
            if max(h, w) > PROXY_MAX_DIM:
                scale = PROXY_MAX_DIM / max(h, w)
                preview_src = cv2.resize(
                    self.original_bgr, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                preview_src = self.original_bgr

        cache_key = f"{self.current_path}|{self._recipe_fingerprint(recipe)}"
        cached = self._preview_cache_get(cache_key)
        if cached is not None:
            result, preview_src = cached
        else:
            result = apply_recipe(
                preview_src, recipe, wb_multipliers=multipliers, meta=meta,
            )
            self._preview_cache_put(cache_key, result, preview_src)

        self.histogram.set_image(result)
        pix = cv_to_qpixmap(result)
        orig_pix = cv_to_qpixmap(preview_src)
        self.preview.set_image(pix, original=orig_pix)
        if hasattr(self, "navigator"):
            self.navigator.set_image(pix)
            self.navigator.set_viewport(0.0, 0.0, 1.0, 1.0)
        self._update_gamut_percent_label()
        if getattr(self, "_pending_history_label", None):
            self._push_history(self._pending_history_label)
            self._pending_history_label = None

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
            self.statusBar().showMessage("Control Point mode: click on image to place a local adjustment")
            self.preview.setCursor(Qt.CursorShape.CrossCursor)
            # Switch to Local tab (index 5)
            if hasattr(self, "_cat_group"):
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
                self._temp_before_mode = getattr(self.preview, "compare_mode", 0)
                self.set_compare_mode(ImageCanvas.MODE_SPLIT)
                self.statusBar().showMessage("Before (release key to restore)")
            e.accept()
            return
        super().keyPressEvent(e)

    def keyReleaseEvent(self, e):
        from PyQt6.QtCore import Qt as _Qt
        if e.key() in (_Qt.Key.Key_Backslash, _Qt.Key.Key_QuoteLeft) and not e.isAutoRepeat():
            if getattr(self, "_temp_before", False):
                self._temp_before = False
                mode = getattr(self, "_temp_before_mode", ImageCanvas.MODE_NORMAL)
                self.set_compare_mode(mode)
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

    def import_photos_dialog(self):
        """Import dialog: copy/move, rename pattern, date folders, optional catalog scan."""
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QFileDialog

        dlg = QDialog(self)
        dlg.setWindowTitle("Import Photos")
        dlg.setMinimumWidth(480)
        form = QFormLayout(dlg)

        src_edit = QLineEdit()
        src_btn = QPushButton("Browse…")
        src_row = QHBoxLayout()
        src_row.addWidget(src_edit, 1)
        src_row.addWidget(src_btn)
        form.addRow("Source folder", src_row)

        dest_edit = QLineEdit()
        dest_edit.setText(self.folder or os.path.expanduser("~"))
        dest_btn = QPushButton("Browse…")
        dest_row = QHBoxLayout()
        dest_row.addWidget(dest_edit, 1)
        dest_row.addWidget(dest_btn)
        form.addRow("Destination", dest_row)

        mode_combo = QComboBox()
        mode_combo.addItem("Copy files", "copy")
        mode_combo.addItem("Move files", "move")
        form.addRow("Mode", mode_combo)

        rename_combo = QComboBox()
        rename_combo.addItem("Keep original names", "keep")
        rename_combo.addItem("Date + sequence (YYYYMMDD_0001)", "date_seq")
        rename_combo.addItem("Date + original (YYYYMMDD_name)", "date_orig")
        form.addRow("Rename", rename_combo)

        subfolder_cb = QCheckBox("Organize into Year / YYYY-MM-DD folders")
        subfolder_cb.setChecked(True)
        form.addRow(subfolder_cb)

        recursive_cb = QCheckBox("Include subfolders of source")
        recursive_cb.setChecked(True)
        form.addRow(recursive_cb)

        scan_cb = QCheckBox("Scan destination into Library after import")
        scan_cb.setChecked(True)
        form.addRow(scan_cb)

        open_cb = QCheckBox("Open destination folder when done")
        open_cb.setChecked(True)
        form.addRow(open_cb)

        count_lbl = QLabel("Choose a source folder to count images.")
        count_lbl.setStyleSheet("color:#9cf; font-size:11px;")
        form.addRow(count_lbl)

        def browse_src():
            d = QFileDialog.getExistingDirectory(dlg, "Source folder", src_edit.text() or "")
            if d:
                src_edit.setText(d)
                refresh_count()

        def browse_dest():
            d = QFileDialog.getExistingDirectory(dlg, "Destination folder", dest_edit.text() or "")
            if d:
                dest_edit.setText(d)

        def refresh_count():
            d = src_edit.text().strip()
            if not d or not os.path.isdir(d):
                count_lbl.setText("Choose a source folder to count images.")
                return
            try:
                from catalog import list_importable_files
                n = len(list_importable_files(d, recursive=recursive_cb.isChecked()))
                count_lbl.setText(f"{n} image(s) found in source.")
            except Exception as e:
                count_lbl.setText(f"Count failed: {e}")

        src_btn.clicked.connect(browse_src)
        dest_btn.clicked.connect(browse_dest)
        recursive_cb.toggled.connect(lambda _=False: refresh_count())

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addRow(buttons)

        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        src = src_edit.text().strip()
        dest = dest_edit.text().strip()
        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "Import", "Valid source folder required.")
            return
        if not dest:
            QMessageBox.warning(self, "Import", "Destination folder required.")
            return
        from catalog import list_importable_files
        sources = list_importable_files(src, recursive=recursive_cb.isChecked())
        if not sources:
            QMessageBox.information(self, "Import", "No images found in source.")
            return
        mode = mode_combo.currentData() or "copy"
        if mode == "move":
            if QMessageBox.question(
                self, "Move files",
                f"Move {len(sources)} file(s) from\n{src}\nto\n{dest}?\n\n"
                "Originals will leave the source folder.",
            ) != QMessageBox.StandardButton.Yes:
                return

        prog = self._make_progress("Importing photos…", maximum=len(sources))
        from workers import ImportWorker
        self._import_worker = ImportWorker(
            sources, dest,
            mode=mode,
            rename_pattern=rename_combo.currentData() or "keep",
            subfolder_by_date=subfolder_cb.isChecked(),
        )

        def on_prog(i, n, p):
            if prog:
                prog.setMaximum(n)
                prog.setValue(i + 1)
                prog.setLabelText(f"Importing {i + 1}/{n}\n{os.path.basename(p)}")

        def on_done(stats):
            if prog:
                prog.close()
            msg = (
                f"Imported {stats.get('ok', 0)} file(s). "
                f"Failed: {stats.get('failed', 0)}. Skipped: {stats.get('skipped', 0)}."
            )
            self.statusBar().showMessage(msg)
            self.log(f"Import: {msg}")
            if scan_cb.isChecked():
                try:
                    self.catalog.scan_folder(dest, recursive=True)
                    self.refresh_library_tree()
                except Exception as e:
                    self.log(f"Post-import scan: {e}", level="ERR")
            if open_cb.isChecked():
                self.open_folder_path(dest)
            QMessageBox.information(self, "Import complete", msg)

        def on_fail(err):
            if prog:
                prog.close()
            QMessageBox.warning(self, "Import failed", err)

        self._import_worker.progress.connect(on_prog)
        self._import_worker.finished_ok.connect(on_done)
        self._import_worker.failed.connect(on_fail)
        if prog:
            prog.canceled.connect(self._import_worker.cancel)
            prog.show()
        self._import_worker.start()

    def start_culling_mode(self):
        """Full-screen culling: rate / reject / next with minimal chrome."""
        paths = self._visible_filmstrip_paths()
        if not paths and self.current_path:
            paths = [self.current_path]
        if not paths:
            # Try library selection
            try:
                paths = self._lib_selected_paths()
            except Exception:
                paths = []
        if not paths:
            QMessageBox.information(
                self, "Culling",
                "Open a folder (filmstrip) or select library photos first.",
            )
            return
        start = 0
        if self.current_path and self.current_path in paths:
            start = paths.index(self.current_path)
        dlg = _CullingDialog(self, paths, start_index=start)
        dlg.exec()
        # Refresh filmstrip badges after culling
        for p in paths:
            try:
                self._refresh_filmstrip_item(p)
            except Exception:
                log.debug("start_culling_mode: non-critical failure, continuing", exc_info=True)
        if dlg.current_path():
            try:
                self._select_filmstrip_path(dlg.current_path())
            except Exception:
                log.debug("start_culling_mode: non-critical failure, continuing", exc_info=True)

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
            log.debug("show_metadata: non-critical failure, continuing", exc_info=True)
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

    def auto_level_horizon(self):
        if self.current_path is None or self.original_bgr is None:
            return
        from imaging import detect_horizon_angle
        src = self.proxy_bgr if self.proxy_bgr is not None else self.original_bgr
        ang = detect_horizon_angle(src)
        # Apply opposite rotation to level the detected tilt
        level = -float(ang)
        level = max(-15.0, min(15.0, level))
        self.recipes[self.current_path].horizon = round(level, 2)
        self.sync_sliders_to_recipe()
        self._push_history("Auto level")
        self.render_preview()
        self.statusBar().showMessage(f"Auto level → horizon {level:.2f}° (detected tilt {ang:.2f}°)")
        self.log(f"Auto level: detected {ang:.2f}°, applied {level:.2f}°")

    def toggle_keystone_mode(self, checked=False):
        on = bool(checked) if isinstance(checked, bool) else not self.preview.keystone_mode
        if hasattr(self, "keystone_btn") and not isinstance(checked, bool):
            self.keystone_btn.setChecked(on)
        corners = None
        if self.current_path is not None:
            corners = getattr(self.recipes[self.current_path], "keystone", None)
        self.preview.set_keystone_mode(on, corners)
        if on:
            self.statusBar().showMessage("Keystone: drag the four corner handles, release to apply")
        else:
            self.statusBar().showMessage("Keystone off")

    def reset_keystone(self):
        if self.current_path is None:
            return
        self.recipes[self.current_path].keystone = None
        self.preview.set_keystone_corners(None)
        if hasattr(self, "keystone_btn"):
            self.keystone_btn.setChecked(False)
        self.preview.set_keystone_mode(False)
        self._push_history("Reset keystone")
        self.render_preview()

    def _on_keystone_changed(self, corners: list):
        if self.current_path is None:
            return
        self.recipes[self.current_path].keystone = [list(c) for c in corners]
        self._push_history("Keystone")
        self.render_preview()
        self.statusBar().showMessage("Keystone corners updated")

    def _apply_filmstrip_filter(self, _idx=None):
        min_r = 0
        if hasattr(self, "film_rating_filter"):
            min_r = int(self.film_rating_filter.currentData() or 0)
        color_f = ""
        if hasattr(self, "film_color_filter"):
            color_f = self.film_color_filter.currentData() or ""
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
                log.debug("_apply_filmstrip_filter: non-critical failure, continuing", exc_info=True)
            hide = min_r > 0 and stars < min_r
            if not hide and color_f:
                cl = self._color_labels.get(path)
                if color_f == "none":
                    hide = cl is not None
                else:
                    hide = cl != color_f
            item.setHidden(hide)

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
        """Rate selected filmstrip images (or current) 0–5."""
        paths = self._target_filmstrip_paths()
        if not paths:
            return
        stars = int(max(0, min(5, stars)))
        for path in paths:
            self._image_ratings[path] = stars
            try:
                if hasattr(self, "catalog") and self.catalog is not None:
                    self.catalog.set_rating(path, stars)
            except Exception:
                log.debug("rate_current: non-critical failure, continuing", exc_info=True)
            self._refresh_filmstrip_item(path)
        n = len(paths)
        msg = f"Rating: {stars} star(s)" if stars else "Rating cleared"
        if n > 1:
            msg += f" × {n}"
        self.statusBar().showMessage(msg)
        self._apply_filmstrip_filter()

    def _prompt_rate_selected(self):
        from PyQt6.QtWidgets import QInputDialog
        stars, ok = QInputDialog.getInt(self, "Rate selected", "Stars (0–5):", 3, 0, 5, 1)
        if ok:
            self.rate_current(stars)

    def set_color_label(self, color: str | None):
        """Apply color label to selected (or current) filmstrip images.

        color: None to clear, or one of red|yellow|green|blue|purple.
        """
        paths = self._target_filmstrip_paths()
        if not paths:
            return
        if color is not None:
            color = str(color).lower().strip()
            if color not in self.COLOR_LABELS:
                self.statusBar().showMessage(f"Unknown color label: {color}")
                return
        for path in paths:
            if color is None:
                self._color_labels.pop(path, None)
            else:
                self._color_labels[path] = color
            self._refresh_filmstrip_item(path)
        n = len(paths)
        label = "cleared" if color is None else color
        self.statusBar().showMessage(f"Color label {label} ({n} image{'s' if n != 1 else ''})")
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
                         "curve_mids", "curve_lights", "curve_highlights"]
            color_keys = ["temperature", "tint", "wb_as_shot", "vibrance", "saturation",
                          "hsl_hue", "hsl_sat", "hsl_lum"]
            detail_keys = ["denoise_luminance", "denoise_chroma", "denoise_strength",
                           "denoise_detail", "denoise_method", "sharpen_intensity",
                           "sharpen_radius", "sharpen_threshold", "sharpen_detail", "output_sharpen"]
            geo_keys = ["horizon", "distortion", "perspective", "perspective_h", "crop",
                        "ca_amount", "lens_auto", "lens_strength", "keystone"]
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
        from imaging import estimate_wb_temp_tint
        temp, tint = estimate_wb_temp_tint(self.original_bgr)
        rcp = self.recipes[self.current_path]
        rcp.wb_as_shot = False
        rcp.temperature = round(temp, 0)
        rcp.tint = round(tint, 0)
        self.sync_sliders_to_recipe()
        self._push_history("Auto WB")
        self.render_preview()

    def match_exposure_selected(self):
        """Match selected filmstrip images' exposure to the current image's median luminance."""
        if self.current_path is None or self.original_bgr is None:
            return
        from imaging import estimate_exposure_stops, load_image
        paths = self._target_filmstrip_paths() if hasattr(self, "_target_filmstrip_paths") else []
        paths = [p for p in paths if p != self.current_path]
        if not paths:
            QMessageBox.information(
                self, "Match Exposure",
                "Select other filmstrip images (Ctrl/Shift+click), then match their exposure to the current image.",
            )
            return
        ref_stops = estimate_exposure_stops(self.original_bgr)
        # Current recipe already has an exposure; target absolute mid-gray relative to ref
        ref_exp = float(self.recipes[self.current_path].exposure)
        n = 0
        for p in paths:
            try:
                img, _meta = load_image(p)
                if img is None:
                    continue
                other_stops = estimate_exposure_stops(img)
                # Align other median to ref median: delta in stops
                delta = ref_stops - other_stops
                r = self.recipes.setdefault(p, Recipe())
                r.exposure = round(ref_exp + delta, 2)
                n += 1
            except Exception as e:
                self.log(f"Match exposure failed for {p}: {e}", level="ERR")
        self.statusBar().showMessage(f"Matched exposure on {n} image(s) to current")
        self.log(f"Match exposure → {n} images")

    def match_wb_selected(self):
        """Copy temperature/tint from current onto selected filmstrip images (or gray-world match)."""
        if self.current_path is None:
            return
        paths = self._target_filmstrip_paths() if hasattr(self, "_target_filmstrip_paths") else []
        paths = [p for p in paths if p != self.current_path]
        if not paths:
            QMessageBox.information(
                self, "Match White Balance",
                "Select other filmstrip images (Ctrl/Shift+click), then match WB to the current image.",
            )
            return
        src = self.recipes[self.current_path]
        temp, tint = float(src.temperature), float(src.tint)
        as_shot = bool(src.wb_as_shot)
        n = 0
        for p in paths:
            r = self.recipes.setdefault(p, Recipe())
            r.wb_as_shot = as_shot
            r.temperature = temp
            r.tint = tint
            n += 1
        self.statusBar().showMessage(f"Matched white balance on {n} image(s) to current")
        self.log(f"Match WB → {n} images")

    def reset_module(self, which: str):
        if self.current_path is None:
            return
        r = self.recipes[self.current_path]
        fresh = Recipe()
        groups = {
            "tone": ["exposure", "smart_light", "contrast", "highlights", "shadows",
                     "whites", "blacks", "clarity", "gamma", "curve_shadows", "curve_darks",
                     "curve_mids", "curve_lights", "curve_highlights",
                     "curve_points", "curve_r_points", "curve_g_points", "curve_b_points"],
            "color": ["temperature", "tint", "wb_as_shot", "vibrance", "saturation",
                      "wb_dual", "temperature2", "tint2", "wb_mix",
                      "split_shadow_hue", "split_shadow_sat", "split_highlight_hue",
                      "split_highlight_sat", "split_balance",
                      "hsl_hue", "hsl_sat", "hsl_lum", "soft_proof", "soft_proof_profile",
                      "soft_proof_gamut", "soft_proof_paper_white", "soft_proof_icc_path",
                      "soft_proof_intent"],
            "detail": ["denoise_luminance", "denoise_chroma", "denoise_strength",
                       "denoise_detail", "denoise_method", "sharpen_intensity",
                       "sharpen_radius", "sharpen_threshold", "sharpen_detail", "output_sharpen",
                       "output_ppi", "output_media", "protect_skin"],
            "geometry": ["horizon", "distortion", "perspective", "perspective_h", "crop",
                         "ca_amount", "lens_auto", "lens_strength", "keystone"],
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
        paths = self._target_filmstrip_paths()
        if not paths:
            return
        primary = self.current_path if self.current_path in paths else paths[0]
        new_val = not self._pick_flags.get(primary, False)
        for path in paths:
            self._pick_flags[path] = new_val
            self._refresh_filmstrip_item(path)
        n = len(paths)
        msg = "Picked" if new_val else "Unpicked"
        self.statusBar().showMessage(f"{msg} ({n} image{'s' if n != 1 else ''})")

    def compare_selected_images(self):
        """Side-by-side compare of 2–4 selected filmstrip images."""
        paths = self._selected_filmstrip_paths()
        if len(paths) < 2:
            QMessageBox.information(
                self, "Compare Selected",
                "Select 2–4 images in the filmstrip (Ctrl/Shift+click), then try again.",
            )
            return
        paths = paths[:4]
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Compare — {len(paths)} images")
        dlg.resize(min(1400, 360 * len(paths) + 40), 520)
        layout = QVBoxLayout(dlg)
        row = QHBoxLayout()
        layout.addLayout(row)
        from imaging import load_image as _load
        max_side = 480
        for path in paths:
            col = QVBoxLayout()
            label = QLabel(os.path.basename(path))
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("color:#ccc; font-size:11px;")
            img_lbl = QLabel()
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            img_lbl.setMinimumSize(280, 320)
            img_lbl.setStyleSheet("background:#111; border:1px solid #333;")
            try:
                img, _meta = _load(path, use_camera_wb=True)
                # Apply recipe if we have one in memory
                r = self.recipes.get(path)
                if r is not None:
                    meta = self.meta_cache.get(path, _meta or {})
                    img = apply_recipe(
                        img, r,
                        wb_multipliers=meta.get("wb_multipliers"),
                        meta=meta,
                    )
                h, w = img.shape[:2]
                if max(h, w) > max_side:
                    scale = max_side / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                pix = cv_to_qpixmap(img)
                img_lbl.setPixmap(pix)
            except Exception as e:
                img_lbl.setText(f"Could not load\n{e}")
            col.addWidget(img_lbl)
            col.addWidget(label)
            # Stars / flags summary
            stars = self._image_ratings.get(path, 0)
            flags = []
            if self._pick_flags.get(path):
                flags.append("✓")
            if self._reject_flags.get(path):
                flags.append("⛔")
            cl = self._color_labels.get(path)
            if cl and cl in self.COLOR_LABELS:
                flags.append(self.COLOR_LABELS[cl][0])
            if stars:
                flags.append("★" * stars)
            meta_l = QLabel(" ".join(flags) if flags else "—")
            meta_l.setAlignment(Qt.AlignmentFlag.AlignCenter)
            meta_l.setStyleSheet("color:#888; font-size:11px;")
            col.addWidget(meta_l)
            open_btn = QPushButton("Open in Develop")
            open_btn.clicked.connect(lambda _=False, p=path: (dlg.accept(), self.load_image(p)))
            col.addWidget(open_btn)
            row.addLayout(col)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        layout.addWidget(close_btn)
        dlg.exec()

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
                    log.debug("_lib_export_selected: non-critical failure, continuing", exc_info=True)
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

        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout
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

        crop_cb = QCheckBox("Crop common area")
        crop_cb.setChecked(True)
        form.addRow(crop_cb)
        depth_cb = QCheckBox("Also save depth map PNG")
        form.addRow(depth_cb)

        tip = QLabel(
            "Frames should be near→far (or far→near) focus brackets on a stable camera.\n"
            "Result opens in Develop for further editing."
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
        self._stack_worker = FocusStackWorker, PanoramaWorker(
            paths,
            out_path,
            align_mode=align_combo.currentData(),
            fusion_mode=fusion_combo.currentData(),
            reference=ref_combo.currentData(),
            max_dim=int(size_combo.currentData() or 0),
            crop_common=crop_cb.isChecked(),
            save_depth=depth_cb.isChecked(),
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
        from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout
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
        base = os.path.splitext(os.path.basename(paths[0]))[0]
        suggested = os.path.join(self.folder or ".", f"{base}_HDR.jpg")
        out_path, _ = QFileDialog.getSaveFileName(
            self, "Save HDR merge", suggested,
            "JPEG (*.jpg);;PNG (*.png);;TIFF (*.tif);;All (*.*)",
        )
        if not out_path:
            return
        self.statusBar().showMessage(f"HDR merge: {len(paths)} frames…")
        self.log(f"HDR merge started ({len(paths)} frames) → {out_path}")
        self._hdr_worker = HdrMergeWorker(paths, out_path, align=align, max_dim=max_dim)
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

    def _update_gamut_percent_label(self):
        """Refresh the Color panel gamut % readout after a soft-proof render."""
        if not hasattr(self, "gamut_pct_label"):
            return
        if self.current_path is None or self.proxy_bgr is None:
            self.gamut_pct_label.setText("Gamut shift: —")
            return
        r = self.recipes.get(self.current_path)
        if not r or not r.soft_proof:
            self.gamut_pct_label.setText("Gamut shift: — (soft proof off)")
            return
        try:
            from imaging import apply_soft_proof
            meta = self.meta_cache.get(self.current_path, {})
            # Compare unproofed baseline (recipe with soft_proof forced off) is expensive;
            # instead measure proof vs current proxy as stored in last render path.
            src = self.proxy_bgr.astype(np.float32) / 255.0
            # Apply non-proof parts lightly: use apply_soft_proof only on source
            _out, stats = apply_soft_proof(
                src,
                r.soft_proof_profile,
                gamut_warning=False,
                paper_white=getattr(r, "soft_proof_paper_white", False),
                icc_path=getattr(r, "soft_proof_icc_path", "") or "",
                intent=getattr(r, "soft_proof_intent", "relative") or "relative",
                return_stats=True,
            )
            pct = float(stats.get("gamut_percent") or 0.0)
            method = stats.get("method") or "?"
            self.gamut_pct_label.setText(f"Gamut shift: {pct:.1f}%  ·  {method}")
        except Exception as e:
            self.gamut_pct_label.setText(f"Gamut shift: (error: {e})")

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
            self.brush_list.addItem(f"Brush {i+1}{inv}  ({n} dabs, exp {m.get('exposure', 0):+.2f})")
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
        if not ok:
            return
        m = masks[self.selected_brush_index]
        for key, row in self.brush_sliders.items():
            row.blockSignals(True)
            if key in ("flow", "opacity"):
                row.set_value(float(m.get(key, 1.0)) * 100.0)
            else:
                row.set_value(float(m.get(key, 0.0)))
            row.blockSignals(False)

    def _on_brush_adj(self, key, val):
        if self.current_path is None or getattr(self, "selected_brush_index", -1) < 0:
            return
        masks = self.recipes[self.current_path].brush_masks
        if not masks or self.selected_brush_index >= len(masks):
            return
        # flow / opacity stored 0..1 internally
        if key in ("flow", "opacity"):
            masks[self.selected_brush_index][key] = float(val) / 100.0
        else:
            masks[self.selected_brush_index][key] = float(val)
        self.preview.set_brush_masks(masks, self.selected_brush_index)
        self._update_brush_list()
        self.render_timer.start()

    def _on_brush_size(self, val):
        # val is 1..30 percent of image
        self.preview.brush_radius = max(0.005, float(val) / 100.0)

    def _on_brush_hard(self, val):
        self.preview.brush_hardness = float(val) / 100.0

    def _on_brush_flow(self, val):
        self.preview.brush_flow = max(0.05, float(val) / 100.0)

    def _on_brush_opacity(self, val):
        self.preview.brush_opacity = max(0.05, float(val) / 100.0)

    def _on_brush_paint_mode(self, _idx=None):
        if not hasattr(self, "brush_mode_combo"):
            return
        mode = self.brush_mode_combo.currentData() or "add"
        self.preview.brush_paint_mode = mode
        # Eraser checkbox mirrors subtract
        if hasattr(self, "brush_erase_cb") and mode == "subtract":
            self.brush_erase_cb.blockSignals(True)
            self.brush_erase_cb.setChecked(True)
            self.brush_erase_cb.blockSignals(False)
            self.preview.brush_erase = True
        elif hasattr(self, "brush_erase_cb") and mode == "add":
            self.brush_erase_cb.blockSignals(True)
            self.brush_erase_cb.setChecked(False)
            self.brush_erase_cb.blockSignals(False)
            self.preview.brush_erase = False

    def _on_auto_subject_mask(self):
        """Create a new brush mask from offline GrabCut subject detection."""
        if self.current_path is None or self.original_bgr is None:
            QMessageBox.information(self, "Subject mask", "Open an image first.")
            return
        try:
            from imaging import generate_subject_mask
            src = self.proxy_bgr if self.proxy_bgr is not None else self.original_bgr
            mask = generate_subject_mask(src)
            if mask is None or float(mask.max()) < 0.01:
                QMessageBox.warning(self, "Subject mask", "Could not detect a subject.")
                return
            entry = {
                "strokes": [],
                "raster": mask,  # float 0..1 at proxy size; apply_brush resizes as needed
                "hardness": 0.8,
                "flow": 1.0,
                "opacity": float(getattr(self.preview, "brush_opacity", 1.0)),
                "mode": "add",
                "exposure": 0.0, "contrast": 0.0, "saturation": 0.0,
                "clarity": 0.0, "temperature": 0.0,
            }
            masks = list(self.recipes[self.current_path].brush_masks or [])
            masks.append(entry)
            self.recipes[self.current_path].brush_masks = masks
            self.selected_brush_index = len(masks) - 1
            self.preview.set_brush_masks(masks, self.selected_brush_index)
            self._update_brush_list()
            self._sync_brush_sliders()
            self._push_history("Auto subject mask")
            self.render_preview()
            self.statusBar().showMessage("Subject mask added — adjust exposure/etc. on the new brush entry")
        except Exception as e:
            QMessageBox.warning(self, "Subject mask", str(e))

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
        if checked:
            self._probe_lensfun(silent=True)
        self._schedule_history("Lensfun")
        self.render_timer.start()

    def _probe_lensfun(self, silent=False):
        meta = self.meta_cache.get(self.current_path, {}) if self.current_path else {}
        from imaging import probe_lensfun
        info = probe_lensfun(meta)
        msg = info.get("message") or "—"
        if hasattr(self, "lensfun_status"):
            color = "#6d6" if info.get("lens_match") else ("#fc6" if info.get("installed") else "#c88")
            self.lensfun_status.setStyleSheet(f"color:{color}; font-size:11px;")
            self.lensfun_status.setText(msg)
        if not silent:
            QMessageBox.information(
                self, "Lensfun match",
                f"{msg}\n\n"
                f"EXIF camera: {info.get('camera_query') or '—'}\n"
                f"EXIF lens: {info.get('lens_query') or '—'}\n"
                f"Matched camera: {info.get('camera_match') or '—'}\n"
                f"Matched lens: {info.get('lens_match') or '—'}\n"
                f"Database: {info.get('db_path') or '—'}\n\n"
                "Install: pip install lensfunpy.\n"
                "Place the Lensfun XML DB in photo_lab/lensfun/ "
                "(or lensfun/data/db/) next to the app.",
            )
        return info

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

    def _on_lib_save_people(self):
        paths = self._lib_selected_paths()
        if not paths:
            self.statusBar().showMessage("Select library photo(s) to tag people")
            return
        people = self.lib_people_edit.text().strip() if hasattr(self, "lib_people_edit") else ""
        for path in paths:
            self.catalog.set_people(path, people)
        self.statusBar().showMessage(f"People tags saved on {len(paths)} photo(s)")
        if hasattr(self, "lib_search") and self.lib_search.text().strip():
            self._on_lib_search()

    def _refresh_collections_combo(self):
        if not hasattr(self, "lib_collections"):
            return
        cur_id = self.lib_collections.currentData()
        self.lib_collections.blockSignals(True)
        self.lib_collections.clear()
        self.lib_collections.addItem("— Collections —", None)
        try:
            for c in self.catalog.list_collections():
                self.lib_collections.addItem(f"{c['name']} ({c.get('count', 0)})", c["id"])
        except Exception:
            log.debug("_refresh_collections_combo: non-critical failure, continuing", exc_info=True)
        if cur_id is not None:
            for i in range(self.lib_collections.count()):
                if self.lib_collections.itemData(i) == cur_id:
                    self.lib_collections.setCurrentIndex(i)
                    break
        self.lib_collections.blockSignals(False)

    def _lib_new_collection(self):
        name, ok = QInputDialog.getText(self, "New collection", "Collection name:")
        if not ok or not (name or "").strip():
            return
        cid = self.catalog.create_collection(name.strip())
        self._refresh_collections_combo()
        for i in range(self.lib_collections.count()):
            if self.lib_collections.itemData(i) == cid:
                self.lib_collections.setCurrentIndex(i)
                break
        self.statusBar().showMessage(f"Collection “{name.strip()}” created")

    def _lib_delete_collection(self):
        cid = self.lib_collections.currentData() if hasattr(self, "lib_collections") else None
        if cid is None:
            self.statusBar().showMessage("Select a collection to delete")
            return
        name = self.lib_collections.currentText()
        if QMessageBox.question(
            self, "Delete collection",
            f"Delete collection “{name}”? (Photos stay in the catalog.)",
        ) != QMessageBox.StandardButton.Yes:
            return
        self.catalog.delete_collection(int(cid))
        self._refresh_collections_combo()
        self.statusBar().showMessage("Collection deleted")

    def _lib_add_to_collection(self):
        cid = self.lib_collections.currentData() if hasattr(self, "lib_collections") else None
        if cid is None:
            self.statusBar().showMessage("Select or create a collection first")
            return
        paths = self._lib_selected_paths()
        if not paths and self.current_path:
            paths = [self.current_path]
        if not paths:
            self.statusBar().showMessage("Select photo(s) to add")
            return
        self.catalog.add_to_collection(int(cid), paths)
        self._refresh_collections_combo()
        self.statusBar().showMessage(f"Added {len(paths)} photo(s) to collection")

    def _on_lib_collection_changed(self, _idx=None):
        cid = self.lib_collections.currentData() if hasattr(self, "lib_collections") else None
        if cid is None:
            return
        include_rej = bool(self.lib_filter_rejected.isChecked())
        recs = self.catalog.images_in_collection(int(cid), include_rejected=include_rej)
        self.lib_heading.setText(f"Collection — {len(recs)} photo(s)")
        self._lib_records = recs
        self.lib_grid.clear()
        for rec in recs:
            label = rec.get("filename") or os.path.basename(rec["path"])
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

    def _lib_find_duplicates(self):
        groups = self.catalog.find_duplicate_groups()
        if not groups:
            QMessageBox.information(self, "Duplicates", "No duplicate groups found.\n\n"
                                    "Duplicates use a content fingerprint built at scan time. "
                                    "Re-scan folders if hashes are missing.")
            return
        # Flatten first groups into grid for review
        flat = []
        for g in groups[:50]:
            flat.extend(g)
        self.lib_heading.setText(f"Duplicates — {len(groups)} group(s), showing {len(flat)} file(s)")
        self._lib_records = flat
        self.lib_grid.clear()
        for rec in flat:
            label = rec.get("filename") or os.path.basename(rec["path"])
            item = QListWidgetItem(f"⧉ {label}")
            item.setData(Qt.ItemDataRole.UserRole, rec["path"])
            item.setToolTip(f"{rec['path']}\nhash={rec.get('content_hash')}")
            item.setSizeHint(QSize(150, 170))
            self.lib_grid.addItem(item)
        if self._lib_thumb_worker and self._lib_thumb_worker.isRunning():
            self._lib_thumb_worker.cancel()
        from workers import CatalogThumbWorker
        self._lib_thumb_worker = CatalogThumbWorker(flat, size=140)
        self._lib_thumb_worker.thumb_ready.connect(self._on_lib_thumb_ready)
        self._lib_thumb_worker.start()
        self.statusBar().showMessage(f"Found {len(groups)} duplicate group(s)")

    def _create_virtual_copy(self):
        if self.current_path is None:
            self.statusBar().showMessage("Open an image in Develop first")
            return
        from imaging import recipe_to_dict
        import json
        r = self.recipes.get(self.current_path)
        recipe_json = ""
        try:
            recipe_json = json.dumps(recipe_to_dict(r) if r else {})
        except Exception:
            recipe_json = "{}"
        n = len(self.catalog.list_virtual_copies(self.current_path)) + 1
        name = f"Copy {n}"
        vc_id = self.catalog.create_virtual_copy(self.current_path, name=name, recipe_json=recipe_json)
        # Also write a sidecar variant path marker in catalog only — edits stay on master
        # until a dedicated VC editor path is built; store recipe for later restore.
        self.statusBar().showMessage(f"Virtual copy “{name}” (id {vc_id}) created from current recipe")
        self.log(f"Virtual copy {vc_id} for {self.current_path}")

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
            if hasattr(self, "act_grad"):
                self.act_grad.setChecked(False)
            self.preview.setCursor(Qt.CursorShape.CrossCursor)
            self.statusBar().showMessage("WB Picker: click on a neutral gray/white area")
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
            if hasattr(self, "act_wb_pick"):
                self.act_wb_pick.setChecked(False)
            self.preview.local_mode = False
            if hasattr(self, "act_local"):
                self.act_local.setChecked(False)
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
            sharpen_detail=15, output_sharpen=5, protect_skin=55,
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

    def save_preset(self):
        if self.current_path is None:
            return
        start = ensure_plugin_dir()
        suggested = os.path.join(start, "preset.json")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Preset", suggested, "JSON (*.json)"
        )
        if path:
            try:
                self.recipes[self.current_path].save_json(path)
                self.statusBar().showMessage(f"Preset saved → {path}")
            except Exception as e:
                QMessageBox.warning(self, "Save Preset", str(e))

    def load_preset(self):
        if self.current_path is None:
            QMessageBox.information(self, "Load Preset", "Open an image first.")
            return
        start_dir = plugin_dir()
        if not os.path.isdir(start_dir):
            start_dir = ""
        path, selected_filter = QFileDialog.getOpenFileName(
            self,
            "Load Preset",
            start_dir,
            "All Presets (*.xmp *.json);;Lightroom XMP (*.xmp);;PhotoLab JSON (*.json);;All (*.*)",
        )
        if not path:
            return
        try:
            r = load_preset_file(path, base=None)
            self.recipes[self.current_path] = r
            self.sync_sliders_to_recipe()
            self._push_history(f"Preset: {os.path.basename(path)}")
            self.render_preview()
            self.statusBar().showMessage(f"Preset loaded ← {os.path.basename(path)}")
        except Exception as e:
            QMessageBox.warning(self, "Load Preset", f"Could not load preset:\n{e}")

    def load_preset_folder(self):
        """Import all .xmp/.json presets from a folder and apply the first one (list in status)."""
        if self.current_path is None:
            QMessageBox.information(self, "Import Presets", "Open an image first.")
            return
        start = plugin_dir() if os.path.isdir(plugin_dir()) else ""
        folder = QFileDialog.getExistingDirectory(self, "Choose preset folder (XMP/JSON)", start)
        if not folder:
            return
        files = list_preset_files(folder)
        if not files:
            QMessageBox.information(self, "Import Presets", "No .xmp or .json presets found in that folder.")
            return
        try:
            r = load_preset_file(files[0])
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
                "Use File → Load Preset… to apply others individually.\n\n"
                f"Bundled plugin folder:\n{plugin_dir()}",
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
        elif name == "local_chroma":
            pt["chroma"] = val
        elif name == "local_luma":
            pt["luma"] = val
        elif name == "local_luma_min":
            pt["luma_min"] = val
        elif name == "local_luma_max":
            pt["luma_max"] = val
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
            if "local_chroma" in self.local_sliders:
                self.local_sliders["local_chroma"].set_value(pt.get("chroma", 100.0))
            if "local_luma" in self.local_sliders:
                self.local_sliders["local_luma"].set_value(pt.get("luma", 100.0))
            if "local_luma_min" in self.local_sliders:
                self.local_sliders["local_luma_min"].set_value(pt.get("luma_min", 0.0))
            if "local_luma_max" in self.local_sliders:
                self.local_sliders["local_luma_max"].set_value(pt.get("luma_max", 100.0))
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

    def closeEvent(self, event):
        try:
            self.catalog.close()
        except Exception:
            log.debug("closeEvent: non-critical failure, continuing", exc_info=True)
        super().closeEvent(event)


class _CullingDialog(QDialog):
    """Full-screen-ish culling: navigate, rate, reject, pick with keyboard."""

    def __init__(self, parent, paths, start_index=0):
        super().__init__(parent)
        self.setWindowTitle("Culling Mode — Esc to exit")
        self.setWindowFlag(Qt.WindowType.Window)
        self._paths = list(paths)
        self._index = max(0, min(int(start_index), len(self._paths) - 1))
        self._parent = parent
        self._pix = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        self.hint = QLabel(
            "← → navigate   0–5 rate   X reject   U pick   Space next   Esc exit"
        )
        self.hint.setStyleSheet("color:#aaa; font-size:12px;")
        self.hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.hint)

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_label.setStyleSheet("background:#0a0a0a;")
        self.img_label.setMinimumSize(640, 400)
        layout.addWidget(self.img_label, stretch=1)

        self.info = QLabel("")
        self.info.setStyleSheet("color:#ddd; font-size:13px;")
        self.info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info)

        self.setStyleSheet("background:#111;")
        self.showMaximized()
        self._load_current()

    def current_path(self):
        if 0 <= self._index < len(self._paths):
            return self._paths[self._index]
        return None

    def _load_current(self):
        path = self.current_path()
        if not path:
            self.img_label.setText("No images")
            return
        try:
            from imaging import extract_embedded_preview, _silent_imread
            img = extract_embedded_preview(path, max_side=1600)
            if img is None:
                img = _silent_imread(path)
                if img is not None:
                    h, w = img.shape[:2]
                    scale = min(1.0, 1600 / max(h, w))
                    if scale < 0.999:
                        img = cv2.resize(img, (int(w * scale), int(h * scale)))
            if img is None:
                self.img_label.setText(f"Could not load\n{path}")
                return
            from qt_utils import cv_to_qpixmap
            pm = cv_to_qpixmap(img)
            self._pix = pm
            self._fit_pixmap()
        except Exception as e:
            self.img_label.setText(str(e))
        self._update_info()

    def _fit_pixmap(self):
        if self._pix is None:
            return
        scaled = self._pix.scaled(
            self.img_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.img_label.setPixmap(scaled)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._fit_pixmap()

    def _update_info(self):
        path = self.current_path()
        if not path:
            self.info.setText("")
            return
        parent = self._parent
        rating = 0
        try:
            rating = int(getattr(parent, "_image_ratings", {}).get(path, 0) or 0)
            if hasattr(parent, "catalog"):
                rec = parent.catalog.get_image(path)
                if rec and rec.get("rating") is not None:
                    rating = int(rec.get("rating") or rating)
        except Exception:
            log.debug("_update_info: non-critical failure, continuing", exc_info=True)
        rejected = False
        try:
            if hasattr(parent, "catalog"):
                rec = parent.catalog.get_image(path) or {}
                rejected = bool(rec.get("reject"))
            # develop-side reject may be on recipe
            r = getattr(parent, "recipes", {}).get(path)
            if r is not None and getattr(r, "reject", False):
                rejected = True
        except Exception:
            log.debug("_update_info: non-critical failure, continuing", exc_info=True)
        picked = bool(getattr(parent, "_pick_flags", {}).get(path, False))
        stars = "★" * rating + "☆" * (5 - rating)
        flags = []
        if picked:
            flags.append("PICK")
        if rejected:
            flags.append("REJECT")
        flag_s = ("  ·  " + " ".join(flags)) if flags else ""
        self.info.setText(
            f"{self._index + 1}/{len(self._paths)}  ·  {os.path.basename(path)}  ·  {stars}{flag_s}"
        )
        self.setWindowTitle(f"Culling — {os.path.basename(path)}")

    def _rate(self, stars: int):
        path = self.current_path()
        if not path or self._parent is None:
            return
        try:
            self._parent.rate_current(stars)
        except Exception:
            # rate may require current_path match
            try:
                if hasattr(self._parent, "catalog"):
                    self._parent.catalog.set_rating(path, stars)
                if hasattr(self._parent, "_image_ratings"):
                    self._parent._image_ratings[path] = stars
            except Exception:
                log.debug("_rate: non-critical failure, continuing", exc_info=True)
        self._update_info()

    def _toggle_reject(self):
        path = self.current_path()
        if not path or self._parent is None:
            return
        # Ensure parent current is this path for shared handlers
        try:
            if self._parent.current_path != path:
                # set without full load if possible
                self._parent.current_path = path
        except Exception:
            log.debug("_toggle_reject: non-critical failure, continuing", exc_info=True)
        try:
            self._parent.toggle_reject_current()
        except Exception:
            try:
                rec = self._parent.catalog.get_image(path) or {}
                new_r = not bool(rec.get("reject"))
                self._parent.catalog.set_reject(path, new_r)
            except Exception:
                log.debug("_toggle_reject: non-critical failure, continuing", exc_info=True)
        self._update_info()

    def _toggle_pick(self):
        path = self.current_path()
        if not path or self._parent is None:
            return
        flags = getattr(self._parent, "_pick_flags", None)
        if flags is None:
            return
        flags[path] = not flags.get(path, False)
        self._update_info()

    def keyPressEvent(self, e):
        key = e.key()
        if key in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.accept()
            return
        if key in (Qt.Key.Key_Right, Qt.Key.Key_Space, Qt.Key.Key_Down):
            if self._index < len(self._paths) - 1:
                self._index += 1
                self._load_current()
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Up, Qt.Key.Key_Backspace):
            if self._index > 0:
                self._index -= 1
                self._load_current()
            return
        if key == Qt.Key.Key_X:
            self._toggle_reject()
            return
        if key in (Qt.Key.Key_U, Qt.Key.Key_P):
            self._toggle_pick()
            return
        # 0-5 ratings
        for n, k in enumerate((
            Qt.Key.Key_0, Qt.Key.Key_1, Qt.Key.Key_2, Qt.Key.Key_3, Qt.Key.Key_4, Qt.Key.Key_5,
        )):
            if key == k:
                self._rate(n)
                return
        super().keyPressEvent(e)


# Backward-compatible alias
PhotoStudio = PhotoLab
