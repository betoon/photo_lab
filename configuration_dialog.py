"""Tools > Configuration / INI Editor for local external paths."""
from __future__ import annotations

import os
import subprocess
import sys

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QDialog, QFileDialog, QGridLayout, QHBoxLayout, QLabel, QLineEdit,
    QMessageBox, QPushButton, QScrollArea, QVBoxLayout, QWidget,
)

from config import get_config, reload_config, user_ini_path
from external_paths import PATH_SPECS, resolve_path, validate_path


class ConfigurationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configuration / INI Editor")
        self.resize(850, 500)
        self._edits = {}
        self._statuses = {}

        root = QVBoxLayout(self)
        intro = QLabel(
            "Configure optional programs and data folders on this computer. "
            "Leave a value empty to use automatic discovery. Invalid or missing paths are safe."
        )
        intro.setWordWrap(True)
        root.addWidget(intro)

        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        panel = QWidget(); grid = QGridLayout(panel)
        grid.setColumnStretch(1, 1)
        for row, spec in enumerate(PATH_SPECS):
            label = QLabel(f"<b>{spec.label}</b><br><span style='color:#999'>{spec.description}</span>")
            label.setWordWrap(True)
            edit = QLineEdit(get_config().path(spec.key)); edit.setPlaceholderText("Automatic discovery")
            browse = QPushButton("Browse…")
            browse.clicked.connect(lambda _=False, s=spec, e=edit: self._browse(s, e))
            status = QLabel(); status.setWordWrap(True)
            test = QPushButton("Test / Validate")
            test.clicked.connect(lambda _=False, s=spec, e=edit: self._validate(s.key, e.text()))
            reset = QPushButton("Reset to Default")
            reset.clicked.connect(lambda _=False, s=spec, e=edit: self._reset(s.key, e))
            opened = QPushButton("Open Folder")
            opened.clicked.connect(lambda _=False, s=spec, e=edit: self._open_folder(s.key, e.text()))
            actions = QHBoxLayout(); actions.addWidget(test); actions.addWidget(opened); actions.addWidget(reset); actions.addStretch(1)
            grid.addWidget(label, row * 3, 0, 2, 1)
            grid.addWidget(edit, row * 3, 1)
            grid.addWidget(browse, row * 3, 2)
            grid.addWidget(status, row * 3 + 1, 1, 1, 2)
            grid.addLayout(actions, row * 3 + 2, 1, 1, 2)
            self._edits[spec.key] = edit; self._statuses[spec.key] = status
            edit.textChanged.connect(lambda text, key=spec.key: self._validate(key, text))
            self._validate(spec.key, edit.text())
        scroll.setWidget(panel); root.addWidget(scroll, 1)

        location = QLabel(f"INI file: {user_ini_path()}"); location.setTextInteractionFlags(location.textInteractionFlags())
        reveal = QPushButton("Reveal INI File Location")
        reveal.clicked.connect(self._reveal_ini)
        bottom = QHBoxLayout(); bottom.addWidget(location, 1); bottom.addWidget(reveal)
        root.addLayout(bottom)
        buttons = QHBoxLayout(); buttons.addStretch(1)
        apply_btn = QPushButton("Apply"); apply_btn.setDefault(True); apply_btn.clicked.connect(self._apply)
        close_btn = QPushButton("Close"); close_btn.clicked.connect(self.reject)
        buttons.addWidget(apply_btn); buttons.addWidget(close_btn); root.addLayout(buttons)

    def _browse(self, spec, edit):
        start = edit.text().strip() or resolve_path(spec.key) or os.path.expanduser("~")
        if spec.kind == "folder":
            chosen = QFileDialog.getExistingDirectory(self, spec.label, start)
        else:
            chosen, _ = QFileDialog.getOpenFileName(
                self, spec.label, start, "Applications and launchers (*.exe *.py);;All files (*)"
            )
            if not chosen:
                chosen = QFileDialog.getExistingDirectory(self, f"Choose {spec.label} folder", start)
        if chosen:
            edit.setText(os.path.normpath(chosen))

    def _validate(self, key, value):
        ok, message = validate_path(key, value)
        status = self._statuses[key]
        status.setText(("✓ " if ok else "⚠ ") + message)
        status.setStyleSheet("color:#43c96b;" if ok else "color:#e6a23c;")
        return ok

    def _reset(self, key, edit):
        edit.clear()
        self._validate(key, "")

    def _open_folder(self, key, value):
        path = value.strip() or resolve_path(key)
        if os.path.isfile(path):
            path = os.path.dirname(path)
        if path and os.path.isdir(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            QMessageBox.information(self, "Open Folder", "No existing folder is available for this entry.")

    def _reveal_ini(self):
        path = get_config().ensure_user_ini()
        QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.dirname(path)))

    def _apply(self):
        cfg = get_config()
        for key, edit in self._edits.items():
            cfg.set("paths", key, edit.text().strip())
        try:
            path = cfg.save_user()
            reload_config()
        except OSError as exc:
            QMessageBox.warning(self, "Configuration", f"Could not save the INI file:\n{exc}")
            return
        for key, edit in self._edits.items():
            self._validate(key, edit.text())
        if self.parent() and hasattr(self.parent(), "statusBar"):
            self.parent().statusBar().showMessage(f"Configuration applied → {path}", 8000)
        QMessageBox.information(self, "Configuration", f"Settings applied and saved to:\n{path}")
