"""print_dialog.py — soft-proof aware print / PDF export for PhotoLab."""
from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPainter, QPageSize, QPageLayout
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QFormLayout, QComboBox, QCheckBox, QDialogButtonBox,
    QFileDialog, QMessageBox, QLabel, QDoubleSpinBox,
)
from PyQt6.QtPrintSupport import QPrinter, QPrintDialog


# Paper sizes in inches (width, height) portrait
_PAPER = {
    "Letter": (8.5, 11.0),
    "Legal": (8.5, 14.0),
    "A4": (8.27, 11.69),
    "A3": (11.69, 16.54),
    "4x6": (4.0, 6.0),
    "5x7": (5.0, 7.0),
    "8x10": (8.0, 10.0),
}


def _bgr_to_qimage(bgr: np.ndarray) -> QImage:
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()


def render_print_image(
    img_bgr: np.ndarray,
    recipe,
    soft_proof: bool = True,
    profile: str = "sRGB",
) -> np.ndarray:
    """Apply recipe (and optional soft proof) for print output."""
    from imaging import apply_recipe, Recipe
    r = recipe if recipe is not None else Recipe()
    # Clone soft-proof flags without mutating user recipe permanently is caller's job
    return apply_recipe(img_bgr, r)


class PrintDialog(QDialog):
    def __init__(self, parent, img_bgr, recipe, default_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Print / PDF")
        self.img_bgr = img_bgr
        self.recipe = recipe
        self.default_path = default_path or "print_output.pdf"

        form = QFormLayout(self)
        self.paper = QComboBox()
        for name in _PAPER:
            self.paper.addItem(name)
        self.paper.setCurrentText("Letter")
        form.addRow("Paper size", self.paper)

        self.orient = QComboBox()
        self.orient.addItems(["Portrait", "Landscape"])
        form.addRow("Orientation", self.orient)

        self.dpi = QDoubleSpinBox()
        self.dpi.setRange(72, 600)
        self.dpi.setValue(300)
        form.addRow("Target DPI (PDF raster)", self.dpi)

        self.proof = QCheckBox("Apply current soft-proof settings from Develop")
        self.proof.setChecked(True)
        form.addRow(self.proof)

        tip = QLabel(
            "Renders the developed image to PDF or sends it to the system printer.\n"
            "For ink/paper soft-proof, enable Soft Proof on the Color tab first."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#888; font-size:11px;")
        form.addRow(tip)

        buttons = QDialogButtonBox()
        pdf_btn = buttons.addButton("Save PDF…", QDialogButtonBox.ButtonRole.AcceptRole)
        print_btn = buttons.addButton("System Print…", QDialogButtonBox.ButtonRole.ActionRole)
        cancel = buttons.addButton(QDialogButtonBox.StandardButton.Cancel)
        pdf_btn.clicked.connect(self._save_pdf)
        print_btn.clicked.connect(self._system_print)
        cancel.clicked.connect(self.reject)
        form.addRow(buttons)

    def _prepared_image(self) -> np.ndarray:
        from imaging import Recipe
        r = self.recipe
        if r is None:
            r = Recipe()
        # Soft proof already part of recipe when enabled in Develop
        return render_print_image(self.img_bgr, r)

    def _page_size(self) -> QPageSize:
        name = self.paper.currentText()
        mapping = {
            "Letter": QPageSize.PageSizeId.Letter,
            "Legal": QPageSize.PageSizeId.Legal,
            "A4": QPageSize.PageSizeId.A4,
            "A3": QPageSize.PageSizeId.A3,
        }
        if name in mapping:
            return QPageSize(mapping[name])
        # Custom inches → points
        w_in, h_in = _PAPER.get(name, (8.5, 11.0))
        from PyQt6.QtCore import QSizeF
        return QPageSize(QSizeF(w_in * 25.4, h_in * 25.4), QPageSize.Unit.Millimeter)

    def _save_pdf(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF", self.default_path, "PDF (*.pdf)"
        )
        if not path:
            return
        try:
            img = self._prepared_image()
            qimg = _bgr_to_qimage(img)
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setOutputFormat(QPrinter.OutputFormat.PdfFormat)
            printer.setOutputFileName(path)
            printer.setPageSize(self._page_size())
            layout = QPageLayout(
                self._page_size(),
                QPageLayout.Orientation.Landscape
                if self.orient.currentText() == "Landscape"
                else QPageLayout.Orientation.Portrait,
                printer.pageLayout().margins(),
            )
            printer.setPageLayout(layout)
            painter = QPainter(printer)
            page = printer.pageRect(QPrinter.Unit.DevicePixel)
            # Fit image into page preserving aspect
            scaled = qimg.scaled(
                int(page.width()), int(page.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = int((page.width() - scaled.width()) / 2)
            y = int((page.height() - scaled.height()) / 2)
            painter.drawImage(x, y, scaled)
            painter.end()
            QMessageBox.information(self, "Print / PDF", f"Saved:\n{path}")
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "Print / PDF", str(e))

    def _system_print(self):
        try:
            img = self._prepared_image()
            qimg = _bgr_to_qimage(img)
            printer = QPrinter(QPrinter.PrinterMode.HighResolution)
            printer.setPageSize(self._page_size())
            dlg = QPrintDialog(printer, self)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                return
            painter = QPainter(printer)
            page = printer.pageRect(QPrinter.Unit.DevicePixel)
            scaled = qimg.scaled(
                int(page.width()), int(page.height()),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            x = int((page.width() - scaled.width()) / 2)
            y = int((page.height() - scaled.height()) / 2)
            painter.drawImage(x, y, scaled)
            painter.end()
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "Print", str(e))
