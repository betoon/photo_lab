"""slideshow.py — fullscreen slideshow with optional Ken Burns motion."""
from __future__ import annotations

import os
from typing import List, Optional

import cv2
import numpy as np

from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QPainter, QImage, QKeyEvent, QColor, QFont
from PyQt6.QtWidgets import QWidget, QApplication


def _to_qimage(bgr: np.ndarray) -> QImage:
    if bgr is None:
        return QImage()
    if bgr.dtype != np.uint8:
        bgr = np.clip(bgr, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


class SlideshowWindow(QWidget):
    """Fullscreen slideshow. Keys: Space pause, Left/Right, Esc exit, K Ken Burns."""

    def __init__(
        self,
        paths: List[str],
        load_fn,
        interval_ms: int = 4000,
        ken_burns: bool = True,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("PhotoLab Slideshow")
        self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setStyleSheet("background:#000;")
        self.paths = list(paths)
        self.load_fn = load_fn  # path -> BGR uint8
        self.idx = 0
        self.paused = False
        self.ken_burns = ken_burns
        self._pix = None
        self._t = 0.0  # 0..1 progress within slide
        self.interval_ms = max(1500, int(interval_ms))

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._anim = QTimer(self)
        self._anim.timeout.connect(self._animate)
        self._anim.start(33)

        self.showFullScreen()
        self._show_current()
        self._timer.start(self.interval_ms)

    def _show_current(self):
        if not self.paths:
            return
        path = self.paths[self.idx % len(self.paths)]
        try:
            img = self.load_fn(path)
            self._pix = _to_qimage(img)
        except Exception:
            self._pix = QImage()
        self._t = 0.0
        self.setWindowTitle(f"Slideshow — {os.path.basename(path)} ({self.idx+1}/{len(self.paths)})")
        self.update()

    def _tick(self):
        if self.paused or not self.paths:
            return
        self.idx = (self.idx + 1) % len(self.paths)
        self._show_current()

    def _animate(self):
        if self.paused or not self.ken_burns:
            return
        self._t = min(1.0, self._t + 33.0 / self.interval_ms)
        self.update()

    def paintEvent(self, _e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0))
        if self._pix is None or self._pix.isNull():
            p.setPen(QColor(180, 180, 180))
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No image")
            p.end()
            return
        ww, wh = self.width(), self.height()
        iw, ih = self._pix.width(), self._pix.height()
        # Ken Burns: slow zoom 1.0 → 1.12 and slight pan
        zoom = 1.0 + 0.12 * self._t if self.ken_burns else 1.0
        scale = max(ww / iw, wh / ih) * zoom
        dw, dh = iw * scale, ih * scale
        pan_x = (ww - dw) / 2.0 - (self._t * 0.04 * dw if self.ken_burns else 0)
        pan_y = (wh - dh) / 2.0 + (self._t * 0.02 * dh if self.ken_burns else 0)
        target = QRectF(pan_x, pan_y, dw, dh)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        p.drawImage(target, self._pix)
        # Caption
        p.setPen(QColor(220, 220, 220))
        font = QFont("Segoe UI", 11)
        p.setFont(font)
        name = os.path.basename(self.paths[self.idx % len(self.paths)])
        p.drawText(16, wh - 24, f"{name}   ·   {self.idx+1}/{len(self.paths)}   ·   Space pause  ·  Esc exit")
        p.end()

    def keyPressEvent(self, e: QKeyEvent):
        k = e.key()
        if k == Qt.Key.Key_Escape:
            self.close()
        elif k == Qt.Key.Key_Space:
            self.paused = not self.paused
            if self.paused:
                self._timer.stop()
            else:
                self._timer.start(self.interval_ms)
            self.update()
        elif k == Qt.Key.Key_Right:
            self.idx = (self.idx + 1) % max(len(self.paths), 1)
            self._show_current()
            self._timer.start(self.interval_ms)
        elif k == Qt.Key.Key_Left:
            self.idx = (self.idx - 1) % max(len(self.paths), 1)
            self._show_current()
            self._timer.start(self.interval_ms)
        elif k == Qt.Key.Key_K:
            self.ken_burns = not self.ken_burns
        else:
            super().keyPressEvent(e)
