"""workers.py — background QThreads so the UI never blocks on I/O or the
image pipeline (thumbnail generation, full-resolution export)."""

import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap

from imaging import apply_recipe
from qt_utils import cv_to_qpixmap


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
