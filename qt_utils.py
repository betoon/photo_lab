"""qt_utils.py — small Qt/OpenCV interop helpers."""

from PyQt6.QtGui import QImage, QPixmap
import cv2
import numpy as np


def cv_to_qpixmap(img_bgr) -> QPixmap:
    if img_bgr is None:
        return QPixmap()
    if img_bgr.dtype != np.uint8:
        img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())
