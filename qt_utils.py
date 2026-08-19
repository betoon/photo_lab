"""qt_utils.py — small Qt/OpenCV interop helpers."""

from PyQt6.QtGui import QImage, QPixmap
import cv2


def cv_to_qpixmap(img_bgr) -> QPixmap:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg.copy())
