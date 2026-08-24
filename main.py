"""main.py — entry point. Run with: python main.py"""

import sys
import os
import traceback


def _excepthook(exc_type, exc, tb):
    traceback.print_exception(exc_type, exc, tb)


sys.excepthook = _excepthook


def main():
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
    os.environ.setdefault("QT_FONT_DPI", "96")
    try:
        import cv2
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(3)
    except Exception:
        pass

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont
    from PyQt6.QtCore import Qt

    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("PhotoLab")
    font = QFont("Segoe UI", 10)
    if font.pointSize() <= 0:
        font.setPixelSize(13)
    app.setFont(font)

    # Import widgets only after QApplication and its valid default font exist.
    from main_window import PhotoLab

    win = PhotoLab()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
