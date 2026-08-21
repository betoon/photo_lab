"""main.py — entry point. Run with: python main.py"""

import sys
import os
import logging
import traceback

from logging_setup import configure_logging, current_log_path

log = logging.getLogger(__name__)


def _excepthook(exc_type, exc, tb):
    # Keep printing to stderr for anyone running from a console, but also
    # make sure crashes land in the rotating log file — previously an
    # unhandled exception was only ever visible if the user happened to be
    # watching the terminal at the time.
    traceback.print_exception(exc_type, exc, tb)
    log.critical(
        "Unhandled exception", exc_info=(exc_type, exc, tb),
    )


sys.excepthook = _excepthook


def main():
    configure_logging()
    log.info("PhotoLab starting (log file: %s)", current_log_path())
    os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
    os.environ.setdefault("QT_FONT_DPI", "96")
    try:
        import cv2
        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(3)
    except Exception:
        log.debug("Could not set OpenCV log level", exc_info=True)

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont
    from PyQt6.QtCore import Qt
    from main_window import PhotoLab

    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )
    except Exception:
        log.debug("Could not set HiDPI rounding policy", exc_info=True)

    app = QApplication(sys.argv)
    app.setApplicationName("PhotoLab")
    font = QFont()
    font.setFamilies(["Segoe UI", "Arial", "Helvetica", "sans-serif"])
    font.setPointSize(10)
    if font.pointSize() <= 0:
        font.setPixelSize(13)
    app.setFont(font)

    win = PhotoLab()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
