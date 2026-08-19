"""main.py — entry point. Run with: python main.py"""

import sys
from PyQt6.QtWidgets import QApplication

from main_window import PhotoStudio


def main():
    app = QApplication(sys.argv)
    win = PhotoStudio()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
