"""Launch Focus Stacker Pro GUI (PySide6) bundled with PhotoLab."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent / "focus_stacker_pro"
sys.path.insert(0, str(ROOT / "src"))
from focus_stacker.app import main

if __name__ == "__main__":
    raise SystemExit(main())
