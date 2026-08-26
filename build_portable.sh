#!/usr/bin/env bash
# Build a portable PhotoLab folder with PyInstaller (Linux / macOS).
set -euo pipefail
cd "$(dirname "$0")"

echo "=== PhotoLab portable build ==="
python3 -c "import PyInstaller, PyQt6, PySide6, cv2, numpy, PIL, rawpy, tifffile" 2>/dev/null || {
  echo "ERROR: Build dependencies are incomplete." >&2
  echo "Install docs/requirements.txt and PyInstaller in this Python environment." >&2
  exit 1
}

mkdir -p plugin docs

echo "Building (one-folder)..."
python3 -m PyInstaller --noconfirm --clean focus_stacker_pro.spec
python3 -m PyInstaller --noconfirm --clean photo_lab.spec

echo
echo "Output: dist/PhotoLab/"
echo "Run: ./dist/PhotoLab/PhotoLab"
echo
echo "Optional: install ffmpeg on PATH for Panorama-to-Video."
echo "Done."
