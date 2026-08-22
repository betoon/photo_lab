#!/usr/bin/env bash
# Build a portable PhotoLab folder with PyInstaller (Linux / macOS).
set -euo pipefail
cd "$(dirname "$0")"

echo "=== PhotoLab portable build ==="
python3 -c "import PyInstaller" 2>/dev/null || pip install pyinstaller

mkdir -p plugin docs

echo "Building (one-folder)..."
pyinstaller --noconfirm photo_lab.spec

echo
echo "Output: dist/PhotoLab/"
echo "Run: ./dist/PhotoLab/PhotoLab"
echo
echo "Optional: install ffmpeg on PATH for Panorama-to-Video."
echo "Done."
