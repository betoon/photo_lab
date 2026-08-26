@echo off
REM Build a portable PhotoLab folder with PyInstaller (Windows).
REM Prerequisites: Python 3.10+, venv with project deps + pyinstaller.

setlocal
cd /d "%~dp0"

echo === PhotoLab portable build ===
python -c "import PyInstaller, PyQt6, PySide6, cv2, numpy, PIL, rawpy, tifffile" 2>nul
if errorlevel 1 (
  echo ERROR: Build dependencies are incomplete.
  echo Install docs\requirements.txt and PyInstaller in this Python environment.
  exit /b 1
)

if not exist plugin mkdir plugin
if not exist docs mkdir docs

echo Building (one-folder)...
python -m PyInstaller --noconfirm --clean focus_stacker_pro.spec
if errorlevel 1 (
  echo Focus Stacker Pro build failed.
  exit /b 1
)
python -m PyInstaller --noconfirm --clean photo_lab.spec
if errorlevel 1 (
  echo Build failed.
  exit /b 1
)

echo.
echo Output: dist\PhotoLab\
echo Copy that folder anywhere to run PhotoLab.exe
echo.
echo Optional: install ffmpeg and ensure it is on PATH for Panorama-to-Video.
echo   https://ffmpeg.org/download.html
echo.
echo Done.
endlocal
