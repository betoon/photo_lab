@echo off
REM Build a portable PhotoLab folder with PyInstaller (Windows).
REM Prerequisites: Python 3.10+, venv with project deps + pyinstaller.

setlocal
cd /d "%~dp0"

echo === PhotoLab portable build ===
python -c "import PyInstaller" 2>nul
if errorlevel 1 (
  echo Installing PyInstaller...
  pip install pyinstaller
)

if not exist plugin mkdir plugin
if not exist docs mkdir docs

echo Building (one-folder)...
pyinstaller --noconfirm photo_lab.spec
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
