# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for PhotoLab (one-folder portable build).

Build (from the project root, with your venv active):

    pip install pyinstaller
    pyinstaller photo_lab.spec

Output: dist/PhotoLab/  — copy that folder anywhere (portable).
Optional: zip dist/PhotoLab for distribution.

Notes
-----
* docs/ and plugin/ are bundled via datas so Help and presets work offline.
* ffmpeg is NOT bundled (license / size). Install system-wide or place
  ffmpeg.exe next to PhotoLab.exe for Panorama-to-Video.
* ArgyllCMS is NOT bundled. Color Calibration Studio locates an external install.
* One-file builds work but start slower; one-folder is recommended.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_submodules
import os

block_cipher = None

# Optional: pull OpenCV/numpy data if hooks miss something
# datas += collect_data_files('cv2')

datas = [
    ("docs", "docs"),
    ("plugin", "plugin"),
]

# Include optional README next to exe
if os.path.isfile("README.md"):
    datas.append(("README.md", "."))
if os.path.isfile("PACKAGING.md"):
    datas.append(("PACKAGING.md", "."))

hiddenimports = [
    "rawpy",
    "PIL",
    "PIL.Image",
    "PIL.ImageCms",
    "numpy",
    "cv2",
    "PyQt6",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "logging_setup",
    "app_paths",
    "imaging",
    "catalog",
    "workers",
    "widgets",
    "presets",
    "focus_stack",
    "panorama",
    "pano_video",
    "audio_editor",
    "qt_utils",
    "color_calibration",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="PhotoLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # windowed app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PhotoLab",
)
