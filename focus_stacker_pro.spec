# -*- mode: python ; coding: utf-8 -*-
"""Standalone Focus Stacker Pro companion bundled inside PhotoLab."""
from PyInstaller.utils.hooks import collect_submodules

a = Analysis(
    ["run_focus_stacker_pro.py"],
    pathex=["focus_stacker_pro/src"],
    binaries=[],
    datas=[],
    hiddenimports=collect_submodules("focus_stacker"),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "Cython", "fsspec", "lxml", "matplotlib", "pandas", "pyarrow",
        "pytest", "tkinter", "IPython", "jupyter", "notebook",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="FocusStackerPro",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
    console=False, disable_windowed_traceback=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas, strip=False, upx=True,
    upx_exclude=[], name="FocusStackerPro",
)
