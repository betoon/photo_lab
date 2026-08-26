# Packaging PhotoLab

## Portable build (recommended)

```bash
# Windows
build_portable.bat

# Linux / macOS
chmod +x build_portable.sh
./build_portable.sh
```

Or manually:

```bash
pip install pyinstaller
pyinstaller --noconfirm photo_lab.spec
```

Result: **`dist/PhotoLab/`** — a self-contained folder you can zip and ship.

Bundled with the build:

- `docs/` — user & developer manuals (Help menu)
- `plugin/` — JSON / Lightroom-style presets

## ffmpeg (Panorama to Video)

ffmpeg is **not** embedded (size and redistribution terms). For **Image → Create Pan Video…**:

1. Install [ffmpeg](https://ffmpeg.org/download.html), **or**
2. Place `ffmpeg` / `ffmpeg.exe` on the system `PATH` or next to `PhotoLab.exe`.

## One-file vs one-folder

The default spec uses **one-folder** (faster startup). For a single executable:

```bash
pyinstaller --noconfirm --windowed --name PhotoLab \
  --add-data "docs;docs" --add-data "plugin;plugin" \
  main.py
```

(On macOS/Linux use `docs:docs` with a colon.)

## Check for updates

**Help → Check for Updates…** opens the GitHub Releases page:

https://github.com/betoon/photo_lab/releases

Publish versioned release zips of `dist/PhotoLab` there.
