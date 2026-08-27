# Packaging PhotoLab

AI restoration models are intentionally not bundled. A portable installation may
point to an external model-pack folder through the Configuration / INI Editor. This
keeps the standard build smaller and avoids silently redistributing third-party model
weights with separate licenses. See `docs/AI_MODEL_PACK.md`.

## Portable build (recommended)

```bash
# Windows
build_portable.bat

# Linux / macOS
chmod +x build_portable.sh
./build_portable.sh

# Reproducible Linux x86_64 build from Windows (Docker Desktop)
build_linux_docker.bat
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
- `lensfun/data/db/` — bundled Lensfun correction database
- `focus_stacker_pro/` — bundled Focus Stacker Pro integration
- `photolab.ini.example` — machine-neutral configuration example

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
