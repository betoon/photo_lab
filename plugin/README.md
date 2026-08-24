# PhotoLab plugin / presets folder

Place **PhotoLab JSON** (`.json`) and **Adobe Lightroom / Camera Raw XMP** (`.xmp`) develop presets here.

## Bundled film-rendering library

The preset browser includes 30 original PhotoLab film interpretations grouped into:

- **Film - Color Negative** — portrait, consumer, editorial, cinema-negative, faded, and high-speed color.
- **Film - Slide and Cinema** — neutral and vivid chrome, classic reversal, bleach bypass, and cinematic color separation.
- **Film - Black and White** — fine-grain, documentary, high-speed, virtual color-filter, orthochromatic, and infrared-like renderings.

These are original aesthetic interpretations built from PhotoLab's own controls. They do not contain or copy proprietary DxO FilmPack profiles, manufacturer profiles, measured film data, or third-party LUTs. All preserve the image's as-shot white balance; use the preset Strength control for a subtler rendering.

To regenerate the bundled JSON files after maintaining the definitions, run:

```text
python tools/generate_film_preset_library.py
```

PhotoLab looks for this folder:

1. Next to the app (or inside a frozen build’s bundle) — `plugin/`
2. Current working directory — `./plugin/`
3. User override — `~/.photolab/plugin/`

**File → Load Preset…** defaults to this folder when it exists.  
**File → Import Preset Folder…** can point at any other directory.

## Shipping with an executable

Include this directory in your packager data files, for example PyInstaller:

```text
--add-data "plugin:plugin"
--add-data "docs:docs"
```

See `docs/DEVELOPER_MANUAL.md` → Packaging.
