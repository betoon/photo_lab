# PhotoLab plugin / presets folder

Place **PhotoLab JSON** (`.json`) and **Adobe Lightroom / Camera Raw XMP** (`.xmp`) develop presets here.

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
