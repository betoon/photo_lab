
## Preferences & photolab.ini

**File → Preferences…** (`Ctrl+,`) edits the user config file:

`~/.photolab/photolab.ini` (Windows: `%USERPROFILE%\.photolab\photolab.ini`)

Sections: paths (plugin, docs, lensfun, ffmpeg, catalog, thumbs, export, scripts), performance, UI, licensing, integrations.

Empty path values mean automatic defaults. Template: `photolab.ini.example` in the app folder.

Environment overrides (optional): `PHOTOLAB_API_KEY`, `PHOTOLAB_SERIAL`, `PHOTOLAB_PLUGIN_DIR`, `PHOTOLAB_FFMPEG`, `PHOTOLAB_CATALOG_DB`, `PHOTOLAB_LENSFUN_DIR`.

Do not commit real API keys or serial numbers to git.
