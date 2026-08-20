# PhotoLab Enhancement Roadmap

Tracked suggestions so nothing is lost. Status is updated as items ship.

| # | Area | Suggestion | Status |
|---|------|------------|--------|
| 1 | Preview | Proxy pipeline (persistent downscale) + recipe-hash cache; optional GPU later | **Done** (proxy + cache; GPU deferred) |
| 2 | Filmstrip | Multi-select rating/reject/pick; color labels; solo/compare selected; keyboard nav respects filter | **Done** |
| 3 | History | Persist named snapshots with sidecar; visual before/after per history entry; copy settings from history entry | **Done** |
| 4 | Soft-proof | Real ICC soft-proof when profile present; gamut %; simulate paper white | **Done** |
| 5 | Local | Control-point luminance range; brush flow/opacity; mask subtract/intersect; optional offline AI subject mask | **Done** |
| 6 | Tone/Color | Full parametric + point curve; split-toning; per-channel RGB curves; match exposure/WB across selection | **Done** |
| 7 | Detail | Print/screen sharpen presets from output PPI; dual-illuminant WB UI; protect skin tones | **Done** |
| 8 | Geometry | Interactive 4-corner keystone; auto level from EXIF/edges; clearer Lensfun match UI | **Done** |
| 9 | Catalog | Saved collections/smart collections; face keywords; duplicate detection; virtual copies | **Done** |
| 10 | Import/culling | Import dialog (copy/move/rename/preset); full-screen culling mode | **Done** |
| 11 | Batch | Apply preset to selected; batch rename/move; export queue on disk | Pending |
| 12 | Stack/Pano | Alignment confidence UI; exclude bad frames; exposure/WB match before stitch; order-by-time helper | Pending |
| 13 | HDR | Bracket detection from EXIF; deghosting strength; optional Debevec path | Pending |
| 14 | Pan Video | Preview-only re-render; ETA; pass current recipe grade from PhotoLab | Pending |
| 15 | Diagnostics | Exportable problem report (debug log + system info); clearer RAW failure messages | Pending |
| 16 | Perf/memory | Clear cache command; limit concurrent workers; optional 16-bit internal path | Pending |
| 17 | Packaging | One-click / portable build; optional update check | Pending (foundation: app_paths + docs/plugin data) |
| 18 | Quality | Golden-image tests for apply_recipe; Recipe/catalog unit tests | Pending |

## Extra (this session)

| Item | Status |
|------|--------|
| plugin/ folder for JSON + Lightroom XMP presets | **Done** |
| docs/ packaged path via app_paths | **Done** |
| Sample presets neutral.json, vivid.json | **Done** |

## Notes for #3

- Sidecar schema: { version, image, recipe, snapshots: [{name, recipe, ts}] }
- History context menu: Preview before/after, Copy settings
- app_paths.plugin_dir / docs_dir work under PyInstaller _MEIPASS

When implementing the next item, mark its status here and update USER/DEVELOPER manuals.
