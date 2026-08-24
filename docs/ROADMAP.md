# PhotoLab Enhancement Roadmap

Tracked suggestions so nothing is lost. Status is updated as items ship.

| # | Area | Suggestion | Status |
|---|------|------------|--------|
| 1 | Preview | Proxy pipeline (persistent downscale) + recipe-hash cache; optional GPU later | **Done** (proxy + cache; GPU deferred) |
| 2 | Filmstrip | Multi-select rating/reject/pick; color labels; solo/compare selected; keyboard nav respects filter | **Done** |
| 3 | History | Persist named snapshots with sidecar; visual before/after per history entry; copy settings from history entry | **Done** |
| 4 | Soft-proof | Real ICC soft-proof when profile present; gamut %; simulate paper white | **Done** |
| 5 | Local | Luminance/color range, flow/opacity, overlays, feather/edge refinement, subtract/intersect, offline subject mask | **Done** |
| 6 | Tone/Color | Parametric + point curve; split-toning; RGB curves; match exposure/WB | **Done** |
| 7 | Detail | Output-PPI sharpening, dual-illuminant WB, skin protection | **Done** |
| 8 | Geometry | Four-corner keystone, auto level, Lensfun match UI | **Done** |
| 9 | Catalog | Collections, smart filters, people tags, duplicates, virtual copies | **Done** |
| 10 | Import/culling | Import workflow, SD-card import, full-screen culling | **Done** |
| 11 | Batch | Preset application, rename/move, persistent export queue | **Done** |
| 12 | Stack/Pano | Confidence/exclusion, exposure/WB matching, order-by-time | **Done** |
| 13 | HDR | Bracket detection, deghosting, Debevec option | **Done** |
| 14 | Pan Video | Live/test preview, ETA, PhotoLab recipe handoff | **Done** |
| 15 | Diagnostics | Problem report, logs/system info, RAW failure guidance | **Done** |
| 16 | Perf/memory | Cache clearing, bounded workers, optional 16-bit decode/export path | **Done** |
| 17 | Packaging | Portable build, bundled resources, update link | **Done** |
| 18 | Quality | Golden-image, Recipe, catalog, mask, import, and RAW regression tests | **Done** |
| 19 | Calibration | Argyll-backed display profiling, camera chart profiling, ICC validation/install workflow | **Done** |

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

## Future candidates

- Optional GPU acceleration with a CPU reference path.
- Higher-fidelity camera/input profiles and a wide-gamut linear working pipeline.
- Pluggable modern denoise/demosaic backends with clear offline licensing.
- Background GPS indexing and optional filmstrip GPS badges for very large folders.
- Embedded/offline map options with explicit privacy and tile-cache controls.
- Panorama projection, seam, exposure-compensation, and manual control-point tools.
- Versioned plugin API and automated packaged-application smoke tests.
- Accessibility pass for scalable type, keyboard traversal, screen readers, and color-safe overlays.

### Panorama quality phase

Exposure/WB reference and strength, capture-time ordering, adjacent overlap diagnostics,
stitch-confidence tuning, horizon wave correction, and optional border cropping are complete.
Dedicated seam painting, custom projection parameters, and manual control points remain future work.
