# PhotoLab Developer Manual

Architecture overview for contributors. PhotoLab is a **PyQt6** desktop app with a **NumPy/OpenCV** processing pipeline and optional **rawpy** for RAW files.

---

## Entry point

### `main.py`

- Creates `QApplication`, forces a valid default font (avoids `QFont::setPointSize(-1)` on Windows).
- Sets OpenCV log level to reduce TIFF noise.
- Installs `sys.excepthook` for visible tracebacks.
- Instantiates `PhotoLab` from `main_window.py` and runs the event loop.

---

## Core processing

### `imaging.py`

Central **non-destructive** image pipeline. No Qt imports.

| Symbol | Role |
|--------|------|
| `Recipe` | Dataclass of all edit parameters for one image |
| `Recipe.soft_proof*` | `soft_proof`, `soft_proof_profile`, `soft_proof_gamut` |
| `load_image` | Loads JPEG/PNG/TIFF via OpenCV; RAW via rawpy under `_rawpy_lock` |
| `is_raw` / `RAW_EXTS` / `IMAGE_EXTS` | Extension helpers |
| `_silent_imread` | `cv2.imread` with stderr silenced (TIFF warnings) |
| `extract_exif` | Best-effort EXIF (Pillow) including datetime, camera, focal, aperture |
| `apply_recipe` | Full pipeline: WB → tone → HSL → local → gradients → brushes → optics → denoise → geometry → sharpen → effects |
| `apply_denoise` | LAB bilateral / NLM + detail recovery |
| `apply_sharpen` / `apply_output_sharpen` | Edge-masked USM + fine detail + output stage |
| `apply_local_points` | Radial control-point masks |
| `apply_gradients` | Linear graduated filters |
| `apply_brush_masks` | Painted dab masks |
| `apply_chromatic_aberration_fix` | Simple radial R/B shift |
| `try_lensfun_correct` | Optional `lensfunpy` geometry correction |
| `merge_hdr_mertens` | Multi-exposure fusion (AlignMTB + MergeMertens) |
| `apply_hdr_look` | Single-image HDR-style tone mapping |
| `recipe_to_dict` / `recipe_from_dict` | JSON serialization |
| `save_recipe_sidecar` / `load_recipe_sidecar` | `.photolab.json` next to the image |
| `apply_watermark` | Text watermark on export |
| `extract_embedded_preview` | Fast RAW thumb via embedded JPEG / half-size decode |
| `apply_soft_proof` | ICC soft-proof (ImageCms) + approximate fallback; paper white; gamut % |
| `soft_proof_gamut_percent` | Fraction of pixels shifted under proof |

**Thread safety:** All rawpy/LibRaw use goes through `_rawpy_lock`.

**Pipeline order (simplified):** load → exposure/tone → WB → HSL → soft proof (optional) → control points → gradients → brushes (supports `invert`) → Lensfun/CA → denoise → geometry (horizon, distortion, crop) → clarity/microcontrast/HDR look → capture sharpen → output sharpen → vignette/grain/B&W.

**Brush masks:** each entry is `{strokes, hardness, exposure, …, invert?}`. When `invert` is true, the painted region is the *protected* area and adjustments apply outside it.

**Localized presets:** a brush entry may also contain `local_preset` (a serialized `Recipe`), `preset_name`, and `preset_strength`. `apply_local_preset_look` explicitly renders only mask-safe tone/color fields, then `apply_brush_masks` blends that result through the rasterized mask. Never call `apply_recipe` recursively for a local preset: it would repeat geometry, denoise, sharpening, other masks, and output stages. New Recipe fields remain excluded until deliberately audited as spatially blend-safe.

**Creative pipeline:** `Recipe.creative_filters` is an ordered list of independent effect dictionaries. `apply_creative_filter_stack` dispatches each enabled block, blends it with `blend_filter_result`, then limits the result through an optional reusable mask. Supported block types are `basic`, `color_grade`, and `monochrome`. Unknown types are ignored for forward compatibility.

**Shared masks:** `Recipe.mask_library` stores named mask specifications with stable IDs. Filters refer to them through `mask_id`; masks may refer to other masks through `intersect_with`. `build_shared_mask` includes cycle protection. A shared mask owns selection geometry and range gates only—correction values remain on filters or legacy local masks.

Color grading is implemented by `apply_four_way_color_grade`; expanded monochrome processing is implemented by `apply_monochrome_workspace`. Both operate on float BGR data and are mask-safe within the creative stack.

`apply_analog_effects` implements the `analog` creative-filter type. Dust uses a stored deterministic seed. Double exposure treats a missing/unreadable source path as a no-op. All analog processing occurs before the stack block's opacity, blend mode, and shared-mask composite.

**Thumbnails:** `ThumbnailWorker` and `CatalogThumbWorker` call `extract_embedded_preview` first (embedded JPEG from RAW, else half-size postprocess, else downscaled image).

---

### `presets.py`

- `xmp_to_recipe` — parse Lightroom-style XMP into a `Recipe` (regex / ElementTree).
- `load_preset_file` — JSON or XMP presets.

---

### `qt_utils.py`

- `cv_to_qpixmap` — BGR NumPy array → `QPixmap` for display.

---

## UI

### `main_window.py` — `PhotoLab(QMainWindow)`

Main application shell.

| Area | Responsibility |
|------|----------------|
| Menus / toolbar | File, Edit, View, Image, Help; mode switches; tools |
| `mode_stack` | Library page vs Develop page |
| Develop layout | Left tools, center `ImageCanvas`, right category tabs, filmstrip |
| Library page | Date tree, search, keywords, rating, thumbnail grid |
| `recipes` | `dict[path, Recipe]` — per-image edit state |
| `meta_cache` | EXIF / RAW metadata per path |
| `render_preview` | Debounced (`QTimer`) apply_recipe → canvas |
| History | Simple undo stack of recipes |
| Help | Opens user/developer manuals in a dialog |

**Important methods**

- `open_folder` / `open_folder_path` — Develop-only folder load  
- `scan_library_folder` — catalog scan worker  
- `toggle_wb_picker` / `toggle_gradient_mode` / `toggle_brush_mode`  
- `export_current` / `batch_export_selected`  
- `save_sidecar` / `reload_sidecar`  
- `_show_user_manual` / `_show_developer_manual`  

Category tabs: Light, Color, Detail, Geometry, Effects, Local (control points, gradients, brushes).

---

### `widgets.py`

Custom Qt widgets:

| Widget | Role |
|--------|------|
| `ImageCanvas` | Zoom/pan, compare modes, crop, control points, spiral, gradients, brush, WB sample |
| `HistogramWidget` | RGB/luma histogram |
| `NavigatorWidget` | Overview + viewport rectangle |
| `ToneCurveWidget` | Curve editor |
| `SliderRow` | Labeled slider + spinbox |
| `HistoryWidget` | History list UI |
| `ColorWheelWidget` | Optional color UI |

**Canvas signals:** `controlPoint*`, `gradientChanged`, `wbPicked`, `brushStrokeFinished`, `brushMaskChanged`.

**Clipping overlay:** `ImageCanvas.show_clipping` scans the displayed pixmap and paints blue/red blocks for shadow/highlight clipping (J key).

**Ratings / flags / color labels:** `rate_current`, `toggle_reject_current`, `toggle_pick_current`, and `set_color_label` operate on **all selected filmstrip paths** (via `_target_filmstrip_paths`), falling back to the current image. Labels are composed by `_filmstrip_label_text` / `_refresh_filmstrip_item` (pick ✓, reject ⛔, color emoji, stars). Color labels live in `_color_labels` (in-memory; not yet a catalog column).

**Filmstrip navigation:** `prev_image` / `next_image` walk `_visible_filmstrip_paths()` so rating/color filters are respected.

**Compare selected:** `compare_selected_images` opens a dialog with 2–4 selected frames (recipes applied when present).

**Auto-save:** `autosave_sidecars` triggers `save_recipe_sidecar` from `_push_history`.

**Selective paste:** `paste_settings` dialog maps checkbox groups to Recipe field subsets.

**History / Undo:** `HistoryWidget` stores `(label, recipe_dict)` snapshots, truncates the redo branch on new `push`, and exposes `undo_index` / `redo_index`. Right-click emits `previewRequested` / `copySettingsRequested`. `PhotoLab._on_history_preview` shows a side-by-side dialog; `_on_history_copy_settings` fills `_copied_recipe`.

**Reject:** `toggle_reject_current` uses `catalog.set_reject` + in-memory `_reject_flags` and updates filmstrip labels (multi-select aware).

**Canvas zoom:** `mouseDoubleClickEvent` toggles fit vs 1:1.

**Horizon line tool:** `horizon_line_mode` + drag; emits `horizonLineFinished(angle)` so `PhotoLab._on_horizon_line` sets `Recipe.horizon`.

**Metadata dialog:** `show_metadata` merges `meta_cache` with `extract_exif`.

**Filmstrip filter:** `_apply_filmstrip_filter` toggles `QListWidgetItem.setHidden` from ratings.

**Focus peaking:** `ImageCanvas.show_peaking` samples the displayed pixmap and draws green points on high local contrast edges (P key).

**Recent folders:** stored in `~/.photolab_recent.json`; `PhotoLab._add_recent_folder` updates the File menu submenu.

**Snapshots:** `_snapshots[path] = [{name, recipe, ts}]` via Edit → Save/Restore Snapshot. Persisted in `.photolab.json` under `"snapshots"`; reloaded in `_on_image_loaded` via `load_snapshots_sidecar`.

**Sync to selected:** `sync_settings_to_selected` clones `recipes[current]` onto other selected filmstrip paths.

**Module reset:** `reset_module(which)` restores default `Recipe` fields for tone/color/detail/geometry/local/effects groups.

**Pick flags:** `_pick_flags` in-memory; filmstrip label prefix `✓`.

**Smart collections:** `_lib_smart_match` filters catalog rows by picked/rejected/rating modes from `lib_smart_combo`.

**Temp before:** hold Backslash/backtick sets split compare; release restores previous mode.

**Export presets:** web/print/archival map to max_dim + jpeg quality + optional extension change.

**Library export:** `_lib_export_selected` builds `BatchExportWorker` jobs, loading sidecars when no in-memory recipe exists.

**Navigator viewport:** `_update_navigator_viewport` maps canvas offset/scale to normalized rect; `zoom_changed` keeps the yellow view rectangle in sync.

**Composition guides:** `show_grid`, interactive `show_spiral` (normalized center/scale/orient). `guide_color` (`yellow|white|cyan|black`) via `set_guide_color`; `_guide_colors()` supplies pens for grid and spiral.

**Library trash:** `_lib_move_to_trash` uses `send2trash` when installed, else moves files into a sibling `.photolab_trash` folder; always calls `catalog.remove_image`. `_lib_remove_from_catalog` is catalog-only.

---

### `workers.py`

`QThread` workers so the UI stays responsive:

| Worker | Job |
|--------|-----|
| `ThumbnailWorker` | Filmstrip thumbs (never `cv2.imread` on RAW) |
| `LoadImageWorker` | Background full-res load |
| `ExportWorker` | Single export + optional watermark / resize |
| `BatchExportWorker` | Multi-file export with per-image recipes |
| `FocusStackWorker` | Align + fuse focus brackets → new file |
| `PanoramaWorker` | OpenCV stitch → new file |
| `HdrMergeWorker` | Bracket merge |
| `CatalogScanWorker` | Recursive library scan |
| `CatalogThumbWorker` | Library grid thumbs + disk cache |

---

### `catalog.py`

SQLite library:

- Table `images`: path, dates, rating, reject, keywords, camera, folder, etc.
- `scan_folder` — incremental recursive scan; EXIF date or file mtime  
- `set_rating` / `set_reject` / `set_keywords` / `remove_image`  
- `search` — filename, keywords, camera, folder  
- Thumb cache paths under a cache directory  

---

## Legacy

### `photo_studio.py`

Older, simpler editor shell. Kept for reference; **`main.py` uses `main_window.PhotoLab`**.

---

## Data flow (edit one image)

```
Open folder → image_paths + filmstrip
     → load_image (worker) → meta_cache + display pixmap
     → Recipe (new or sidecar)
User moves slider → recipes[path].field = value → render_timer
     → apply_recipe(preview_src, recipe, meta=…) → ImageCanvas
Export → ExportWorker → load full → apply_recipe → watermark → imwrite
```

---

## Extending PhotoLab

1. **New global adjustment**  
   - Add field on `Recipe`  
   - Implement pure function in `imaging.py`  
   - Call it from `apply_recipe`  
   - Add slider in the appropriate `_build_*_tab`  

2. **New local tool**  
   - Store geometry + params on `Recipe` (list of dicts)  
   - Draw/interact in `ImageCanvas`  
   - Apply mask in `imaging.py`  

3. **New background job**  
   - Add a `QThread` in `workers.py`  
   - Emit progress/finished signals to `PhotoLab`  

4. **Keep `imaging.py` free of Qt** so batch and tests stay simple.

---

## Dependencies

- **Required:** Python 3.10+, PyQt6, NumPy, OpenCV, Pillow  
- **RAW:** rawpy  
- **RAW metadata fallback:** ExifRead (recommended for NEF and other containers Pillow cannot open)
- **Optional:** lensfunpy (optics auto-correct), send2trash, pygame, ffmpeg

## GPS metadata and map workflow

`imaging.extract_exif` first uses Pillow and then performs a best-effort ExifRead pass when GPS was not found. This fallback reads metadata IFDs without decoding sensor pixels. `extract_gps` normalizes DMS ratios to signed decimal degrees. Keep metadata extraction independent of RAW decode success.

The Metadata panel updates its red/green GPS indicator from `meta_cache`. The bulk action scans filmstrip paths, selects only geotagged items, and passes those paths to `MapDialog`. `map_view.py` writes a temporary Leaflet document; map tiles are loaded from OpenStreetMap, while PhotoLab does not upload the image files.

## Color Calibration Studio

`color_calibration.py` is a PyQt wrapper around ArgyllCMS, not a replacement color engine. Pure helpers locate executables, build deterministic argument lists, and validate ICC files with Pillow ImageCms. `ColorCalibrationDialog` exposes:

- display profiling through interactive `dispcal -o`, with display, white-point, luminance, gamma, and quality targets;
- camera chart recognition through `scanin`, followed by matrix/shaper input-profile creation through `colprof`;
- ICC validation and confirmed operating-system installation through `dispwin -I`.

Display calibration opens a separate console because Argyll's instrument workflow is interactive. Camera profiling uses `QProcess`, merges output channels into the visible log, and sequences `colprof` only after successful chart recognition. Never mark a profile successful solely because a process exited: require the expected ICC file and validate it.

Profile installation is a material external-state change and must remain behind explicit confirmation. Keep generated calibration/profile files in user-selected locations. ArgyllCMS is an external executable dependency and is not bundled by PyInstaller.

## SD-card import

`SdImportWorker` recursively enumerates supported image/video extensions, copies to a user-selected destination, optionally preserves relative card folders, skips identical destination files, and creates unique names for collisions. It emits progress and supports cooperative cancellation. Keep import copy-only unless a separately confirmed move workflow is introduced; a card should never be erased by PhotoLab.

## White-balance invariant

rawpy output rendered with camera white balance already has the camera multipliers baked into RGB. `meta["wb_baked"]` prevents `apply_recipe` from multiplying by `camera_whitebalance` a second time. Recipe temperature/tint are creative offsets. Preset parsing must preserve as-shot WB unless a preset explicitly declares absolute white balance. Pipeline reordering requires golden-image and preset/WB regression tests.

## Release checklist

1. Review the Git diff and preserve unrelated user changes.
2. Run the complete test suite with a repository-local temporary directory.
3. Open representative RGB, RAW, DNG, and embedded-preview-fallback files.
4. Verify as-shot WB, presets, masks, sidecar reload, SD import, GPS selection/map, and full-resolution export.
5. Build the portable package and confirm manuals, presets, Lensfun data, and companion applications resolve correctly.
6. Update both manuals and the roadmap; commit/push only when explicitly requested.

---

## Files checklist

| File | Layer |
|------|--------|
| `main.py` | Bootstrap |
| `main_window.py` | Application UI / orchestration |
| `widgets.py` | Reusable UI controls + canvas |
| `imaging.py` | Image science + Recipe |
| `workers.py` | Background threads |
| `catalog.py` | Library database |
| `focus_stack.py` | Focus-stack alignment + fusion engine (no Qt) |
| `panorama.py` | OpenCV Stitcher v1 panorama engine (no Qt) |
| `presets.py` | Preset I/O |
| `qt_utils.py` | Qt image helpers |
| `docs/USER_MANUAL.md` | End-user guide |
| `docs/DEVELOPER_MANUAL.md` | This document |

---

## Conventions

- Normalized geometry (crop, points, gradients, brush dabs) uses **0..1** image coordinates.  
- Preview may run on a downscaled copy; export always uses full resolution.  
- Prefer `getattr(r, "new_field", default)` in the pipeline so older sidecars still load.

## Focus stacking

`focus_stack.py` is a Qt-free engine:

- `align_ecc` / `align_orb` — registration
- `focus_measure` — Laplacian energy + local variance
- `fuse_depth_map` / `fuse_weighted` / `fuse_pyramid` / `fuse_average`
- `focus_stack(paths, ...)` — load → align to reference → fuse → report

`FocusStackWorker` runs this off the UI thread. PhotoLab opens the output path in Develop with a fresh recipe (stacking is **not** part of `apply_recipe`).

## Panorama (OpenCV Stitcher v1)

`panorama.stitch_panorama` loads frames, runs `cv2.Stitcher`, crops empty borders, returns BGR uint8 + report.

Known status codes are mapped to readable errors (`ERR_NEED_MORE_IMGS`, homography failure, etc.).

Same product pattern as HDR / focus stack: **output is a new file**, then opened in Develop. Future work: projection controls, exposure compensation UI, manual control points.

## Panorama to Video integration

PhotoLab does **not** reimplement Ken Burns / LUT / ffmpeg logic. `create_pan_video` launches:

```text
python pano_video.py --image <current> --output-folder <folder>
```

`pano_video.main()` accepts `--image`, `--output-folder`, `--preset`, `--title`.  
`PanoramaToVideoApp.load_image_path` preloads the still and sets the output folder.

`open_audio_editor` similarly spawns `audio_editor.py` as a separate process.

This preserves 100% of the standalone tool features while linking the workflow from Develop.

---

## Resource paths & packaging

### `app_paths.py`

Resolves bundled folders for **source** runs and **frozen** executables (`sys._MEIPASS`):

| Helper | Purpose |
|--------|---------|
| `app_root()` | App / bundle root |
| `docs_dir()` / `manual_file()` | User & developer manuals |
| `plugin_dir()` / `ensure_plugin_dir()` | JSON / XMP presets |
| `list_bundled_presets()` | `.json` / `.xmp` in `plugin/` |

### Folders to ship with the executable

```text
docs/           USER_MANUAL.md, DEVELOPER_MANUAL.md
plugin/         *.json, *.xmp develop presets (+ README.md)
```

**PyInstaller example:**

```bash
pyinstaller --noconfirm --windowed --name PhotoLab \
  --add-data "docs:docs" \
  --add-data "plugin:plugin" \
  main.py
```

On Windows use `;` instead of `:` in `--add-data`. One-folder builds keep `docs/` and `plugin/` next to the binary; one-file extracts them under `_MEIPASS` (read-only). User-writable presets still go to `~/.photolab/plugin/`.


### Tone / color (#6)

- Parametric + point curves (`curve_points`, `curve_r/g/b_points`)
- Split toning fields on `Recipe`
- `match_exposure_selected` / `match_wb_selected` in main window


### Detail / WB (#7)

- `output_sharpen_params(ppi, media)` → amount + radius for screen, matte, glossy, canvas, or custom delivery
- `build_portrait_skin_mask` → Lab chroma selection with saturation/luminance gates and edge suppression
- `apply_portrait_detail` → three-scale smoothing, texture recovery, and optional shared-mask intersection
- `measure_noise_profile` → robust flat-region luminance/chroma estimates plus row/column banding score
- `apply_denoise` also accepts edge preservation, debanding orientation/strength, and JPEG artifact reduction; the default edge-preservation value reproduces the legacy detail-recovery formula
- `ImageCanvas.set_sharpen_proof` → 100% proof label; main-window proof rendering targets output pixel width with a 4000-pixel cap
- Dual illuminant: `wb_dual`, `temperature2`, `tint2`, `wb_mix` in `apply_white_balance`


### Geometry (#8)

- `apply_perspective` — vertical keystone correction
- `apply_advanced_geometry` — horizontal perspective plus independent four-edge projective warp
- `apply_keystone` — rectifies a normalized TL,TR,BR,BL source quadrilateral
- `detect_architectural_upright` — Hough-line estimate for horizon and vertical convergence
- `geometry_auto_crop_bounds` — conservative crop for transform-generated edge margins
- `apply_wide_angle_stretch` — horizontal nonlinear remap for wide-angle edge control
- `apply_diorama` — rotatable, feathered selective-focus band
- `detect_horizon_angle` — Hough near-horizontal edges
- `probe_lensfun` / `try_lensfun_correct` — status + strength blend
- Canvas: `keystone_mode`, `keystoneChanged`, `keystoneFinished`


### Lensfun local DB

- `app_paths.lensfun_db_paths()` / `primary_lensfun_db()` discover `./lensfun` layouts
- `imaging._open_lensfun_database()` prefers those paths, then system/bundled

### Catalog (#9)

- Tables: `collections`, `collection_members`, `virtual_copies`
- Columns: `people`, `content_hash`, `color_label`
- APIs: create/list collections, find_duplicate_groups, create_virtual_copy, set_people


### Import / culling (#10)

- `catalog.list_importable_files`, `format_import_name`, `import_photos`
- `workers.ImportWorker` background copy/move
- `_CullingDialog` full-screen keyboard culling


### Batch (#11)

- `catalog.relocate_image(old, new)` updates images, collection_members, virtual_copies
- `export_queue.py` — JSON queue under `~/.photolab/export_queue.json`
- `workers.ExportQueueWorker` processes pending jobs with stored recipe dicts
- UI: apply_preset_to_selected, batch_rename_move, export_queue_* 


### Stack / Pano (#12)

- `focus_stack(..., min_align_score=)` excludes weak alignments; report has scores/excluded/used_indices
- `panorama.match_exposure_wb`, `order_paths_by_capture_time`
- UI report dialogs + `_match_style_from_source`


### HDR (#13)

- `detect_exposure_brackets`, `exposure_signature`
- `deghost_stack` median motion blend
- `merge_hdr_mertens` / `merge_hdr_debevec` / `merge_hdr` dispatcher
- `HdrMergeWorker` accepts method, deghost, tonemap


### Pan Video (#14)

- Dark ttk theme in `PanoramaToVideoApp._apply_photolab_theme`
- Progress callback computes ETA from elapsed / fraction
- PhotoLab `create_pan_video` writes temp `*_photolab_graded.jpg` via `apply_recipe` then launches CLI


### Diagnostics (#15)

- `imaging.format_raw_error` — actionable RAW decode messages
- `logging_setup.recent_log_lines` / `system_info_text` / `build_problem_report`
- UI: `export_problem_report` from Help and Debug Console
