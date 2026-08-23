# PhotoLab User Manual

PhotoLab is a non-destructive photo editor inspired by applications such as DxO PhotoLab. Edits are stored as a *recipe* and applied on demand—your original files stay untouched.

---

## Getting started

1. Start the app: `python main.py`
2. **File → Open Folder for Editing…** (Ctrl+O) to load images into **Develop** mode.
   - **File → Recent Folders** reopens folders you used before.
3. Click a thumbnail in the filmstrip to edit it.
4. Use the right-hand panels (**Light, Color, Detail, Geometry, Effects, Local**) to adjust.
5. Export when ready: **File → Export to Disk…** (Ctrl+E).

**Open Folder** only fills the Develop filmstrip. It does **not** add photos to the Library catalog.

---

## Develop vs Library

| Mode | Shortcut | Purpose |
|------|----------|---------|
| **Develop** | Ctrl+D | Edit the current folder’s images |
| **Library** | Ctrl+L | Browse a scanned catalog by date, rate, keyword, search |

- **Scan Folder into Library…** (Ctrl+Shift+O) recursively indexes a folder tree into a local SQLite catalog.
- Double-click a library thumbnail to open it in Develop.

---

## Main interface

- **Left:** histogram, navigator (click/drag to pan), history
- **Center:** image canvas (zoom with mouse wheel, pan with Space+drag or middle mouse)
- **Right:** tool categories
- **Bottom (Develop):** filmstrip of the open folder

### Compare

- **C** — before/after split
- **B** — side-by-side (toolbar may also offer compare modes)
- **Hold `\` or `` ` ``** — temporary before view; release to restore

### Zoom / pan

- Mouse wheel: zoom  
- Space + drag or middle-button drag: pan  
- Fit / 100% controls when available in the toolbar  

### Clipping warning

- **J** or **View → Clipping Warning**  
- **Blue** = crushed shadows · **Red** = blown highlights  
- Toggle off with **J** again  

### Metadata

- **I** or **View → Image Metadata…** — path, size, camera, lens, exposure, dates, and other tags  

### Filmstrip filters & multi-select

- Under the filmstrip: **Filmstrip min ★** hides lower-rated images (All, 1+, … 5+)  
- **Color** filter: All / None / Red / Yellow / Green / Blue / Purple  
- **Ctrl/Shift+click** to multi-select thumbnails  
- Keys **0–5**, **X** (reject), **U** (pick) apply to **all selected** images (or the current one if nothing is multi-selected)  
- **Ctrl+Shift+1…5** set a color label; **Ctrl+Shift+0** clears it  
- Filmstrip bar buttons: **★ Rate**, **✓ Pick**, **⛔ Reject**, **Compare**  
- **Left / Right** arrows skip images hidden by the current filters  

### Compare selected

- Multi-select 2–4 filmstrip images → **Image → Compare Selected…** (**Ctrl+Shift+B**) or the **Compare** button  
- Side-by-side preview (with current recipes applied when available); open any frame in Develop from the dialog  

### Focus peaking

- **P** or **View → Focus Peaking**  
- Green edge overlay highlights sharp regions (useful for checking focus)  

### Star ratings

- Keys **0–5** set a star rating on **selected** images (or current); 0 clears  
- Shown on the filmstrip label; also written to the Library catalog when that image is indexed  

### Reject / Pick / Color labels

- **X** — reject / unreject selected (⛔ on filmstrip); synced to Library when indexed  
- **U** — pick / unpick selected (✓ on filmstrip) for keepers  
- **Image → Color Label** or **Ctrl+Shift+0–5** — red / yellow / green / blue / purple (or clear)  

### Undo / Redo / History

- **Ctrl+Z** / **Ctrl+Y** step through the History panel  
- Click any history entry to jump to that state  
- **Right-click** a history entry:
  - **Preview before / after…** — side-by-side of that state vs the current recipe  
  - **Copy settings from this entry** — then **Ctrl+Shift+V** to paste onto the current image  
- **Ctrl+R** — reset entire image  
- **Edit → Reset Module** — reset Tone, Color, Detail, Geometry, Local, or Effects only  

### Named snapshots

- **Edit → Save Snapshot…** — stores a named recipe checkpoint  
- Snapshots are **persisted** in the image’s `.photolab.json` sidecar and reload when you open the image again  
- **Edit → Restore Snapshot…** — pick a named checkpoint to apply  

### Presets / plugin folder

- **File → Load Preset…** / **Save Preset…** default to the app’s **`plugin/`** folder  
- Drop **`.json`** (PhotoLab) or **`.xmp`** (Lightroom / Camera Raw) presets into `plugin/`  
- User override: `~/.photolab/plugin/`  
- Sample presets: `neutral.json`, `vivid.json`  

---

## Light

Global exposure and tone:

- Exposure, Smart Lighting, Contrast  
- Highlights, Shadows, Whites, Blacks  
- Clarity, Microcontrast, Gamma  
- Tone curve points (when shown)

Use **Auto Exposure** from the Image menu for a starting point.

### Tone curve

- **Parametric** — five region handles (shadows → highlights)  
- **Luma / R / G / B** — free point curves; double-click to add a point, double-click a point to remove  
- **Reset** clears the active curve channel  

### Match across selection

- **Image → Match Exposure to Current** — align median luminance of selected filmstrip images to the current one  
- **Image → Match White Balance to Current** — copy temperature / tint to the selection  



---

## Color

- **White Balance:** As Shot, Temperature, Tint  
- **W** — **White Balance Picker**: click a neutral gray/white in the image  
- Vibrance / Saturation  
- Selective HSL per color family (hue, saturation, luminance)  
- **Split Toning** (Color panel): independent shadow / highlight hue & saturation, plus balance  
- **Soft Proofing** (Color panel):
  - Enable soft proof and choose a profile: **sRGB**, **DisplayP3**, **AdobeRGB**, **CMYK**, or **Gray**
  - **Rendering intent:** Relative / Perceptual / Saturation / Absolute  
  - **Load ICC…** — use a custom `.icc` / `.icm` profile (Pillow ImageCms); otherwise PhotoLab tries a system profile, then an approximation  
  - **Gamut warning (magenta)** highlights colors that change strongly under the simulation  
  - **Simulate paper white** — slight warm paper tint after conversion  
  - **Gamut shift %** readout shows how much of the image moved under the proof (and whether ICC or approximate was used)  
  - Useful for relative checks; not a substitute for press certification

---

## Detail

### White balance extras

- **Dual illuminant** — blend primary and secondary temperature/tint (mixed lighting)  
- **Mix toward Temp 2** — 0% primary only, 100% secondary only  

### Noise reduction

- **Luminance** — grain / luminance noise  
- **Chrominance** — color blotches  
- **Strength** — biases toward stronger NLM denoise  
- **Detail Recovery** — restores fine structure after NR (edge-aware)  
- **Method:** Auto · Bilateral (fast) · NLM (stronger, slower)

### Sharpening

- **Capture sharpening:** Amount, Radius, Masking/Threshold, Detail  
- **Output sharpening:** final amount for screen/print  

### Presets

Light NR, Strong NR, Portrait, Landscape — one-click starting points. Always check at **100% zoom**.

---

### Output sharpening (PPI)

- Set **Output PPI** and **Media** (Screen / Matte / Glossy)  
- **Apply suggestion** fills Output amount from PPI + media  
- Quick buttons: Screen 96, Print 240 / 300 / 360  

### Skin protection

- **Protect skin tones** reduces capture/output sharpening and vibrance on skin-like hues  
- Portrait detail preset enables moderate skin protection  

## Batch tools

- **Apply Preset to Selected…** — load one JSON/XMP preset onto all selected filmstrip/library images (writes sidecars)  
- **Batch Rename / Move…** — rename patterns (`keep`, date+seq, date+orig), optional destination folder; **catalog paths update**  
- **Export queue** (survives restart; stored in `~/.photolab/export_queue.json`):  
  - **Add Selected to Export Queue**  
  - **View Export Queue…**  
  - **Process Export Queue…**  

Import also accepts an optional **default preset** applied to every imported file.

## Import & culling

### Import Photos (Ctrl+Shift+I)

- Choose **source** and **destination** folders  
- **Copy** or **Move**  
- Rename: keep names, `YYYYMMDD_0001`, or `YYYYMMDD_original`  
- Optional **Year / YYYY-MM-DD** subfolders  
- Optionally scan into Library and open the destination when done  

### Culling Mode (F7)

Full-screen review of the filmstrip (or library selection):

| Key | Action |
|-----|--------|
| ← → / Space | Previous / next |
| 0–5 | Rating |
| X | Reject toggle |
| U | Pick toggle |
| Esc | Exit |

## Library catalog

- **Collections** — create, add selected photos, browse members  
- **People / faces** — comma-separated person tags (saved on selected thumbs)  
- **Find duplicates** — groups images with the same content fingerprint (built at scan)  
- **Virtual copy** — stores a named recipe clone of the current Develop image in the catalog  

## Lensfun database

Place the Lensfun XML database next to the app, for example:

- `photo_lab/lensfun/`
- `photo_lab/lensfun/data/db/`
- `photo_lab/lensfun/version_1/`

Also: `pip install lensfunpy`. **Test match…** in Geometry shows which DB path was used.

## Geometry

- Horizon angle, Distortion, Perspective  
- **Level horizon (L):** drag a line along the real horizon; angle is applied for you  
- Or use the **Level: draw line on image** button under Geometry → Horizon  
- **Show grid** — rule of thirds + center crosshair  
- **Fibonacci / golden spiral** — composition guide  
  - Drag center handle to move  
  - Drag blue corner to resize  
  - Size slider and Orientation (A–D + mirrored)  
- **Guide color** — Yellow, **White**, Cyan, or Black (easier to see on busy images)  
- Crop tool with aspect presets: Free, Original, 1:1, 5:4, 4:3, 3:2, 16:9, 16:10, portraits, A4  
- **Optics / Lens:** chromatic aberration; optional Lensfun auto-correct if `lensfunpy` is installed  

---

## Effects

- ClearView-style dehaze, vignette, film grain  
- Black & white  
- HDR Look (single-image HDR-style tone mapping)  
- Rotate 90°  

---

## Batch tools

- **Apply Preset to Selected…** — load one JSON/XMP preset onto all selected filmstrip/library images (writes sidecars)  
- **Batch Rename / Move…** — rename patterns (`keep`, date+seq, date+orig), optional destination folder; **catalog paths update**  
- **Export queue** (survives restart; stored in `~/.photolab/export_queue.json`):  
  - **Add Selected to Export Queue**  
  - **View Export Queue…**  
  - **Process Export Queue…**  

Import also accepts an optional **default preset** applied to every imported file.

## Import & culling

### Import Photos (Ctrl+Shift+I)

- Choose **source** and **destination** folders  
- **Copy** or **Move**  
- Rename: keep names, `YYYYMMDD_0001`, or `YYYYMMDD_original`  
- Optional **Year / YYYY-MM-DD** subfolders  
- Optionally scan into Library and open the destination when done  

### Culling Mode (F7)

Full-screen review of the filmstrip (or library selection):

| Key | Action |
|-----|--------|
| ← → / Space | Previous / next |
| 0–5 | Rating |
| X | Reject toggle |
| U | Pick toggle |
| Esc | Exit |

## Library catalog

- **Collections** — create, add selected photos, browse members  
- **People / faces** — comma-separated person tags (saved on selected thumbs)  
- **Find duplicates** — groups images with the same content fingerprint (built at scan)  
- **Virtual copy** — stores a named recipe clone of the current Develop image in the catalog  

## Lensfun database

Place the Lensfun XML database next to the app, for example:

- `photo_lab/lensfun/`
- `photo_lab/lensfun/data/db/`
- `photo_lab/lensfun/version_1/`

Also: `pip install lensfunpy`. **Test match…** in Geometry shows which DB path was used.

## Geometry

- **Horizon** — angle slider, **Draw level line**, or **Auto level** (Hough edge detection)  
- **Perspective** — vertical + horizontal trapezoid sliders  
- **Keystone (4 corners)** — drag TL/TR/BR/BL handles on the image  
- **Lensfun** — enable correction, strength slider, **Test match…** shows camera/lens resolution from EXIF  
- **Distortion**, crop, CA as before  

## Local adjustments

### Control points

Enable **Local Adjustments**, click the image to place a point, then refine:

- Size, Feather  
- **Chroma range** / **Luma similarity** — limit the effect to similar colors/tones at the point  
- **Luma min / Luma max** — absolute luminance range (only tones in this band are affected)  
- Exposure, Contrast, Saturation, Clarity  

### Graduated filter (G)

1. Press **G** or use the toolbar.  
2. Drag on the image to place a linear gradient.  
3. Drag end handles to reshape.  
4. Adjust exposure, contrast, saturation, clarity, temp, feather in the Local panel.

### Adjustment brush (Shift+B)

1. Press **Shift+B**.  
2. Paint on the image (size / hardness / **flow** / **opacity** in Local panel).  
3. **Mask mode:** Add · Subtract · Intersect (or Eraser / right-drag).  
4. **Invert selected mask** — apply adjustments outside the painted area.  
5. **Auto subject mask** — offline OpenCV GrabCut (no neural net); creates a new brush entry you can refine.  
6. Show mask only to preview coverage.


## HDR

- **Effects → HDR Look** for a single photo.  
- Multi-exposure: select 2+ filmstrip images → **Image → Merge HDR…** (Ctrl+Shift+H).

## Panorama (OpenCV v1)

Stitch overlapping frames into a wide image:

1. Multi-select ordered frames in the filmstrip (or Library) — ideally **left → right**.  
2. **Image → Panorama…** (**Ctrl+Shift+P**), or **Panorama…** in the Library toolbar.  
3. Choose mode (**Panorama** / Scans / Auto) and optional working size.  
4. Save (TIFF recommended). The result opens in **Develop**.

**Expectations (important):**
- OpenCV automatic stitch works best with **~25–40% overlap**, steady exposure/WB, and **little parallax**.  
- Moving subjects, strong foreground parallax, or sparse overlap often fail or show seams.  
- This is **not** a PTGui/Hugin replacement; quality will improve in later versions.

**Tips:** tripod or nodal head, manual exposure, consistent focal length, orderly capture.

### Create Pan Video

Turn a panorama (or any wide still) into a motion video with the full **Panorama to Video** tool:

1. Open the image in Develop (often the result of **Panorama…**).  
2. **Image → Create Pan Video…**  
3. The companion app opens with your image and output folder already set.  
4. Use Ken Burns, cinematic looks, film effects, audio, ffmpeg encoders — **all original features**.  

Optional: **Image → Audio Editor…** to prepare a soundtrack (cut / fade / normalize) before attaching it in Pan Video.

Requires `pano_video.py` (and optionally `audio_editor.py`) next to PhotoLab, plus **ffmpeg** on PATH for encoding.

---

## Focus Stack

Combine a near-to-far (or far-to-near) focus bracket into one sharp image:

1. Open a folder of focus frames (or multi-select in the filmstrip).  
2. **Image → Focus Stack…** (**Ctrl+Shift+F**).  
3. Choose:
   - **Alignment:** ECC Affine (default), Translation, Rigid, Homography, or ORB variants  
   - **Fusion:** Depth map, Weighted, Pyramid, or Average  
   - **Reference frame:** middle / first / last  
   - Optional working size, common-area crop, depth-map PNG  
4. Save the result (TIFF recommended). It opens in **Develop** for further edits.

From **Library**, multi-select brackets and use the **Focus Stack…** toolbar button (same dialog).

Tips: use a tripod, manual exposure/WB, and modest focus steps. Large viewpoint changes may need ORB Homography; expect soft scores on very hard frames.

---

## Library

1. **Scan Folder into Library…**  
2. Browse the **date tree** or view all.  
3. Rate / reject photos.  
4. Search by filename, keywords, camera, folder.  
5. Select photos → enter keywords → **Save KW**.
6. **Smart collection** dropdown: All, Picked, Rejected only, Rated 3+, Rated 5, Unrated.
7. **Focus Stack…** in the library toolbar stacks the selected photos.  
6. **Remove from Library** — drops catalog entries only (files stay on disk).  
7. **Move to Trash** — sends files to the system trash (or a `.photolab_trash` folder if OS trash is unavailable) and removes them from the catalog. Confirm the dialog carefully.

Optional: `pip install send2trash` for better OS recycle-bin integration.

---

## Export and recipes

Edits are non-destructive **recipes**.

- **Ctrl+S** — save a `.photolab.json` sidecar next to the file  
- Sidecars **auto-load** when you open that image again  
- **File → Auto-Save Sidecars** — writes the sidecar after each history step while editing  
- **Export** options: watermark text, long-edge size, save sidecar  
- **Export presets:** Web (2048px JPEG 85), Print (full JPEG 95), Archival (TIFF)  
- **Copy / Paste Settings** (Ctrl+Shift+C / V) — paste everything or only Tone / Color / Detail / Geometry / Local / Effects  
- **Sync Settings to Selected…** (Ctrl+Shift+S) — copy the current recipe onto other filmstrip selections  
- **Save / Restore Snapshot** — named checkpoints for the current image (separate from linear history)  

**Batch Export Selected…** (Ctrl+Shift+E): multi-select filmstrip thumbnails, choose output folder.

---

## Keyboard shortcuts

| Key | Action |
|-----|--------|
| Ctrl+O | Open folder (Develop) |
| Ctrl+L / Ctrl+D | Library / Develop |
| Ctrl+E | Export |
| Ctrl+Shift+E | Batch export |
| Ctrl+S | Save recipe sidecar |
| Ctrl+Shift+D | Debug console |
| Ctrl+Shift+H | Merge HDR |
| Ctrl+Shift+F | Focus Stack |
| Ctrl+Shift+O | Scan into Library |
| W | White balance picker |
| G | Graduated filter |
| Shift+B | Adjustment brush |
| C | Split compare |
| F1 | User manual (Help) |
| J | Clipping warning |
| P | Focus peaking |
| L | Level horizon (draw line) |
| I | Image metadata |
| F11 | Full screen |
| Ctrl+Shift+S | Sync settings to selected |
| 0–5 | Star rating (selected or current) |
| X | Reject / unreject (selected or current) |
| U | Pick / unpick (selected or current) |
| Ctrl+Shift+0–5 | Color label clear / R / Y / G / B / Purple |
| Ctrl+Shift+B | Compare selected filmstrip images |
| Left / Right | Prev / next visible filmstrip image |
| Ctrl+R | Reset image |
| Ctrl+Z / Ctrl+Y | Undo / Redo |
| Ctrl+Shift+C / V | Copy / paste settings |

---

## Tips

- Library and filmstrip thumbnails use **embedded RAW previews** when available (much faster than full decode).  
- Judge noise and sharpening at **100% zoom**.  
- Prefer original camera RAW files; partial “Copy” NEFs often fail to decode.  
- Library and Develop are separate workflows—scan only when you want catalog features.  
- Use the **Debug Console** (View menu) if something misbehaves.

---

## About

PhotoLab is an open, Python-based editor. It aims for a DxO-like workflow without claiming full parity with commercial products (e.g. DeepPRIME-class denoise or proprietary lens modules).


## Focus stack & panorama

### Focus stack
- Alignment modes (ECC / ORB) and fusion (depth / weighted / pyramid / average)
- **Min align score** — frames below the threshold are excluded from fusion (reference always kept)
- After completion: **report** with per-frame confidence and **Match style of source frame**

### Panorama
- Modes: spherical-like (panoramas) / planar scans / auto  
- **Match exposure / WB** before stitch  
- **Order by capture time** helper  
- Report notes projection behavior; optional match style from first source frame  


## HDR merge

- **Detect brackets from open folder** — groups by EXIF exposure bias / shutter  
- **Mertens** (default exposure fusion) or **Debevec + tonemap** (Reinhard / Drago / Mantiuk)  
- **Deghost strength** — blends moving regions toward the stack median before merge  
- Align frames for handheld brackets  



## Panorama to Video

- Dark UI aligned with PhotoLab Develop  
- Launched from **Image → Create Pan Video…** with a **graded still** baked from the current Develop recipe  
- Progress shows **percent + ETA**  
- **Refresh Still Preview** re-renders the canvas still without a full encode  
- Live preview, 3s test clip, and full render unchanged  



## Diagnostics

- **Help → Report a Problem…** (also on the Debug Console) saves a text report with system info, the last 50 log-file lines, and the Debug Console buffer. The report is also copied to the clipboard.
- RAW failures show clearer messages (partial file, unsupported compression, camera-specific tips) instead of a bare path error.



## Updates & packaging

- **Help → Check for Updates…** opens GitHub Releases.
- Portable builds: see **PACKAGING.md** / `build_portable.bat`.
- Pan Video needs **ffmpeg** on PATH.


## Zone System (Ansel Adams)

**Effects → Black & White** includes zone mapping: placement (Zone V ≈ mid-gray), expansion (N−/N+), spectral filters, snap, and false-color overlay.

## Help manuals

**Help → User Manual** (F1) and **Developer Manual** load Markdown from the `docs/` folder next to the app (or the path set in Preferences).

## Toolbar

| Button | Action |
|--------|--------|
| Open / Scan | Open folder / scan into Library |
| Library / Develop | Switch modes |
| Edit / Split / Side-by-Side | Compare views |
| Fit / 1:1 | Fit window / actual pixels (Ctrl+1) |
| Prev / Next | Navigate filmstrip |
| Local / Grad / Brush / WB | Tool modes (exclusive) |
| Reset / Preset / Save Preset | Recipe |
| Tools ▾ | HDR, stack, pano, pan video, map, slideshow, print, scripts |
| Export | Export current image |

Number keys **0–5** set star rating (so 1:1 zoom uses **Ctrl+1**).

## Infrared & Astro (Effects tab)

### Infrared
- **Channel swap R↔B** — classic false-color IR starting point
- **False color** — blend toward a warm foliage / cool sky remix
- **Mono IR** — red-weighted black & white
- Preset buttons: Wood effect, Gold/Blue, Mono IR

### Astro
- **Stretch (asinh)** — lift faint nebula / Milky Way structure
- **Background / gradient remove** — soften sky gradients and light pollution
- **Star emphasis** — mild detail boost on point sources
- Presets: Milky Way, DSO soft

These are stored in the recipe sidecar like any other edit. Full calibration stacking (darks/flats) can be added later.
