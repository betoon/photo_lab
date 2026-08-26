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

### Import from an SD card

Choose **File → Import from SD Card…**, then choose the card/camera folder as the source and a normal local folder as the destination. PhotoLab discovers supported photos and videos, skips identical files already copied, gives safe unique names to collisions, and can preserve the card's folder structure. The progress dialog can cancel the remaining copies. Verify the import and backup before formatting a card.

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

The Metadata panel also shows GPS availability:

- A **green dot** means valid coordinates are present; **Map this photo** opens that location.
- A **red dot** means no coordinates were found.
- **Select GPS photos & show map** scans the filmstrip, selects geotagged images, and opens the map list.
- The browser map uses Leaflet/OpenStreetMap and needs internet access for map tiles.

RAW metadata is read independently of sensor decoding when possible, so camera and GPS fields may remain available even when a new RAW compression requires a newer LibRaw build.

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
- **Edge preservation** controls whether recovered detail is concentrated on strong edges or restored more uniformly.
- **Banding correction** reduces repeated horizontal or vertical sensor bands; direction may be automatic or selected manually.
- **JPEG artifact reduction** softens block boundaries and ringing independently of sensor-noise reduction.
- **Measure Image Noise** analyzes flatter image regions and records luminance noise, chroma noise, and banding orientation without changing the image.
- **Apply Measured Profile** transfers the stored measurements into editable starting values.

Noise profiles are stored in the image recipe. Measurement and application are separate so an automatic estimate never changes an edit without confirmation.

### Sharpening

- **Capture sharpening:** Amount, Radius, Masking/Threshold, Detail  
- **Output sharpening:** final amount for screen/print  

### Presets

Light NR, Strong NR, Portrait, Landscape — one-click starting points. Always check at **100% zoom**.

---

### Output sharpening (PPI)

- Choose **Custom, Screen, Matte paper, Glossy paper, or Canvas**.
- Set the intended **PPI** and output width in inches.
- **Apply Suggestion** fills Output amount and calculates a media/PPI-aware sharpening radius.
- **Sharpening Proof** renders toward the intended pixel width (safely capped at 4000 preview pixels), switches the image to 100%, and displays the active delivery conditions.
- Choose **Custom** to retain the original fixed-radius output-sharpening behavior.

### Portrait Detail

- Enable **Portrait Detail**, then choose **Pick Skin Color** and click representative, evenly lit skin.
- **Color reach** expands or narrows the sampled-color selection.
- **Small, Medium, and Large details** control smoothing at separate spatial scales.
- **Edge preservation** protects eyes, lips, hair boundaries, jewelry, and other strong features.
- **Texture recovery** restores controlled fine texture after smoothing.
- **Show Mask** displays the calculated coverage as a red overlay.
- **Limit with shared mask** intersects skin detection with a named mask—for example, a painted face or subject mask—to prevent similar-colored backgrounds from being affected.

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
- **Auto level** uses Hough edge detection as a starting suggestion
- **Keystone (4 corners)** lets you drag TL/TR/BR/BL handles directly on the image
- **Show grid** — rule of thirds + center crosshair  
- **Fibonacci / golden spiral** — composition guide  
- **Distortion** — correct barrel/pincushion curvature and independently adjust wide-angle edge stretching
- **Perspective** — correct converging verticals and horizontal perspective independently
- **Auto Upright** — analyzes strong architectural lines to level the image and correct converging verticals
- **4-Corner Tool** — drag the four labeled handles around a photographed rectangle, then release to rectify it
- **Edge Warp** — shift the top, bottom, left, and right edges for off-axis frames, signs, and architecture
- **Tilt-Shift / Diorama** — set the strength, position, width, and angle of a selective-focus band
- **Auto-crop transformed edge margins** — removes reflected safety borders created by strong geometry corrections

The Fibonacci guide is a continuous golden spiral. Move its yellow center handle, resize it with the blue corner, and use the eight orientation choices to rotate or mirror it.
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

### Apply a preset to a selected area

1. Paint an Adjustment Brush mask and refine it with feather, edge-aware refinement, luminance range, color range, inversion, or intersection.
2. Select that brush in the mask list.
3. Under **Preset on selected mask**, choose **Apply preset…** and select a PhotoLab JSON or Lightroom/Camera Raw XMP preset.
4. Adjust **Strength** from 0–100% while viewing the image and mask overlay.
5. Use **Clear preset** to remove the look without deleting the mask.

Localized presets apply tone and color operations only. Crop, geometry, denoise, sharpening, grain, vignette, HDR, output proofing, and other whole-image operations are intentionally ignored because they cannot be safely blended inside a painted boundary. The mask, preset, and strength are stored non-destructively in the sidecar.

## Creative filter stack

Open the **Creative** tab to build an ordered, non-destructive finishing stack. Filters run from top to bottom and may be added more than once.

- **Basic Tone & Color** — exposure, contrast, saturation, and clarity.
- **Four-Way Color Grade** — independent global, shadow, midtone, and highlight color plus tonal luminance controls.
- **Monochrome Workspace** — RGB channel mixing, virtual colored filters, brightness, contrast, structure, split toning, grain, burned edges, and borders.
- **Analog Effects** — halation, diffusion, positional bokeh, colored light leaks, chromatic shift, motion/zoom/rotation blur, dust and scratches, and double exposure.

Select a filter to change its opacity, enable or disable it, choose a blending mode, invert its assigned mask, duplicate it, delete it, or move it up and down. Available blending modes are Normal, Multiply, Screen, Overlay, Soft Light, Luminosity, and Color.

### Shared Mask Library

Shared masks are named selections that can be assigned to any number of creative filters. Editing one shared mask updates every filter that references it.

- **From Brush** promotes the selected painted mask from the Local tab while leaving its original local correction intact.
- **Luminance** creates a reusable tonal-range mask.
- **Full** creates a whole-image mask useful as the starting point for inversion or intersections.
- Shared masks can be inverted, intersected, duplicated, and deleted.
- Selecting a shared mask displays its calculated coverage as a red overlay, including luminance restrictions and intersections.

Deleting a shared mask safely removes its references from filters and other mask intersections; it does not delete painted Local masks.

### Analog Effects workflow

Add **Analog Effects** from the Creative Filter Stack. Because it is a stack filter, the complete result supports opacity, blending modes, duplication, ordering, and shared masks.

- **Halation** adds a restrained warm halo around bright regions.
- **Diffusion / bloom** softens and spreads highlights; Radius controls its scale.
- **Bokeh blur** uses X/Y center and focus-size controls to preserve a chosen area.
- **Light leak** provides hue, position, size, and strength controls.
- **Chromatic shift** separates red and blue channels along an adjustable angle.
- **Motion, zoom, and rotation blur** can be used independently or together.
- **Dust & scratches** is deterministic: the same recipe produces the same marks on every render.
- **Double exposure** loads a second image and supports Screen, Multiply, Normal, Overlay, and Soft Light blending.

If a double-exposure source is moved or deleted, PhotoLab safely skips that component while retaining the rest of the analog filter.


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

### Local application paths

Use **Tools → Configuration / INI Editor…** to configure the plugin folder,
ArgyllCMS `bin` folder, Lensfun database, and Focus Stacker Pro. **Apply** saves
the machine-specific values to `~/.photolab/photolab.ini`; leave a field empty
to use automatic discovery. The editor can validate entries, open their folders,
reset them to automatic defaults, and reveal the INI file location. Missing
optional tools are reported but do not prevent PhotoLab from starting.

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

### Focus Stacker Pro and microscope stacks

Select a focus sequence in the PhotoLab filmstrip or Library, then choose **Tools → Open Focus Stacker Pro…**. PhotoLab passes the selected files in their current order, opens the advanced application, and selects its **Microscope 2D** workspace automatically. The handoff does not copy or modify the source files.

If no sequence is selected, Focus Stacker Pro still opens normally and you can add images or a folder there. Its microscope workflow includes illumination normalization, hot-pixel cleanup, flat/dark calibration, multiscale focus analysis, color-selective stacking, calibration scale bars, parameter comparison, retouching, and scientific export controls.

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
| Ctrl+= / Ctrl+- | Increase / decrease interface text |
| Ctrl+0 | Reset interface text to 100% |

---

## Accessibility and interface scaling

Use **View → Increase Interface Text**, **Decrease Interface Text**, or **Reset Interface Text** to change text immediately. The selected scale is saved and restored the next time PhotoLab starts. Preset sizes from 80% through 160% are also available under **File → Preferences… → UI → Interface text size**.

Keyboard focus is shown with a high-contrast amber outline around buttons, input fields, lists, trees, tabs, and selection controls. The main preview, filmstrip, folder browser, histogram, and navigator also expose descriptive names to screen readers.

---

## Tips

- Library and filmstrip thumbnails use **embedded RAW previews** when available (much faster than full decode).  
- Judge noise and sharpening at **100% zoom**.  
- Prefer original camera RAW files; partial “Copy” NEFs often fail to decode.  
- Library and Develop are separate workflows—scan only when you want catalog features.  
- Use the **Debug Console** (View menu) if something misbehaves.
- If a RAW sensor decode is unsupported, PhotoLab may open the full embedded JPEG and labels the fallback clearly. It remains editable, but it does not provide true RAW highlight recovery or 16-bit sensor latitude.
- DNG is an openly documented container, but vendor-specific compression and tags can still require a newer rawpy/LibRaw version.

---

## About

PhotoLab is an open, Python-based editor. It aims for a DxO-like workflow without claiming full parity with commercial products (e.g. DeepPRIME-class denoise or proprietary lens modules).


## Focus stack & panorama

### Focus stack
- Alignment modes (ECC / ORB) and fusion (depth / weighted / pyramid / average)
- **Min align score** — frames below the threshold are excluded from fusion (reference always kept)
- After completion: **report** with per-frame confidence and **Match style of source frame**

### Panorama
- Modes: spherical-like panorama / planar scans / auto
- **Analyze Overlap** inspects every adjacent transition and reports reliable feature matches plus a good/fair/weak confidence score before stitching
- **Match exposure and white balance** uses a selectable reference frame and adjustable strength before stitching
- **Order frames by capture time** can repair filename ordering from cameras that use separate folders or prefixes
- **Match confidence** permits careful recovery of difficult sets; lower values accept weaker geometry and should be used cautiously
- **Straighten wavy horizon** enables OpenCV wave correction for panoramas, while scan mode stays planar
- **Crop empty warped borders** can be disabled when you want to retain the full warped canvas for later geometry work
- **Output projection** optionally finishes the stitched result as cylindrical, rectilinear, or Mercator; Automatic/unchanged preserves the standard OpenCV result exactly
- **Projection strength** blends continuously from the automatic result, while **Field of view** controls the geometric intensity
- **Projection edges** can reflect or extend nearby pixels, or retain black edges for a later crop/compositing workflow
- **Automatically soften suspected stitch seams** is an optional, conservative finishing pass. It detects column-wide tonal discontinuities, reports their count, and blends only a narrow band around them
- **Seam refinement** controls blend strength and **Seam blend width** controls the affected pixel radius. Leave the option off when the panorama contains intentional full-height hard edges
- Report notes projection behavior; optional match style from first source frame  


## HDR merge

- **Detect brackets from open folder** — groups by EXIF exposure bias / shutter  
- **Mertens** (default exposure fusion) or **Debevec + tonemap** (Reinhard / Drago / Mantiuk)  
- **Analyze Bracket** — shows relative exposure, recorded shutter, estimated frame shift, alignment confidence, and a magenta motion/ghost overlay before merging
- **Deghost strength and reference** — blends moving regions toward either the automatically chosen middle exposure or a frame you select
- **Chromatic fringe** — corrects red/blue edge fringing on every source before alignment and merge
- Align frames for handheld brackets  

The automatic deghost reference is the middle-brightness exposure. Choose another reference when a person, animal, leaf, or other moving subject is best in a different frame. Magenta is only a diagnostic overlay and is never baked into the merged image.



## Panorama to Video

- Dark UI aligned with PhotoLab Develop  
- Launched from **Image → Create Pan Video…** with a **graded still** baked from the current Develop recipe  
- Progress shows **percent + ETA**  
- **Refresh Still Preview** re-renders the canvas still without a full encode  
- Live preview, 3s test clip, and full render unchanged  



## Diagnostics

- **Help → Report a Problem…** (also on the Debug Console) saves a text report with system info, the last 50 log-file lines, and the Debug Console buffer. The report is also copied to the clipboard.
- RAW failures show clearer messages (partial file, unsupported compression, camera-specific tips) instead of a bare path error.

## Color Calibration Studio

Open **Tools → Color Calibration Studio…** for measurement-based color profiling. PhotoLab uses the external, open-source ArgyllCMS tools; it does not attempt visual software-only calibration.

### Display calibration

1. Install ArgyllCMS and choose its `bin` folder if PhotoLab does not find it automatically.
2. Connect a supported colorimeter or spectrophotometer and warm the display for about 30 minutes.
3. Disable Night Light, HDR, automatic brightness, and competing calibration loaders.
4. Choose the display, white point, brightness, response curve, quality, and output name.
5. Select **Start guided display calibration…** and follow Argyll's measured prompts.
6. In **ICC profiles**, validate the result and optionally install it for the selected display. Installation requires confirmation because it changes the operating-system profile.

For a typical dim-to-moderate editing room, D65, gamma 2.2, and about 100-120 cd/m² are reasonable starting targets. Print-matching environments may use D50 and brightness chosen to match controlled viewing light.

### Camera chart profile

Use an evenly lit, glare-free photograph of a supported color chart. Provide the Argyll `.cht` recognition layout and corresponding reference-values file. A neutral TIFF converted from RAW with automatic and creative corrections disabled is preferred. **Build camera ICC profile** runs `scanin` followed by `colprof`; review the Delta E fit report in the log before using the result.

Camera profiles are specific to the camera response and lighting spectrum. Create separate profiles for materially different illuminants when accuracy matters.



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
