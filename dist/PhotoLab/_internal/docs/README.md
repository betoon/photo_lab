# PhotoLab — DxO-inspired photo editor (Python)

PhotoLab includes a guided **Restore & Colorize** workspace. Its built-in Restoration
Studio repairs creases, scratches, tears, stains, fading, silvering, modest defocus,
grain, and damaged detail without AI. The optional AI Restoration Lab connects to a
user-configured external local model pack for colorization, face restoration,
reconstruction, enhancement, and super-resolution. AI weights are never required by
the standard application and results are explicitly labeled as interpretive.

## Documentation set

- `USER_MANUAL.md` — in-app and printable end-user guide.
- `DEVELOPER_MANUAL.md` — architecture, testing, packaging, and extension annex.
- `PhotoLab_User_Manual_with_Developer_Annex.pdf` — combined printable manual.
- `PACKAGING.md` — portable-build notes.
- `ROADMAP.md` — completed work and future candidates.
- `PhotoLab_Workflow_Acceptance_Test_Plan.docx` — printable 25-scenario release and regression checklist.

Display and camera profiling require an external ArgyllCMS installation plus measurement hardware or reference-chart files.

The Help menu reads the Markdown manuals directly. Update them whenever controls, shortcuts, formats, sidecar fields, dependencies, or processing order change.

## Layout (from DxO PhotoLab)
- **Left:** Histogram (R/G/B/L toggles) · Move/Zoom navigator · Advanced History
- **Center:** Large preview with zoom/pan/crop/compare
- **Right:** Light · Color · Detail · Geometry · Effects
- **Bottom:** Filmstrip
- **Top:** Compare · Fit/1:1/zoom% · Prev/Next · Local Adjustments · Presets · Export

## Tools
- Exposure, Smart Lighting, Selective Tone, ClearView Plus
- Contrast / Microcontrast / Clarity, Tone Curve
- White Balance, Vibrancy, **HSL Color Wheel** (8 channels)
- Soft Proofing (sRGB / Display P3 / Gray)
- Denoise + Unsharp Mask
- Horizon, Crop, Distortion, Perspective
- **Control Points** (local radial adjustments)
- History restore, RAW support, JSON presets
- **Remove Distractions workspace** — heal/clone, content-aware object and wire removal,
  automatic dust detection, reusable folder dust maps, editable reflection layers,
  and experimental aligned multi-image reflection separation

## Shortcuts
| Key | Action |
|-----|--------|
| Ctrl+O | Open folder |
| Ctrl+E | Export |
| Ctrl+R | Reset |
| Ctrl+Shift+R | Remove Distractions workspace |
| C | Split compare |
| B | Side-by-side |
| F | Fit |
| 1 | 1:1 |
| ← → | Prev / Next image |
| Wheel | Zoom |
| Space+drag | Pan |

## Run
```bash
pip install -r requirements.txt
python main.py
```


## Presets
- **PhotoLab JSON** — File → Save Preset…
- **Lightroom Classic / ACR XMP** — File → Load Preset… (select `.xmp`)
- **Batch folder** — File → Import Preset Folder…

Mapped from XMP: Exposure, Contrast, Highlights/Shadows/Whites/Blacks, Clarity, Dehaze→ClearView, Vibrance, Saturation, Temperature/Tint, Sharpening, Luminance/Color NR, Grain, HSL channel adjustments, B&W.

## RAW support (via rawpy / LibRaw)
Canon CR2/CR3 · Nikon NEF · Sony ARW · Fujifilm RAF · Olympus ORF · Panasonic RW2 · Pentax PEF · DNG · and more.
