# Lightroom XMP presets in PhotoLab

PhotoLab reads XMP editing values and renders them with its own NumPy/OpenCV pipeline.
It does not contain Adobe Camera Raw, Adobe camera profiles, or Adobe Sensei. Matching
setting values is not a promise of identical Lightroom pixels.

## Supported additions

- `ToneCurvePV2012` and its Red/Green/Blue variants, including their older `ToneCurve`
  names: RDF point sequences in 0–255 coordinates are normalized and stored in the
  recipe. The master curve is applied to RGB channels rather than Lab lightness.
  Channel curves follow the master curve. Interpolation is linear, not Adobe's spline.
  Identity sequences reset previous curves; omitted curves preserve existing edits.
  Invalid/nonfinite/out-of-range coordinates or duplicate inputs produce an import
  error without changing the current recipe. Empty RDF sequences clear a curve.
- Parametric Shadows/Darks/Lights/Highlights map to PhotoLab's existing five-region
  curve. Lightroom's custom parametric split positions are not implemented.
- Legacy `SplitToning*` hue, saturation and balance plus modern `ColorGrade*` midtone,
  global, luminance and blending values. Imported grading uses PhotoLab's four-way
  tonal weighting and runs after monochrome conversion so sepia/cyanotype survive.
  Edit it in **Color > Color Grading**. Zero-valued controls clear those controls;
  omitted controls preserve the existing recipe. Older PhotoLab JSON recipes retain
  their previous split-toning behavior until the new grading mode is enabled.
- `Dehaze` accepts -100 through +100. Negative values reduce local contrast through
  the existing ClearView approximation; they are not clamped to zero. The control
  in the Light panel is now **Dehaze / haze** and allows negative values.
- `AutoTone="True"` analyzes a bounded, white-balance-adjusted sample of the open
  original image. PhotoLab chooses Exposure, Contrast, Highlights, Shadows, Whites
  and Blacks using luminance percentiles and highlight limits. Explicit tone values
  in the same XMP override the automatic values. Saturation and Vibrance are not
  automatically estimated. This is PhotoLab Auto Tone, not Adobe's Auto algorithm.

## Applying and saving

Open a photograph before applying an Auto Tone preset. The browser preview, Apply
Preset, folder import and local-mask preset paths provide the image to the importer.
Auto Tone values are calculated once and become normal editable sliders. Export,
history and JSON sidecars use those saved numbers without re-running Auto. Copying
that recipe copies its numbers; reapply the original XMP to analyze a different image.
For a local-mask preset, Auto analyzes the whole source image, not only the mask.

The importer APIs accept optional keyword arguments `image_bgr` and `meta`. Without
image pixels, an Auto Tone XMP raises a clear `ValueError` instead of silently doing
nothing. Image data may be uint8, uint16, or normalized float BGR. Metadata carries
camera white-balance multipliers and the `wb_baked` flag.

Tone curves participate in the Tone module and interpolate with preset strength.
Grading participates in Color, and Dehaze in Effects. Reset Tone clears imported
curves; Reset Color clears grading. All imported fields serialize with the recipe.
Existing luma curves and imported RGB master curves use different math; blending
between these modes uses the target curve mode rather than a two-render crossfade.

## Remaining differences

Camera profiles, LUT/profile tables, Adobe masks, custom parametric split positions,
gray-mixer controls, sharpening masks and several other XMP fields remain unsupported.
Positive and negative Dehaze, grading and Auto Tone are PhotoLab approximations.
Clipped capture detail cannot be recreated. Strong automatic brightening may reveal
noise. Judge results on the photograph, not a requirement that the histogram be flat.

Regression tests cover namespaces, master/channel curves, curve errors, partial
presets, strength/module selection, grading after B&W, negative Dehaze, Auto Tone
at multiple bit depths, JSON round trips and application controls.
