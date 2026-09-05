# Composition guides and reflection

Geometry already provided a rule-of-thirds/center guide, an adjustable golden
spiral, horizon leveling, vertical/horizontal perspective, four-corner correction,
Auto Upright, edge warp, lens distortion, crop, and tilt-shift blur. The existing
reflection adjustment in Remove Distractions reduces glare; it does not mirror
image content and remains a separate feature.

## New composition guides

Under **Geometry > Composition Guides**, enable the grid and choose:

| Guide | Intended use |
| --- | --- |
| 1-point | Corridors, roads, centered facades, recession toward one point |
| 2-point | Building corners and objects with two converging directions |
| 3-point | Looking up or down at tall structures |
| Isometric | Parallel construction guides with vertical and ±30° axes |
| Diagonal / 45° | Diagonal rhythm and angular alignment |
| Golden ratio / phi | Straight 38.2% / 61.8% divisions, complementing the spiral |
| Architectural elevation | Even horizontal and vertical subdivisions |

Rule of thirds and the interactive spiral remain available. Density controls
line count; horizon and vanishing center position the perspective arrangement.
Side vanishing points sit outside the frame; the third point sits above it.
Guides use the existing colors, track the image through zoom/pan, and never export.
Guide settings are view controls, not recipe edits or undo steps.

## Reflection Under a Line

1. Open an image and enable **Geometry > Reflection Under a Line**.
2. Choose **Draw / Edit Reflection…** and drag a line on the image.
3. Drag either blue endpoint to refine it, or drag elsewhere to redraw.
4. Use **Swap source / reflected**, or enable **Click source side** and click
   the half to keep. Disable that mode to resume editing endpoints.
5. Refine opacity and seam feathering while watching the live preview.
   **Show original** provides a comparison.
6. Choose **Apply** to create one undoable recipe edit, or **Cancel / Esc**
   to discard the dialog's changes. Reopen to edit the stored line.

The segment defines an infinite mirror line, shown dashed beyond the handles.
By default a left-to-right horizontal segment keeps the top and reflects below.
Canvas dimensions stay fixed. Source pixels remain unchanged; destination pixels
whose mirror falls outside the image also remain unchanged. Feathering extends
only into the destination, measured as a percentage of the shorter image side.

The operation runs on the developed image after crop and other corrections.
Normalized endpoints are converted to actual pixel coordinates before computing
the reflection, so arbitrary slopes work correctly on rectangular photographs.
Changing geometry/crop afterward changes the content under that normalized line;
reopen the tool to refine placement. Export reapplies the recipe at full resolution
without an 8-bit intermediate; bilinear sampling avoids ringing. The interactive
preview is reduced in resolution for responsiveness.

**Clear Line Reflection**, Reset Geometry, geometry presets, copy/paste settings,
sidecar saving, and Undo/Redo include the new recipe fields. Clearing leaves the
opacity/feather preferences available for the next line; Reset Geometry restores
all defaults.

## Recommended next additions

Curvilinear/fisheye guides should use explicit projection and field-of-view
controls; generic arcs would be a misleading alignment reference. Independently
draggable vanishing points, an above/below third point, and selectable axonometric
axis angles are useful future refinements. Canvas expansion is intentionally not
part of this tool because it would also require decisions about crop, masks, and
output framing.

## Validation

Tests cover an independent arbitrary-angle reflection oracle on a rectangular
image, horizontal/vertical reflection, endpoint reversal, source preservation,
clipping, opacity, feathering, invalid lines, uint8/uint16/float precision, recipe
round-tripping, cropped full-precision output, guide angles and vanishing points,
Qt mouse interaction, original comparison, resizing, Apply/Cancel/Esc, and the
actual main-window Undo/Redo/Clear/Reset handlers. Existing geometry and image
golden tests are included in the targeted regression run.
