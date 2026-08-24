from __future__ import annotations

import numpy as np

from imaging import Recipe, apply_recipe, build_brush_mask


def _image():
    img = np.zeros((80, 120, 3), np.float32)
    img[:, :60] = (0.15, 0.15, 0.15)
    img[:, 60:] = (0.8, 0.2, 0.1)
    return img


def test_luminance_range_restricts_painted_mask():
    img = _image()
    spec = {"strokes": [{"x": .5, "y": .5, "r": .7}], "hardness": 1,
            "luminance_min": .18, "luminance_max": 1.0, "range_feather": .02}
    mask = build_brush_mask(img, spec)
    assert mask[:, 80:].mean() > mask[:, 20:40].mean() * 5


def test_color_range_and_edge_refinement_respect_color_boundary():
    img = _image()
    spec = {"strokes": [{"x": .75, "y": .5, "r": .7}], "hardness": 1,
            "color_range": True, "color_tolerance": .15, "edge_refine": 1.0}
    mask = build_brush_mask(img, spec)
    assert mask[:, 80:].mean() > mask[:, 20:40].mean() * 8


def test_brush_adjustments_are_part_of_recipe_pipeline():
    src = np.full((50, 50, 3), 100, np.uint8)
    recipe = Recipe(brush_masks=[{
        "id": "paint", "strokes": [{"x": .5, "y": .5, "r": .25}],
        "hardness": 1.0, "exposure": 1.0,
    }])
    out = apply_recipe(src, recipe, meta={"wb_baked": True})
    assert out[25, 25].mean() > out[0, 0].mean() * 1.5


def test_reusable_intersection_limits_second_mask():
    src = np.full((60, 100, 3), 90, np.uint8)
    first = {"id": "left", "strokes": [{"x": .25, "y": .5, "r": .28}], "hardness": 1}
    second = {"id": "wide", "strokes": [{"x": .5, "y": .5, "r": .7}], "hardness": 1,
              "intersect_with": ["left"], "exposure": 1.0}
    out = apply_recipe(src, Recipe(brush_masks=[first, second]), meta={"wb_baked": True})
    assert out[30, 25].mean() > out[30, 80].mean() * 1.4
