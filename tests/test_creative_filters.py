from __future__ import annotations

import numpy as np

from imaging import (
    Recipe, apply_creative_filter_stack, apply_four_way_color_grade,
    apply_monochrome_workspace, apply_recipe, blend_filter_result,
    build_shared_mask,
)


def _image(height=64, width=96):
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    return np.dstack((x / width, y / height, (x + y) / (width + height))).astype(np.float32)


def test_empty_creative_stack_is_exact_identity():
    source = _image()
    assert np.array_equal(apply_creative_filter_stack(source, []), source)


def test_filter_order_is_meaningful_and_repeatable():
    source = _image()
    brighten = {"type": "basic", "settings": {"exposure": 1.0}, "opacity": 100}
    contrast = {"type": "basic", "settings": {"contrast": 70}, "opacity": 100}
    first = apply_creative_filter_stack(source, [brighten, contrast])
    second = apply_creative_filter_stack(source, [contrast, brighten])
    assert not np.allclose(first, second)
    assert np.array_equal(first, apply_creative_filter_stack(source, [brighten, contrast]))


def test_disabled_filter_and_zero_opacity_are_noops():
    source = _image()
    disabled = {"type": "basic", "enabled": False, "settings": {"exposure": 2}}
    transparent = {"type": "basic", "opacity": 0, "settings": {"exposure": 2}}
    assert np.array_equal(apply_creative_filter_stack(source, [disabled]), source)
    assert np.array_equal(apply_creative_filter_stack(source, [transparent]), source)


def test_supported_blending_modes_stay_finite_and_bounded():
    base = _image()
    filtered = np.flip(base, axis=1).copy()
    for mode in ("normal", "multiply", "screen", "overlay", "soft light", "luminosity", "color"):
        result = blend_filter_result(base, filtered, 0.65, mode)
        assert result.shape == base.shape
        assert np.isfinite(result).all()
        assert 0 <= result.min() <= result.max() <= 1


def test_shared_luminance_mask_can_drive_multiple_filters():
    source = _image()
    library = [{
        "id": "bright", "name": "Bright areas", "kind": "luminance",
        "luminance_min": 0.65, "luminance_max": 1.0,
    }]
    mask = build_shared_mask(source, library[0], library)
    assert mask[:, :10].mean() < mask[:, -10:].mean()
    filters = [
        {"type": "basic", "mask_id": "bright", "settings": {"exposure": 0.5}, "opacity": 100},
        {"type": "basic", "mask_id": "bright", "settings": {"saturation": 30}, "opacity": 50},
    ]
    result = apply_creative_filter_stack(source, filters, library)
    assert not np.array_equal(result, source)


def test_shared_mask_intersection_and_cycle_protection():
    source = _image()
    library = [
        {"id": "high", "kind": "luminance", "luminance_min": 0.5, "luminance_max": 1.0, "intersect_with": ["right"]},
        {"id": "right", "kind": "brush", "strokes": [{"x": .8, "y": .5, "r": .35}], "intersect_with": ["high"]},
    ]
    mask = build_shared_mask(source, library[0], library)
    assert np.isfinite(mask).all()
    assert 0 <= mask.min() <= mask.max() <= 1


def test_four_way_grade_targets_tonal_regions():
    source = np.tile(np.linspace(0.05, 0.95, 90, dtype=np.float32)[None, :, None], (20, 1, 3))
    result = apply_four_way_color_grade(source, {"shadow_hue": 220, "shadow_sat": 70})
    shadow_change = np.abs(result[:, :20] - source[:, :20]).mean()
    highlight_change = np.abs(result[:, -20:] - source[:, -20:]).mean()
    assert shadow_change > highlight_change


def test_monochrome_workspace_has_neutral_channels_and_controls():
    source = _image()
    result = apply_monochrome_workspace(source, {
        "mix_red": 50, "mix_green": 40, "mix_blue": 10,
        "filter_hue": 45, "filter_strength": 30, "structure": 20,
    })
    assert np.allclose(result[..., 0], result[..., 1])
    assert np.allclose(result[..., 1], result[..., 2])


def test_recipe_round_trips_creative_filters_and_mask_library(tmp_path):
    path = tmp_path / "creative.json"
    recipe = Recipe(
        mask_library=[{"id": "m1", "name": "Sky", "kind": "luminance", "luminance_min": .5}],
        creative_filters=[{"id": "f1", "type": "color_grade", "mask_id": "m1", "settings": {"highlight_hue": 40, "highlight_sat": 25}}],
    )
    recipe.save_json(str(path))
    loaded = Recipe.load_json(str(path))
    assert loaded.mask_library == recipe.mask_library
    assert loaded.creative_filters == recipe.creative_filters
    output = apply_recipe((_image() * 255).astype(np.uint8), loaded, meta={"wb_baked": True})
    assert output.shape == _image().shape
