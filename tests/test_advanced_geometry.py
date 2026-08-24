from __future__ import annotations

import cv2
import numpy as np

from imaging import (
    Recipe, apply_advanced_geometry, apply_diorama, apply_recipe,
    apply_wide_angle_stretch,
)


def _grid(height=90, width=140):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[::10, :] = 255
    image[:, ::10] = 255
    image[height // 2 - 2:height // 2 + 3, :, 1] = 180
    return image


def test_advanced_geometry_identity_is_exact():
    source = _grid()
    assert np.array_equal(apply_advanced_geometry(source), source)
    assert np.array_equal(apply_wide_angle_stretch(source, 0), source)
    assert np.array_equal(apply_diorama(source, 0), source)


def test_horizontal_perspective_and_edge_warp_preserve_canvas():
    source = _grid()
    result = apply_advanced_geometry(source, horizontal=18, top=9, bottom=-7, left=5, right=-4)
    assert result.shape == source.shape
    assert result.dtype == source.dtype
    assert not np.array_equal(result, source)


def test_wide_angle_supports_both_directions():
    source = _grid()
    stretched = apply_wide_angle_stretch(source, 55)
    compressed = apply_wide_angle_stretch(source, -55)
    assert stretched.shape == compressed.shape == source.shape
    assert not np.array_equal(stretched, source)
    assert not np.array_equal(compressed, source)
    assert not np.array_equal(stretched, compressed)


def test_diorama_keeps_focus_band_sharper_than_outer_area():
    source = _grid(120, 160)
    result = apply_diorama(source, strength=100, position=50, width=16, angle=0)

    def detail(region):
        return cv2.Laplacian(region, cv2.CV_32F).var()

    center = result[52:68]
    outer = np.concatenate((result[:16], result[-16:]), axis=0)
    assert detail(center) > detail(outer)


def test_recipe_serializes_and_applies_new_geometry_fields(tmp_path):
    path = tmp_path / "geometry.json"
    recipe = Recipe(
        perspective_horizontal=12, warp_top=5, warp_bottom=-4,
        warp_left=3, warp_right=-2, wide_angle=20,
        diorama_strength=35, diorama_position=45, diorama_width=28,
        diorama_angle=8,
    )
    recipe.save_json(str(path))
    loaded = Recipe.load_json(str(path))
    assert loaded.perspective_horizontal == 12
    assert loaded.diorama_angle == 8
    output = apply_recipe(_grid(), loaded)
    assert output.shape == _grid().shape
