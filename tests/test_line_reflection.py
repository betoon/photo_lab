import copy
import json

import numpy as np
import pytest

from imaging import Recipe, apply_recipe
from line_reflection import line_frame, reflect_under_line


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_horizontal_reflection_preserves_source_and_precision(dtype):
    source = np.arange(7*11*3).reshape(7, 11, 3).astype(dtype)
    if np.issubdtype(dtype, np.floating):
        source /= 251.3
    result = reflect_under_line(source, [[0, .5], [1, .5]])
    assert result.dtype == source.dtype
    assert np.array_equal(result[:4], source[:4])
    assert np.array_equal(result[4:], source[2::-1])


def test_vertical_reverse_endpoints_and_side_are_equivalent():
    source = np.arange(5*9).reshape(5, 9).astype(np.uint16)
    line = [[.5, 0], [.5, 1]]
    result = reflect_under_line(source, line, 1)
    assert np.array_equal(result[:, 5:], source[:, 3::-1])
    assert np.array_equal(result, reflect_under_line(source, line[::-1], -1))


def test_oblique_non_square_image_matches_independent_householder_oracle():
    h, w = 31, 73
    yy, xx = np.mgrid[:h, :w]
    # Linear ramp: bilinear interpolation must reproduce the analytic value.
    source = (xx*.002 + yy*.017).astype(np.float64)
    line = [[.16, .13], [.78, .81]]
    a, b = np.array(line)*[w-1, h-1]
    d = b-a
    matrix = 2*np.outer(d, d)/np.dot(d, d)-np.eye(2)
    positions = np.stack([xx, yy], axis=-1)
    mirrored = (positions-a)@matrix.T+a
    side = d[0]*(yy-a[1])-d[1]*(xx-a[0])
    valid = (side > 0) & (mirrored[..., 0] >= 0) & (mirrored[..., 0] <= w-1) & (mirrored[..., 1] >= 0) & (mirrored[..., 1] <= h-1)
    expected = source.copy()
    expected[valid] = (mirrored[..., 0]*.002 + mirrored[..., 1]*.017)[valid]
    before = source.copy()
    assert np.allclose(reflect_under_line(source, line), expected, atol=1e-12)
    assert np.array_equal(source, before)


def test_opacity_feather_and_off_canvas_leave_source_untouched():
    source = np.zeros((21, 31, 3), np.float32)
    source[:10] = 1
    line = [[0, .5], [1, .5]]
    half = reflect_under_line(source, line, opacity=50)
    assert np.all(half[11:] == .5)
    soft = reflect_under_line(source, line, feather=20)
    assert 0 < soft[11, 15, 0] < soft[13, 15, 0] < 1
    assert np.array_equal(soft[:11], source[:11])
    edge = reflect_under_line(source, [[0, .1], [1, .1]])
    assert np.array_equal(edge[5:], source[5:])


@pytest.mark.parametrize("line", [[], [[.5,.5],[.5,.5]], [[float('nan'),0],[1,1]], [[0,0]], [[-1,0],[1,1]]])
def test_invalid_lines_are_safe_identity(line):
    source = np.ones((12, 20, 3), np.uint16)
    assert line_frame(line, 20, 12) is None
    assert np.array_equal(reflect_under_line(source, line), source)


def test_recipe_export_roundtrip_and_crop_coordinates(tmp_path):
    image = np.random.default_rng(19).integers(0, 65536, (41, 81, 3), dtype=np.uint16)
    base = Recipe(crop=(.1,.1,.9,.9), exposure=.3)
    recipe = copy.deepcopy(base)
    recipe.line_reflection_points = [[.1,.2],[.9,.7]]
    recipe.line_reflection_opacity = 76
    recipe.line_reflection_feather = 3
    path = tmp_path/'reflection.json'
    recipe.save_json(str(path))
    loaded = Recipe.load_json(str(path))
    assert json.dumps(loaded.to_dict(), sort_keys=True) == json.dumps(recipe.to_dict(), sort_keys=True)
    developed = apply_recipe(image, base, output_dtype=np.float32)
    expected = reflect_under_line(developed, recipe.line_reflection_points, -1, 76, 3)
    actual = apply_recipe(image, loaded, output_dtype=np.uint16)
    assert np.max(np.abs(actual.astype(float)-np.rint(expected*65535))) <= 1
    assert actual.dtype == np.uint16
