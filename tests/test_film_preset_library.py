from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from imaging import Recipe, apply_recipe
from presets import load_preset_file


ROOT = Path(__file__).resolve().parents[1]
FILM_FOLDERS = {
    "Film - Color Negative": 12,
    "Film - Slide and Cinema": 8,
    "Film - Black and White": 10,
}


def _film_files():
    return [path for folder in FILM_FOLDERS for path in sorted((ROOT / "plugin" / folder).glob("*.json"))]


def _color_test_chart():
    levels = np.linspace(8, 247, 32, dtype=np.uint8)
    b, g = np.meshgrid(levels, levels)
    r = np.flipud(g)
    return np.dstack((b, g, r))


def test_library_has_expected_categories_and_unique_names():
    names = []
    for folder, expected_count in FILM_FOLDERS.items():
        paths = sorted((ROOT / "plugin" / folder).glob("*.json"))
        assert len(paths) == expected_count
        for path in paths:
            data = json.loads(path.read_text(encoding="utf-8"))
            assert data["preset_kind"] == "film_rendering"
            assert data["film_family"] == folder
            assert data["description"].strip()
            assert data["wb_as_shot"] is True
            names.append(data["name"])
    assert len(names) == len(set(names)) == 30


def test_every_film_preset_loads_and_renders():
    source = _color_test_chart()
    assert len(_film_files()) == 30
    for index, path in enumerate(_film_files()):
        recipe = load_preset_file(str(path), base=Recipe())
        np.random.seed(index)
        rendered = apply_recipe(source, recipe)
        assert rendered.shape == source.shape, path.name
        assert rendered.dtype == np.uint8, path.name
        assert np.isfinite(rendered).all(), path.name


def test_black_and_white_presets_render_neutral_channels():
    source = _color_test_chart()
    for path in sorted((ROOT / "plugin" / "Film - Black and White").glob("*.json")):
        np.random.seed(123)
        rendered = apply_recipe(source, load_preset_file(str(path)))
        assert np.array_equal(rendered[..., 0], rendered[..., 1]), path.name
        assert np.array_equal(rendered[..., 1], rendered[..., 2]), path.name


def test_decimal_point_curve_coordinates_are_valid():
    source = _color_test_chart()
    recipe = Recipe(curve_points=[[0.0, 0.02], [0.18, 0.15], [0.5, 0.5], [1.0, 0.98]])
    rendered = apply_recipe(source, recipe)
    assert rendered.shape == source.shape
