"""Unit tests: Recipe serialize / deserialize round-trips."""
from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from imaging import Recipe, recipe_to_dict, recipe_from_dict


def test_default_recipe_roundtrip_dict():
    r = Recipe()
    d = r.to_dict()
    r2 = Recipe.from_dict(d)
    assert r2.exposure == 0.0
    assert r2.gamma == 1.0
    assert r2.temperature == 5500.0
    assert r2.wb_as_shot is True
    assert r2.crop is None
    assert r2.hsl_hue == (0.0,) * 8


def test_recipe_with_edits_roundtrip():
    r = Recipe()
    r.exposure = 0.75
    r.contrast = 20.0
    r.shadows = 30.0
    r.highlights = -15.0
    r.saturation = 10.0
    r.temperature = 6200.0
    r.tint = -5.0
    r.wb_as_shot = False
    r.curve_shadows = 10.0
    r.curve_highlights = -8.0
    r.black_and_white = True
    r.crop = (0.1, 0.1, 0.9, 0.9)
    r.hsl_hue = (5.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    r.local_points = [{"x": 0.5, "y": 0.5, "radius": 0.1, "exposure": 0.5}]

    d = r.to_dict()
    assert isinstance(d["crop"], list)
    r2 = Recipe.from_dict(d)
    assert r2.exposure == pytest.approx(0.75)
    assert r2.contrast == pytest.approx(20.0)
    assert r2.crop == (0.1, 0.1, 0.9, 0.9)
    assert r2.black_and_white is True
    assert r2.hsl_hue[0] == pytest.approx(5.0)
    assert len(r2.local_points) == 1
    assert r2.local_points[0]["exposure"] == pytest.approx(0.5)


def test_recipe_json_file_roundtrip():
    r = Recipe()
    r.exposure = -0.3
    r.clarity = 25.0
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "preset.json")
        r.save_json(path)
        assert os.path.isfile(path)
        r2 = Recipe.load_json(path)
        assert r2.exposure == pytest.approx(-0.3)
        assert r2.clarity == pytest.approx(25.0)


def test_recipe_from_dict_ignores_unknown_keys():
    d = {"exposure": 1.0, "not_a_real_field": 99}
    r = Recipe.from_dict(d)
    assert r.exposure == pytest.approx(1.0)
    assert not hasattr(r, "not_a_real_field") or getattr(r, "not_a_real_field", None) is None


def test_recipe_to_dict_helper():
    r = Recipe()
    r.vibrance = 12.0
    d = recipe_to_dict(r)
    assert d["vibrance"] == pytest.approx(12.0)
    r2 = recipe_from_dict(d)
    assert r2.vibrance == pytest.approx(12.0)


def test_reset_clears_edits():
    r = Recipe()
    r.exposure = 2.0
    r.saturation = 50.0
    r.reset()
    assert r.exposure == 0.0
    assert r.saturation == 0.0
