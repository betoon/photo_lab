"""tests/test_recipe.py — unit tests for imaging.Recipe (dataclass, not pixels).

Complements test_imaging_golden.py: this file checks the *data* layer
(serialization, reset, field coverage) rather than pixel output.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from imaging import Recipe  # noqa: E402


def test_default_recipe_round_trips_through_dict():
    r = Recipe()
    d = r.to_dict()
    r2 = Recipe.from_dict(d)
    assert r2.to_dict() == d


def test_modified_recipe_round_trips_through_dict():
    r = Recipe(
        exposure=1.25,
        contrast=-15.0,
        temperature=3200.0,
        black_and_white=True,
        hsl_hue=(5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -10.0),
        local_points=[{"x": 0.3, "y": 0.4, "radius": 0.2, "exposure": 0.5}],
        crop=(0.1, 0.1, 0.8, 0.8),
    )
    d = r.to_dict()
    r2 = Recipe.from_dict(d)

    assert r2.exposure == r.exposure
    assert r2.contrast == r.contrast
    assert r2.temperature == r.temperature
    assert r2.black_and_white is True
    assert r2.hsl_hue == r.hsl_hue
    assert r2.local_points == r.local_points
    # crop is stored as a list in the dict (for JSON) but should come back
    # as a tuple, matching the field's declared type.
    assert r2.crop == r.crop
    assert isinstance(r2.crop, tuple)


def test_to_dict_stores_crop_as_list_for_json_compatibility():
    r = Recipe(crop=(0.0, 0.0, 1.0, 1.0))
    d = r.to_dict()
    assert isinstance(d["crop"], list)
    assert d["crop"] == [0.0, 0.0, 1.0, 1.0]


def test_to_dict_leaves_none_crop_as_none():
    r = Recipe()
    d = r.to_dict()
    assert d["crop"] is None


def test_from_dict_ignores_unknown_keys():
    d = Recipe().to_dict()
    d["some_future_field_this_version_does_not_know_about"] = 123
    r = Recipe.from_dict(d)
    assert not hasattr(r, "some_future_field_this_version_does_not_know_about")


def test_from_dict_partial_dict_keeps_other_defaults():
    r = Recipe.from_dict({"exposure": 2.0})
    assert r.exposure == 2.0
    assert r.contrast == 0.0  # untouched fields keep their normal default


def test_reset_restores_all_fields_to_defaults():
    r = Recipe(exposure=1.0, contrast=50.0, black_and_white=True, vignette=30.0)
    r.reset()
    blank = Recipe()
    for f in fields(Recipe):
        assert getattr(r, f.name) == getattr(blank, f.name), f"field '{f.name}' not reset"


def test_save_and_load_json_round_trip(tmp_path):
    r = Recipe(exposure=0.5, saturation=10.0, black_and_white=True)
    path = tmp_path / "recipe.json"
    r.save_json(str(path))

    loaded = Recipe.load_json(str(path))
    assert loaded.exposure == r.exposure
    assert loaded.saturation == r.saturation
    assert loaded.black_and_white == r.black_and_white
