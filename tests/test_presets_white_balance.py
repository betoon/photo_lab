from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from imaging import Recipe, apply_recipe
from presets import xmp_to_recipe


def _xmp(tmp_path: Path, attrs: str) -> str:
    path = tmp_path / "preset.xmp"
    path.write_text(
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" '
        'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
        'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">'
        f'<rdf:RDF><rdf:Description {attrs}/></rdf:RDF></x:xmpmeta>',
        encoding="utf-8",
    )
    return str(path)


def test_relative_xmp_wb_does_not_replace_as_shot_wb(tmp_path):
    base = Recipe(exposure=0.25, temperature=6100, tint=-7, wb_as_shot=True)
    path = _xmp(tmp_path, 'crs:Temperature="+6" crs:Tint="+2" crs:Contrast2012="20"')

    result = xmp_to_recipe(path, base=base)

    assert result is not base
    assert result.contrast == 20
    assert result.temperature == 6100
    assert result.tint == -7
    assert result.wb_as_shot is True


def test_absolute_xmp_wb_is_applied(tmp_path):
    path = _xmp(tmp_path, 'crs:Temperature="7200" crs:Tint="12"')
    result = xmp_to_recipe(path, base=Recipe(wb_as_shot=True))
    assert result.temperature == 7200
    assert result.tint == 12
    assert result.wb_as_shot is False


def test_baked_raw_camera_wb_is_not_applied_twice():
    src = np.full((12, 12, 3), 128, dtype=np.uint8)
    multipliers = [2.0, 1.0, 1.5, 1.0]

    baked = apply_recipe(
        src, Recipe(), wb_multipliers=multipliers,
        meta={"is_raw": True, "wb_baked": True},
    )
    unbaked = apply_recipe(
        src, Recipe(), wb_multipliers=multipliers,
        meta={"is_raw": True, "wb_baked": False},
    )

    assert np.array_equal(baked, src)
    assert not np.array_equal(unbaked, src)
