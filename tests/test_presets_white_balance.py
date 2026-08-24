from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from imaging import Recipe, apply_recipe
from presets import apply_preset_file, xmp_to_recipe


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
    assert result.creative_temperature == 6
    assert result.creative_tint == 2


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


def test_preset_strength_and_module_filtering(tmp_path):
    path = tmp_path / "controlled.json"
    Recipe(exposure=2.0, saturation=80.0, sharpen_intensity=40.0).save_json(str(path))
    base = Recipe(exposure=0.0, saturation=10.0, sharpen_intensity=5.0)

    result = apply_preset_file(str(path), base=base, strength=0.5, modules=["Tone", "Color"])

    assert result.exposure == pytest.approx(1.0)
    assert result.saturation == pytest.approx(45.0)
    assert result.sharpen_intensity == 5.0


def test_apply_recipe_can_return_true_uint16_precision():
    src = np.linspace(0, 65535, 256 * 3, dtype=np.uint16).reshape(1, 256, 3)
    out = apply_recipe(src, Recipe(), output_dtype=np.uint16)

    assert out.dtype == np.uint16
    assert np.unique(out).size > 256
    assert np.any((out % 257) != 0)  # not an 8-bit image expanded to 16-bit
