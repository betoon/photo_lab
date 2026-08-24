import numpy as np

from imaging import Recipe, apply_brush_masks, apply_local_preset_look


def test_local_preset_look_ignores_spatial_and_detail_fields():
    image = np.full((8, 8, 3), 0.4, dtype=np.float32)
    preset = Recipe(exposure=1.0, crop=(0.2, 0.2, 0.8, 0.8), horizon=20,
                    denoise_luminance=100, sharpen_intensity=200).to_dict()
    out = apply_local_preset_look(image, preset)
    assert out.shape == image.shape
    assert np.allclose(out, 0.8)


def test_brush_preset_changes_only_painted_area():
    image = np.full((64, 64, 3), 0.25, dtype=np.float32)
    mask = {
        "strokes": [{"x": 0.5, "y": 0.5, "r": 0.18}],
        "hardness": 1.0,
        "local_preset": Recipe(exposure=1.0).to_dict(),
        "preset_strength": 1.0,
    }
    out = apply_brush_masks(image, [mask])
    assert np.allclose(out[0, 0], image[0, 0])
    assert np.allclose(out[32, 32], 0.5, atol=0.02)


def test_local_preset_strength_blends_look_inside_mask():
    image = np.full((32, 32, 3), 0.25, dtype=np.float32)
    mask = {
        "strokes": [{"x": 0.5, "y": 0.5, "r": 0.5}],
        "hardness": 1.0,
        "local_preset": Recipe(exposure=1.0).to_dict(),
        "preset_strength": 0.5,
    }
    out = apply_brush_masks(image, [mask])
    assert np.allclose(out[16, 16], 0.375, atol=0.02)
