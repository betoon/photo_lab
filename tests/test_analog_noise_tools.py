from __future__ import annotations

import cv2
import numpy as np

from imaging import (
    Recipe, apply_analog_effects, apply_creative_filter_stack, apply_denoise,
    measure_noise_profile,
)


def _image(height=96, width=128):
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    image = np.dstack((xx / width, yy / height, (xx + yy) / (width + height)))
    image[25:48, 46:78] = 0.95
    return image.astype(np.float32)


def test_analog_defaults_are_exact_identity():
    source = _image()
    assert np.array_equal(apply_analog_effects(source, {}), source)


def test_each_primary_analog_family_changes_the_image():
    source = _image()
    settings = [
        {"halation": 60},
        {"diffusion": 55, "diffusion_radius": 5},
        {"bokeh": 70, "bokeh_x": 50, "bokeh_y": 45, "bokeh_size": 25},
        {"light_leak": 70, "leak_hue": 18, "leak_size": 55},
        {"chromatic_shift": 65, "chromatic_angle": 30},
        {"motion_blur": 65, "motion_angle": 20},
        {"zoom_blur": 60},
        {"rotation_blur": 60},
        {"dust_scratches": 60, "dust_seed": 7},
    ]
    for effect in settings:
        result = apply_analog_effects(source, effect)
        assert result.shape == source.shape
        assert np.isfinite(result).all()
        assert not np.array_equal(result, source), effect


def test_dust_and_scratches_are_deterministic():
    source = _image()
    settings = {"dust_scratches": 75, "dust_seed": 42}
    assert np.array_equal(apply_analog_effects(source, settings), apply_analog_effects(source, settings))


def test_double_exposure_loads_secondary_image(tmp_path):
    source = _image()
    secondary = np.full((30, 40, 3), (20, 80, 240), dtype=np.uint8)
    path = tmp_path / "second.png"
    assert cv2.imwrite(str(path), secondary)
    result = apply_analog_effects(source, {
        "double_exposure": 70, "double_exposure_path": str(path), "double_blend": "screen",
    })
    assert result.shape == source.shape
    assert not np.array_equal(result, source)


def test_analog_filter_uses_existing_stack_mask_and_blend():
    source = _image()
    library = [{"id": "left", "kind": "brush", "strokes": [{"x": .2, "y": .5, "r": .35}]}]
    filters = [{
        "type": "analog", "mask_id": "left", "opacity": 80, "blend_mode": "screen",
        "settings": {"light_leak": 80, "leak_x": 0, "leak_size": 60},
    }]
    result = apply_creative_filter_stack(source, filters, library)
    assert np.abs(result[:, :50] - source[:, :50]).mean() > np.abs(result[:, -35:] - source[:, -35:]).mean()


def test_noise_measurement_distinguishes_noisy_from_clean():
    clean = (_image() * 255).astype(np.uint8)
    rng = np.random.default_rng(12)
    noisy = np.clip(clean.astype(np.float32) + rng.normal(0, 18, clean.shape), 0, 255).astype(np.uint8)
    clean_profile = measure_noise_profile(clean)
    noisy_profile = measure_noise_profile(noisy)
    assert noisy_profile["luminance_sigma"] > clean_profile["luminance_sigma"]
    assert noisy_profile["chroma_sigma"] > clean_profile["chroma_sigma"]
    assert noisy_profile["suggested_luminance"] > clean_profile["suggested_luminance"]


def test_horizontal_deband_reduces_row_pattern():
    base = np.full((120, 140, 3), 120, dtype=np.uint8)
    base[::4] = 150
    before = np.std(base.mean(axis=(1, 2)))
    result = apply_denoise(base, 0, 0, deband=100, deband_orientation="horizontal")
    after = np.std(result.mean(axis=(1, 2)))
    assert after < before


def test_jpeg_artifact_reduction_smooths_block_boundaries():
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    for y in range(0, 96, 8):
        for x in range(0, 96, 8):
            image[y:y+8, x:x+8] = 90 if (x // 8 + y // 8) % 2 else 125
    result = apply_denoise(image, 0, 0, detail_preserve=0, jpeg_artifacts=100)
    assert result.var() < image.var()


def test_noise_profile_fields_round_trip(tmp_path):
    path = tmp_path / "noise.json"
    profile = {"luminance_sigma": .02, "suggested_luminance": 22, "banding_orientation": "vertical"}
    recipe = Recipe(
        noise_profile=profile, denoise_edge_preserve=82, denoise_deband=25,
        denoise_deband_orientation="vertical", denoise_jpeg_artifacts=18,
        creative_filters=[{"type": "analog", "settings": {"halation": 30}}],
    )
    recipe.save_json(str(path))
    loaded = Recipe.load_json(str(path))
    assert loaded.noise_profile == profile
    assert loaded.denoise_deband_orientation == "vertical"
    assert loaded.creative_filters[0]["type"] == "analog"
