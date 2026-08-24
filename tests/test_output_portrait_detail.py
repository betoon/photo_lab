from __future__ import annotations

import numpy as np

from imaging import (
    Recipe, apply_output_sharpen, apply_portrait_detail, apply_recipe,
    build_portrait_skin_mask, output_sharpen_params,
)


SKIN_BGR = (0.48, 0.62, 0.78)


def _portrait_test_image():
    image = np.full((96, 128, 3), (0.38, 0.42, 0.32), dtype=np.float32)
    image[18:82, 34:94] = SKIN_BGR
    yy, xx = np.mgrid[0:64, 0:60]
    texture = (((xx + yy) % 5) - 2).astype(np.float32) * 0.012
    image[18:82, 34:94] = np.clip(image[18:82, 34:94] + texture[..., None], 0, 1)
    image[42:49, 48:57] = (0.08, 0.09, 0.10)  # eye-like strong edge/detail
    return image


def test_output_sharpen_profiles_are_media_and_ppi_aware():
    screen_amount, screen_radius = output_sharpen_params(96, "screen")
    matte_amount, matte_radius = output_sharpen_params(300, "matte")
    canvas_amount, canvas_radius = output_sharpen_params(300, "canvas")
    assert screen_amount < matte_amount < canvas_amount
    assert screen_radius < matte_radius < canvas_radius
    high_ppi = output_sharpen_params(600, "glossy")
    low_ppi = output_sharpen_params(150, "glossy")
    assert high_ppi[1] > low_ppi[1]


def test_custom_output_profile_preserves_legacy_radius():
    source = _portrait_test_image()
    expected = apply_output_sharpen(source, 40, radius=0.8)
    recipe = Recipe(output_sharpen=40, output_sharpen_media="custom")
    actual = apply_recipe((source * 255).astype(np.uint8), recipe, meta={"wb_baked": True}).astype(np.float32) / 255.0
    assert np.abs(actual - expected).mean() < 0.01


def test_skin_mask_selects_sampled_skin_more_than_background():
    source = _portrait_test_image()
    mask = build_portrait_skin_mask(source, SKIN_BGR, color_reach=28, edge_preserve=75)
    skin = mask[25:75, 40:90].mean()
    background = np.concatenate((mask[:, :20].ravel(), mask[:, -20:].ravel())).mean()
    assert skin > background * 3


def test_edge_preservation_reduces_mask_on_strong_features():
    source = _portrait_test_image()
    protected = build_portrait_skin_mask(source, SKIN_BGR, 35, edge_preserve=100)
    unprotected = build_portrait_skin_mask(source, SKIN_BGR, 35, edge_preserve=0)
    feature = (slice(40, 51), slice(46, 59))
    assert protected[feature].mean() < unprotected[feature].mean()


def test_portrait_smoothing_reduces_skin_variation_and_preserves_shape():
    source = _portrait_test_image()
    result = apply_portrait_detail(source, {
        "skin_color": SKIN_BGR, "color_reach": 35,
        "small_smooth": 70, "medium_smooth": 45, "large_smooth": 10,
        "edge_preserve": 85, "texture_recovery": 25,
    })
    assert result.shape == source.shape
    before = source[24:38, 42:86].var()
    after = result[24:38, 42:86].var()
    assert after < before


def test_shared_mask_limits_portrait_processing():
    source = _portrait_test_image()
    settings = {
        "skin_color": SKIN_BGR, "color_reach": 40,
        "small_smooth": 80, "medium_smooth": 50, "edge_preserve": 70,
        "texture_recovery": 0,
    }
    shared = np.zeros(source.shape[:2], dtype=np.float32)
    shared[:, :source.shape[1] // 2] = 1.0
    result = apply_portrait_detail(source, settings, shared)
    left_change = np.abs(result[:, :64] - source[:, :64]).mean()
    right_change = np.abs(result[:, 64:] - source[:, 64:]).mean()
    assert left_change > right_change * 5


def test_new_detail_fields_round_trip(tmp_path):
    path = tmp_path / "detail.json"
    recipe = Recipe(
        output_sharpen=48, output_sharpen_media="matte", output_sharpen_ppi=360,
        output_sharpen_width_in=16, output_sharpen_proof=True,
        portrait_detail_enabled=True, portrait_skin_color=SKIN_BGR,
        portrait_color_reach=36, portrait_small_smooth=40,
        portrait_medium_smooth=25, portrait_large_smooth=5,
        portrait_edge_preserve=82, portrait_texture_recovery=50,
        portrait_mask_id="face-mask",
    )
    recipe.save_json(str(path))
    loaded = Recipe.load_json(str(path))
    assert loaded.output_sharpen_media == "matte"
    assert loaded.output_sharpen_ppi == 360
    assert loaded.portrait_detail_enabled is True
    assert tuple(loaded.portrait_skin_color) == SKIN_BGR
    assert loaded.portrait_mask_id == "face-mask"
