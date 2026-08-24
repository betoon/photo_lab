import numpy as np

from imaging import Recipe, apply_recipe, apply_zebra_highlight_exposure


def test_zebra_adjustment_changes_only_pixels_in_selected_range():
    # Neutral BGR pixels whose luminance is exactly the channel value.
    image = np.array([[[0.40, 0.40, 0.40], [0.80, 0.80, 0.80], [0.96, 0.96, 0.96]]], dtype=np.float32)
    out = apply_zebra_highlight_exposure(image, exposure=-1.0, threshold=95.0, feather=0.0)
    assert np.allclose(out[0, 0], image[0, 0])
    assert np.allclose(out[0, 1], image[0, 1])
    assert np.allclose(out[0, 2], image[0, 2] * 0.5)


def test_zebra_feather_creates_smooth_transition_without_touching_shadows():
    image = np.array([[[0.50, 0.50, 0.50], [0.92, 0.92, 0.92], [0.98, 0.98, 0.98]]], dtype=np.float32)
    out = apply_zebra_highlight_exposure(image, exposure=-1.0, threshold=95.0, feather=10.0)
    assert np.allclose(out[0, 0], image[0, 0])
    assert 0.46 < out[0, 1, 0] < 0.92
    assert np.allclose(out[0, 2], image[0, 2] * 0.5)


def test_recipe_pipeline_persists_and_applies_zebra_settings():
    src = np.full((3, 3, 3), 250, dtype=np.uint8)
    recipe = Recipe(zebra_threshold=95.0, zebra_exposure=-1.0, zebra_feather=0.0)
    restored = Recipe.from_dict(recipe.to_dict())
    out = apply_recipe(src, restored, meta={"wb_baked": True})
    assert restored.zebra_exposure == -1.0
    assert 123 <= int(out[0, 0, 0]) <= 126
