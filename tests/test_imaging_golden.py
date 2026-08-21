"""tests/test_imaging_golden.py — pixel-level regression tests for apply_recipe.

Why this exists: PhotoLab's ROADMAP (#18) calls out that the processing
pipeline has no correctness tests, only GUI smoke tests. imaging.py is pure
NumPy/OpenCV with no Qt dependency, so it's cheap to test directly: build a
deterministic synthetic image, run it through apply_recipe with a handful of
representative Recipes, and compare against a stored "golden" output.

If one of these fails after a refactor, either:
  (a) you introduced an unintended change to the pipeline math — fix it, or
  (b) the change was intentional — regenerate the golden files:

      python tests/test_imaging_golden.py --update

Golden files are small (a few KB each, uint8 PNGs) and are checked into the
repo under tests/golden/ so CI can compare against them without needing to
regenerate anything.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imaging import Recipe, apply_recipe  # noqa: E402
from imaging_fixtures import make_test_image  # noqa: E402

try:
    import pytest
except ImportError:  # pragma: no cover - `--update` doesn't need pytest
    pytest = None

    class _DummyMark:
        @staticmethod
        def parametrize(*_a, **_k):
            def _decorator(fn):
                return fn
            return _decorator

    pytest = type("pytest", (), {"mark": _DummyMark()})

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"

# Recipes chosen to each exercise a different, largely independent part of
# apply_recipe, so a regression in one stage doesn't get masked by another.
RECIPES = {
    "identity": Recipe(),
    "exposure_contrast_wb": Recipe(
        exposure=0.8, contrast=25.0, temperature=6800.0, tint=8.0,
    ),
    "hsl_vibrance_saturation": Recipe(
        vibrance=40.0, saturation=15.0,
        hsl_hue=(10.0, 0.0, -15.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        hsl_sat=(0.0, 0.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        hsl_lum=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ),
    "curve_split_tone": Recipe(
        curve_shadows=-10.0, curve_lights=15.0,
        split_shadow_hue=210.0, split_shadow_sat=25.0,
        split_highlight_hue=45.0, split_highlight_sat=20.0,
        split_balance=-10.0,
    ),
    "denoise_sharpen": Recipe(
        denoise_luminance=30.0, denoise_chroma=20.0, denoise_strength=50.0,
        sharpen_intensity=60.0, sharpen_radius=1.2, sharpen_threshold=10.0,
    ),
    "geometry_crop_horizon": Recipe(
        horizon=3.5, distortion=-8.0, crop=(0.05, 0.05, 0.9, 0.9),
    ),
    "bw_vignette_grain": Recipe(
        black_and_white=True, vignette=40.0, film_grain=15.0, microcontrast=20.0,
    ),
    "gradient_filter": Recipe(
        gradients=[
            {"x0": 0.5, "y0": 0.0, "x1": 0.5, "y1": 1.0, "feather": 0.6,
             "exposure": -0.6, "saturation": -20.0},
        ],
    ),
}


def _run(name: str) -> np.ndarray:
    img = make_test_image()
    recipe = RECIPES[name]
    # film_grain uses numpy's global, unseeded RNG (see apply_recipe's
    # np.random.randn call) — real renders are intentionally different each
    # time, but the golden-image comparison needs determinism, so pin the
    # seed here rather than changing app behavior.
    np.random.seed(0)
    out = apply_recipe(img, recipe)
    if out.dtype != np.uint8:
        out = (np.clip(out, 0, 1) * 255).round().astype(np.uint8)
    return out


def _golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.png"


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_apply_recipe_matches_golden(name):
    golden_path = _golden_path(name)
    assert golden_path.exists(), (
        f"No golden file for '{name}' — run "
        f"'python tests/test_imaging_golden.py --update' once to generate it, "
        f"review the output images, then commit tests/golden/."
    )
    actual = _run(name)
    expected = cv2.imread(str(golden_path), cv2.IMREAD_COLOR)
    assert expected is not None, f"Could not read golden file {golden_path}"
    assert actual.shape == expected.shape, (
        f"{name}: output shape {actual.shape} != golden shape {expected.shape}"
    )

    # Allow a small tolerance for platform/OpenCV-version floating point
    # differences rather than requiring byte-exact output. A real pipeline
    # regression will blow well past this; JPEG-level noise won't.
    diff = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
    max_diff = int(diff.max())
    mean_diff = float(diff.mean())
    assert max_diff <= 6 and mean_diff <= 0.5, (
        f"{name}: output diverged from golden (max_diff={max_diff}, "
        f"mean_diff={mean_diff:.3f}). If this divergence is expected, "
        f"regenerate with: python tests/test_imaging_golden.py --update"
    )


def test_default_recipe_white_balance_is_not_neutral():
    """Documents current behavior, doesn't assert it's correct.

    A fresh Recipe() has temperature=5500.0, and with no camera
    multipliers available apply_white_balance falls back to
    kelvin_to_rgb(5500), which is *not* neutral (approx RGB
    (1.0, 0.93, 0.87) — a visible warm cast). For comparison,
    kelvin_to_rgb(6500) — the default for the *secondary* dual-WB
    temperature — is close to neutral (1.0, 0.997, 0.981).

    In other words: opening an image with no EXIF/as-shot WB metadata
    currently renders with a warm cast out of the box, purely from the
    default Recipe field values. That may be intentional (e.g. matching a
    specific reference illuminant) — this test just pins the current
    behavior down so it can't drift silently. If it turns out to be a bug,
    fixing it likely means changing Recipe.temperature's default, and this
    test should be updated alongside that fix.
    """
    img = make_test_image()
    out = apply_recipe(img, Recipe())
    if out.dtype != np.uint8:
        out = (np.clip(out, 0, 1) * 255).round().astype(np.uint8)
    diff = np.abs(out.astype(np.int16) - img.astype(np.int16))
    # Currently ~5.6 mean / ~17 max; assert it stays roughly in that
    # ballpark rather than silently drifting further from neutral.
    assert 3.0 < diff.mean() < 8.0, (
        f"Default-recipe WB drift changed (mean_diff={diff.mean():.2f}); "
        f"re-check whether this is expected before updating the bounds."
    )


def _update_golden_files():
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    for name in RECIPES:
        out = _run(name)
        path = _golden_path(name)
        cv2.imwrite(str(path), out)
        print(f"wrote {path} ({out.shape[1]}x{out.shape[0]})")


if __name__ == "__main__":
    if "--update" in sys.argv:
        _update_golden_files()
    else:
        raise SystemExit(
            "Run with pytest to check against golden files, or with "
            "--update to (re)generate them after an intentional pipeline change."
        )
