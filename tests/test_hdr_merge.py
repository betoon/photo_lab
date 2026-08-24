import cv2
import numpy as np
import pytest

from imaging import (
    _exposure_seconds_from_meta,
    analyze_hdr_stack,
    deghost_stack,
    hdr_ghost_preview,
    merge_hdr_debevec,
    merge_hdr_mertens,
)


def _write_bracket(tmp_path):
    paths = []
    yy, xx = np.mgrid[:96, :128]
    base = np.dstack((xx * 1.3, yy * 1.8, (xx + yy) * 0.8)).astype(np.float32)
    for index, gain in enumerate((0.45, 0.9, 1.6)):
        image = np.clip(base * gain + 8, 0, 255).astype(np.uint8)
        path = tmp_path / f"bracket_{index}.png"
        assert cv2.imwrite(str(path), image)
        paths.append(str(path))
    return paths


@pytest.mark.parametrize("value, expected", [("1/125", 1 / 125), ("2s", 2.0), (0.25, 0.25)])
def test_exposure_parser(value, expected):
    assert _exposure_seconds_from_meta({"shutter": value}) == pytest.approx(expected)


def test_analyze_hdr_stack_orders_exposures(tmp_path):
    report = analyze_hdr_stack(_write_bracket(tmp_path), max_dim=80)
    evs = [item["relative_ev"] for item in report["frames"]]
    assert report["reference_index"] == 1
    assert evs[0] < evs[1] < evs[2]
    assert all(np.isfinite(item["alignment_confidence"]) for item in report["frames"])


def test_deghost_can_use_selected_reference():
    dark = np.zeros((32, 32, 3), np.uint8)
    moved = dark.copy()
    moved[8:24, 8:24] = 255
    result = deghost_stack([dark, moved], strength=100, reference_index=0)
    assert result[1][16, 16].mean() < moved[16, 16].mean()


def test_ghost_preview_returns_overlay_and_mask(tmp_path):
    paths = _write_bracket(tmp_path)
    preview, mask = hdr_ghost_preview(paths, align=False, max_dim=100, reference_index=1)
    assert preview.shape[:2] == mask.shape
    assert preview.dtype == np.uint8
    assert mask.dtype == np.float32
    assert 0.0 <= float(mask.min()) <= float(mask.max()) <= 1.0


def test_both_hdr_methods_produce_images(tmp_path):
    paths = _write_bracket(tmp_path)
    mertens = merge_hdr_mertens(paths, align=False, deghost=30, reference_index=1, ca_correction=10)
    debevec = merge_hdr_debevec(paths, align=False, deghost=20, reference_index=1)
    assert mertens.shape == debevec.shape == (96, 128, 3)
    assert mertens.dtype == debevec.dtype == np.uint8
    assert mertens.max() > mertens.min()
    assert debevec.max() > debevec.min()
