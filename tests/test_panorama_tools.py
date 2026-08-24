import cv2
import numpy as np

import panorama
from panorama import analyze_panorama_sequence, match_exposure_wb, reproject_panorama, stitch_panorama


def _overlap_paths(tmp_path):
    rng = np.random.default_rng(42)
    scene = rng.integers(0, 256, (140, 300, 3), dtype=np.uint8)
    cv2.circle(scene, (130, 70), 35, (20, 220, 80), 4)
    paths = []
    for index, x0 in enumerate((0, 40, 80)):
        frame = scene[:, x0:x0 + 140]
        path = tmp_path / f"pano_{index}.png"
        assert cv2.imwrite(str(path), frame)
        paths.append(str(path))
    return paths


def test_overlap_analysis_finds_adjacent_matches(tmp_path):
    report = analyze_panorama_sequence(_overlap_paths(tmp_path), max_dim=0)
    assert report["frames"] == 3
    assert len(report["pairs"]) == 2
    assert all(pair["inliers"] >= 4 for pair in report["pairs"])
    assert all(0.0 <= pair["score"] <= 1.0 for pair in report["pairs"])


def test_exposure_match_strength_and_reference():
    reference = np.full((30, 40, 3), (60, 100, 140), np.uint8)
    darker = np.full((30, 40, 3), (30, 50, 70), np.uint8)
    unchanged = match_exposure_wb([reference, darker], ref_idx=0, strength=0.0)
    matched = match_exposure_wb([reference, darker], ref_idx=0, strength=1.0)
    assert np.array_equal(unchanged[1], darker)
    assert np.allclose(np.median(matched[1], axis=(0, 1)), np.median(reference, axis=(0, 1)), atol=1)


def test_stitch_options_are_forwarded(monkeypatch, tmp_path):
    paths = _overlap_paths(tmp_path)[:2]

    class FakeStitcher:
        def __init__(self):
            self.confidence = None
            self.wave = None

        def setPanoConfidenceThresh(self, value):
            self.confidence = value

        def setWaveCorrection(self, value):
            self.wave = value

        def stitch(self, images):
            return 0, np.hstack(images)

    fake = FakeStitcher()
    monkeypatch.setattr(panorama, "_make_stitcher", lambda mode: fake)
    result, report = stitch_panorama(
        paths, match_exposure=False, confidence_threshold=0.7,
        wave_correction=False, crop_borders=False,
    )
    assert result.shape[1] == 280
    assert fake.confidence == 0.7
    assert fake.wave is False
    assert report["crop_borders"] is False


def test_original_projection_is_exact_noop():
    image = np.arange(60 * 120 * 3, dtype=np.uint8).reshape(60, 120, 3)
    assert np.array_equal(reproject_panorama(image, "original"), image)
    assert np.array_equal(reproject_panorama(image, "cylindrical", strength=0), image)


def test_supported_projections_preserve_canvas_and_dtype():
    yy, xx = np.mgrid[:80, :180]
    image = np.dstack((xx, yy * 2, (xx + yy) % 255)).astype(np.uint8)
    for name in ("cylindrical", "rectilinear", "mercator"):
        result = reproject_panorama(image, name, strength=0.7, field_of_view=110)
        assert result.shape == image.shape
        assert result.dtype == image.dtype
        assert np.isfinite(result).all()


def test_projection_settings_appear_in_stitch_report(monkeypatch, tmp_path):
    paths = _overlap_paths(tmp_path)[:2]

    class FakeStitcher:
        def setPanoConfidenceThresh(self, _value): pass
        def setWaveCorrection(self, _value): pass
        def stitch(self, images): return 0, np.hstack(images)

    monkeypatch.setattr(panorama, "_make_stitcher", lambda _mode: FakeStitcher())
    result, report = stitch_panorama(
        paths, match_exposure=False, crop_borders=False,
        output_projection="cylindrical", projection_strength=0.5,
        projection_fov=100, projection_border="replicate",
    )
    assert result.shape[:2] == (140, 280)
    assert report["output_projection"] == "cylindrical"
    assert report["projection_strength"] == 0.5
    assert report["projection_fov"] == 100
