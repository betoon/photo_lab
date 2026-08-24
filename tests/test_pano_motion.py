from __future__ import annotations

import numpy as np

from pano_video import _render_crop_frame, evaluate_motion_keyframes


def test_motion_keyframes_interpolate_focus_zoom_and_rotation():
    frames = [
        {"time": 0.0, "x": 0.1, "y": 0.2, "zoom": 1.0, "rotation": 0.0, "easing": "Linear"},
        {"time": 1.0, "x": 0.9, "y": 0.8, "zoom": 2.0, "rotation": 10.0},
    ]
    focus, zoom, rotation = evaluate_motion_keyframes(frames, 0.5)
    assert focus == (0.5, 0.5)
    assert zoom == 1.5
    assert rotation == 5.0


def test_motion_keyframe_hold_delays_segment_motion():
    frames = [
        {"time": 0.0, "x": 0.2, "y": 0.5, "zoom": 1.0, "hold": 0.5, "easing": "Linear"},
        {"time": 1.0, "x": 0.8, "y": 0.5, "zoom": 1.5},
    ]
    focus, zoom, _rotation = evaluate_motion_keyframes(frames, 0.25)
    assert focus == (0.2, 0.5)
    assert zoom == 1.0


def test_keyframed_crop_preserves_requested_frame_shape():
    source = np.zeros((240, 800, 3), dtype=np.uint8)
    source[:, :, 1] = np.arange(800, dtype=np.uint16).clip(0, 255).astype(np.uint8)
    frame = _render_crop_frame(
        source, 800, 240, 16 / 9, 1.4, 0.5, 320, 180,
        focus_override=(0.75, 0.5), rotation=4.0,
    )
    assert frame.shape == (180, 320, 3)
    assert frame.dtype == np.uint8
