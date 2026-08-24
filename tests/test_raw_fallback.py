from __future__ import annotations

import sys
import types

import cv2
import numpy as np

from imaging import load_image


def test_unsupported_raw_uses_full_embedded_jpeg(monkeypatch, tmp_path):
    source = np.full((48, 72, 3), (20, 80, 180), np.uint8)
    ok, encoded = cv2.imencode(".jpg", source)
    assert ok

    class FakeRaw:
        camera_whitebalance = [2.0, 1.0, 1.5, 1.0]

        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def postprocess(self, **_kwargs): raise RuntimeError(b"Unsupported file format or not RAW file")
        def extract_thumb(self):
            return types.SimpleNamespace(format="jpeg", data=encoded.tobytes())

    fake_rawpy = types.SimpleNamespace(
        imread=lambda _path: FakeRaw(),
        ThumbFormat=types.SimpleNamespace(JPEG="jpeg", BITMAP="bitmap"),
    )
    monkeypatch.setitem(sys.modules, "rawpy", fake_rawpy)
    path = tmp_path / "unsupported.nef"
    path.write_bytes(b"synthetic raw container")

    image, meta = load_image(str(path), output_bps=16)

    assert image.shape == source.shape
    assert image.dtype == np.uint8
    assert meta["is_raw"] is True
    assert meta["wb_baked"] is True
    assert meta["raw_fallback"] == "embedded_preview"
    assert meta["decode_bps"] == 8
