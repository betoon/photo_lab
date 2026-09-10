import struct
import sys
import types
import numpy as np
import pytest
import nikon_raw
from imaging import load_image


def test_nef_uses_sdk_before_jpeg_fallback(monkeypatch, tmp_path):
    def unsupported(_path):
        raise RuntimeError('Unsupported RAW compression')
    monkeypatch.setitem(sys.modules, 'rawpy', types.SimpleNamespace(imread=unsupported))
    pixels = np.full((10, 20, 3), 40000, dtype=np.uint16)
    calls = []
    def decode(path, bps):
        calls.append((path, bps))
        return pixels
    monkeypatch.setattr(nikon_raw, 'decode_nef', decode)
    source = tmp_path / 'camera.NEF'
    source.write_bytes(b'raw')
    image, meta = load_image(str(source), output_bps=16)
    assert image is pixels
    assert calls == [(str(source), 16)]
    assert meta['raw_decoder'] == 'nikon_sdk'
    assert meta['wb_baked'] is True
    assert meta['decode_bps'] == 16
    assert 'raw_fallback' not in meta


@pytest.mark.parametrize('depth', [1, 2])
def test_bridge_output_preserves_precision_and_converts_rgb_to_bgr(tmp_path, depth):
    path = tmp_path / 'image.rgb'
    rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8 if depth == 1 else '<u2')
    path.write_bytes(struct.pack('<4I', 0x4e4b5247, 2, 1, depth) + rgb.tobytes())
    image = nikon_raw._read_output(path)
    np.testing.assert_array_equal(image, rgb[:, :, ::-1])
    assert image.dtype == rgb.dtype


def test_bridge_rejects_truncated_pixels(tmp_path):
    path = tmp_path / 'image.rgb'
    path.write_bytes(struct.pack('<4I', 0x4e4b5247, 20, 10, 2) + b'bad')
    with pytest.raises(RuntimeError, match='incomplete image'):
        nikon_raw._read_output(path)
