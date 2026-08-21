import struct
import zlib

from PIL import Image

from imaging import safe_pil_open


def _minimal_png(path, width, height):
    # A valid PNG header/IHDR/IEND is enough for Image.open() to inspect size.
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IEND", b"")
    )


def test_safe_pil_open_allows_large_dimensions(tmp_path):
    path = tmp_path / "huge.png"
    _minimal_png(path, 100_000, 100_000)

    previous = Image.MAX_IMAGE_PIXELS
    with safe_pil_open(str(path)) as image:
        assert image.size == (100_000, 100_000)
    assert Image.MAX_IMAGE_PIXELS == previous


def test_safe_pil_open_restores_limit_after_failure(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")
    previous = Image.MAX_IMAGE_PIXELS

    try:
        with safe_pil_open(str(path)):
            pass
    except Exception:
        pass

    assert Image.MAX_IMAGE_PIXELS == previous
