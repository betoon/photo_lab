"""tests/imaging_fixtures.py — deterministic synthetic test image for imaging tests.

No external image files needed (and none checked into the repo): the image
is generated from a fixed seed so it is byte-identical on every run and on
every machine, which is what golden-image comparisons need.
"""

from __future__ import annotations

import numpy as np


def make_test_image(h: int = 96, w: int = 128, seed: int = 1234) -> np.ndarray:
    """Deterministic synthetic BGR uint8 image exercising several code paths.

    - A smooth diagonal gradient exercises tone/WB/curve math over a
      continuous range rather than a single flat value.
    - Fixed-seed Gaussian noise gives denoise/sharpen something real to do
      (a flat image makes those effectively no-ops).
    - A checkerboard block in one corner gives edge-aware operations
      (sharpen, clarity, microcontrast) real edges to react to.
    """
    rng = np.random.RandomState(seed)
    y, x = np.mgrid[0:h, 0:w]
    b = x / w * 255
    g = y / h * 255
    r = (x + y) / (w + h) * 255
    img = np.stack([b, g, r], axis=-1).astype(np.float32)

    noise = rng.normal(0, 8, size=img.shape).astype(np.float32)
    img = np.clip(img + noise, 0, 255)

    ch, cw = 24, 32
    cb = ((x // 8 + y // 8) % 2) * 255
    img[h - ch:h, w - cw:w, :] = cb[h - ch:h, w - cw:w, None]

    return img.astype(np.uint8)
