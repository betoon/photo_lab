import math

import numpy as np
import pytest

from geometry_guides import GUIDES, guide_segments


@pytest.mark.parametrize("kind", [key for key, _ in GUIDES])
def test_guides_are_finite_and_bounded_in_count(kind):
    lines = guide_segments(kind, 960, 640)
    assert 1 < len(lines) < 300
    assert np.isfinite(lines).all()
    assert guide_segments(kind, 0, 100) == []


def test_vanishing_points_move_and_fans_share_origins():
    lines = guide_segments("one", 900, 500, horizon=.3, center=.6)
    assert all(line[:2] == (540, 150) for line in lines[1:])
    three = guide_segments("three", 900, 500, density=8)
    assert any(line[:2] == (450, -500) for line in three)
    assert len(guide_segments("two", 900, 500, density=20)) > len(guide_segments("two", 900, 500, density=6))


def test_parallel_angles_are_pixel_correct_and_phi_complements_spiral():
    for kind, slope in (("iso", math.tan(math.pi/6)), ("diagonal", 1)):
        for x0, y0, x1, y1 in guide_segments(kind, 1100, 300):
            if x1 != x0:
                assert abs((y1-y0)/(x1-x0)) == pytest.approx(slope)
    phi = guide_segments("phi", 1000, 500)
    assert len(phi) == 4
    assert phi[0][0] == pytest.approx(381.96601125)
