"""Image-space composition guides; never included in exported pixels."""
import math


GUIDES = (
    ("thirds", "Rule of thirds + center"),
    ("one", "1-point perspective"),
    ("two", "2-point perspective"),
    ("three", "3-point perspective"),
    ("iso", "Isometric (30° axes)"),
    ("diagonal", "Diagonal / 45°"),
    ("phi", "Golden ratio / phi"),
    ("elevation", "Architectural elevation"),
)


def guide_segments(kind, width, height, density=10, horizon=0.4, center=0.5):
    """Return pixel-space segments (clipped by the painter).

    Perspective fans share true vanishing points. Two side VPs sit outside
    the image; the third sits above it. Parallel grids use pixel-space angles
    so 30/45 degree slopes remain correct on non-square images.
    """
    w, h = float(width), float(height)
    if w <= 0 or h <= 0:
        return []
    n = max(4, min(30, int(density)))
    cy, cx = h * max(.05, min(.95, horizon)), w * max(.05, min(.95, center))
    lines = []
    def add(x0, y0, x1, y1):
        lines.append((x0, y0, x1, y1))
    if kind in ("thirds", "phi", "elevation"):
        fractions = ([1/3, 2/3] if kind == "thirds" else
                     [1 - 1/((1+math.sqrt(5))/2), 1/((1+math.sqrt(5))/2)]
                     if kind == "phi" else [i/n for i in range(1, n)])
        for f in fractions:
            add(w*f, 0, w*f, h)
            add(0, h*f, w, h*f)
    elif kind in ("one", "two", "three"):
        add(0, cy, w, cy)
        if kind == "one":
            for i in range(n+1):
                t = i/n
                for x, y in ((w*t, 0), (w*t, h), (0, h*t), (w, h*t)):
                    add(cx, cy, x, y)
        else:
            for vx, target_x in ((cx-w, w), (cx+w, 0)):
                for i in range(n+1):
                    add(vx, cy, target_x, h*i/n)
            if kind == "three":
                for i in range(n+1):
                    add(cx, -h, w*i/n, h)
            else:
                for i in range(1, n):
                    add(w*i/n, 0, w*i/n, h)
    elif kind in ("iso", "diagonal"):
        slope = math.tan(math.pi/6) if kind == "iso" else 1.0
        step = min(w, h)/n
        intercept_step = 2*slope*step if kind == "iso" else step
        reach = int(math.ceil((h + slope*w)/intercept_step))
        for i in range(-reach, reach+1):
            for m in (-slope, slope):
                add(0, i*intercept_step, w, i*intercept_step + m*w)
        if kind == "iso":
            for i in range(1, int(w/step)+1):
                add(i*step, 0, i*step, h)
    return lines
