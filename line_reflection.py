"""Non-destructive arbitrary-line reflection in native image precision."""
import numpy as np


def line_frame(line, width, height):
    """Normalized endpoints -> pixel origin and unit normal.

    Pixel centers span [0, width-1] and [0, height-1]. Computing the normal
    AFTER this conversion is essential for non-square photographs.
    """
    try:
        points = np.asarray(line, dtype=np.float64)
        if points.shape != (2, 2) or not np.isfinite(points).all():
            return None
        if (points < 0).any() or (points > 1).any():
            return None
        a, b = points * [max(width-1, 0), max(height-1, 0)]
        delta = b-a
        length = np.linalg.norm(delta)
        if length < 1e-6:
            return None
        return a, np.array([-delta[1], delta[0]])/length
    except (ValueError, TypeError):
        return None


def reflect_under_line(image, line, source_side=-1, opacity=100., feather=0.):
    """Reflect the selected half-plane across the infinite endpoint line.

    Source side is the sign of the signed normal distance. Only opposite-side
    pixels whose mirror lies inside the canvas are replaced. Feather is a
    percentage of the shorter image side, confined to the destination side.
    Row tiles bound temporary memory. Bilinear sampling avoids ringing and
    does not quantize float/16-bit input to an 8-bit intermediate.
    """
    h, w = image.shape[:2]
    frame = line_frame(line, w, h)
    if frame is None or not np.isfinite([opacity, feather]).all() or opacity <= 0:
        return image
    a, normal = frame
    side = -1 if source_side < 0 else 1
    strength = float(np.clip(opacity/100., 0., 1.))
    fade = max(0., feather)/100. * min(w, h)
    output = image.copy()
    xx = np.arange(w, dtype=np.float64)[None, :]
    # Explicit bilinear sampling also supports images beyond remap's 32767 limit.
    for top in range(0, h, 128):
        yy = np.arange(top, min(top+128, h), dtype=np.float64)[:, None]
        distance = (xx-a[0])*normal[0] + (yy-a[1])*normal[1]
        mx = xx - 2*distance*normal[0]
        my = yy - 2*distance*normal[1]
        valid = (distance*side < -1e-7) & (mx >= -1e-7) & (mx <= w-1+1e-7) & (my >= -1e-7) & (my <= h-1+1e-7)
        mx, my = np.clip(mx, 0, w-1), np.clip(my, 0, h-1)
        x0, y0 = mx.astype(np.intp), my.astype(np.intp)
        x1, y1 = np.minimum(x0+1, w-1), np.minimum(y0+1, h-1)
        fx, fy = mx-x0, my-y0
        alpha = valid.astype(np.float64)*strength
        if fade > 0:
            t = np.clip(np.abs(distance)/fade, 0, 1)
            alpha *= t*t*(3-2*t)
        if image.ndim == 3:
            fx, fy, alpha = fx[..., None], fy[..., None], alpha[..., None]
        reflected = ((image[y0, x0]*(1-fx) + image[y0, x1]*fx)*(1-fy)
                     + (image[y1, x0]*(1-fx) + image[y1, x1]*fx)*fy)
        base = image[top:top+128]
        result = base*(1-alpha) + reflected*alpha
        if np.issubdtype(image.dtype, np.integer):
            limits = np.iinfo(image.dtype)
            result = np.clip(np.rint(result), limits.min, limits.max)
        output[top:top+128] = result.astype(image.dtype)
    return output
