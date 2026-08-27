"""Original, non-destructive helpers for PhotoLab's distraction-removal workspace.

All public functions accept OpenCV BGR images.  Masks are uint8 (0..255), while
operation coordinates are normalized so recipes remain valid for previews and
full-resolution exports.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


def _u8(image: np.ndarray) -> tuple[np.ndarray, float, np.dtype]:
    dtype = image.dtype
    if np.issubdtype(dtype, np.integer):
        maximum = float(np.iinfo(dtype).max)
    else:
        maximum = 1.0 if float(np.nanmax(image)) <= 1.5 else 255.0
    return np.clip(image.astype(np.float32) / max(maximum, 1.0) * 255.0, 0, 255).astype(np.uint8), maximum, dtype


def _restore(image: np.ndarray, maximum: float, dtype: np.dtype) -> np.ndarray:
    out = image.astype(np.float32) / 255.0 * maximum
    if np.issubdtype(dtype, np.integer):
        out = np.rint(out)
    return np.clip(out, 0, maximum).astype(dtype)


def _circle(mask: np.ndarray, x: float, y: float, radius: float, value: int = 255) -> None:
    h, w = mask.shape[:2]
    px, py = int(round(x * (w - 1))), int(round(y * (h - 1)))
    pr = max(1, int(round(radius * min(h, w))))
    cv2.circle(mask, (px, py), pr, int(value), -1, cv2.LINE_AA)


def operations_mask(shape: Sequence[int], operations: Iterable[dict], kinds=None) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    mask = np.zeros((h, w), np.uint8)
    allowed = set(kinds or ("heal", "inpaint", "smart", "wire"))
    for op in operations or []:
        kind = str(op.get("type", ""))
        if kind not in allowed:
            continue
        if kind == "wire":
            pts = op.get("points") or []
            if len(pts) >= 2:
                p1 = (int(float(pts[0][0]) * (w - 1)), int(float(pts[0][1]) * (h - 1)))
                p2 = (int(float(pts[1][0]) * (w - 1)), int(float(pts[1][1]) * (h - 1)))
                width = max(1, int(float(op.get("radius", .005)) * min(h, w) * 2))
                cv2.line(mask, p1, p2, 255, width, cv2.LINE_AA)
        elif kind == "smart" and op.get("polygon"):
            points = np.asarray([[int(float(px) * (w - 1)), int(float(py) * (h - 1))]
                                 for px, py in op["polygon"]], np.int32)
            if len(points) >= 3:
                cv2.fillPoly(mask, [points], 255, cv2.LINE_AA)
        elif kind == "smart" and op.get("rect"):
            x1, y1, x2, y2 = op["rect"]
            cv2.rectangle(mask, (int(x1*w), int(y1*h)), (int(x2*w), int(y2*h)), 255, -1)
        else:
            _circle(mask, float(op.get("x", .5)), float(op.get("y", .5)), float(op.get("radius", .01)))
    return mask


def _clone(image: np.ndarray, op: dict) -> np.ndarray:
    h, w = image.shape[:2]
    radius = max(2, int(float(op.get("radius", .02)) * min(h, w)))
    dx, dy = int(float(op.get("x", .5)) * (w - 1)), int(float(op.get("y", .5)) * (h - 1))
    sx, sy = int(float(op.get("source_x", .5)) * (w - 1)), int(float(op.get("source_y", .5)) * (h - 1))
    yy, xx = np.mgrid[-radius:radius+1, -radius:radius+1]
    alpha = np.clip((radius - np.sqrt(xx*xx + yy*yy)) / max(radius * .35, 1), 0, 1).astype(np.float32)
    out = image.copy()
    for oy in range(-radius, radius + 1):
        ty, syy = dy + oy, sy + oy
        if not (0 <= ty < h and 0 <= syy < h):
            continue
        for ox in range(-radius, radius + 1):
            tx, sxx = dx + ox, sx + ox
            if 0 <= tx < w and 0 <= sxx < w and alpha[oy+radius, ox+radius] > 0:
                a = alpha[oy+radius, ox+radius]
                out[ty, tx] = out[ty, tx] * (1-a) + image[syy, sxx] * a
    return out


def apply_distraction_operations(image: np.ndarray, operations: Iterable[dict], inpaint_radius: float = 3.0) -> np.ndarray:
    """Apply ordered clone/heal/object/wire operations while preserving bit depth."""
    work, maximum, dtype = _u8(image)
    pending = np.zeros(work.shape[:2], np.uint8)
    for op in operations or []:
        if not bool(op.get("enabled", True)):
            continue
        kind = str(op.get("type", "heal"))
        if kind == "clone":
            if np.any(pending):
                work = cv2.inpaint(work, pending, float(inpaint_radius), cv2.INPAINT_TELEA)
                pending[:] = 0
            work = _clone(work.astype(np.float32), op).astype(np.uint8)
        elif kind in {"heal", "inpaint", "smart", "wire"}:
            pending = cv2.max(pending, operations_mask(work.shape, [op]))
    if np.any(pending):
        work = cv2.inpaint(work, pending, float(inpaint_radius), cv2.INPAINT_TELEA)
    return _restore(work, maximum, dtype)


def detect_dust_spots(image: np.ndarray, sensitivity: float = 60.0, max_radius: int = 20) -> list[dict]:
    """Find compact dark/bright spots using a robust local-background residual."""
    work, _, _ = _u8(image)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY) if work.ndim == 3 else work
    sigma = max(5.0, min(gray.shape[:2]) / 100.0)
    background = cv2.GaussianBlur(gray, (0, 0), sigma)
    residual = cv2.absdiff(gray, background).astype(np.float32)
    median, mad = float(np.median(residual)), float(np.median(np.abs(residual - np.median(residual))))
    threshold = median + max(2.0, 7.0 - float(sensitivity) / 18.0) * max(1.4826 * mad, 1.0)
    binary = (residual >= threshold).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, centers = cv2.connectedComponentsWithStats(binary)
    h, w = gray.shape
    spots = []
    for label in range(1, count):
        x, y, bw, bh, area = stats[label]
        radius = max(bw, bh) / 2.0
        fill = float(area) / max(float(bw * bh), 1.0)
        if 3 <= area <= np.pi * max_radius * max_radius and radius <= max_radius and fill >= .18:
            cx, cy = centers[label]
            spots.append({"type": "heal", "x": cx/(w-1), "y": cy/(h-1),
                          "radius": min(.08, (radius+2)/min(h, w)),
                          "confidence": min(1.0, float(residual[labels == label].mean()) / max(threshold*2, 1))})
    return sorted(spots, key=lambda s: s["confidence"], reverse=True)


def build_sensor_dust_map(images: Sequence[np.ndarray], sensitivity: float = 60.0,
                          recurrence: float = .5, max_radius: int = 20) -> list[dict]:
    """Return spots recurring at fixed sensor coordinates across a folder/sequence."""
    if not images:
        return []
    shape = images[0].shape[:2]
    if any(im.shape[:2] != shape for im in images):
        raise ValueError("Dust-map images must have matching dimensions")
    h, w = shape
    votes = np.zeros(shape, np.float32)
    for image in images:
        mask = operations_mask(shape, detect_dust_spots(image, sensitivity, max_radius))
        votes += (mask > 0).astype(np.float32)
    binary = (votes >= max(2 if len(images) > 1 else 1, int(np.ceil(len(images)*recurrence)))).astype(np.uint8)*255
    count, _, stats, centers = cv2.connectedComponentsWithStats(binary)
    result = []
    for idx in range(1, count):
        x, y, bw, bh, area = stats[idx]
        if area > 0:
            cx, cy = centers[idx]
            result.append({"type":"heal", "x":cx/(w-1), "y":cy/(h-1),
                           "radius":(max(bw, bh)/2+2)/min(h,w), "dust_map":True,
                           "confidence":float(votes[int(cy), int(cx)]/len(images))})
    return result


def reflection_mask(image: np.ndarray, sensitivity: float = 55.0, blur: float = 8.0) -> np.ndarray:
    work, _, _ = _u8(image)
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV).astype(np.float32)
    value, saturation = hsv[..., 2] / 255.0, hsv[..., 1] / 255.0
    local = cv2.GaussianBlur(value, (0, 0), max(2.0, min(work.shape[:2])/80.0))
    bright = np.clip((value - (.92 - sensitivity/500.0)) * 6.0, 0, 1)
    veiling = np.clip((local - .55) * 2.2, 0, 1) * np.clip((.65 - saturation)*2.0, 0, 1)
    mask = np.maximum(bright, veiling)
    if blur > 0:
        mask = cv2.GaussianBlur(mask, (0, 0), max(.5, float(blur)))
    return np.clip(mask*255, 0, 255).astype(np.uint8)


def edit_reflection_mask(mask: np.ndarray, strokes: Iterable[dict]) -> np.ndarray:
    """Replay normalized add/erase brush marks over an automatic mask."""
    result = mask.astype(np.float32) / 255.0
    h, w = result.shape
    for stroke in strokes or []:
        dab = np.zeros((h, w), np.float32)
        _circle(dab, float(stroke.get("x", .5)), float(stroke.get("y", .5)),
                float(stroke.get("radius", .02)), 1)
        feather = max(.5, float(stroke.get("feather", .25)) *
                      float(stroke.get("radius", .02)) * min(h, w))
        dab = cv2.GaussianBlur(dab, (0, 0), feather)
        if str(stroke.get("mode", "add")) == "erase":
            result *= 1.0 - np.clip(dab, 0, 1)
        else:
            result = np.maximum(result, np.clip(dab, 0, 1))
    return np.clip(result * 255, 0, 255).astype(np.uint8)


def apply_reflection_adjustment(image: np.ndarray, mask: np.ndarray, strength: float = 50.0,
                                highlights: float = -35.0, saturation: float = 0.0,
                                neutralize: float = 20.0, contrast: float = 10.0) -> np.ndarray:
    work, maximum, dtype = _u8(image)
    src = work.astype(np.float32)/255.0
    m = cv2.resize(mask, (work.shape[1], work.shape[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)/255.0
    m = np.clip(m * float(strength)/100.0, 0, 1)[..., None]
    target = src.copy()
    lum = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)[..., None]
    target += float(highlights)/100.0 * .45 * np.clip((lum-.45)*2, 0, 1)
    target = (target-.5)*(1+float(contrast)/100.0)+.5
    gray = np.repeat(lum, 3, axis=2)
    target = target*(1-float(neutralize)/100.0)+gray*(float(neutralize)/100.0)
    hsv = cv2.cvtColor(np.clip(target,0,1), cv2.COLOR_BGR2HSV)
    hsv[...,1] = np.clip(hsv[...,1]*(1+float(saturation)/100.0),0,1)
    target = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    out = np.clip(src*(1-m)+target*m, 0, 1)
    return _restore((out*255).astype(np.uint8), maximum, dtype)


def align_to_reference(reference: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    ref, _, _ = _u8(reference); mov, _, _ = _u8(moving)
    if mov.shape[:2] != ref.shape[:2]:
        mov = cv2.resize(mov, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    rg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY); mg = cv2.cvtColor(mov, cv2.COLOR_BGR2GRAY)
    warp = np.eye(2, 3, dtype=np.float32)
    score = 0.0
    try:
        score, warp = cv2.findTransformECC(rg.astype(np.float32)/255, mg.astype(np.float32)/255,
                                           warp, cv2.MOTION_AFFINE,
                                           (cv2.TERM_CRITERIA_EPS|cv2.TERM_CRITERIA_COUNT, 80, 1e-5))
        aligned = cv2.warpAffine(mov, warp, (ref.shape[1],ref.shape[0]), flags=cv2.INTER_LANCZOS4|cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REFLECT)
    except cv2.error:
        aligned = mov
    return aligned, warp, float(score)


def separate_reflections(images: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list]:
    """Experimental multi-image separation using robust aligned statistics."""
    if len(images) < 2:
        raise ValueError("Reflection separation needs at least two photographs")
    reference = images[0]
    aligned, diagnostics = [], []
    for index, image in enumerate(images):
        if index == 0:
            frame, warp, score = _u8(image)[0], np.eye(2, 3, dtype=np.float32), 1.0
        else:
            frame, warp, score = align_to_reference(reference, image)
        aligned.append(frame.astype(np.float32)/255.0)
        diagnostics.append({"frame":index, "score":score, "matrix":warp.tolist()})
    stack = np.stack(aligned)
    base = np.percentile(stack, 25, axis=0)
    reflection = np.clip(np.mean(np.maximum(stack-base[None,...], 0), axis=0)*2.0, 0, 1)
    energy = cv2.cvtColor(reflection.astype(np.float32), cv2.COLOR_BGR2GRAY)
    mask = np.clip(energy*5.0,0,1)
    spread = np.mean(np.std(stack, axis=0), axis=2)
    confidence = np.clip(spread*5.0,0,1)
    return ((base*255).astype(np.uint8), (reflection*255).astype(np.uint8),
            (mask*255).astype(np.uint8), (confidence*255).astype(np.uint8), diagnostics)


def smart_object_mask(image: np.ndarray, rect: Sequence[float]) -> np.ndarray:
    """GrabCut smart selection; rectangle is normalized x1,y1,x2,y2."""
    work, _, _ = _u8(image); h,w = work.shape[:2]
    x1,y1,x2,y2 = rect
    x, y = max(0,int(x1*w)), max(0,int(y1*h))
    rw, rh = max(2,min(w-x,int((x2-x1)*w))), max(2,min(h-y,int((y2-y1)*h)))
    mask = np.zeros((h,w),np.uint8); bg=np.zeros((1,65),np.float64); fg=np.zeros((1,65),np.float64)
    try:
        cv2.grabCut(work,mask,(x,y,rw,rh),bg,fg,4,cv2.GC_INIT_WITH_RECT)
        return np.where((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD),255,0).astype(np.uint8)
    except cv2.error:
        cv2.rectangle(mask,(x,y),(x+rw,y+rh),255,-1)
        return mask


def load_images(paths: Sequence[str]) -> list[np.ndarray]:
    images=[]
    for path in paths:
        data=np.fromfile(str(Path(path)),dtype=np.uint8); image=cv2.imdecode(data,cv2.IMREAD_COLOR)
        if image is None: raise ValueError(f"Could not read {path}")
        images.append(image)
    return images
