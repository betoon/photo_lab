"""Focus stacking engine for PhotoLab.

Lightweight align + fuse pipeline inspired by typical focus-stack workflows:
  - ECC / ORB registration
  - Laplacian / variance focus measures
  - Depth-map, weighted, pyramid, and average fusion

Produces a new BGR uint8 (or float) image. Keeps Qt out of this module.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from imaging import load_image, is_raw, _silent_imread


def _to_gray32(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        g = img.astype(np.float32)
    else:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    if g.max() > 1.5:
        g = g / 255.0
    return g


def _to_bgr8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    x = np.clip(img, 0, 1) if img.max() <= 1.5 else np.clip(img, 0, 255)
    if x.max() <= 1.5:
        x = x * 255.0
    return x.astype(np.uint8)


def load_stack_frame(path: str, max_dim: int = 0) -> np.ndarray:
    """Load a frame as BGR uint8, optionally downscaling long edge."""
    try:
        img, _ = load_image(path, use_camera_wb=True)
    except Exception:
        img = _silent_imread(path)
        if img is None:
            raise RuntimeError(f"Could not load {path}")
    img = _to_bgr8(img)
    if max_dim and max(img.shape[:2]) > max_dim:
        h, w = img.shape[:2]
        scale = max_dim / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return img


def align_ecc(
    ref_gray: np.ndarray,
    mov_gray: np.ndarray,
    mode: str = "affine",
    iterations: int = 80,
    eps: float = 1e-5,
) -> Tuple[np.ndarray, float]:
    """Return (2x3 or 3x3 warp matrix, confidence-ish score)."""
    mode = (mode or "affine").lower()
    if mode == "translation":
        warp_mode = cv2.MOTION_TRANSLATION
        warp = np.eye(2, 3, dtype=np.float32)
    elif mode == "euclidean" or mode == "rigid":
        warp_mode = cv2.MOTION_EUCLIDEAN
        warp = np.eye(2, 3, dtype=np.float32)
    elif mode == "homography":
        warp_mode = cv2.MOTION_HOMOGRAPHY
        warp = np.eye(3, 3, dtype=np.float32)
    else:
        warp_mode = cv2.MOTION_AFFINE
        warp = np.eye(2, 3, dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, eps)
    try:
        cc, warp = cv2.findTransformECC(
            ref_gray, mov_gray, warp, warp_mode, criteria, None, 5
        )
        return warp, float(cc)
    except cv2.error:
        # identity on failure
        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            return np.eye(3, 3, dtype=np.float32), 0.0
        return np.eye(2, 3, dtype=np.float32), 0.0


def align_orb(
    ref_gray: np.ndarray,
    mov_gray: np.ndarray,
    perspective: bool = False,
) -> Tuple[np.ndarray, float]:
    """ORB + RANSAC affine or homography."""
    ref_u8 = np.clip(ref_gray * 255, 0, 255).astype(np.uint8)
    mov_u8 = np.clip(mov_gray * 255, 0, 255).astype(np.uint8)
    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(ref_u8, None)
    k2, d2 = orb.detectAndCompute(mov_u8, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return np.eye(2, 3, dtype=np.float32), 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)
    if len(good) < 8:
        return np.eye(2, 3, dtype=np.float32), 0.0
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    if perspective:
        H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 4.0)
        if H is None:
            return np.eye(3, 3, dtype=np.float32), 0.0
        conf = float(mask.mean()) if mask is not None else 0.5
        return H.astype(np.float32), conf
    M, mask = cv2.estimateAffinePartial2D(dst, src, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    if M is None:
        return np.eye(2, 3, dtype=np.float32), 0.0
    conf = float(mask.mean()) if mask is not None else 0.5
    return M.astype(np.float32), conf


def warp_image(img: np.ndarray, M: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    if M.shape == (3, 3):
        return cv2.warpPerspective(img, M, (w, h), flags=cv2.INTER_LANCZOS4,
                                   borderMode=cv2.BORDER_REFLECT)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LANCZOS4,
                          borderMode=cv2.BORDER_REFLECT)


def focus_measure(gray: np.ndarray, radius: int = 3) -> np.ndarray:
    """Hybrid Laplacian energy + local variance focus map."""
    g = gray.astype(np.float32)
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    energy = lap * lap
    k = max(3, radius * 2 + 1)
    mean = cv2.GaussianBlur(g, (k, k), 0)
    mean2 = cv2.GaussianBlur(g * g, (k, k), 0)
    var = np.maximum(mean2 - mean * mean, 0)
    energy_s = cv2.GaussianBlur(energy, (k, k), 0)
    return energy_s + 0.5 * var


def fuse_average(frames: List[np.ndarray]) -> np.ndarray:
    acc = np.zeros_like(frames[0], dtype=np.float32)
    for f in frames:
        acc += f.astype(np.float32)
    return np.clip(acc / max(len(frames), 1), 0, 255).astype(np.uint8)


def fuse_weighted(frames: List[np.ndarray], radius: int = 3) -> np.ndarray:
    h, w = frames[0].shape[:2]
    weights = []
    for f in frames:
        g = _to_gray32(f)
        weights.append(focus_measure(g, radius))
    stack_w = np.stack(weights, axis=0)
    # soft max
    stack_w = stack_w - stack_w.max(axis=0, keepdims=True)
    stack_w = np.exp(stack_w / 0.15)
    stack_w = stack_w / (stack_w.sum(axis=0, keepdims=True) + 1e-8)
    out = np.zeros((h, w, 3), dtype=np.float32)
    for i, f in enumerate(frames):
        out += f.astype(np.float32) * stack_w[i][..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def fuse_depth_map(
    frames: List[np.ndarray],
    radius: int = 3,
    smooth: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Hard-ish winner map with boundary smoothing. Returns (result, depth_u8)."""
    measures = [focus_measure(_to_gray32(f), radius) for f in frames]
    stack = np.stack(measures, axis=0)
    winner = np.argmax(stack, axis=0).astype(np.int32)
    if smooth and smooth > 0:
        k = smooth * 2 + 1
        # majority filter via blur of one-hot
        depth = winner.astype(np.float32)
        depth = cv2.medianBlur(depth.astype(np.uint8), k if k % 2 else k + 1).astype(np.float32)
        winner = np.clip(np.round(depth), 0, len(frames) - 1).astype(np.int32)
    h, w = frames[0].shape[:2]
    out = np.zeros((h, w, 3), dtype=np.float32)
    for i, f in enumerate(frames):
        mask = (winner == i).astype(np.float32)
        if smooth and smooth > 0:
            mask = cv2.GaussianBlur(mask, (smooth * 2 + 1, smooth * 2 + 1), 0)
        out += f.astype(np.float32) * mask[..., None]
    # renormalize soft masks
    norm = np.zeros((h, w), dtype=np.float32)
    for i in range(len(frames)):
        mask = (winner == i).astype(np.float32)
        if smooth and smooth > 0:
            mask = cv2.GaussianBlur(mask, (smooth * 2 + 1, smooth * 2 + 1), 0)
        norm += mask
    out /= (norm[..., None] + 1e-8)
    depth_u8 = (winner.astype(np.float32) * (255.0 / max(len(frames) - 1, 1))).astype(np.uint8)
    return np.clip(out, 0, 255).astype(np.uint8), depth_u8


def fuse_pyramid(frames: List[np.ndarray], levels: int = 5, radius: int = 3) -> np.ndarray:
    """Laplacian pyramid fusion guided by focus measure."""
    levels = max(2, min(levels, 7))

    def build_gauss_pyr(img, n):
        pyr = [img.astype(np.float32)]
        for _ in range(n - 1):
            pyr.append(cv2.pyrDown(pyr[-1]))
        return pyr

    def build_lap_pyr(img, n):
        g = build_gauss_pyr(img, n)
        lap = []
        for i in range(n - 1):
            size = (g[i].shape[1], g[i].shape[0])
            up = cv2.pyrUp(g[i + 1], dstsize=size)
            lap.append(g[i] - up)
        lap.append(g[-1])
        return lap

    # Focus maps at full res, then pyramid of weights
    measures = [focus_measure(_to_gray32(f), radius) for f in frames]
    # Softmax weights
    stack_w = np.stack(measures, axis=0)
    stack_w = stack_w - stack_w.max(axis=0, keepdims=True)
    stack_w = np.exp(stack_w / 0.2)
    stack_w = stack_w / (stack_w.sum(axis=0, keepdims=True) + 1e-8)

    lap_pyrs = [build_lap_pyr(f, levels) for f in frames]
    weight_pyrs = [build_gauss_pyr(stack_w[i], levels) for i in range(len(frames))]

    fused_lap = []
    for lvl in range(levels):
        acc = np.zeros_like(lap_pyrs[0][lvl])
        wsum = np.zeros(lap_pyrs[0][lvl].shape[:2], dtype=np.float32)
        for i in range(len(frames)):
            w = weight_pyrs[i][lvl]
            if w.shape[:2] != acc.shape[:2]:
                w = cv2.resize(w, (acc.shape[1], acc.shape[0]), interpolation=cv2.INTER_LINEAR)
            acc += lap_pyrs[i][lvl] * w[..., None]
            wsum += w
        fused_lap.append(acc / (wsum[..., None] + 1e-8))

    # Reconstruct
    img = fused_lap[-1]
    for lvl in range(levels - 2, -1, -1):
        size = (fused_lap[lvl].shape[1], fused_lap[lvl].shape[0])
        img = cv2.pyrUp(img, dstsize=size) + fused_lap[lvl]
    return np.clip(img, 0, 255).astype(np.uint8)


def common_area_crop(frames: List[np.ndarray], margin: int = 2) -> List[np.ndarray]:
    """Conservative crop removing a small border (proxy for valid overlap)."""
    if not frames:
        return frames
    h, w = frames[0].shape[:2]
    m = max(0, min(margin, h // 20, w // 20))
    if m == 0:
        return frames
    return [f[m:h - m, m:w - m] for f in frames]


def focus_stack(
    paths: List[str],
    align_mode: str = "ecc_affine",
    fusion_mode: str = "depth",
    reference: str = "middle",
    max_dim: int = 0,
    focus_radius: int = 3,
    boundary_smooth: int = 5,
    pyramid_levels: int = 5,
    crop_common: bool = True,
    progress_cb=None,
) -> Tuple[np.ndarray, Optional[np.ndarray], dict]:
    """
    Align and fuse a focus stack.

    Returns (result_bgr_u8, depth_u8_or_None, report_dict).
    """
    if len(paths) < 2:
        raise ValueError("Need at least 2 frames to focus-stack")

    n = len(paths)
    if reference == "first":
        ref_idx = 0
    elif reference == "last":
        ref_idx = n - 1
    else:
        ref_idx = n // 2

    report = {
        "frames": n,
        "reference": ref_idx,
        "align_mode": align_mode,
        "fusion_mode": fusion_mode,
        "scores": [],
    }

    def prog(msg, frac=None):
        if progress_cb:
            progress_cb(msg, frac)

    prog("Loading frames…", 0.02)
    frames = []
    for i, p in enumerate(paths):
        frames.append(load_stack_frame(p, max_dim=max_dim))
        prog(f"Loaded {i + 1}/{n}", 0.05 + 0.2 * (i + 1) / n)

    # Match sizes to reference
    rh, rw = frames[ref_idx].shape[:2]
    sized = []
    for f in frames:
        if f.shape[0] != rh or f.shape[1] != rw:
            sized.append(cv2.resize(f, (rw, rh), interpolation=cv2.INTER_AREA))
        else:
            sized.append(f)
    frames = sized

    ref_gray = _to_gray32(frames[ref_idx])
    aligned = [None] * n
    aligned[ref_idx] = frames[ref_idx]
    report["scores"].append({"index": ref_idx, "score": 1.0, "path": paths[ref_idx]})

    for i, f in enumerate(frames):
        if i == ref_idx:
            continue
        prog(f"Aligning frame {i + 1}/{n}…", 0.25 + 0.35 * i / n)
        g = _to_gray32(f)
        mode = (align_mode or "ecc_affine").lower()
        if mode.startswith("orb"):
            M, score = align_orb(ref_gray, g, perspective=("homography" in mode or "perspective" in mode))
        else:
            ecc = "affine"
            if "translation" in mode:
                ecc = "translation"
            elif "rigid" in mode or "euclidean" in mode:
                ecc = "euclidean"
            elif "homography" in mode:
                ecc = "homography"
            M, score = align_ecc(ref_gray, g, mode=ecc)
        aligned[i] = warp_image(f, M, (rh, rw))
        report["scores"].append({"index": i, "score": score, "path": paths[i]})

    frames = aligned
    if crop_common:
        frames = common_area_crop(frames, margin=4)

    prog(f"Fusing ({fusion_mode})…", 0.7)
    depth = None
    fm = (fusion_mode or "depth").lower()
    if fm in ("average", "mean"):
        result = fuse_average(frames)
    elif fm in ("weighted", "weight"):
        result = fuse_weighted(frames, radius=focus_radius)
    elif fm in ("pyramid", "laplacian"):
        result = fuse_pyramid(frames, levels=pyramid_levels, radius=focus_radius)
    else:
        result, depth = fuse_depth_map(frames, radius=focus_radius, smooth=boundary_smooth)

    prog("Done", 1.0)
    return result, depth, report
