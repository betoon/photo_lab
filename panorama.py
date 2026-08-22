"""Panorama stitching for PhotoLab (OpenCV Stitcher v1).

Uses cv2.Stitcher for feature matching, warping, seam finding, and multi-band
blending. Suitable for ordered rows with good overlap and limited parallax.
Not a replacement for dedicated tools (PTGui, Hugin) on hard sets.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np

from imaging import load_image, _silent_imread


def _to_bgr8(img: np.ndarray) -> np.ndarray:
    if img is None:
        raise ValueError("empty image")
    if img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    if x.max() <= 1.5:
        x = x * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def load_pano_frame(path: str, max_dim: int = 0) -> np.ndarray:
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


def _make_stitcher(mode: str = "auto"):
    """Create OpenCV Stitcher. mode: auto | panoramas | scans."""
    mode = (mode or "auto").lower()
    # OpenCV 4.x
    if hasattr(cv2, "Stitcher_create"):
        if mode == "scans":
            try:
                return cv2.Stitcher_create(cv2.Stitcher_SCANS)
            except Exception:
                return cv2.Stitcher_create()
        if mode == "panoramas":
            try:
                return cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
            except Exception:
                return cv2.Stitcher_create()
        return cv2.Stitcher_create()
    # Older API
    if hasattr(cv2, "createStitcher"):
        try:
            return cv2.createStitcher(False)
        except Exception:
            return cv2.createStitcher()
    raise RuntimeError("This OpenCV build has no Stitcher module")


_STATUS_NAMES = {
    0: "OK",
    1: "ERR_NEED_MORE_IMGS",
    2: "ERR_HOMOGRAPHY_EST_FAIL",
    3: "ERR_CAMERA_PARAMS_ADJUST_FAIL",
}


def order_paths_by_capture_time(paths: List[str]) -> List[str]:
    """Sort image paths by EXIF/capture datetime (then filename)."""
    from catalog import resolve_capture_datetime

    keyed = []
    for p in paths:
        try:
            dt, _ = resolve_capture_datetime(p)
            keyed.append((dt, os.path.basename(p).lower(), p))
        except Exception:
            keyed.append((None, os.path.basename(p).lower(), p))
    # None datetimes last
    keyed.sort(key=lambda t: (t[0] is None, t[0] or 0, t[1]))
    return [p for _dt, _n, p in keyed]


def match_exposure_wb(images: List[np.ndarray], ref_idx: int = 0) -> List[np.ndarray]:
    """Simple per-channel gain match toward the reference frame (mean RGB).

    Helps OpenCV stitch when brackets or auto-exposure drifted between frames.
    """
    if not images or ref_idx < 0 or ref_idx >= len(images):
        return images
    ref = images[ref_idx].astype(np.float32)
    ref_mean = ref.reshape(-1, ref.shape[-1]).mean(axis=0) + 1e-3
    out = []
    for i, img in enumerate(images):
        if i == ref_idx:
            out.append(img)
            continue
        x = img.astype(np.float32)
        mean = x.reshape(-1, x.shape[-1]).mean(axis=0) + 1e-3
        gains = ref_mean / mean
        # Soften extreme gains
        gains = np.clip(gains, 0.5, 2.0)
        y = np.clip(x * gains, 0, 255).astype(np.uint8)
        out.append(y)
    return out


def stitch_panorama(
    paths: List[str],
    mode: str = "panoramas",
    max_dim: int = 0,
    try_use_gpu: bool = False,
    match_exposure: bool = True,
    order_by_time: bool = False,
    progress_cb=None,
) -> Tuple[np.ndarray, dict]:
    """
    Stitch images into a panorama.

    Returns (result_bgr_u8, report_dict).
    Raises RuntimeError with a clear message on failure.
    """
    if len(paths) < 2:
        raise ValueError("Need at least 2 images to stitch a panorama")

    def prog(msg, frac=None):
        if progress_cb:
            progress_cb(msg, frac)

    ordered = list(paths)
    if order_by_time:
        prog("Ordering by capture time…", 0.02)
        ordered = order_paths_by_capture_time(ordered)

    prog("Loading frames…", 0.05)
    images = []
    for i, p in enumerate(ordered):
        images.append(load_pano_frame(p, max_dim=max_dim))
        prog(f"Loaded {i + 1}/{len(ordered)}", 0.05 + 0.25 * (i + 1) / len(ordered))

    if match_exposure and len(images) >= 2:
        prog("Matching exposure / WB…", 0.32)
        images = match_exposure_wb(images, ref_idx=0)

    prog("Stitching (OpenCV)…", 0.4)
    stitcher = _make_stitcher(mode)
    # try_use_gpu is ignored on most CPU builds; kept for future
    try:
        status, pano = stitcher.stitch(images)
    except cv2.error as e:
        raise RuntimeError(f"OpenCV stitch failed: {e}") from e

    status_i = int(status) if status is not None else -1
    name = _STATUS_NAMES.get(status_i, f"status={status_i}")
    report = {
        "frames": len(ordered),
        "mode": mode,
        "max_dim": max_dim,
        "status": status_i,
        "status_name": name,
        "paths": list(ordered),
        "match_exposure": bool(match_exposure),
        "order_by_time": bool(order_by_time),
        "projection_note": (
            "OpenCV Stitcher_PANORAMA uses a spherical-like projection for wide "
            "horizons; SCANS is closer to a planar/cylindrical flat-copy. "
            "For true cylindrical control use Hugin/PTGui."
        ),
    }

    if status_i != 0 and getattr(cv2, "Stitcher_OK", 0) != status_i:
        # cv2.Stitcher_OK is 0
        hints = {
            1: "Need more images or more overlap.",
            2: "Could not estimate geometry — check order, overlap (~30%+), and parallax.",
            3: "Camera parameter adjustment failed — try fewer frames or lower resolution.",
        }
        hint = hints.get(status_i, "Stitching failed.")
        raise RuntimeError(f"Panorama stitch failed ({name}). {hint}")

    if pano is None or pano.size == 0:
        raise RuntimeError("Stitch returned an empty image")

    # Crop mostly-black borders from the warped canvas
    prog("Cropping borders…", 0.9)
    pano = _crop_black_borders(pano)
    report["result_size"] = [int(pano.shape[1]), int(pano.shape[0])]
    prog("Done", 1.0)
    return pano, report


def _crop_black_borders(img: np.ndarray, thresh: int = 8) -> np.ndarray:
    """Remove large empty borders left by warping."""
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    mask = gray > thresh
    coords = np.column_stack(np.where(mask))
    if coords.size == 0:
        return img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    # small pad safety
    h, w = gray.shape[:2]
    y0, x0 = max(0, y0), max(0, x0)
    y1, x1 = min(h, y1), min(w, x1)
    if y1 - y0 < 10 or x1 - x0 < 10:
        return img
    return img[y0:y1, x0:x1]
