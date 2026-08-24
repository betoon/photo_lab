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


def match_exposure_wb(images: List[np.ndarray], ref_idx: int = 0,
                      strength: float = 1.0) -> List[np.ndarray]:
    """Simple per-channel gain match toward the reference frame (mean RGB).

    Helps OpenCV stitch when brackets or auto-exposure drifted between frames.
    """
    if not images or ref_idx < 0 or ref_idx >= len(images):
        return images
    ref = images[ref_idx].astype(np.float32)
    ref_mean = np.median(ref.reshape(-1, ref.shape[-1]), axis=0) + 1e-3
    amount = float(np.clip(strength, 0.0, 1.0))
    out = []
    for i, img in enumerate(images):
        if i == ref_idx:
            out.append(img)
            continue
        x = img.astype(np.float32)
        mean = np.median(x.reshape(-1, x.shape[-1]), axis=0) + 1e-3
        gains = ref_mean / mean
        # Soften extreme gains
        gains = np.clip(gains, 0.5, 2.0)
        gains = 1.0 + (gains - 1.0) * amount
        y = np.clip(x * gains, 0, 255).astype(np.uint8)
        out.append(y)
    return out


def analyze_panorama_sequence(paths: List[str], max_dim: int = 900) -> dict:
    """Estimate adjacent-frame overlap and homography confidence with ORB."""
    if len(paths) < 2:
        raise ValueError("Need at least 2 panorama frames")
    images = [load_pano_frame(path, max_dim=max_dim) for path in paths]
    orb = cv2.ORB_create(nfeatures=1800)
    pairs = []
    for index in range(len(images) - 1):
        left = cv2.cvtColor(images[index], cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(images[index + 1], cv2.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(left, None)
        kp2, des2 = orb.detectAndCompute(right, None)
        good, inliers = [], 0
        if des1 is not None and des2 is not None:
            # Mutual nearest-neighbour matching is dependable on panorama strips,
            # including repeated detail where a strict ratio test rejects too much.
            matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(des1, des2)
            matches = sorted(matches, key=lambda match: match.distance)
            if matches:
                cutoff = max(28.0, min(64.0, float(np.median([m.distance for m in matches]))))
                good = [match for match in matches[:300] if match.distance <= cutoff]
            if len(good) >= 4:
                src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
                dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
                _matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 4.0)
                inliers = int(mask.sum()) if mask is not None else 0
        ratio = float(inliers / max(len(good), 1))
        score = float(np.clip((inliers / 35.0) * ratio, 0.0, 1.0))
        pairs.append({
            "left_index": index, "right_index": index + 1,
            "features_left": len(kp1), "features_right": len(kp2),
            "matches": len(good), "inliers": inliers,
            "inlier_ratio": ratio, "score": score,
            "quality": "good" if score >= 0.55 else "fair" if score >= 0.25 else "weak",
        })
    return {"frames": len(paths), "pairs": pairs,
            "weak_pairs": sum(pair["quality"] == "weak" for pair in pairs)}


def reproject_panorama(image: np.ndarray, projection: str = "original",
                       strength: float = 1.0, field_of_view: float = 120.0,
                       border_mode: str = "reflect") -> np.ndarray:
    """Apply a conservative post-stitch projection adjustment.

    This does not replace OpenCV's camera warper. It gives photographers a
    continuous finishing control for edge stretch and vertical mapping while
    preserving the stitched canvas and an exact no-op default.
    """
    name = (projection or "original").lower()
    amount = float(np.clip(strength, 0.0, 1.0))
    if image is None or name in ("original", "automatic", "none") or amount <= 0:
        return image.copy() if image is not None else image
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx / max(w - 1, 1)) * 2.0 - 1.0
    v = (yy / max(h - 1, 1)) * 2.0 - 1.0
    half_fov = np.deg2rad(float(np.clip(field_of_view, 45.0, 170.0))) * 0.5
    tan_half = max(float(np.tan(half_fov)), 1e-4)
    src_u, src_v = u.copy(), v.copy()
    if name == "cylindrical":
        src_u = np.tan(u * half_fov) / tan_half
        src_v = v * np.sqrt(1.0 + (src_u * tan_half) ** 2)
    elif name == "rectilinear":
        src_u = np.arctan(u * tan_half) / half_fov
        src_v = v / np.sqrt(1.0 + (u * tan_half) ** 2)
    elif name == "mercator":
        latitude = np.deg2rad(np.clip(v * 80.0, -80.0, 80.0))
        mercator = np.log(np.tan(np.pi / 4.0 + latitude / 2.0))
        max_mercator = np.log(np.tan(np.pi / 4.0 + np.deg2rad(80.0) / 2.0))
        src_v = mercator / max_mercator
    else:
        raise ValueError(f"Unknown panorama projection: {projection}")
    src_u = u * (1.0 - amount) + src_u * amount
    src_v = v * (1.0 - amount) + src_v * amount
    map_x = ((src_u + 1.0) * 0.5 * max(w - 1, 1)).astype(np.float32)
    map_y = ((src_v + 1.0) * 0.5 * max(h - 1, 1)).astype(np.float32)
    borders = {
        "black": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT_101,
    }
    return cv2.remap(
        image, map_x, map_y, cv2.INTER_LANCZOS4,
        borderMode=borders.get((border_mode or "reflect").lower(), cv2.BORDER_REFLECT_101),
    )


def detect_panorama_seams(image: np.ndarray, max_seams: int = 8) -> List[dict]:
    """Find suspicious column-wide tonal discontinuities in a stitched result."""
    if image is None or image.ndim != 3 or image.shape[1] < 16:
        return []
    lab = cv2.cvtColor(_to_bgr8(image), cv2.COLOR_BGR2LAB).astype(np.float32)
    delta = np.mean(np.abs(lab[:, 1:] - lab[:, :-1]), axis=2)
    column_energy = np.median(delta, axis=0)
    baseline = cv2.GaussianBlur(column_energy.reshape(1, -1), (0, 0), 9.0).ravel()
    excess = np.maximum(column_energy - baseline, 0.0)
    median = float(np.median(excess))
    mad = float(np.median(np.abs(excess - median))) + 1e-4
    robust_z = (excess - median) / (1.4826 * mad)
    h, w = image.shape[:2]
    candidates = []
    for x in np.argsort(robust_z)[::-1]:
        if robust_z[x] < 6.0:
            break
        # A stitch seam normally affects a substantial portion of the image
        # height; this rejects isolated poles, windows, and other short edges.
        local_threshold = max(float(np.median(delta[:, x]) + 2.5 * np.std(delta[:, x])), 8.0)
        coverage = float(np.mean(delta[:, x] > min(local_threshold, column_energy[x] * 0.8)))
        if coverage < 0.18:
            continue
        xpos = int(x + 1)
        if xpos < max(4, int(w * 0.02)) or xpos > min(w - 5, int(w * 0.98)):
            continue
        if any(abs(xpos - item["x"]) < max(8, w // 80) for item in candidates):
            continue
        candidates.append({
            "x": xpos, "x_normalized": float(xpos / max(w - 1, 1)),
            "score": float(np.clip(robust_z[x] / 20.0, 0.0, 1.0)),
            "coverage": coverage,
        })
        if len(candidates) >= int(max_seams):
            break
    return sorted(candidates, key=lambda item: item["x"])


def refine_panorama_seams(image: np.ndarray, strength: float = 0.0,
                          radius: int = 12, seams=None) -> Tuple[np.ndarray, List[dict]]:
    """Softly blend only automatically detected, column-wide stitch seams."""
    amount = float(np.clip(strength, 0.0, 1.0))
    found = list(detect_panorama_seams(image) if seams is None else seams)
    if image is None or amount <= 0.0 or not found:
        return image.copy() if image is not None else image, found
    radius = max(2, int(radius))
    source = _to_bgr8(image)
    softened = cv2.GaussianBlur(source, (0, 0), sigmaX=max(1.0, radius / 3.0), sigmaY=0.15)
    h, w = source.shape[:2]
    weight = np.zeros((h, w), np.float32)
    for seam in found:
        x = int(seam["x"])
        lo, hi = max(0, x - radius), min(w, x + radius + 1)
        offsets = np.arange(lo, hi, dtype=np.float32) - x
        band = np.exp(-0.5 * (offsets / max(radius / 2.5, 1.0)) ** 2)
        weight[:, lo:hi] = np.maximum(weight[:, lo:hi], band[None, :])
    weight = (weight * amount)[..., None]
    result = source.astype(np.float32) * (1.0 - weight) + softened.astype(np.float32) * weight
    return np.clip(result, 0, 255).astype(np.uint8), found


def stitch_panorama(
    paths: List[str],
    mode: str = "panoramas",
    max_dim: int = 0,
    try_use_gpu: bool = False,
    match_exposure: bool = True,
    order_by_time: bool = False,
    exposure_reference: int = 0,
    exposure_strength: float = 1.0,
    confidence_threshold: float = 1.0,
    wave_correction: bool = True,
    crop_borders: bool = True,
    output_projection: str = "original",
    projection_strength: float = 1.0,
    projection_fov: float = 120.0,
    projection_border: str = "reflect",
    seam_refine_strength: float = 0.0,
    seam_refine_radius: int = 12,
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
        images = match_exposure_wb(
            images, ref_idx=int(np.clip(exposure_reference, 0, len(images) - 1)),
            strength=exposure_strength,
        )

    prog("Stitching (OpenCV)…", 0.4)
    stitcher = _make_stitcher(mode)
    if hasattr(stitcher, "setPanoConfidenceThresh"):
        stitcher.setPanoConfidenceThresh(float(confidence_threshold))
    if hasattr(stitcher, "setWaveCorrection"):
        stitcher.setWaveCorrection(bool(wave_correction and mode != "scans"))
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
        "exposure_reference": int(exposure_reference),
        "exposure_strength": float(exposure_strength),
        "confidence_threshold": float(confidence_threshold),
        "wave_correction": bool(wave_correction),
        "crop_borders": bool(crop_borders),
        "output_projection": str(output_projection),
        "projection_strength": float(projection_strength),
        "projection_fov": float(projection_fov),
        "projection_border": str(projection_border),
        "seam_refine_strength": float(seam_refine_strength),
        "seam_refine_radius": int(seam_refine_radius),
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

    if (output_projection or "original").lower() not in ("original", "automatic", "none"):
        prog(f"Applying {output_projection} projection…", 0.86)
        pano = reproject_panorama(
            pano, output_projection, projection_strength,
            projection_fov, projection_border,
        )

    seams = detect_panorama_seams(pano)
    report["suspected_seams"] = seams
    if seam_refine_strength and seams:
        prog(f"Refining {len(seams)} suspected seam(s)…", 0.88)
        pano, _ = refine_panorama_seams(
            pano, strength=seam_refine_strength,
            radius=seam_refine_radius, seams=seams,
        )

    # Crop mostly-black borders from the warped canvas
    prog("Cropping borders…", 0.9)
    if crop_borders:
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
