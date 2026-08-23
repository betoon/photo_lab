from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
import cv2
import numpy as np


@dataclass
class AlignmentResult:
    images: list[np.ndarray]
    transforms: list[np.ndarray]
    common_roi: tuple[int, int, int, int]
    scores: list[float]
    used_indices: list[int]
    warnings: list[str]


def _gray8(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
    return np.clip(gray * 255, 0, 255).astype(np.uint8)


def _ecc(reference: np.ndarray, moving: np.ndarray, mode: int, iterations: int, epsilon: float,
         initial: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    shape = (3, 3) if mode == cv2.MOTION_HOMOGRAPHY else (2, 3)
    warp = (np.eye(3, dtype=np.float32) if shape == (3, 3) else np.eye(2, 3, dtype=np.float32)) if initial is None else initial.astype(np.float32).copy()
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon)
    score, warp = cv2.findTransformECC(_gray8(reference), _gray8(moving), warp, mode, criteria, None, 5)
    return warp, float(score)


def _ecc_multiscale(reference: np.ndarray, moving: np.ndarray, mode: int, iterations: int,
                    epsilon: float, levels: int = 3, proxy_dimension: int = 1800) -> tuple[np.ndarray, float]:
    """Coarse-to-fine ECC with transform propagation for a wider convergence basin."""
    levels = max(1, min(int(levels), 5))
    warp = np.eye(3, dtype=np.float32) if mode == cv2.MOTION_HOMOGRAPHY else np.eye(2, 3, dtype=np.float32)
    score = 0.0
    initial = min(1.0 / (2 ** (levels - 1)), proxy_dimension / max(reference.shape[:2])) if proxy_dimension > 0 else 1.0 / (2 ** (levels - 1))
    scales = []
    scale = initial
    while scale < .999 and len(scales) < levels: scales.append(scale); scale = min(1.0, scale * 2)
    if not scales or scales[-1] < .999: scales.append(1.0)
    previous_scale = None
    for scale in scales:
        if min(reference.shape[:2]) * scale < 48:
            continue
        if previous_scale is not None:
            ratio = scale / previous_scale
            warp[0, 2] *= ratio; warp[1, 2] *= ratio
        size = (max(16, round(reference.shape[1] * scale)), max(16, round(reference.shape[0] * scale)))
        ref_small = cv2.resize(reference, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        mov_small = cv2.resize(moving, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        warp, score = _ecc(ref_small, mov_small, mode, max(40, iterations // levels), epsilon, warp)
        previous_scale = scale
    return warp, score


def _features(reference: np.ndarray, moving: np.ndarray, homography: bool) -> tuple[np.ndarray, float]:
    orb = cv2.ORB_create(nfeatures=5000, fastThreshold=7)
    k1, d1 = orb.detectAndCompute(_gray8(reference), None)
    k2, d2 = orb.detectAndCompute(_gray8(moving), None)
    if d1 is None or d2 is None:
        raise RuntimeError("Not enough features for alignment")
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d2, d1, k=2)
    good = [a for a, b in pairs if a.distance < 0.75 * b.distance]
    if len(good) < (8 if homography else 4):
        raise RuntimeError("Insufficient reliable feature matches")
    src = np.float32([k2[m.queryIdx].pt for m in good])
    dst = np.float32([k1[m.trainIdx].pt for m in good])
    if homography:
        matrix, mask = cv2.findHomography(src, dst, cv2.RANSAC, 2.5)
    else:
        matrix, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if matrix is None:
        raise RuntimeError("Transform estimation failed")
    return matrix.astype(np.float32), float(mask.mean())


def align_stack(images: list[np.ndarray], method: str = "ecc_affine", reference_index: int | None = None,
                crop_common: bool = True, iterations: int = 150, epsilon: float = 1e-6,
                multiscale: bool = True, pyramid_scales: int = 3,
                proxy_dimension: int = 1800, recover_failed: bool = False,
                cancelled: Callable[[], bool] = lambda: False,
                progress: Callable[[int, str], None] = lambda *_: None) -> AlignmentResult:
    if not images:
        raise ValueError("No images supplied")
    h, w = images[0].shape[:2]
    if any(im.shape[:2] != (h, w) for im in images):
        raise ValueError("All images must have identical dimensions before alignment")
    ref_i = len(images) // 2 if reference_index is None else reference_index
    ref = images[ref_i]
    aligned, transforms, masks, scores, used_indices, warnings = [], [], [], [], [], []
    mode_map = {"ecc_translation": cv2.MOTION_TRANSLATION, "ecc_rigid": cv2.MOTION_EUCLIDEAN,
                "ecc_affine": cv2.MOTION_AFFINE, "ecc_homography": cv2.MOTION_HOMOGRAPHY}
    for i, image in enumerate(images):
        if cancelled():
            raise InterruptedError("Cancelled")
        progress(int(45 * i / len(images)), f"Aligning {i + 1}/{len(images)}")
        try:
            if i == ref_i or method == "none":
                matrix, score = np.eye(2, 3, dtype=np.float32), 1.0
            elif method.startswith("feature"):
                matrix, score = _features(ref, image, method == "feature_homography")
            else:
                matrix, score = (_ecc_multiscale(ref, image, mode_map[method], iterations, epsilon, pyramid_scales, proxy_dimension)
                                 if multiscale else _ecc(ref, image, mode_map[method], iterations, epsilon))
        except Exception as exc:
            if not recover_failed or i == ref_i: raise
            warnings.append(f"Frame {i + 1} skipped after alignment failure: {exc}"); progress(int(45 * i / len(images)), warnings[-1]); continue
        is_h = matrix.shape == (3, 3)
        flags = cv2.INTER_LANCZOS4 | cv2.WARP_INVERSE_MAP if method.startswith("ecc") else cv2.INTER_LANCZOS4
        warp_fn = cv2.warpPerspective if is_h else cv2.warpAffine
        warped = warp_fn(image, matrix, (w, h), flags=flags, borderMode=cv2.BORDER_REFLECT101)
        mask = warp_fn(np.ones((h, w), np.uint8), matrix, (w, h), flags=(cv2.INTER_NEAREST | (cv2.WARP_INVERSE_MAP if method.startswith("ecc") else 0)), borderValue=0)
        aligned.append(warped); transforms.append(matrix); masks.append(mask); scores.append(score); used_indices.append(i)
    valid = np.logical_and.reduce(masks)
    if crop_common and valid.any():
        ys, xs = np.where(valid)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        aligned = [im[y0:y1, x0:x1] for im in aligned]
    else:
        x0, y0, x1, y1 = 0, 0, w, h
    return AlignmentResult(aligned, transforms, (x0, y0, x1 - x0, y1 - y0), scores, used_indices, warnings)
