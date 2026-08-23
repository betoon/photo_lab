from __future__ import annotations

from typing import Callable
import cv2
import numpy as np


def focus_measure(image: np.ndarray, radius: int = 5) -> np.ndarray:
    gray = cv2.cvtColor(image.astype(np.float32), cv2.COLOR_RGB2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    energy = cv2.GaussianBlur(lap * lap, (0, 0), max(0.8, radius / 3))
    local_var = cv2.GaussianBlur(gray * gray, (0, 0), radius) - cv2.GaussianBlur(gray, (0, 0), radius) ** 2
    return np.maximum(energy + 0.25 * np.maximum(local_var, 0), 1e-8)


def weighted_fusion(images: list[np.ndarray], radius: int = 5, smooth: int = 7, temperature: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    scores = np.stack([focus_measure(im, radius) for im in images])
    normalized = scores / (scores.mean(axis=0, keepdims=True) + 1e-8)
    logits = np.clip(normalized * temperature, -40, 40)
    weights = np.exp(logits - logits.max(axis=0, keepdims=True))
    if smooth:
        weights = np.stack([cv2.GaussianBlur(w, (0, 0), smooth) for w in weights])
    weights /= weights.sum(axis=0, keepdims=True) + 1e-8
    result = np.sum(np.stack(images) * weights[..., None], axis=0)
    return result, np.argmax(scores, axis=0).astype(np.uint16)


def depth_map_fusion(images: list[np.ndarray], radius: int = 5, smooth: int = 7, cleanup: int = 5) -> tuple[np.ndarray, np.ndarray]:
    scores = np.stack([focus_measure(im, radius) for im in images])
    depth = np.argmax(scores, axis=0).astype(np.float32)
    if cleanup >= 3:
        k = cleanup if cleanup % 2 else cleanup + 1
        depth = cv2.medianBlur(depth, k)
    sigma = max(0.5, smooth / 3)
    weights = []
    for i in range(len(images)):
        mask = (np.abs(depth - i) < 0.5).astype(np.float32)
        weights.append(cv2.GaussianBlur(mask, (0, 0), sigma))
    weights = np.stack(weights)
    weights /= weights.sum(axis=0, keepdims=True) + 1e-8
    return np.sum(np.stack(images) * weights[..., None], axis=0), depth.astype(np.uint16)


def pyramid_fusion(images: list[np.ndarray], radius: int = 5, levels: int = 5) -> tuple[np.ndarray, np.ndarray]:
    scores = np.stack([focus_measure(im, radius) for im in images])
    weights = scores / (scores.sum(axis=0, keepdims=True) + 1e-8)
    levels = max(1, min(levels, int(np.log2(min(images[0].shape[:2]))) - 2))
    gp_w, lp_i = [], []
    for weight, image in zip(weights, images):
        gw = [weight]; gi = [image]
        for _ in range(levels):
            gw.append(cv2.pyrDown(gw[-1])); gi.append(cv2.pyrDown(gi[-1]))
        lp = [gi[-1]]
        for j in range(levels, 0, -1):
            lp.append(gi[j - 1] - cv2.pyrUp(gi[j], dstsize=(gi[j - 1].shape[1], gi[j - 1].shape[0])))
        # Both lists are fine-to-coarse after reversing the Laplacian list.
        gp_w.append(gw); lp_i.append(list(reversed(lp)))
    blended = []
    for level in range(levels + 1):
        denom = sum(gp_w[i][level] for i in range(len(images))) + 1e-8
        blended.append(sum(lp_i[i][level] * (gp_w[i][level] / denom)[..., None] for i in range(len(images))))
    result = blended[-1]
    for level in range(levels - 1, -1, -1):
        result = cv2.pyrUp(result, dstsize=(blended[level].shape[1], blended[level].shape[0])) + blended[level]
    return np.clip(result, 0, 1), np.argmax(scores, axis=0).astype(np.uint16)


def fuse(images: list[np.ndarray], algorithm: str, radius: int = 5, smooth: int = 7,
         temperature: float = 8, levels: int = 5, cleanup: int = 5,
         progress: Callable[[int, str], None] = lambda *_: None) -> tuple[np.ndarray, np.ndarray]:
    progress(50, "Computing focus measures")
    if algorithm == "weighted":
        out = weighted_fusion(images, radius, smooth, temperature)
    elif algorithm == "pyramid":
        out = pyramid_fusion(images, radius, levels)
    elif algorithm == "average":
        out = (np.mean(images, axis=0), np.zeros(images[0].shape[:2], np.uint16))
    else:
        out = depth_map_fusion(images, radius, smooth, cleanup)
    progress(85, "Applying finishing filters")
    return out


def finish(image: np.ndarray, sharpen: float = 0.25, denoise: float = 0.0) -> np.ndarray:
    out = np.clip(image, 0, 1).astype(np.float32)
    if denoise > 0:
        out = cv2.bilateralFilter(out, 7, denoise * 0.15, 3 + denoise * 5)
    if sharpen > 0:
        blur = cv2.GaussianBlur(out, (0, 0), 1.0)
        out = cv2.addWeighted(out, 1 + sharpen, blur, -sharpen, 0)
    return np.clip(out, 0, 1)
