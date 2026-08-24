"""Microscope-specific 2D fusion, parameter exploration, calibration overlays, and Z-order tools."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
from typing import Callable

import cv2
import numpy as np

from .fusion import focus_measure
from .models import MicroscopeOptions


def natural_key(path: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", Path(path).name)]


def natural_sort(paths: list[str]) -> list[str]: return sorted(paths, key=natural_key)


def diagnose_focus_order(images: list[np.ndarray]) -> dict:
    """Estimate focus progression and irregular steps from sharp-region centroids."""
    centroids, strengths = [], []
    for image in images:
        score = focus_measure(image, 4); threshold = np.percentile(score, 85); weights = np.maximum(score - threshold, 0)
        yy, xx = np.indices(score.shape); total = float(weights.sum()) + 1e-8
        centroids.append([float((xx * weights).sum() / total), float((yy * weights).sum() / total)]); strengths.append(float(np.percentile(score, 95)))
    changes = np.linalg.norm(np.diff(np.asarray(centroids), axis=0), axis=1) if len(centroids) > 1 else np.array([])
    irregular = [int(i + 1) for i, value in enumerate(changes) if value > (np.median(changes) * 3 + 1e-6)]
    strength_trend = float(np.corrcoef(np.arange(len(strengths)), strengths)[0, 1]) if len(strengths) > 2 and np.std(strengths) > 0 else 0.0
    return {"centroids": centroids, "strengths": strengths, "irregular_transitions": irregular,
            "likely_reversed": bool(strength_trend < -.35), "trend": strength_trend}


def synthesize_intermediate_planes(images: list[np.ndarray], count: int = 1) -> tuple[list[np.ndarray], list[dict]]:
    """Create clearly marked presentation-only planes between captures."""
    count = max(1, min(int(count), 3)); output, provenance = [], []
    for index, (first, second) in enumerate(zip(images[:-1], images[1:])):
        output.append(first); provenance.append({"synthetic": False, "source": index})
        for step in range(1, count + 1):
            alpha = step / (count + 1); blended = first * (1 - alpha) + second * alpha
            blur = cv2.GaussianBlur(blended, (0, 0), .8); blended = np.clip(cv2.addWeighted(blended, 1.2, blur, -.2, 0), 0, 1)
            output.append(blended.astype(np.float32)); provenance.append({"synthetic": True, "between": [index, index + 1], "fraction": alpha})
    output.append(images[-1]); provenance.append({"synthetic": False, "source": len(images) - 1}); return output, provenance


def _color_scores(images: list[np.ndarray], target: list[int], tolerance: float, space: str) -> np.ndarray:
    target_rgb = np.asarray(target, np.uint8).reshape(1, 1, 3)
    conversion = cv2.COLOR_RGB2LAB if space.lower() == "lab" else cv2.COLOR_RGB2HSV
    converted_target = cv2.cvtColor(target_rgb, conversion).astype(np.float32)[0, 0]
    scores = []
    for image in images:
        converted = cv2.cvtColor(np.clip(image * 255, 0, 255).astype(np.uint8), conversion).astype(np.float32)
        distance = np.linalg.norm(converted - converted_target, axis=2); scores.append(np.exp(-.5 * (distance / max(1, tolerance)) ** 2))
    return np.stack(scores).astype(np.float32)


def smart_focus_scores(images: list[np.ndarray], options: MicroscopeOptions) -> tuple[np.ndarray, np.ndarray]:
    radii = {"fine": [options.fine_radius], "medium": [options.medium_radius], "coarse": [options.coarse_radius],
             "smart": [options.fine_radius, options.medium_radius, options.coarse_radius]}[options.focus_scale_mode]
    per_scale = np.stack([np.stack([focus_measure(image, radius) for image in images]) for radius in radii])
    normalized = per_scale / (np.median(per_scale, axis=(2, 3), keepdims=True) + 1e-8)
    scale_choice = np.argmax(np.max(normalized, axis=1), axis=0).astype(np.uint8)
    scores = np.max(normalized, axis=0)
    if options.color_selective:
        color = _color_scores(images, options.target_color, options.color_tolerance, options.color_space)
        color /= color.max(axis=0, keepdims=True) + 1e-8; mix = np.clip(options.color_focus_mix, 0, 1); scores = scores * mix + color * (1 - mix)
    depth_axis = np.linspace(-1, 1, len(images), dtype=np.float32)[:, None, None]
    scores *= np.exp(np.clip(options.depth_preference, -1, 1) * depth_axis)
    return scores.astype(np.float32), scale_choice


def microscope_fuse(images: list[np.ndarray], options: MicroscopeOptions, smooth_radius: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    scores, scale_map = smart_focus_scores(images, options); winner = np.argmax(scores, axis=0).astype(np.uint16)
    ordered = np.partition(scores, -2, axis=0); best = ordered[-1]; second = ordered[-2] if len(images) > 1 else np.zeros_like(best)
    confidence = np.clip((best - second) / (np.abs(best) + 1e-8), 0, 1)
    structure = best / (np.percentile(best, 99) + 1e-8); localized = (structure >= options.minimum_structure) & (confidence >= options.minimum_confidence)
    amount = int(options.patch_morphology)
    if amount:
        size = abs(amount) * 2 + 1; kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size)); adjusted = []
        for i in range(len(images)):
            mask = (winner == i).astype(np.uint8); mask = cv2.dilate(mask, kernel) if amount > 0 else cv2.erode(mask, kernel); adjusted.append(mask.astype(np.float32) * scores[i])
        winner = np.argmax(np.stack(adjusted), axis=0).astype(np.uint16)
    sigma = max(.5, smooth_radius / 3); weights = np.stack([cv2.GaussianBlur(((winner == i) & localized).astype(np.float32), (0, 0), sigma) for i in range(len(images))])
    weights /= weights.sum(axis=0, keepdims=True) + 1e-8; stack = np.stack(images); fused = np.sum(stack * weights[..., None], axis=0)
    if options.uncertain_mode == "median": uncertain = np.median(stack, axis=0)
    elif options.uncertain_mode == "reference": uncertain = stack[len(images) // 2]
    else: uncertain = np.mean(stack, axis=0)
    fused = np.where(localized[..., None], fused, uncertain); depth = winner.copy(); depth[~localized] = np.iinfo(np.uint16).max
    diagnostics = {"scale_map": scale_map, "localized": localized, "structure": np.clip(structure, 0, 1), "unlocalized_count": int((~localized).sum())}
    return np.clip(fused, 0, 1), depth, confidence.astype(np.float32), diagnostics


def parameter_comparison(images: list[np.ndarray], options: MicroscopeOptions,
                         progress: Callable[[int, str], None] = lambda *_: None) -> list[tuple[str, np.ndarray, MicroscopeOptions]]:
    variants = [
        ("Fine / strict", replace(options, focus_scale_mode="fine", minimum_confidence=min(1, options.minimum_confidence + .08))),
        ("Smart / balanced", replace(options, focus_scale_mode="smart")),
        ("Smart / smooth", replace(options, focus_scale_mode="smart", patch_morphology=max(1, options.patch_morphology), minimum_confidence=max(0, options.minimum_confidence - .05))),
        ("Coarse / clean", replace(options, focus_scale_mode="coarse", minimum_structure=min(1, options.minimum_structure + .06))),
    ]
    results = []
    for index, (name, variant) in enumerate(variants):
        progress(int(index * 100 / len(variants)), name); result, _, _, _ = microscope_fuse(images, variant); results.append((name, result, variant))
    return results


def draw_scale_bar(image: np.ndarray, microns_per_pixel: float, length_microns: float,
                   color: list[int], position: str = "bottom-right") -> np.ndarray:
    if microns_per_pixel <= 0 or length_microns <= 0: return image.copy()
    output = image.copy(); h, w = output.shape[:2]; pixels = max(1, round(length_microns / microns_per_pixel)); margin = max(12, min(h, w) // 40); thickness = max(3, h // 300)
    x0 = margin if "left" in position else max(margin, w - margin - pixels); y = margin + thickness if "top" in position else h - margin
    rgb = np.asarray(color, np.float32) / 255; mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (x0, y - thickness), (min(w - margin, x0 + pixels), y), 255, -1)
    label = f"{length_microns:g} um"; cv2.putText(mask, label, (x0, max(margin, y - thickness - 7)), cv2.FONT_HERSHEY_SIMPLEX, max(.4, h / 1800), 255, max(1, thickness // 2), cv2.LINE_AA)
    alpha = mask.astype(np.float32) / 255; output = output * (1 - alpha[..., None]) + rgb * alpha[..., None]
    return output
