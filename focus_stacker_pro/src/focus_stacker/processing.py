"""Preprocessing, resource estimation, and reproducible reporting."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import json
import platform
import os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .models import Project


def estimate_memory_bytes(count: int, height: int, width: int, algorithm: str, levels: int = 5) -> int:
    """Conservative working-set estimate for normalized images and intermediates."""
    pixels = count * height * width
    base = pixels * (12 + 4 + 4)  # RGB float, focus score, mask/temporary
    aligned_copy = pixels * 12
    pyramid = int(pixels * 16 * 4 / 3) if algorithm == "pyramid" else 0
    output_and_overhead = height * width * 40
    return int((base + aligned_copy + pyramid + output_and_overhead) * 1.25)


def human_bytes(value: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(value)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{value} B"


def normalize_stack(images: list[np.ndarray], reference_index: int, exposure: bool, color: bool,
                    progress: Callable[[int, str], None] = lambda *_: None) -> tuple[list[np.ndarray], list[dict]]:
    """Robustly match central-tone gain/offset to a reference without clipping highlights."""
    if not exposure and not color:
        return images, [{"gain": [1, 1, 1], "offset": [0, 0, 0]} for _ in images]
    reference = images[reference_index]
    channels = 3 if color else 1
    ref_stats = []
    ref_data = reference if channels == 3 else cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)[..., None]
    for c in range(channels):
        values = ref_data[..., c]
        ref_stats.append((float(np.percentile(values, 10)), float(np.percentile(values, 50)), float(np.percentile(values, 90))))
    output, transforms = [], []
    for i, image in enumerate(images):
        progress(i, f"Normalizing frame {i + 1}/{len(images)}")
        data = image if channels == 3 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[..., None]
        gains, offsets = [], []
        for c in range(channels):
            low, median, high = [float(v) for v in np.percentile(data[..., c], (10, 50, 90))]
            rlow, rmedian, rhigh = ref_stats[c]
            gain = np.clip((rhigh - rlow) / max(high - low, 1e-5), 0.5, 2.0) if exposure else 1.0
            offset = np.clip(rmedian - median * gain, -0.2, 0.2)
            gains.append(float(gain)); offsets.append(float(offset))
        if channels == 1:
            gains *= 3; offsets *= 3
        corrected = image * np.asarray(gains, np.float32) + np.asarray(offsets, np.float32)
        output.append(np.clip(corrected, 0, 1)); transforms.append({"gain": gains, "offset": offsets})
    return output, transforms


def build_report(project: Project, details: dict, output_shape: tuple[int, ...], timings: dict,
                 warnings: list[str] | None = None) -> dict:
    alignment = details.get("alignment")
    try:
        import psutil
        memory = psutil.virtual_memory(); system_memory = {"total": memory.total, "available": memory.available}
        cpu = {"logical": psutil.cpu_count(), "physical": psutil.cpu_count(logical=False)}
    except Exception:
        system_memory = {}; cpu = {"logical": os.cpu_count()}
    dependencies = {}
    for package in ("numpy", "opencv-python", "PySide6", "Pillow", "tifffile", "rawpy"):
        try: dependencies[package] = version(package)
        except PackageNotFoundError: dependencies[package] = None
    source_summaries = []
    for metadata in details.get("metadata", []):
        source_summaries.append({key: (f"<{len(value)} binary bytes>" if isinstance(value, bytes) else value) for key, value in metadata.items()})
    return {
        "report_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "application": "Focus Stacker Pro 1.5",
        "system": {"platform": platform.platform(), "python": platform.python_version(), "processor": platform.processor(), "memory": system_memory, "cpu": cpu},
        "dependencies": dependencies,
        "project": asdict(project),
        "result": {"shape": [int(v) for v in output_shape], "common_roi": [int(v) for v in alignment.common_roi] if alignment else None},
        "alignment": {"scores": alignment.scores, "transforms": [m.tolist() for m in alignment.transforms]} if alignment else {},
        "normalization": details.get("normalization", []),
        "sources": source_summaries,
        "acceleration": details.get("acceleration", {}),
        "peak_memory_estimate": details.get("peak_memory_estimate"),
        "timings_seconds": timings,
        "warnings": warnings or [],
        "microscope_diagnostics": {key: value for key, value in details.get("microscope_diagnostics", {}).items() if key not in {"scale_map", "localized", "structure"}},
        "synthetic_provenance": details.get("synthetic_provenance", []),
    }


def save_report(path: str | Path, report: dict) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
