"""Large-stack execution helpers: tiling, disk caching, pause gates, ETA and acceleration."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import threading
from time import perf_counter
from typing import Callable

import cv2
import numpy as np

from .fusion import focus_measure, fuse
from .fusion import finish


class PauseGate:
    def __init__(self): self._running = threading.Event(); self._running.set(); self.cancelled = False
    def pause(self): self._running.clear()
    def resume(self): self._running.set()
    def cancel(self): self.cancelled = True; self._running.set()
    def checkpoint(self):
        self._running.wait()
        if self.cancelled: raise InterruptedError("Cancelled")


@dataclass
class AccelerationStatus:
    requested_gpu: bool
    opencl_available: bool
    opencl_enabled: bool
    cpu_threads: int


def configure_acceleration(use_gpu: bool, cpu_threads: int) -> AccelerationStatus:
    cv2.setUseOptimized(True)
    cv2.setNumThreads(int(cpu_threads))
    available = bool(cv2.ocl.haveOpenCL())
    cv2.ocl.setUseOpenCL(bool(use_gpu and available))
    return AccelerationStatus(use_gpu, available, bool(cv2.ocl.useOpenCL()), cv2.getNumThreads())


def finish_accelerated(image: np.ndarray, sharpen: float, denoise: float, opencl_enabled: bool) -> np.ndarray:
    """Run finishing filters through UMat/OpenCL when available, otherwise use the CPU implementation."""
    if not opencl_enabled: return finish(image, sharpen, denoise)
    work = cv2.UMat(np.clip(image, 0, 1).astype(np.float32))
    if denoise > 0: work = cv2.bilateralFilter(work, 7, denoise * .15, 3 + denoise * 5)
    if sharpen > 0:
        blur = cv2.GaussianBlur(work, (0, 0), 1.0); work = cv2.addWeighted(work, 1 + sharpen, blur, -sharpen, 0)
    return np.clip(work.get(), 0, 1)


class DiskBackedStack:
    """Temporary float32 arrays backed by individual memory-mapped files."""
    def __init__(self, images: list[np.ndarray], directory: str = ""):
        self._temp = tempfile.TemporaryDirectory(prefix="focusstack_", dir=directory or None)
        self.paths, self.images = [], []
        for i, image in enumerate(images):
            path = Path(self._temp.name) / f"aligned_{i:05d}.npy"; mapped = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=image.shape)
            mapped[:] = image; mapped.flush(); del mapped; self.paths.append(path); self.images.append(np.load(path, mmap_mode="r"))
    def close(self):
        for image in self.images:
            mmap = getattr(image, "_mmap", None)
            if mmap is not None:
                try: mmap.close()
                except Exception: pass
        self.images.clear()
        try: self._temp.cleanup()
        except PermissionError: pass
    def __del__(self):
        try: self.close()
        except Exception: pass


def _tile_ranges(length: int, tile: int):
    for start in range(0, length, tile): yield start, min(length, start + tile)


def fuse_tiled(images: list[np.ndarray], algorithm: str, radius: int, smooth: int, temperature: float,
               levels: int, cleanup: int, tile_size: int = 1024, gate: PauseGate | None = None,
               progress: Callable[[int, str, float | None], None] = lambda *_: None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse overlapping tiles and retain only seam-safe centers."""
    h, w = images[0].shape[:2]; tile_size = max(128, int(tile_size))
    overlap = max(radius * 4, smooth * 4, cleanup * 2, (2 ** min(levels, 7)) if algorithm == "pyramid" else 24)
    result = np.empty((h, w, 3), np.float32); depth = np.empty((h, w), np.uint16); confidence = np.empty((h, w), np.float32)
    tiles = [(y0, y1, x0, x1) for y0, y1 in _tile_ranges(h, tile_size) for x0, x1 in _tile_ranges(w, tile_size)]
    started = perf_counter()
    for index, (y0, y1, x0, x1) in enumerate(tiles):
        if gate: gate.checkpoint()
        sy0, sy1, sx0, sx1 = max(0, y0 - overlap), min(h, y1 + overlap), max(0, x0 - overlap), min(w, x1 + overlap)
        parts = [np.asarray(im[sy0:sy1, sx0:sx1], dtype=np.float32) for im in images]
        fused, tile_depth = fuse(parts, algorithm, radius, smooth, temperature, levels, cleanup)
        scores = np.stack([focus_measure(part, radius) for part in parts]); ordered = np.partition(scores, -2, axis=0)
        tile_conf = np.clip((ordered[-1] - ordered[-2]) / (ordered[-1] + 1e-8), 0, 1) if len(parts) > 1 else np.ones(parts[0].shape[:2], np.float32)
        cy0, cy1, cx0, cx1 = y0 - sy0, y1 - sy0, x0 - sx0, x1 - sx0
        result[y0:y1, x0:x1] = fused[cy0:cy1, cx0:cx1]; depth[y0:y1, x0:x1] = tile_depth[cy0:cy1, cx0:cx1]; confidence[y0:y1, x0:x1] = tile_conf[cy0:cy1, cx0:cx1]
        elapsed = perf_counter() - started; done = index + 1; eta = elapsed / done * (len(tiles) - done)
        progress(int(done * 100 / len(tiles)), f"Fusing tile {done}/{len(tiles)}", eta)
    return result, depth, confidence
