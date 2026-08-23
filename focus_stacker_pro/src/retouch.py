"""Nondestructive source-frame retouching for aligned focus stacks."""
from __future__ import annotations

from dataclasses import dataclass, field
import cv2
import numpy as np


@dataclass
class RetouchSession:
    base: np.ndarray
    sources: list[np.ndarray]
    result: np.ndarray = field(init=False)
    ownership: np.ndarray = field(init=False)
    _undo: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list, init=False)
    _redo: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list, init=False)

    def __post_init__(self):
        self.result = self.base.copy()
        self.ownership = np.full(self.base.shape[:2], -1, np.int16)

    def snapshot(self):
        self._undo.append((self.result.copy(), self.ownership.copy()))
        if len(self._undo) > 30: self._undo.pop(0)
        self._redo.clear()

    def paint(self, x: int, y: int, source_index: int, radius: int, hardness: float, opacity: float):
        if not 0 <= source_index < len(self.sources): return
        self.snapshot()
        h, w = self.result.shape[:2]; radius = max(1, int(radius))
        x0, x1 = max(0, x - radius), min(w, x + radius + 1); y0, y1 = max(0, y - radius), min(h, y + radius + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]; distance = np.sqrt((xx - x) ** 2 + (yy - y) ** 2) / radius
        edge = max(0.02, 1.0 - np.clip(hardness, 0, 1))
        alpha = np.clip((1.0 - distance) / edge, 0, 1) if hardness < .98 else (distance <= 1).astype(np.float32)
        alpha = (alpha * np.clip(opacity, 0, 1)).astype(np.float32)
        region = self.result[y0:y1, x0:x1]; source = self.sources[source_index][y0:y1, x0:x1]
        self.result[y0:y1, x0:x1] = region * (1 - alpha[..., None]) + source * alpha[..., None]
        self.ownership[y0:y1, x0:x1][alpha > .5] = source_index

    def undo(self):
        if self._undo:
            self._redo.append((self.result.copy(), self.ownership.copy())); self.result, self.ownership = self._undo.pop()

    def redo(self):
        if self._redo:
            self._undo.append((self.result.copy(), self.ownership.copy())); self.result, self.ownership = self._redo.pop()

    def overlay(self) -> np.ndarray:
        colors = cv2.applyColorMap(np.clip((self.ownership + 1) * 31, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        mask = self.ownership >= 0
        return np.where(mask[..., None], self.result * .45 + colors * .55, self.result)

