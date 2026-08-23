"""Advanced output encoding and auxiliary-asset export."""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import tifffile
from PIL import Image

from .models import OutputOptions


def _resize(image: np.ndarray, percent: float) -> np.ndarray:
    if abs(percent - 100) < .01: return image
    scale = max(.01, percent / 100); size = (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LANCZOS4)


def write_advanced(path: str, image: np.ndarray, options: OutputOptions, metadata: dict | None = None,
                   alpha: np.ndarray | None = None) -> None:
    p = Path(path); metadata = metadata or {}; work = np.clip(_resize(image, options.resize_percent), 0, 1)
    if options.grayscale: work = cv2.cvtColor(work.astype(np.float32), cv2.COLOR_RGB2GRAY)
    if options.include_alpha:
        if alpha is None: alpha = np.ones(work.shape[:2], np.float32)
        else: alpha = _resize(alpha, options.resize_percent)
        work = np.dstack([work, np.clip(alpha, 0, 1)])
    bits = 16 if options.preset == "archival" else options.bit_depth
    dtype, maximum = (np.uint16, 65535) if bits == 16 else (np.uint8, 255)
    encoded = np.round(work * maximum).astype(dtype)
    suffix = p.suffix.lower()
    if suffix in {".tif", ".tiff"}:
        extra = []
        icc = metadata.get("icc_profile") if options.preserve_icc else None
        if icc: extra.append((34675, "B", len(icc), icc, False))
        kwargs = {"photometric": "minisblack" if encoded.ndim == 2 else "rgb", "bigtiff": bool(options.bigtiff)}
        if options.preserve_dpi and metadata.get("dpi"): kwargs["resolution"] = metadata["dpi"]; kwargs["resolutionunit"] = "INCH"
        if extra: kwargs["extratags"] = extra
        tifffile.imwrite(p, encoded, **kwargs)
    elif suffix == ".png":
        if encoded.dtype == np.uint16: cv2.imwrite(str(p), cv2.cvtColor(encoded, cv2.COLOR_RGBA2BGRA if encoded.ndim == 3 and encoded.shape[2] == 4 else cv2.COLOR_RGB2BGR) if encoded.ndim == 3 else encoded)
        else: Image.fromarray(encoded).save(p, icc_profile=metadata.get("icc_profile") if options.preserve_icc else None, dpi=metadata.get("dpi") if options.preserve_dpi else None)
    else:
        rgb = encoded if encoded.dtype == np.uint8 else np.round(work * 255).astype(np.uint8)
        if rgb.ndim == 2: rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.shape[2] == 4: rgb = rgb[..., :3]
        quality = 88 if options.preset == "web" else 96
        Image.fromarray(rgb, "RGB").save(p, quality=quality, optimize=True, subsampling=0,
            exif=metadata.get("exif") or b"", icc_profile=metadata.get("icc_profile") if options.preserve_icc else None,
            dpi=metadata.get("dpi") if options.preserve_dpi else None)


def export_auxiliary(base_path: str, aligned: list[np.ndarray], depth: np.ndarray | None,
                     confidence: np.ndarray | None, masks: list[np.ndarray] | None, options: OutputOptions) -> list[str]:
    base = Path(base_path); folder = base.with_name(base.stem + "_assets"); folder.mkdir(parents=True, exist_ok=True); written = []
    if options.export_aligned:
        for i, image in enumerate(aligned):
            path = folder / f"aligned_{i + 1:04d}.tif"; write_advanced(str(path), image, options); written.append(str(path))
    if options.export_depth and depth is not None:
        path = folder / "depth_map.tif"; tifffile.imwrite(path, depth.astype(np.uint16), photometric="minisblack", bigtiff=options.bigtiff); written.append(str(path))
    if options.export_confidence and confidence is not None:
        path = folder / "confidence_map.tif"; tifffile.imwrite(path, np.round(np.clip(confidence, 0, 1) * 65535).astype(np.uint16), photometric="minisblack", bigtiff=options.bigtiff); written.append(str(path))
    if options.export_masks and masks:
        for i, mask in enumerate(masks):
            path = folder / f"mask_{i + 1:04d}.png"; cv2.imwrite(str(path), np.round(np.clip(mask, 0, 1) * 255).astype(np.uint8)); written.append(str(path))
    return written
