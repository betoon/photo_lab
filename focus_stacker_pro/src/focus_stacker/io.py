from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import tifffile
from PIL import Image

RAW_EXTENSIONS = {".3fr", ".arw", ".cr2", ".cr3", ".dng", ".erf", ".nef", ".nrw", ".orf", ".pef", ".raf", ".rw2", ".srw"}
IMAGE_EXTENSIONS = RAW_EXTENSIONS | {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def read_image(path: str) -> tuple[np.ndarray, dict]:
    p = Path(path)
    metadata: dict = {"source": str(p), "exif": None}
    if p.suffix.lower() in RAW_EXTENSIONS:
        try:
            import rawpy
        except ImportError as exc:
            raise RuntimeError("RAW input requires: pip install rawpy") from exc
        with rawpy.imread(str(p)) as raw:
            rgb = raw.postprocess(output_bps=16, use_camera_wb=True, no_auto_bright=True)
        return rgb, metadata
    if p.suffix.lower() in {".tif", ".tiff"}:
        arr = tifffile.imread(p)
        if arr.ndim > 3:
            arr = arr[0]
        if arr.ndim == 2:
            arr = np.repeat(arr[..., None], 3, axis=2)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        return np.ascontiguousarray(arr), metadata
    with Image.open(p) as im:
        metadata["exif"] = im.info.get("exif")
        metadata["icc_profile"] = im.info.get("icc_profile")
        metadata["dpi"] = im.info.get("dpi")
        metadata["mode"] = im.mode
        return np.asarray(im.convert("RGB")), metadata


def to_float(image: np.ndarray) -> np.ndarray:
    if np.issubdtype(image.dtype, np.integer):
        return image.astype(np.float32) / np.iinfo(image.dtype).max
    image = image.astype(np.float32)
    return np.clip(image, 0, 1)


def write_image(path: str, image: np.ndarray, *, bit_depth: int = 16, exif: bytes | None = None) -> None:
    p = Path(path)
    a = np.clip(image, 0, 1)
    if p.suffix.lower() in {".tif", ".tiff"}:
        out = np.round(a * (65535 if bit_depth == 16 else 255)).astype(np.uint16 if bit_depth == 16 else np.uint8)
        tifffile.imwrite(p, out, photometric="rgb")
    elif p.suffix.lower() == ".png":
        out = np.round(a * (65535 if bit_depth == 16 else 255)).astype(np.uint16 if bit_depth == 16 else np.uint8)
        if out.dtype == np.uint16:
            cv2.imwrite(str(p), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        else:
            Image.fromarray(out, "RGB").save(p)
    else:
        out = np.round(a * 255).astype(np.uint8)
        kwargs = {"quality": 95, "subsampling": 0}
        if exif:
            kwargs["exif"] = exif
        Image.fromarray(out, "RGB").save(p, **kwargs)
