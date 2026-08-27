"""workers.py — background QThreads so the UI never blocks on I/O or the
image pipeline (thumbnail generation, full-resolution export, RAW decode)."""

from __future__ import annotations

import os
import shutil
import hashlib
import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap

from imaging import apply_recipe, load_image, is_raw, IMAGE_EXTS, _silent_imread, extract_embedded_preview
from qt_utils import cv_to_qpixmap

import threading
import copy

_HEAVY_SEM = threading.Semaphore(2)
_HEAVY_LIMIT = 2
# Thumbnail extraction and full RAW decoding both serialize through LibRaw.
# This flag prevents a folder's thumbnail scan from repeatedly jumping ahead
# of the image the user explicitly selected.
_FULL_LOAD_PENDING = threading.Event()


def write_restored_image_atomic(image, path, progress=None, cancelled=None):
    """Encode and atomically write a restored image with byte-level progress."""
    progress = progress or (lambda _value, _message: None)
    cancelled = cancelled or (lambda: False)
    path = os.path.abspath(os.fspath(path))
    extension = os.path.splitext(path)[1].lower() or ".png"
    temporary = path + ".photolab-part"
    progress(8, "Encoding restored image…")
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise ValueError("The selected output format could not be encoded")
    if cancelled():
        return False
    payload = memoryview(encoded).cast("B")
    total = max(1, len(payload))
    chunk_size = 1024 * 1024
    try:
        progress(30, "Writing restored image…")
        with open(temporary, "wb") as stream:
            for offset in range(0, total, chunk_size):
                if cancelled():
                    return False
                stream.write(payload[offset:min(total, offset + chunk_size)])
                fraction = min(1.0, (offset + chunk_size) / total)
                progress(30 + int(fraction * 62), f"Writing restored image… {min(total, offset + chunk_size):,} of {total:,} bytes")
            stream.flush()
            os.fsync(stream.fileno())
        if cancelled():
            return False
        progress(96, "Finalizing restored image…")
        os.replace(temporary, path)
        progress(100, "Restored image saved")
        return True
    finally:
        if os.path.exists(temporary):
            try:
                os.remove(temporary)
            except OSError:
                pass


class RestorationSaveWorker(QThread):
    """Encode and save Restoration Studio output without freezing the UI."""
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(str)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()

    def __init__(self, image, path):
        super().__init__()
        self.image = image
        self.path = os.path.abspath(os.fspath(path))
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            saved = write_restored_image_atomic(
                self.image, self.path,
                progress=lambda value, message: self.progress.emit(value, message),
                cancelled=lambda: self._cancel,
            )
            if saved:
                self.completed.emit(self.path)
            else:
                self.cancelled.emit()
        except Exception as exc:
            self.failed.emit(str(exc))


class SdImportWorker(QThread):
    """Copy camera media safely; never deletes or modifies the source card."""
    progress = pyqtSignal(int, int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, source, destination, preserve_folders=False, video_exts=()):
        super().__init__()
        self.source = os.path.abspath(source)
        self.destination = os.path.abspath(destination)
        self.preserve_folders = bool(preserve_folders)
        self.extensions = {e.lower() for e in tuple(IMAGE_EXTS) + tuple(video_exts)}
        self._cancel = False

    def cancel(self):
        self._cancel = True

    @staticmethod
    def _same_file(source, destination, source_size):
        if os.path.getsize(destination) != source_size:
            return False
        def digest(path):
            value = hashlib.sha256()
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    value.update(chunk)
            return value.digest()
        return digest(source) == digest(destination)

    @classmethod
    def _unique_destination(cls, path, source_path, source_size):
        if not os.path.exists(path):
            return path, False
        if cls._same_file(source_path, path, source_size):
            return path, True
        stem, ext = os.path.splitext(path)
        number = 1
        while True:
            candidate = f"{stem}_{number}{ext}"
            if not os.path.exists(candidate):
                return candidate, False
            if cls._same_file(source_path, candidate, source_size):
                return candidate, True
            number += 1

    def run(self):
        try:
            files = []
            for root, _dirs, names in os.walk(self.source):
                for name in names:
                    if os.path.splitext(name)[1].lower() in self.extensions:
                        files.append(os.path.join(root, name))
            files.sort(key=str.lower)
            copied = skipped = renamed = 0
            errors = []
            os.makedirs(self.destination, exist_ok=True)
            for index, source_path in enumerate(files, 1):
                if self._cancel:
                    break
                relative = os.path.relpath(source_path, self.source)
                target = os.path.join(self.destination, relative if self.preserve_folders else os.path.basename(source_path))
                os.makedirs(os.path.dirname(target), exist_ok=True)
                source_size = os.path.getsize(source_path)
                chosen, exists = self._unique_destination(target, source_path, source_size)
                if exists:
                    skipped += 1
                    self.progress.emit(index, len(files), os.path.basename(source_path))
                    continue
                if os.path.normcase(chosen) != os.path.normcase(target):
                    renamed += 1
                temporary = chosen + ".photolab-part"
                try:
                    shutil.copy2(source_path, temporary)
                    if os.path.getsize(temporary) != source_size:
                        raise IOError("copied size does not match source")
                    os.replace(temporary, chosen)
                    copied += 1
                except Exception as exc:
                    errors.append(f"{os.path.basename(source_path)}: {exc}")
                    try:
                        if os.path.exists(temporary):
                            os.remove(temporary)
                    except Exception:
                        pass
                self.progress.emit(index, len(files), os.path.basename(source_path))
            self.completed.emit({"found": len(files), "copied": copied, "skipped": skipped,
                                 "renamed": renamed, "errors": errors, "cancelled": self._cancel,
                                 "destination": self.destination})
        except Exception as exc:
            self.failed.emit(str(exc))


def set_max_concurrent_workers(n: int) -> None:
    """Max concurrent heavy jobs (RAW load / export), 1–8."""
    global _HEAVY_SEM, _HEAVY_LIMIT
    n = max(1, min(8, int(n)))
    _HEAVY_LIMIT = n
    _HEAVY_SEM = threading.Semaphore(n)


def get_max_concurrent_workers() -> int:
    return int(_HEAVY_LIMIT)


class PreviewRenderWorker(QThread):
    """Single latest-wins preview queue; stale slider renders are discarded."""
    rendered = pyqtSignal(int, str, object, object)  # generation, path, result, source
    failed = pyqtSignal(int, str, str)

    def __init__(self):
        super().__init__()
        self._condition = threading.Condition()
        self._pending = None
        self._stopping = False

    def submit(self, generation, path, source, recipe, meta):
        with self._condition:
            self._pending = (
                generation, path, source, copy.deepcopy(recipe), copy.deepcopy(meta or {})
            )
            self._condition.notify()

    def stop(self):
        with self._condition:
            self._stopping = True
            self._condition.notify()

    def run(self):
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._stopping:
                    return
                job = self._pending
                self._pending = None
            generation, path, source, recipe, meta = job
            try:
                result = apply_recipe(
                    source, recipe, wb_multipliers=meta.get("wb_multipliers"), meta=meta
                )
                with self._condition:
                    stale = self._pending is not None and self._pending[0] > generation
                if not stale:
                    self.rendered.emit(generation, path, result, source)
            except Exception as exc:
                self.failed.emit(generation, path, str(exc))


class ThumbnailWorker(QThread):
    thumb_ready = pyqtSignal(str, QPixmap)

    def __init__(self, paths):
        super().__init__()
        self.paths = paths
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        for p in self.paths:
            if self._cancel:
                break
            try:
                while _FULL_LOAD_PENDING.is_set() and not self._cancel:
                    self.msleep(20)
                if self._cancel:
                    break
                # Prefer embedded JPEG from RAW — much faster than full decode
                img = extract_embedded_preview(p, max_side=120)
                if img is None:
                    continue
                self.thumb_ready.emit(p, cv_to_qpixmap(img))
            except Exception:
                continue


class ExportWorker(QThread):
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)
    progress = pyqtSignal(int, str)

    def __init__(self, path, recipe, out_path, wb_multipliers=None,
                 watermark_text="", watermark_opacity=0.45, max_dim=0, jpeg_quality=92,
                 export_16bit=False, source_image=None, source_meta=None):
        super().__init__()
        self.path = path
        self.recipe = recipe
        self.out_path = out_path
        self.wb_multipliers = wb_multipliers
        self.watermark_text = watermark_text or ""
        self.watermark_opacity = watermark_opacity
        self.max_dim = max_dim
        self.jpeg_quality = jpeg_quality
        self.export_16bit = export_16bit
        self.source_image = source_image
        self.source_meta = dict(source_meta or {})

    def run(self):
        try:
            import numpy as np
            from imaging import apply_watermark
            with _HEAVY_SEM:
                bps = 16 if self.export_16bit else None
                self.progress.emit(10, "Preparing source image…")
                if self.source_image is not None and (
                    not self.export_16bit or self.source_image.dtype == np.uint16
                ):
                    img, meta = self.source_image, dict(self.source_meta)
                else:
                    self.progress.emit(20, "Decoding RAW file…" if is_raw(self.path) else "Loading image…")
                    img, meta = load_image(self.path, use_camera_wb=True, output_bps=bps)
                multipliers = self.wb_multipliers or meta.get("wb_multipliers")
                out_dtype = np.uint16 if self.export_16bit else np.uint8
                self.progress.emit(40, "Applying PhotoLab adjustments…")
                out = apply_recipe(
                    img, self.recipe, wb_multipliers=multipliers, meta=meta,
                    output_dtype=out_dtype,
                )
                if self.max_dim and max(out.shape[:2]) > self.max_dim:
                    self.progress.emit(70, "Resizing output…")
                    h, w = out.shape[:2]
                    scale = self.max_dim / max(h, w)
                    out = cv2.resize(out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                if self.watermark_text:
                    out = apply_watermark(out, self.watermark_text, opacity=self.watermark_opacity)
                ext = self.out_path.lower().rsplit(".", 1)[-1]
                self.progress.emit(85, f"Encoding {ext.upper()}…")
                if ext in ("jpg", "jpeg"):
                    saved = cv2.imwrite(self.out_path, out, [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)])
                elif ext in ("tif", "tiff") and self.export_16bit:
                    saved = cv2.imwrite(self.out_path, out)
                else:
                    saved = cv2.imwrite(self.out_path, out)
                if not saved or not os.path.isfile(self.out_path):
                    raise RuntimeError(f"The {ext.upper()} encoder could not write the selected file.")
            self.progress.emit(100, "Export complete")
            self.finished_ok.emit(self.out_path)
        except Exception as e:
            self.failed.emit(str(e))


class LoadImageWorker(QThread):
    """Background full-resolution load (especially useful for large RAWs)."""
    loaded = pyqtSignal(str, object, object)  # path, img_bgr, meta
    failed = pyqtSignal(str, str)

    def __init__(self, path: str, output_bps: int = 8):
        super().__init__()
        self.path = path
        self.output_bps = 16 if int(output_bps or 8) >= 16 else 8

    def run(self):
        _FULL_LOAD_PENDING.set()
        try:
            with _HEAVY_SEM:
                img, meta = load_image(
                    self.path, use_camera_wb=True, output_bps=self.output_bps
                )
            self.loaded.emit(self.path, img, meta)
        except Exception as e:
            self.failed.emit(self.path, str(e))
        finally:
            _FULL_LOAD_PENDING.clear()


class HdrMergeWorker(QThread):
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, paths, out_path, align=True, max_dim=0,
                 method="mertens", deghost=0, reference_index=None,
                 ca_correction=0):
        super().__init__()
        self.paths = list(paths)
        self.out_path = out_path
        self.align = align
        self.max_dim = max_dim
        self.method = (method or "mertens").lower()
        self.deghost = float(deghost or 0)
        self.reference_index = reference_index
        self.ca_correction = float(ca_correction or 0)

    def run(self):
        try:
            from imaging import merge_hdr_mertens, merge_hdr_debevec, deghost_stack, load_image
            self.progress.emit(f"Merging {len(self.paths)} exposures ({self.method})…")
            if self.method.startswith("debevec"):
                out = merge_hdr_debevec(
                    self.paths, align=self.align, max_dim=self.max_dim or 0,
                    deghost=self.deghost, reference_index=self.reference_index,
                    ca_correction=self.ca_correction,
                )
            else:
                # Mertens path; optional deghost via pre-load if strength > 0
                if self.deghost > 0:
                    try:
                        # Prefer merge_hdr_mertens deghost kw if supported
                        out = merge_hdr_mertens(
                            self.paths, align=self.align, max_dim=self.max_dim or 0,
                            deghost=self.deghost, reference_index=self.reference_index,
                            ca_correction=self.ca_correction,
                        )
                    except TypeError:
                        out = merge_hdr_mertens(
                            self.paths, align=self.align, max_dim=self.max_dim or 0,
                        )
                else:
                    out = merge_hdr_mertens(
                        self.paths, align=self.align, max_dim=self.max_dim or 0,
                        ca_correction=self.ca_correction,
                    )
            if out is None:
                raise RuntimeError("HDR merge returned empty result")
            ext = self.out_path.lower().rsplit(".", 1)[-1]
            if ext in ("jpg", "jpeg"):
                cv2.imwrite(self.out_path, out, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(self.out_path, out)
            self.finished_ok.emit(self.out_path)
        except Exception as e:
            self.failed.emit(str(e))


class CatalogScanWorker(QThread):
    progress = pyqtSignal(dict, str)
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, root, recursive=True, db_path=None):
        super().__init__()
        self.root = root
        self.recursive = recursive
        self.db_path = db_path
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            from catalog import Catalog
            cat = Catalog(self.db_path)
            stats = cat.scan_folder(
                self.root,
                recursive=self.recursive,
                progress_cb=lambda s, path: self.progress.emit(dict(s), path),
                should_cancel=lambda: self._cancel,
            )
            cat.close()
            self.finished_ok.emit(dict(stats))
        except Exception as e:
            self.failed.emit(str(e))


class CatalogThumbWorker(QThread):
    thumb_ready = pyqtSignal(str, object)
    finished_ok = pyqtSignal()

    def __init__(self, records, size=160):
        super().__init__()
        self.records = records
        self.size = size
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        import os
        from catalog import thumb_cache_path
        for rec in self.records:
            if self._cancel:
                break
            path = rec.get("path") if isinstance(rec, dict) else rec
            mtime = float(rec.get("file_mtime") or 0) if isinstance(rec, dict) else 0
            try:
                if mtime <= 0 and path and os.path.isfile(path):
                    mtime = os.path.getmtime(path)
                cache = thumb_cache_path(path, mtime)
                if os.path.isfile(cache):
                    pix = QPixmap(cache)
                    if not pix.isNull():
                        self.thumb_ready.emit(path, pix)
                        continue
                img = extract_embedded_preview(path, max_side=self.size)
                if img is None:
                    continue
                pix = cv_to_qpixmap(img)
                try:
                    pix.save(cache, "JPEG", 85)
                except Exception:
                    pass
                self.thumb_ready.emit(path, pix)
            except Exception:
                continue
        self.finished_ok.emit()


class BatchExportWorker(QThread):
    """Export many images with their recipes (or a shared recipe)."""
    progress = pyqtSignal(int, int, str)  # done, total, path
    finished_ok = pyqtSignal(int)  # count
    failed = pyqtSignal(str)

    def __init__(self, jobs, max_dim=0, jpeg_quality=92):
        """jobs: list of dicts {path, recipe, out_path, wb_multipliers}"""
        super().__init__()
        self.jobs = jobs
        self.max_dim = max_dim
        self.jpeg_quality = jpeg_quality
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        ok = 0
        total = len(self.jobs)
        for i, job in enumerate(self.jobs):
            if self._cancel:
                break
            path = job["path"]
            try:
                self.progress.emit(i, total, path)
                img, meta = load_image(path, use_camera_wb=True)
                mult = job.get("wb_multipliers") or meta.get("wb_multipliers")
                out = apply_recipe(img, job["recipe"], wb_multipliers=mult, meta=meta)
                if self.max_dim and max(out.shape[:2]) > self.max_dim:
                    h, w = out.shape[:2]
                    scale = self.max_dim / max(h, w)
                    out = cv2.resize(out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                out_path = job["out_path"]
                ext = out_path.lower().rsplit(".", 1)[-1]
                if ext in ("jpg", "jpeg"):
                    cv2.imwrite(out_path, out, [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)])
                else:
                    cv2.imwrite(out_path, out)
                ok += 1
            except Exception as e:
                self.failed.emit(f"{path}: {e}")
        self.finished_ok.emit(ok)


class FocusStackWorker(QThread):
    """Background focus stack (align + fuse)."""
    finished_ok = pyqtSignal(str, object)  # out_path, report
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        paths,
        out_path,
        align_mode="ecc_affine",
        fusion_mode="depth",
        reference="middle",
        max_dim=0,
        focus_radius=5,
        boundary_smooth=7,
        pyramid_levels=5,
        crop_common=True,
        save_depth=False,
        min_align_score=0.0,
        normalize_exposure=False,
    ):
        super().__init__()
        self.paths = list(paths)
        self.out_path = out_path
        self.align_mode = align_mode
        self.fusion_mode = fusion_mode
        self.reference = reference
        self.max_dim = max_dim
        self.focus_radius = focus_radius
        self.boundary_smooth = boundary_smooth
        self.pyramid_levels = pyramid_levels
        self.crop_common = crop_common
        self.save_depth = save_depth
        self.min_align_score = float(min_align_score or 0.0)
        self.normalize_exposure = bool(normalize_exposure)

    def run(self):
        try:
            from focus_stack import focus_stack
            import cv2
            import numpy as np

            def cb(msg, frac=None):
                self.progress.emit(msg)

            result, depth, report = focus_stack(
                self.paths,
                align_mode=self.align_mode,
                fusion_mode=self.fusion_mode,
                reference=self.reference,
                max_dim=self.max_dim or 0,
                focus_radius=self.focus_radius,
                boundary_smooth=self.boundary_smooth,
                pyramid_levels=self.pyramid_levels,
                crop_common=self.crop_common,
                min_align_score=self.min_align_score,
                normalize_exposure=self.normalize_exposure,
                progress_cb=cb,
            )
            ext = self.out_path.lower().rsplit(".", 1)[-1]
            if ext in ("jpg", "jpeg"):
                cv2.imwrite(self.out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(self.out_path, result)
            if self.save_depth and depth is not None:
                depth_path = self.out_path.rsplit(".", 1)[0] + "_depth.png"
                d = depth
                if d.dtype != np.uint8:
                    d = np.clip(d, 0, 255).astype(np.uint8)
                cv2.imwrite(depth_path, d)
                report["depth_path"] = depth_path
            report["out_path"] = self.out_path
            self.finished_ok.emit(self.out_path, report)
        except Exception as e:
            self.failed.emit(str(e))



class PanoramaWorker(QThread):
    """Background OpenCV panorama stitch."""
    finished_ok = pyqtSignal(str, object)  # out_path, report
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, paths, out_path, mode="panoramas", max_dim=0,
                 match_exposure=True, order_by_time=False, exposure_reference=0,
                 exposure_strength=1.0, confidence_threshold=1.0,
                 wave_correction=True, crop_borders=True,
                 output_projection="original", projection_strength=1.0,
                 projection_fov=120.0, projection_border="reflect",
                 seam_refine_strength=0.0, seam_refine_radius=12):
        super().__init__()
        self.paths = list(paths)
        self.out_path = out_path
        self.mode = mode
        self.max_dim = max_dim
        self.match_exposure = bool(match_exposure)
        self.order_by_time = bool(order_by_time)
        self.exposure_reference = int(exposure_reference)
        self.exposure_strength = float(exposure_strength)
        self.confidence_threshold = float(confidence_threshold)
        self.wave_correction = bool(wave_correction)
        self.crop_borders = bool(crop_borders)
        self.output_projection = output_projection
        self.projection_strength = float(projection_strength)
        self.projection_fov = float(projection_fov)
        self.projection_border = projection_border
        self.seam_refine_strength = float(seam_refine_strength)
        self.seam_refine_radius = int(seam_refine_radius)

    def run(self):
        try:
            from panorama import stitch_panorama
            import cv2

            def cb(msg, frac=None):
                self.progress.emit(msg)

            result, report = stitch_panorama(
                self.paths,
                mode=self.mode,
                max_dim=self.max_dim or 0,
                match_exposure=self.match_exposure,
                order_by_time=self.order_by_time,
                exposure_reference=self.exposure_reference,
                exposure_strength=self.exposure_strength,
                confidence_threshold=self.confidence_threshold,
                wave_correction=self.wave_correction,
                crop_borders=self.crop_borders,
                output_projection=self.output_projection,
                projection_strength=self.projection_strength,
                projection_fov=self.projection_fov,
                projection_border=self.projection_border,
                seam_refine_strength=self.seam_refine_strength,
                seam_refine_radius=self.seam_refine_radius,
                progress_cb=cb,
            )
            ext = self.out_path.lower().rsplit(".", 1)[-1]
            if ext in ("jpg", "jpeg"):
                cv2.imwrite(self.out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(self.out_path, result)
            report["out_path"] = self.out_path
            self.finished_ok.emit(self.out_path, report)
        except Exception as e:
            self.failed.emit(str(e))
