"""workers.py — background QThreads so the UI never blocks on I/O or the
image pipeline (thumbnail generation, full-resolution export, RAW decode)."""

from __future__ import annotations

import os
import cv2
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QPixmap

from imaging import apply_recipe, load_image, is_raw, _silent_imread, extract_embedded_preview
from qt_utils import cv_to_qpixmap


class ThumbnailWorker(QThread):
    thumb_ready = pyqtSignal(str, QPixmap)

    def __init__(self, paths):
        super().__init__()
        self.paths = paths

    def run(self):
        for p in self.paths:
            try:
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

    def __init__(self, path, recipe, out_path, wb_multipliers=None,
                 watermark_text="", watermark_opacity=0.45, max_dim=0, jpeg_quality=92):
        super().__init__()
        self.path = path
        self.recipe = recipe
        self.out_path = out_path
        self.wb_multipliers = wb_multipliers
        self.watermark_text = watermark_text or ""
        self.watermark_opacity = watermark_opacity
        self.max_dim = max_dim
        self.jpeg_quality = jpeg_quality

    def run(self):
        try:
            from imaging import apply_watermark
            img, meta = load_image(self.path, use_camera_wb=True)
            multipliers = self.wb_multipliers or meta.get("wb_multipliers")
            out = apply_recipe(img, self.recipe, wb_multipliers=multipliers, meta=meta)
            if self.max_dim and max(out.shape[:2]) > self.max_dim:
                h, w = out.shape[:2]
                scale = self.max_dim / max(h, w)
                out = cv2.resize(out, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            if self.watermark_text:
                out = apply_watermark(out, self.watermark_text, opacity=self.watermark_opacity)
            ext = self.out_path.lower().rsplit(".", 1)[-1]
            if ext in ("jpg", "jpeg"):
                cv2.imwrite(self.out_path, out, [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)])
            else:
                cv2.imwrite(self.out_path, out)
            self.finished_ok.emit(self.out_path)
        except Exception as e:
            self.failed.emit(str(e))


class LoadImageWorker(QThread):
    """Background full-resolution load (especially useful for large RAWs)."""
    loaded = pyqtSignal(str, object, object)  # path, img_bgr, meta
    failed = pyqtSignal(str, str)

    def __init__(self, path: str):
        super().__init__()
        self.path = path

    def run(self):
        try:
            img, meta = load_image(self.path, use_camera_wb=True)
            self.loaded.emit(self.path, img, meta)
        except Exception as e:
            self.failed.emit(self.path, str(e))


class HdrMergeWorker(QThread):
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, paths, out_path, align=True, max_dim=0):
        super().__init__()
        self.paths = list(paths)
        self.out_path = out_path
        self.align = align
        self.max_dim = max_dim

    def run(self):
        try:
            from imaging import merge_hdr_mertens
            self.progress.emit(f"Merging {len(self.paths)} exposures…")
            out = merge_hdr_mertens(self.paths, align=self.align, max_dim=self.max_dim or 0)
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
    """Background focus-stack: align + fuse selected frames."""
    finished_ok = pyqtSignal(str, object)  # out_path, report dict
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
        focus_radius=3,
        boundary_smooth=5,
        pyramid_levels=5,
        crop_common=True,
        save_depth=False,
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

    def run(self):
        try:
            from focus_stack import focus_stack
            import cv2

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
                progress_cb=cb,
            )
            ext = self.out_path.lower().rsplit(".", 1)[-1]
            if ext in ("tif", "tiff"):
                cv2.imwrite(self.out_path, result)
            elif ext in ("png",):
                cv2.imwrite(self.out_path, result)
            else:
                cv2.imwrite(self.out_path, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if self.save_depth and depth is not None:
                depth_path = self.out_path.rsplit(".", 1)[0] + "_depth.png"
                cv2.imwrite(depth_path, depth)
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

    def __init__(self, paths, out_path, mode="panoramas", max_dim=0):
        super().__init__()
        self.paths = list(paths)
        self.out_path = out_path
        self.mode = mode
        self.max_dim = max_dim

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


class ImportWorker(QThread):
    """Copy/move files into a destination with rename rules."""
    progress = pyqtSignal(int, int, str)  # index, total, src
    finished_ok = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(
        self,
        sources,
        dest_dir,
        mode="copy",
        rename_pattern="keep",
        subfolder_by_date=True,
    ):
        super().__init__()
        self.sources = list(sources)
        self.dest_dir = dest_dir
        self.mode = mode
        self.rename_pattern = rename_pattern
        self.subfolder_by_date = subfolder_by_date
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        try:
            from catalog import import_photos

            def cb(i, n, src):
                self.progress.emit(i, n, src)

            stats = import_photos(
                self.sources,
                self.dest_dir,
                mode=self.mode,
                rename_pattern=self.rename_pattern,
                subfolder_by_date=self.subfolder_by_date,
                progress_cb=cb,
                should_cancel=lambda: self._cancel,
            )
            self.finished_ok.emit(stats)
        except Exception as e:
            self.failed.emit(str(e))
