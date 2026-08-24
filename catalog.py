"""catalog.py — SQLite photo library (scan, date index, ratings).

No Qt dependency. Used by CatalogScanWorker and PhotoLab library UI.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from imaging import IMAGE_EXTS, is_raw, extract_exif

# Default DB location
def default_db_path() -> str:
    try:
        from config import get_config
        override = get_config().path("catalog_db")
        if override:
            parent = os.path.dirname(override) or "."
            os.makedirs(parent, exist_ok=True)
            return override
    except Exception:
        pass
    root = os.path.join(os.path.expanduser("~"), ".photolab")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "catalog.db")


def default_thumb_dir() -> str:
    try:
        from config import get_config
        override = get_config().path("thumb_cache")
        if override:
            os.makedirs(override, exist_ok=True)
            return override
    except Exception:
        pass
    d = os.path.join(os.path.expanduser("~"), ".photolab", "cache", "thumbs")
    os.makedirs(d, exist_ok=True)
    return d


def _parse_exif_datetime(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).strip()
    for fmt in (
        "%Y:%m:%d %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y:%m:%d %H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(s[:26].replace("T", " "), fmt[: len(s) + 2] if False else fmt)
        except Exception:
            continue
    # Loose: take first 19 chars as EXIF classic
    try:
        return datetime.strptime(s[:19], "%Y:%m:%d %H:%M:%S")
    except Exception:
        return None


def resolve_capture_datetime(path: str, exif: Optional[dict] = None) -> Tuple[datetime, str]:
    """Return (datetime, source) where source is 'exif' or 'file'."""
    if exif is None:
        exif = {}
    for key in ("datetime_original", "datetime", "DateTimeOriginal", "DateTime"):
        raw = exif.get(key) or exif.get(key.lower())
        if raw:
            dt = _parse_exif_datetime(str(raw))
            if dt:
                return dt, "exif"
    # File mtime fallback
    try:
        mtime = os.path.getmtime(path)
        return datetime.fromtimestamp(mtime), "file"
    except Exception:
        return datetime.now(), "file"


def thumb_cache_path(image_path: str, mtime: float, thumb_dir: Optional[str] = None) -> str:
    thumb_dir = thumb_dir or default_thumb_dir()
    key = hashlib.sha1(f"{image_path}|{mtime}".encode("utf-8", errors="ignore")).hexdigest()
    return os.path.join(thumb_dir, f"{key}.jpg")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    folder TEXT,
    filename TEXT,
    file_mtime REAL,
    file_size INTEGER,
    exif_datetime TEXT,
    date_key TEXT,
    year INTEGER,
    month INTEGER,
    day INTEGER,
    camera TEXT,
    lens TEXT,
    iso TEXT,
    aperture TEXT,
    shutter TEXT,
    focal TEXT,
    is_raw INTEGER DEFAULT 0,
    width INTEGER,
    height INTEGER,
    rating INTEGER DEFAULT 0,
    reject INTEGER DEFAULT 0,
    last_scan_ts REAL,
    date_source TEXT,
    keywords TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_images_date_key ON images(date_key);
CREATE INDEX IF NOT EXISTS idx_images_folder ON images(folder);
CREATE INDEX IF NOT EXISTS idx_images_year_month ON images(year, month);
CREATE INDEX IF NOT EXISTS idx_images_rating ON images(rating);
CREATE INDEX IF NOT EXISTS idx_images_reject ON images(reject);

CREATE TABLE IF NOT EXISTS roots (
    path TEXT PRIMARY KEY,
    last_scan_ts REAL
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class Catalog:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or default_db_path()
        self.thumb_dir = default_thumb_dir()
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()

    def _migrate(self):
        """Add columns/indexes introduced after the first release.

        CREATE TABLE IF NOT EXISTS never alters an existing table, so new
        columns must be added with ALTER TABLE, and indexes that depend on
        them must be created only after that succeeds.
        """
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(images)").fetchall()
        }
        # (column_name, SQL type + default)
        wanted = [
            ("keywords", "TEXT DEFAULT ''"),
        ]
        for name, decl in wanted:
            if name not in cols:
                try:
                    self._conn.execute(f"ALTER TABLE images ADD COLUMN {name} {decl}")
                except Exception:
                    pass
        # Indexes that may depend on migrated columns
        for sql in (
            "CREATE INDEX IF NOT EXISTS idx_images_keywords ON images(keywords)",
            "CREATE INDEX IF NOT EXISTS idx_images_date_key ON images(date_key)",
            "CREATE INDEX IF NOT EXISTS idx_images_folder ON images(folder)",
            "CREATE INDEX IF NOT EXISTS idx_images_year_month ON images(year, month)",
            "CREATE INDEX IF NOT EXISTS idx_images_rating ON images(rating)",
            "CREATE INDEX IF NOT EXISTS idx_images_reject ON images(reject)",
        ):
            try:
                self._conn.execute(sql)
            except Exception:
                pass
        self._conn.commit()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            pass

    def connection(self) -> sqlite3.Connection:
        return self._conn

    # ----- roots -----
    def add_root(self, folder: str):
        folder = os.path.abspath(folder)
        self._conn.execute(
            "INSERT OR REPLACE INTO roots(path, last_scan_ts) VALUES (?, ?)",
            (folder, time.time()),
        )
        self._conn.commit()

    def list_roots(self) -> List[str]:
        rows = self._conn.execute("SELECT path FROM roots ORDER BY path").fetchall()
        return [r["path"] for r in rows]

    # ----- scan helpers -----
    def get_file_fingerprint(self, path: str) -> Optional[Tuple[float, int]]:
        row = self._conn.execute(
            "SELECT file_mtime, file_size FROM images WHERE path = ?", (path,)
        ).fetchone()
        if row is None:
            return None
        return float(row["file_mtime"] or 0), int(row["file_size"] or 0)

    def upsert_image(self, record: Dict[str, Any]):
        cols = [
            "path", "folder", "filename", "file_mtime", "file_size",
            "exif_datetime", "date_key", "year", "month", "day",
            "camera", "lens", "iso", "aperture", "shutter", "focal",
            "is_raw", "width", "height", "last_scan_ts", "date_source",
        ]
        # Preserve rating/reject on update
        existing = self._conn.execute(
            "SELECT rating, reject FROM images WHERE path = ?", (record["path"],)
        ).fetchone()
        rating = existing["rating"] if existing else record.get("rating", 0)
        reject = existing["reject"] if existing else record.get("reject", 0)

        values = [record.get(c) for c in cols] + [rating, reject]
        self._conn.execute(
            f"""
            INSERT INTO images ({", ".join(cols)}, rating, reject)
            VALUES ({", ".join("?" for _ in cols)}, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
              folder=excluded.folder,
              filename=excluded.filename,
              file_mtime=excluded.file_mtime,
              file_size=excluded.file_size,
              exif_datetime=excluded.exif_datetime,
              date_key=excluded.date_key,
              year=excluded.year,
              month=excluded.month,
              day=excluded.day,
              camera=excluded.camera,
              lens=excluded.lens,
              iso=excluded.iso,
              aperture=excluded.aperture,
              shutter=excluded.shutter,
              focal=excluded.focal,
              is_raw=excluded.is_raw,
              width=excluded.width,
              height=excluded.height,
              last_scan_ts=excluded.last_scan_ts,
              date_source=excluded.date_source
            """,
            values,
        )

    def commit(self):
        self._conn.commit()

    def build_record(self, path: str) -> Dict[str, Any]:
        path = os.path.abspath(path)
        st = os.stat(path)
        exif = extract_exif(path)
        # Prefer DateTimeOriginal if PIL exposed it — extract_exif may only have DateTime
        dt, source = resolve_capture_datetime(path, exif)
        date_key = dt.strftime("%Y-%m-%d")
        width = height = None
        try:
            from PIL import Image
            with Image.open(path) as im:
                width, height = im.size
        except Exception:
            pass
        return {
            "path": path,
            "folder": os.path.dirname(path),
            "filename": os.path.basename(path),
            "file_mtime": st.st_mtime,
            "file_size": st.st_size,
            "exif_datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "date_key": date_key,
            "year": dt.year,
            "month": dt.month,
            "day": dt.day,
            "camera": exif.get("camera"),
            "lens": exif.get("lens"),
            "iso": exif.get("iso"),
            "aperture": exif.get("aperture"),
            "shutter": exif.get("shutter"),
            "focal": exif.get("focal"),
            "is_raw": 1 if is_raw(path) else 0,
            "width": width,
            "height": height,
            "last_scan_ts": time.time(),
            "date_source": source,
            "rating": 0,
            "reject": 0,
        }

    def scan_folder(
        self,
        root: str,
        recursive: bool = True,
        progress_cb=None,
        should_cancel=None,
    ) -> Dict[str, int]:
        """Walk folder, upsert images. Returns stats dict."""
        root = os.path.abspath(root)
        self.add_root(root)
        stats = {"seen": 0, "added": 0, "updated": 0, "skipped": 0}
        exts = tuple(e.lower() for e in IMAGE_EXTS)

        def iter_files():
            if recursive:
                for dirpath, _dirnames, filenames in os.walk(root):
                    for name in filenames:
                        if name.lower().endswith(exts):
                            yield os.path.join(dirpath, name)
            else:
                for name in os.listdir(root):
                    p = os.path.join(root, name)
                    if os.path.isfile(p) and name.lower().endswith(exts):
                        yield p

        for path in iter_files():
            if should_cancel and should_cancel():
                break
            stats["seen"] += 1
            try:
                st = os.stat(path)
                fp = self.get_file_fingerprint(path)
                if fp and abs(fp[0] - st.st_mtime) < 0.5 and fp[1] == st.st_size:
                    stats["skipped"] += 1
                    if progress_cb:
                        progress_cb(stats, path)
                    continue
                rec = self.build_record(path)
                existed = fp is not None
                self.upsert_image(rec)
                if existed:
                    stats["updated"] += 1
                else:
                    stats["added"] += 1
                if stats["seen"] % 25 == 0:
                    self.commit()
                if progress_cb:
                    progress_cb(stats, path)
            except Exception:
                stats["skipped"] += 1
                if progress_cb:
                    progress_cb(stats, path)
        self.commit()
        return stats

    # ----- queries -----
    def count(self, include_rejected: bool = True) -> int:
        if include_rejected:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM images").fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS c FROM images WHERE reject = 0"
            ).fetchone()
        return int(row["c"])

    def date_tree(self, include_rejected: bool = False) -> List[Dict[str, Any]]:
        """Nested structure: [{year, count, months: [{month, count, days: [{day, date_key, count}]}]}]"""
        where = "" if include_rejected else "WHERE reject = 0"
        rows = self._conn.execute(
            f"""
            SELECT year, month, day, date_key, COUNT(*) AS c
            FROM images
            {where}
            GROUP BY year, month, day, date_key
            ORDER BY year DESC, month DESC, day DESC
            """
        ).fetchall()
        years: Dict[int, Dict] = {}
        for r in rows:
            y, m, d = int(r["year"]), int(r["month"]), int(r["day"])
            if y not in years:
                years[y] = {"year": y, "count": 0, "months": {}}
            years[y]["count"] += int(r["c"])
            months = years[y]["months"]
            if m not in months:
                months[m] = {"month": m, "count": 0, "days": []}
            months[m]["count"] += int(r["c"])
            months[m]["days"].append(
                {"day": d, "date_key": r["date_key"], "count": int(r["c"])}
            )
        out = []
        for y in sorted(years.keys(), reverse=True):
            yd = years[y]
            month_list = []
            for m in sorted(yd["months"].keys(), reverse=True):
                md = yd["months"][m]
                month_list.append(md)
            yd["months"] = month_list
            out.append(yd)
        return out

    def images_for_date(
        self,
        date_key: Optional[str] = None,
        year: Optional[int] = None,
        month: Optional[int] = None,
        include_rejected: bool = False,
        only_rejected: bool = False,
        min_rating: int = 0,
    ) -> List[Dict[str, Any]]:
        clauses = []
        params: list = []
        if date_key:
            clauses.append("date_key = ?")
            params.append(date_key)
        if year is not None:
            clauses.append("year = ?")
            params.append(year)
        if month is not None:
            clauses.append("month = ?")
            params.append(month)
        if only_rejected:
            clauses.append("reject = 1")
        elif not include_rejected:
            clauses.append("reject = 0")
        if min_rating > 0:
            clauses.append("rating >= ?")
            params.append(min_rating)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"""
            SELECT * FROM images
            {where}
            ORDER BY exif_datetime DESC, filename ASC
            """,
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def set_rating(self, path: str, rating: int):
        rating = max(0, min(5, int(rating)))
        self._conn.execute("UPDATE images SET rating = ? WHERE path = ?", (rating, path))
        self._conn.commit()

    def set_reject(self, path: str, reject: bool):
        self._conn.execute(
            "UPDATE images SET reject = ? WHERE path = ?", (1 if reject else 0, path)
        )
        self._conn.commit()

    def get_image(self, path: str) -> Optional[Dict[str, Any]]:
        row = self._conn.execute("SELECT * FROM images WHERE path = ?", (path,)).fetchone()
        return dict(row) if row else None

    def remove_image(self, path: str):
        """Remove a path from the catalog (does not delete the file)."""
        self._conn.execute("DELETE FROM images WHERE path = ?", (path,))
        self._conn.commit()

    def get(self, path: str):
        row = self._conn.execute("SELECT * FROM images WHERE path = ?", (path,)).fetchone()
        return dict(row) if row else None

    def set_keywords(self, path: str, keywords: str):
        self._conn.execute(
            "UPDATE images SET keywords = ? WHERE path = ?",
            (keywords.strip(), path),
        )
        self._conn.commit()

    def search(self, query: str, include_rejected: bool = False):
        """Simple search across filename, keywords, camera, folder."""
        q = f"%{(query or '').strip()}%"
        if not query or not query.strip():
            return self.images_for_date(include_rejected=include_rejected)
        where = [
            "(filename LIKE ? OR keywords LIKE ? OR camera LIKE ? OR folder LIKE ? OR path LIKE ?)"
        ]
        params = [q, q, q, q, q]
        if not include_rejected:
            where.append("reject = 0")
        sql = f"SELECT * FROM images WHERE {' AND '.join(where)} ORDER BY exif_datetime DESC, filename ASC"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
