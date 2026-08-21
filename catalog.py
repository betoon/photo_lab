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

from imaging import IMAGE_EXTS, is_raw, extract_exif, safe_pil_open
import logging

log = logging.getLogger(__name__)

# Default DB location
def default_db_path() -> str:
    root = os.path.join(os.path.expanduser("~"), ".photolab")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "catalog.db")


def default_thumb_dir() -> str:
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

CREATE TABLE IF NOT EXISTS collections (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    parent_id INTEGER,
    created_ts REAL,
    UNIQUE(name, parent_id)
);

CREATE TABLE IF NOT EXISTS collection_members (
    collection_id INTEGER NOT NULL,
    path TEXT NOT NULL,
    PRIMARY KEY (collection_id, path)
);

CREATE TABLE IF NOT EXISTS virtual_copies (
    id INTEGER PRIMARY KEY,
    master_path TEXT NOT NULL,
    name TEXT,
    recipe_json TEXT,
    created_ts REAL,
    rating INTEGER DEFAULT 0,
    reject INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_vc_master ON virtual_copies(master_path);
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
            ("people", "TEXT DEFAULT ''"),  # face / person tags, comma-separated
            ("color_label", "TEXT DEFAULT ''"),
            ("content_hash", "TEXT DEFAULT ''"),  # short hash for duplicate detection
        ]
        for name, decl in wanted:
            if name not in cols:
                try:
                    self._conn.execute(f"ALTER TABLE images ADD COLUMN {name} {decl}")
                except Exception:
                    log.debug("_migrate: non-critical failure, continuing", exc_info=True)
        # Indexes that may depend on migrated columns
        for sql in (
            "CREATE INDEX IF NOT EXISTS idx_images_keywords ON images(keywords)",
            "CREATE INDEX IF NOT EXISTS idx_images_people ON images(people)",
            "CREATE INDEX IF NOT EXISTS idx_images_content_hash ON images(content_hash)",
            "CREATE INDEX IF NOT EXISTS idx_images_date_key ON images(date_key)",
            "CREATE INDEX IF NOT EXISTS idx_images_folder ON images(folder)",
            "CREATE INDEX IF NOT EXISTS idx_images_year_month ON images(year, month)",
            "CREATE INDEX IF NOT EXISTS idx_images_rating ON images(rating)",
            "CREATE INDEX IF NOT EXISTS idx_images_reject ON images(reject)",
            """CREATE TABLE IF NOT EXISTS collections (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                parent_id INTEGER,
                created_ts REAL,
                UNIQUE(name, parent_id)
            )""",
            """CREATE TABLE IF NOT EXISTS collection_members (
                collection_id INTEGER NOT NULL,
                path TEXT NOT NULL,
                PRIMARY KEY (collection_id, path)
            )""",
            """CREATE TABLE IF NOT EXISTS virtual_copies (
                id INTEGER PRIMARY KEY,
                master_path TEXT NOT NULL,
                name TEXT,
                recipe_json TEXT,
                created_ts REAL,
                rating INTEGER DEFAULT 0,
                reject INTEGER DEFAULT 0
            )""",
            "CREATE INDEX IF NOT EXISTS idx_vc_master ON virtual_copies(master_path)",
        ):
            try:
                self._conn.execute(sql)
            except Exception:
                log.debug("_migrate: non-critical failure, continuing", exc_info=True)
        self._conn.commit()

    def close(self):
        try:
            self._conn.close()
        except Exception:
            log.debug("close: non-critical failure, continuing", exc_info=True)

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
            "content_hash",
        ]
        # Preserve rating/reject/keywords/people on update
        existing = self._conn.execute(
            "SELECT rating, reject, keywords, people, color_label FROM images WHERE path = ?",
            (record["path"],),
        ).fetchone()
        rating = existing["rating"] if existing else record.get("rating", 0)
        reject = existing["reject"] if existing else record.get("reject", 0)
        keywords = (existing["keywords"] if existing else "") or record.get("keywords", "")
        people = (existing["people"] if existing else "") or record.get("people", "")
        color_label = (existing["color_label"] if existing else "") or record.get("color_label", "")

        values = [record.get(c) for c in cols] + [rating, reject, keywords, people, color_label]
        self._conn.execute(
            f"""
            INSERT INTO images ({", ".join(cols)}, rating, reject, keywords, people, color_label)
            VALUES ({", ".join("?" for _ in cols)}, ?, ?, ?, ?, ?)
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
              date_source=excluded.date_source,
              content_hash=excluded.content_hash
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
            with safe_pil_open(path) as im:
                width, height = im.size
        except Exception:
            log.debug("build_record: non-critical failure, continuing", exc_info=True)
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
            "content_hash": self._quick_content_hash(path, st.st_size, st.st_mtime),
        }

    @staticmethod
    def _quick_content_hash(path: str, size: int, mtime: float) -> str:
        """Fast fingerprint: size + sample of file head/mid/tail (not cryptographic)."""
        h = hashlib.sha1()
        h.update(f"{size}|".encode())
        try:
            with open(path, "rb") as f:
                head = f.read(65536)
                h.update(head)
                if size > 131072:
                    f.seek(max(0, size // 2 - 32768))
                    h.update(f.read(65536))
                if size > 65536:
                    f.seek(max(0, size - 65536))
                    h.update(f.read(65536))
        except Exception:
            h.update(f"{mtime}".encode())
        return h.hexdigest()[:20]

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
            except Exception as exc:
                stats["skipped"] += 1
                stats.setdefault("errors", []).append({"path": path, "error": str(exc)})
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

    def set_people(self, path: str, people: str):
        """Face / person tags (comma-separated names)."""
        self._conn.execute(
            "UPDATE images SET people = ? WHERE path = ?",
            (people.strip(), path),
        )
        self._conn.commit()

    def search(self, query: str, include_rejected: bool = False):
        """Simple search across filename, keywords, people, camera, folder."""
        q = f"%{(query or '').strip()}%"
        if not query or not query.strip():
            return self.images_for_date(include_rejected=include_rejected)
        where = [
            "(filename LIKE ? OR keywords LIKE ? OR people LIKE ? OR camera LIKE ? OR folder LIKE ? OR path LIKE ?)"
        ]
        params = [q, q, q, q, q, q]
        if not include_rejected:
            where.append("reject = 0")
        sql = f"SELECT * FROM images WHERE {' AND '.join(where)} ORDER BY exif_datetime DESC, filename ASC"
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ----- collections -----
    def create_collection(self, name: str, parent_id: Optional[int] = None) -> int:
        name = (name or "").strip() or "Untitled"
        cur = self._conn.execute(
            "INSERT INTO collections(name, parent_id, created_ts) VALUES (?, ?, ?)",
            (name, parent_id, time.time()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list_collections(self) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT c.*, (SELECT COUNT(*) FROM collection_members m WHERE m.collection_id = c.id) AS count "
            "FROM collections c ORDER BY c.name"
        ).fetchall()
        return [dict(r) for r in rows]

    def rename_collection(self, collection_id: int, name: str):
        self._conn.execute(
            "UPDATE collections SET name = ? WHERE id = ?",
            ((name or "").strip() or "Untitled", collection_id),
        )
        self._conn.commit()

    def delete_collection(self, collection_id: int):
        self._conn.execute("DELETE FROM collection_members WHERE collection_id = ?", (collection_id,))
        self._conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
        self._conn.commit()

    def add_to_collection(self, collection_id: int, paths: Iterable[str]):
        for p in paths:
            try:
                self._conn.execute(
                    "INSERT OR IGNORE INTO collection_members(collection_id, path) VALUES (?, ?)",
                    (collection_id, p),
                )
            except Exception:
                log.debug("add_to_collection: non-critical failure, continuing", exc_info=True)
        self._conn.commit()

    def remove_from_collection(self, collection_id: int, paths: Iterable[str]):
        for p in paths:
            self._conn.execute(
                "DELETE FROM collection_members WHERE collection_id = ? AND path = ?",
                (collection_id, p),
            )
        self._conn.commit()

    def images_in_collection(self, collection_id: int, include_rejected: bool = False) -> List[Dict[str, Any]]:
        where = "m.collection_id = ?"
        params: List[Any] = [collection_id]
        if not include_rejected:
            where += " AND COALESCE(i.reject, 0) = 0"
        sql = (
            f"SELECT i.* FROM collection_members m "
            f"JOIN images i ON i.path = m.path "
            f"WHERE {where} ORDER BY i.exif_datetime DESC, i.filename ASC"
        )
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    # ----- duplicates -----
    def find_duplicate_groups(self, min_group_size: int = 2) -> List[List[Dict[str, Any]]]:
        """Group images that share the same content_hash (and non-empty hash)."""
        rows = self._conn.execute(
            """
            SELECT content_hash, COUNT(*) AS n FROM images
            WHERE content_hash IS NOT NULL AND content_hash != ''
            GROUP BY content_hash HAVING n >= ?
            ORDER BY n DESC
            """,
            (min_group_size,),
        ).fetchall()
        groups = []
        for row in rows:
            members = self._conn.execute(
                "SELECT * FROM images WHERE content_hash = ? ORDER BY path",
                (row["content_hash"],),
            ).fetchall()
            groups.append([dict(m) for m in members])
        return groups

    # ----- virtual copies -----
    def create_virtual_copy(
        self, master_path: str, name: str = "", recipe_json: str = ""
    ) -> int:
        cur = self._conn.execute(
            """
            INSERT INTO virtual_copies(master_path, name, recipe_json, created_ts)
            VALUES (?, ?, ?, ?)
            """,
            (master_path, name or "Copy", recipe_json or "", time.time()),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def list_virtual_copies(self, master_path: Optional[str] = None) -> List[Dict[str, Any]]:
        if master_path:
            rows = self._conn.execute(
                "SELECT * FROM virtual_copies WHERE master_path = ? ORDER BY created_ts",
                (master_path,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM virtual_copies ORDER BY created_ts DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_virtual_copy(self, vc_id: int) -> Optional[Dict[str, Any]]:
        row = self._conn.execute(
            "SELECT * FROM virtual_copies WHERE id = ?", (vc_id,)
        ).fetchone()
        return dict(row) if row else None

    def update_virtual_copy_recipe(self, vc_id: int, recipe_json: str):
        self._conn.execute(
            "UPDATE virtual_copies SET recipe_json = ? WHERE id = ?",
            (recipe_json, vc_id),
        )
        self._conn.commit()

    def delete_virtual_copy(self, vc_id: int):
        self._conn.execute("DELETE FROM virtual_copies WHERE id = ?", (vc_id,))
        self._conn.commit()

    def list_people_tags(self) -> List[str]:
        """Unique person names appearing in the people column."""
        rows = self._conn.execute(
            "SELECT people FROM images WHERE people IS NOT NULL AND people != ''"
        ).fetchall()
        names = set()
        for r in rows:
            for part in str(r["people"]).split(","):
                p = part.strip()
                if p:
                    names.add(p)
        return sorted(names, key=str.lower)


def list_importable_files(source_dir: str, recursive: bool = True) -> List[str]:
    """Image paths under source_dir suitable for import."""
    source_dir = os.path.abspath(source_dir)
    exts = tuple(e.lower() for e in IMAGE_EXTS)
    out: List[str] = []
    if recursive:
        for dirpath, _dns, filenames in os.walk(source_dir):
            for name in filenames:
                if name.lower().endswith(exts):
                    out.append(os.path.join(dirpath, name))
    else:
        for name in os.listdir(source_dir):
            p = os.path.join(source_dir, name)
            if os.path.isfile(p) and name.lower().endswith(exts):
                out.append(p)
    out.sort()
    return out


def format_import_name(
    src_path: str,
    sequence: int,
    pattern: str = "keep",
    capture_dt: Optional[datetime] = None,
) -> str:
    """Build destination filename from pattern.

    pattern:
      keep — original basename
      date_seq — YYYYMMDD_0001.ext
      date_orig — YYYYMMDD_originalname.ext
    """
    base = os.path.basename(src_path)
    stem, ext = os.path.splitext(base)
    if capture_dt is None:
        capture_dt, _ = resolve_capture_datetime(src_path)
    date_s = capture_dt.strftime("%Y%m%d")
    if pattern == "date_seq":
        return f"{date_s}_{sequence:04d}{ext.lower()}"
    if pattern == "date_orig":
        return f"{date_s}_{stem}{ext.lower()}"
    return base


def import_photos(
    sources: List[str],
    dest_dir: str,
    mode: str = "copy",
    rename_pattern: str = "keep",
    subfolder_by_date: bool = True,
    progress_cb=None,
    should_cancel=None,
) -> Dict[str, Any]:
    """Copy or move images into dest_dir with optional rename / date folders.

    Returns stats: {ok, failed, skipped, paths: [dest paths]}.
    """
    import shutil

    dest_dir = os.path.abspath(dest_dir)
    os.makedirs(dest_dir, exist_ok=True)
    stats: Dict[str, Any] = {"ok": 0, "failed": 0, "skipped": 0, "paths": []}
    seq = 1
    for i, src in enumerate(sources):
        if should_cancel and should_cancel():
            break
        if progress_cb:
            try:
                progress_cb(i, len(sources), src)
            except Exception:
                log.debug("import_photos: non-critical failure, continuing", exc_info=True)
        try:
            if not os.path.isfile(src):
                stats["skipped"] += 1
                continue
            dt, _ = resolve_capture_datetime(src)
            name = format_import_name(src, seq, rename_pattern, dt)
            if subfolder_by_date:
                sub = os.path.join(dest_dir, dt.strftime("%Y"), dt.strftime("%Y-%m-%d"))
                os.makedirs(sub, exist_ok=True)
                dest = os.path.join(sub, name)
            else:
                dest = os.path.join(dest_dir, name)
            # Avoid overwrite: unique suffix
            if os.path.exists(dest):
                stem, ext = os.path.splitext(dest)
                n = 1
                while os.path.exists(f"{stem}_{n}{ext}"):
                    n += 1
                dest = f"{stem}_{n}{ext}"
            if mode == "move":
                shutil.move(src, dest)
            else:
                shutil.copy2(src, dest)
            stats["ok"] += 1
            stats["paths"].append(dest)
            seq += 1
        except Exception:
            stats["failed"] += 1
    return stats
