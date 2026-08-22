"""Unit tests: Catalog upsert, migrate, and basic queries."""
from __future__ import annotations

import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from catalog import Catalog


@pytest.fixture
def catalog(tmp_path):
    db = tmp_path / "test_catalog.db"
    cat = Catalog(db_path=str(db))
    yield cat
    cat.close()


def _sample_record(path: str, **extra):
    rec = {
        "path": path,
        "folder": os.path.dirname(path),
        "filename": os.path.basename(path),
        "file_mtime": time.time(),
        "file_size": 12345,
        "exif_datetime": "2024-01-15T12:00:00",
        "date_key": "2024-01-15",
        "year": 2024,
        "month": 1,
        "day": 15,
        "camera": "TestCam",
        "lens": "50mm",
        "iso": "100",
        "aperture": "f/2.8",
        "shutter": "1/125",
        "focal": "50mm",
        "is_raw": 0,
        "width": 100,
        "height": 80,
        "last_scan_ts": time.time(),
        "date_source": "exif",
    }
    rec.update(extra)
    return rec


def test_upsert_insert_and_update(catalog):
    path = "/tmp/fake/photo_a.jpg"
    catalog.upsert_image(_sample_record(path, rating=0))
    catalog.commit()
    row = catalog.connection().execute(
        "SELECT * FROM images WHERE path = ?", (path,)
    ).fetchone()
    assert row is not None
    assert row["filename"] == "photo_a.jpg"
    assert row["camera"] == "TestCam"
    assert row["rating"] == 0

    # Update metadata; rating/reject preserved
    catalog.connection().execute(
        "UPDATE images SET rating = 4, reject = 0 WHERE path = ?", (path,)
    )
    catalog.commit()
    catalog.upsert_image(_sample_record(path, camera="UpdatedCam"))
    catalog.commit()
    row = catalog.connection().execute(
        "SELECT camera, rating FROM images WHERE path = ?", (path,)
    ).fetchone()
    assert row["camera"] == "UpdatedCam"
    assert row["rating"] == 4


def test_migrate_adds_keywords_column(tmp_path):
    """Open catalog twice: migration is idempotent and keywords column exists."""
    db = str(tmp_path / "migrate.db")
    cat = Catalog(db_path=db)
    cols = {r[1] for r in cat.connection().execute("PRAGMA table_info(images)")}
    assert "keywords" in cols
    cat.close()
    cat2 = Catalog(db_path=db)
    cols2 = {r[1] for r in cat2.connection().execute("PRAGMA table_info(images)")}
    assert "keywords" in cols2
    cat2.close()


def test_roots(catalog):
    catalog.add_root("/tmp/photos")
    roots = catalog.list_roots()
    assert any(r.endswith("photos") or r == "/tmp/photos" for r in roots)


def test_fingerprint(catalog):
    path = "/tmp/fake/fp.jpg"
    catalog.upsert_image(_sample_record(path, file_mtime=1000.0, file_size=50))
    catalog.commit()
    fp = catalog.get_file_fingerprint(path)
    assert fp is not None
    assert fp[1] == 50
    assert catalog.get_file_fingerprint("/no/such") is None
