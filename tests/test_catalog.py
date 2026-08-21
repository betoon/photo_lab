"""tests/test_catalog.py — unit tests for catalog.Catalog.

Uses a fresh temp-file sqlite DB per test (catalog.py uses
check_same_thread=False and opens its own connection, so an in-memory
":memory:" DB would need extra wiring to share across helpers — a temp
file is simpler and still fast). Records are built by hand rather than
via build_record()/scan_folder(), since those need real image files on
disk and EXIF extraction; upsert_image() takes a plain dict, so the
catalog layer can be tested independently of imaging/EXIF.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catalog import Catalog  # noqa: E402


def _make_record(path, **overrides):
    record = {
        "path": path,
        "folder": str(Path(path).parent),
        "filename": Path(path).name,
        "file_mtime": 1_700_000_000.0,
        "file_size": 12345,
        "exif_datetime": "2024:03:15 10:30:00",
        "date_key": "2024-03-15",
        "year": 2024,
        "month": 3,
        "day": 15,
        "camera": "TestCam",
        "lens": "TestLens 24-70mm",
        "iso": 100,
        "aperture": 4.0,
        "shutter": "1/200",
        "focal": 50.0,
        "is_raw": 0,
        "width": 4000,
        "height": 3000,
        "last_scan_ts": 1_700_000_100.0,
        "date_source": "exif",
        "content_hash": "abc123",
    }
    record.update(overrides)
    return record


def _catalog(tmp_path) -> Catalog:
    return Catalog(db_path=str(tmp_path / "catalog.db"))


def test_new_catalog_creates_schema_and_starts_empty(tmp_path):
    cat = _catalog(tmp_path)
    try:
        assert cat.count() == 0
        assert cat.list_roots() == []
    finally:
        cat.close()


def test_opening_existing_db_again_does_not_fail_migration(tmp_path):
    """_migrate() runs ALTER TABLE for columns that may already exist —
    opening the same DB twice should be a no-op, not an error."""
    db_path = str(tmp_path / "catalog.db")
    cat1 = Catalog(db_path=db_path)
    cat1.upsert_image(_make_record("/photos/a.jpg"))
    cat1.commit()
    cat1.close()

    cat2 = Catalog(db_path=db_path)
    try:
        assert cat2.count() == 1
    finally:
        cat2.close()


def test_upsert_and_get_image(tmp_path):
    cat = _catalog(tmp_path)
    try:
        cat.upsert_image(_make_record("/photos/a.jpg", camera="Nikon Z6"))
        cat.commit()

        row = cat.get_image("/photos/a.jpg")
        assert row is not None
        assert row["camera"] == "Nikon Z6"
        assert row["rating"] == 0
        assert row["reject"] == 0

        assert cat.get_image("/photos/missing.jpg") is None
    finally:
        cat.close()


def test_upsert_preserves_rating_reject_keywords_on_rescan(tmp_path):
    """A rescan (e.g. after re-editing the file) re-upserts EXIF fields but
    must not clobber user edits like rating/keywords — upsert_image reads
    the existing row first specifically to preserve these."""
    cat = _catalog(tmp_path)
    try:
        cat.upsert_image(_make_record("/photos/a.jpg"))
        cat.commit()
        cat.set_rating("/photos/a.jpg", 4)
        cat.set_keywords("/photos/a.jpg", "sunset, beach")

        # Simulate a rescan with a changed file_size (e.g. re-exported)
        cat.upsert_image(_make_record("/photos/a.jpg", file_size=99999))
        cat.commit()

        row = cat.get_image("/photos/a.jpg")
        assert row["file_size"] == 99999  # EXIF/file fields updated
        assert row["rating"] == 4          # user rating preserved
        assert row["keywords"] == "sunset, beach"  # user keywords preserved
    finally:
        cat.close()


def test_set_rating_clamps_to_valid_range(tmp_path):
    cat = _catalog(tmp_path)
    try:
        cat.upsert_image(_make_record("/photos/a.jpg"))
        cat.commit()

        cat.set_rating("/photos/a.jpg", 99)
        assert cat.get_image("/photos/a.jpg")["rating"] == 5

        cat.set_rating("/photos/a.jpg", -5)
        assert cat.get_image("/photos/a.jpg")["rating"] == 0
    finally:
        cat.close()


def test_set_reject_and_count_include_exclude(tmp_path):
    cat = _catalog(tmp_path)
    try:
        cat.upsert_image(_make_record("/photos/a.jpg"))
        cat.upsert_image(_make_record("/photos/b.jpg"))
        cat.commit()
        cat.set_reject("/photos/b.jpg", True)

        assert cat.count(include_rejected=True) == 2
        assert cat.count(include_rejected=False) == 1
    finally:
        cat.close()


def test_remove_image(tmp_path):
    cat = _catalog(tmp_path)
    try:
        cat.upsert_image(_make_record("/photos/a.jpg"))
        cat.commit()
        assert cat.count() == 1

        cat.remove_image("/photos/a.jpg")
        assert cat.count() == 0
        assert cat.get_image("/photos/a.jpg") is None
    finally:
        cat.close()


def test_date_tree_groups_by_year_month_day(tmp_path):
    cat = _catalog(tmp_path)
    try:
        cat.upsert_image(_make_record(
            "/photos/a.jpg", year=2024, month=3, day=15, date_key="2024-03-15",
        ))
        cat.upsert_image(_make_record(
            "/photos/b.jpg", year=2024, month=3, day=15, date_key="2024-03-15",
        ))
        cat.upsert_image(_make_record(
            "/photos/c.jpg", year=2024, month=1, day=1, date_key="2024-01-01",
        ))
        cat.commit()

        tree = cat.date_tree()
        assert len(tree) == 1  # one year
        assert tree[0]["year"] == 2024
        assert tree[0]["count"] == 3
        months = {m["month"]: m for m in tree[0]["months"]}
        assert months[3]["count"] == 2
        assert months[1]["count"] == 1
    finally:
        cat.close()


def test_images_for_date_filters_by_rating_and_rejection(tmp_path):
    cat = _catalog(tmp_path)
    try:
        cat.upsert_image(_make_record("/photos/a.jpg", date_key="2024-03-15"))
        cat.upsert_image(_make_record("/photos/b.jpg", date_key="2024-03-15"))
        cat.commit()
        cat.set_rating("/photos/a.jpg", 5)
        cat.set_reject("/photos/b.jpg", True)

        all_for_date = cat.images_for_date(date_key="2024-03-15", include_rejected=True)
        assert len(all_for_date) == 2

        not_rejected = cat.images_for_date(date_key="2024-03-15")
        assert [r["path"] for r in not_rejected] == ["/photos/a.jpg"]

        highly_rated = cat.images_for_date(date_key="2024-03-15", min_rating=5, include_rejected=True)
        assert [r["path"] for r in highly_rated] == ["/photos/a.jpg"]
    finally:
        cat.close()


def test_search_matches_filename_keywords_and_camera(tmp_path):
    cat = _catalog(tmp_path)
    try:
        cat.upsert_image(_make_record("/photos/sunset.jpg", camera="Fujifilm X-T5"))
        cat.upsert_image(_make_record("/photos/portrait.jpg", camera="Nikon Z6"))
        cat.commit()
        cat.set_keywords("/photos/portrait.jpg", "family, birthday")

        assert [r["path"] for r in cat.search("sunset")] == ["/photos/sunset.jpg"]
        assert [r["path"] for r in cat.search("Fujifilm")] == ["/photos/sunset.jpg"]
        assert [r["path"] for r in cat.search("birthday")] == ["/photos/portrait.jpg"]
        assert cat.search("nonexistent-term") == []
        # Empty query behaves like images_for_date(): everything not rejected
        assert len(cat.search("")) == 2
    finally:
        cat.close()


def test_collections_create_add_remove(tmp_path):
    cat = _catalog(tmp_path)
    try:
        cat.upsert_image(_make_record("/photos/a.jpg"))
        cat.upsert_image(_make_record("/photos/b.jpg"))
        cat.commit()

        cid = cat.create_collection("Favorites")
        assert isinstance(cid, int)

        cat.add_to_collection(cid, ["/photos/a.jpg", "/photos/b.jpg"])
        members = cat.images_in_collection(cid, include_rejected=True)
        assert {m["path"] for m in members} == {"/photos/a.jpg", "/photos/b.jpg"}

        cat.remove_from_collection(cid, ["/photos/a.jpg"])
        members = cat.images_in_collection(cid, include_rejected=True)
        assert {m["path"] for m in members} == {"/photos/b.jpg"}

        names = {c["name"] for c in cat.list_collections()}
        assert "Favorites" in names

        cat.delete_collection(cid)
        assert cat.list_collections() == []
    finally:
        cat.close()


def test_virtual_copies_lifecycle(tmp_path):
    cat = _catalog(tmp_path)
    try:
        cat.upsert_image(_make_record("/photos/a.jpg"))
        cat.commit()

        vc_id = cat.create_virtual_copy("/photos/a.jpg", name="B&W version", recipe_json="{}")
        copies = cat.list_virtual_copies(master_path="/photos/a.jpg")
        assert len(copies) == 1
        assert copies[0]["name"] == "B&W version"

        cat.update_virtual_copy_recipe(vc_id, '{"black_and_white": true}')
        vc = cat.get_virtual_copy(vc_id)
        assert vc["recipe_json"] == '{"black_and_white": true}'

        cat.delete_virtual_copy(vc_id)
        assert cat.list_virtual_copies(master_path="/photos/a.jpg") == []
    finally:
        cat.close()


def test_add_and_list_roots(tmp_path):
    cat = _catalog(tmp_path)
    try:
        cat.add_root(str(tmp_path / "PhotoLibrary"))
        roots = cat.list_roots()
        assert len(roots) == 1
        assert roots[0].endswith("PhotoLibrary")
    finally:
        cat.close()
