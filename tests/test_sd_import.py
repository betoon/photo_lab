from __future__ import annotations

from workers import SdImportWorker


def test_sd_import_copies_media_skips_duplicates_and_renames_collisions(tmp_path):
    source = tmp_path / "card"
    destination = tmp_path / "photos"
    (source / "DCIM" / "100NIKON").mkdir(parents=True)
    destination.mkdir()
    (source / "DCIM" / "100NIKON" / "one.NEF").write_bytes(b"raw-one")
    (source / "DCIM" / "100NIKON" / "two.JPG").write_bytes(b"jpeg-two")
    (source / "notes.txt").write_text("ignore me", encoding="utf-8")
    (destination / "one.NEF").write_bytes(b"raw-one")
    (destination / "two.JPG").write_bytes(b"different")

    summaries = []
    worker = SdImportWorker(str(source), str(destination), preserve_folders=False)
    worker.completed.connect(summaries.append)
    worker.run()

    summary = summaries[0]
    assert summary["found"] == 2
    assert summary["copied"] == 1
    assert summary["skipped"] == 1
    assert summary["renamed"] == 1
    assert (destination / "two_1.JPG").read_bytes() == b"jpeg-two"
    assert not list(destination.glob("*.photolab-part"))


def test_sd_import_can_preserve_card_folders(tmp_path):
    source = tmp_path / "card"
    destination = tmp_path / "photos"
    nested = source / "DCIM" / "100NIKON"
    nested.mkdir(parents=True)
    (nested / "frame.NEF").write_bytes(b"raw")

    summaries = []
    worker = SdImportWorker(str(source), str(destination), preserve_folders=True)
    worker.completed.connect(summaries.append)
    worker.run()

    assert summaries[0]["copied"] == 1
    assert (destination / "DCIM" / "100NIKON" / "frame.NEF").read_bytes() == b"raw"
