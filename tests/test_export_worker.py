from pathlib import Path

import numpy as np
import pytest

from imaging import Recipe
from workers import ExportWorker


@pytest.mark.parametrize("suffix", [".jpg", ".png", ".tif"])
def test_export_worker_saves_loaded_source_with_progress(tmp_path, suffix):
    source = np.full((24, 32, 3), 128, dtype=np.uint8)
    destination = tmp_path / f"converted{suffix}"
    progress = []
    completed = []
    errors = []
    worker = ExportWorker(
        "camera.raw", Recipe(), str(destination), source_image=source,
        source_meta={"is_raw": True, "wb_baked": True},
    )
    worker.progress.connect(lambda value, message: progress.append((value, message)))
    worker.finished_ok.connect(completed.append)
    worker.failed.connect(errors.append)

    worker.run()

    assert not errors
    assert completed == [str(destination)]
    assert destination.stat().st_size > 0
    assert progress[0][0] == 10
    assert progress[-1][0] == 100


def test_export_worker_reports_encoder_failure(tmp_path, monkeypatch):
    errors = []
    worker = ExportWorker(
        "camera.raw", Recipe(), str(tmp_path / "bad.jpg"),
        source_image=np.zeros((8, 8, 3), dtype=np.uint8),
        source_meta={"is_raw": True, "wb_baked": True},
    )
    monkeypatch.setattr("workers.cv2.imwrite", lambda *args, **kwargs: False)
    worker.failed.connect(errors.append)

    worker.run()

    assert errors
    assert "encoder could not write" in errors[0]
