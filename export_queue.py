"""Persistent export job queue stored under ~/.photolab/export_queue.json."""

from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, List, Optional


def _queue_path() -> str:
    root = os.path.join(os.path.expanduser("~"), ".photolab")
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, "export_queue.json")


def load_queue() -> List[Dict[str, Any]]:
    path = _queue_path()
    if not os.path.isfile(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        jobs = data.get("jobs") if isinstance(data, dict) else data
        return list(jobs) if isinstance(jobs, list) else []
    except Exception:
        return []


def save_queue(jobs: List[Dict[str, Any]]) -> None:
    path = _queue_path()
    payload = {"version": 1, "updated": time.time(), "jobs": jobs}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def add_job(
    src_path: str,
    out_path: str,
    recipe_dict: Optional[dict] = None,
    max_dim: int = 0,
    jpeg_quality: int = 92,
    watermark_text: str = "",
) -> Dict[str, Any]:
    jobs = load_queue()
    job = {
        "id": str(uuid.uuid4())[:8],
        "src": os.path.abspath(src_path),
        "out": os.path.abspath(out_path),
        "recipe": recipe_dict or {},
        "max_dim": int(max_dim or 0),
        "jpeg_quality": int(jpeg_quality or 92),
        "watermark_text": watermark_text or "",
        "status": "pending",
        "created": time.time(),
        "error": "",
    }
    jobs.append(job)
    save_queue(jobs)
    return job


def update_job(job_id: str, **fields) -> None:
    jobs = load_queue()
    for j in jobs:
        if j.get("id") == job_id:
            j.update(fields)
            break
    save_queue(jobs)


def remove_job(job_id: str) -> None:
    jobs = [j for j in load_queue() if j.get("id") != job_id]
    save_queue(jobs)


def clear_completed() -> int:
    jobs = load_queue()
    keep = [j for j in jobs if j.get("status") not in ("done", "failed")]
    n = len(jobs) - len(keep)
    save_queue(keep)
    return n


def pending_jobs() -> List[Dict[str, Any]]:
    return [j for j in load_queue() if j.get("status") == "pending"]
