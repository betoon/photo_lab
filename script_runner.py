"""script_runner.py — run user scripts with current path + recipe."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from typing import Optional


def scripts_dir() -> str:
    base = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base, "scripts")


def list_scripts() -> list:
    d = scripts_dir()
    if not os.path.isdir(d):
        return []
    return sorted(
        os.path.join(d, f)
        for f in os.listdir(d)
        if f.endswith(".py") and not f.startswith("_")
    )


def run_script(script_path: str, image_path: str, recipe) -> tuple:
    """Run script; return (returncode, stdout, stderr)."""
    from imaging import recipe_to_dict
    env = os.environ.copy()
    env["PHOTOLAB_IMAGE"] = image_path or ""
    recipe_path = None
    try:
        fd, recipe_path = tempfile.mkstemp(suffix=".json", prefix="photolab_recipe_")
        os.close(fd)
        data = recipe_to_dict(recipe) if recipe is not None else {}
        with open(recipe_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        env["PHOTOLAB_RECIPE_JSON"] = recipe_path
        cmd = [
            sys.executable,
            script_path,
            "--path", image_path or "",
            "--recipe", recipe_path,
        ]
        proc = subprocess.run(
            cmd, capture_output=True, text=True, env=env, timeout=120,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    finally:
        if recipe_path and os.path.isfile(recipe_path):
            try:
                os.remove(recipe_path)
            except OSError:
                pass
