"""Central discovery and validation for optional local PhotoLab integrations."""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from typing import Callable, Iterable

from config import get_config


@dataclass(frozen=True)
class PathSpec:
    key: str
    label: str
    kind: str
    description: str
    discover: Callable[[], Iterable[str]]


def _app_root() -> str:
    from app_paths import app_root
    return app_root()


def _plugin_candidates():
    yield os.path.join(_app_root(), "plugin")
    yield os.path.join(os.path.expanduser("~"), ".photolab", "plugin")


def _argyll_candidates():
    for name in ("dispcal", "dispcal.exe"):
        found = shutil.which(name)
        if found:
            yield os.path.dirname(found)
    yield os.path.join(_app_root(), "Argyll_V3.5.0", "bin")
    if os.name == "nt":
        yield r"C:\Program Files\ArgyllCMS\bin"
        yield os.path.join(os.path.expanduser("~"), "Argyll", "bin")


def _lensfun_candidates():
    from app_paths import lensfun_db_paths
    yield from lensfun_db_paths(include_config=False)


def _focus_candidates():
    root = _app_root()
    yield os.path.join(root, "run_focus_stacker_pro.py")
    yield os.path.join(root, "focus_stacker_pro", "run.py")
    for name in ("focus-stacker-pro", "focus-stacker-pro.exe"):
        found = shutil.which(name)
        if found:
            yield found


PATH_SPECS = (
    PathSpec("plugin_dir", "Plugin / presets folder", "folder",
             "JSON and XMP presets available to PhotoLab.", _plugin_candidates),
    PathSpec("argyllcms_dir", "ArgyllCMS bin folder", "folder",
             "Folder containing dispcal, dispwin, scanin, and colprof.", _argyll_candidates),
    PathSpec("lensfun_dir", "Lensfun database folder", "folder",
             "Database root containing XML profiles or version_N folders.", _lensfun_candidates),
    PathSpec("focus_stacker_pro", "Focus Stacker Pro", "file_or_folder",
             "Application, Python launcher, or its containing folder.", _focus_candidates),
)

_BY_KEY = {spec.key: spec for spec in PATH_SPECS}


def configured_path(key: str) -> str:
    return os.path.expandvars(os.path.expanduser(get_config().path(key).strip()))


def _focus_launcher(path: str) -> str:
    if os.path.isfile(path):
        return path
    if os.path.isdir(path):
        for rel in ("run_focus_stacker_pro.py", "run.py", os.path.join("focus_stacker_pro", "run.py"),
                    "Focus Stacker Pro.exe", "focus-stacker-pro.exe"):
            candidate = os.path.join(path, rel)
            if os.path.isfile(candidate):
                return candidate
    return ""


def validate_path(key: str, path: str) -> tuple[bool, str]:
    path = os.path.abspath(os.path.expandvars(os.path.expanduser(path.strip()))) if path.strip() else ""
    if not path:
        resolved = resolve_path(key)
        return (True, f"Auto-detected: {resolved}") if resolved else (False, "Not configured or detected")
    if key == "argyllcms_dir":
        names = ("dispcal", "dispwin", "scanin", "colprof")
        missing = [n for n in names if not os.path.isfile(os.path.join(path, n + (".exe" if os.name == "nt" else "")))]
        return (not missing, "All required ArgyllCMS tools found" if not missing else "Missing: " + ", ".join(missing))
    if key == "lensfun_dir":
        from app_paths import looks_like_lensfun_db
        return (looks_like_lensfun_db(path), "Lensfun database found" if looks_like_lensfun_db(path) else "No Lensfun XML database found")
    if key == "focus_stacker_pro":
        launcher = _focus_launcher(path)
        return (bool(launcher), f"Launcher found: {launcher}" if launcher else "No application or launcher found")
    ok = os.path.isdir(path)
    return ok, "Folder found" if ok else "Folder does not exist"


def resolve_path(key: str) -> str:
    spec = _BY_KEY[key]
    candidates = [configured_path(key), *spec.discover()]
    seen = set()
    for raw in candidates:
        if not raw:
            continue
        path = os.path.abspath(os.path.expandvars(os.path.expanduser(raw)))
        norm = os.path.normcase(path)
        if norm in seen:
            continue
        seen.add(norm)
        if key == "focus_stacker_pro":
            launcher = _focus_launcher(path)
            if launcher:
                return launcher
        elif validate_path(key, path)[0]:
            return path
    return ""


def focus_stacker_command() -> tuple[list[str], str]:
    """Return launch command and working directory, or ([], '') when unavailable."""
    launcher = resolve_path("focus_stacker_pro")
    if not launcher:
        return [], ""
    if launcher.lower().endswith(".py"):
        return [sys.executable, launcher], os.path.dirname(launcher)
    return [launcher], os.path.dirname(launcher)
