"""app_paths.py — resolve bundled resources for source runs and frozen executables.

When packaged with PyInstaller (or similar), data files listed in the spec
(e.g. docs/, plugin/) are extracted under sys._MEIPASS. This helper finds
them whether the app is running from source or from a one-file/one-folder
build.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional
import logging

log = logging.getLogger(__name__)


def app_root() -> str:
    """Directory that contains main.py (source) or the frozen bundle root."""
    if getattr(sys, "frozen", False):
        # PyInstaller one-folder: exe dir; one-file: _MEIPASS
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return meipass
        return os.path.dirname(os.path.abspath(sys.executable))
    # Source: this file lives next to main.py
    return os.path.dirname(os.path.abspath(__file__))


def resource_path(*parts: str) -> str:
    """Join path segments under the app root."""
    return os.path.join(app_root(), *parts)


def docs_dir() -> str:
    """User / developer manuals directory."""
    candidates = [
        resource_path("docs"),
        resource_path(),  # manuals may sit next to the binary
        os.path.join(os.getcwd(), "docs"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    return candidates[0]


def plugin_dir() -> str:
    """JSON / Lightroom XMP preset folder shipped with the app."""
    candidates = [
        resource_path("plugin"),
        os.path.join(os.getcwd(), "plugin"),
        os.path.join(os.path.expanduser("~"), ".photolab", "plugin"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    # Ensure a writable user fallback exists when the bundle is read-only
    user = candidates[-1]
    try:
        os.makedirs(user, exist_ok=True)
    except Exception:
        log.debug("plugin_dir: non-critical failure, continuing", exc_info=True)
    return user if os.path.isdir(user) else candidates[0]


def ensure_plugin_dir() -> str:
    """Create plugin dir if missing (source tree or user fallback)."""
    d = plugin_dir()
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        log.debug("ensure_plugin_dir: non-critical failure, continuing", exc_info=True)
    return d


def list_bundled_presets() -> List[str]:
    """All .json / .xmp files in the plugin folder (non-recursive)."""
    d = plugin_dir()
    out: List[str] = []
    if not os.path.isdir(d):
        return out
    for name in sorted(os.listdir(d)):
        if name.lower().endswith((".json", ".xmp")):
            out.append(os.path.join(d, name))
    return out


def manual_file(name: str) -> Optional[str]:
    """Path to a manual markdown file if it exists."""
    for base in (docs_dir(), resource_path(), os.getcwd()):
        p = os.path.join(base, name)
        if os.path.isfile(p):
            return p
        p2 = os.path.join(base, "docs", name)
        if os.path.isfile(p2):
            return p2
    return None


def _looks_like_lensfun_db(path: str) -> bool:
    """True if path is a Lensfun database root (XML profiles or version_N/)."""
    if not path or not os.path.isdir(path):
        return False
    try:
        names = os.listdir(path)
    except Exception:
        return False
    lower = {n.lower() for n in names}
    if any(n.startswith("version_") for n in lower):
        return True
    if any(n.endswith(".xml") for n in lower):
        return True
    # Nested version_1 with xml
    for n in names:
        sub = os.path.join(path, n)
        if os.path.isdir(sub) and n.lower().startswith("version_"):
            try:
                if any(f.lower().endswith(".xml") for f in os.listdir(sub)):
                    return True
            except Exception:
                log.debug("_looks_like_lensfun_db: non-critical failure, continuing", exc_info=True)
    return False


def lensfun_db_paths() -> List[str]:
    """Candidate Lensfun database directories next to the app / cwd / user data.

    Order matters: first existing, valid path wins for callers that take one path.
    Typical layouts:
      <app>/lensfun/
      <app>/lensfun/data/db/
      <app>/lensfun/version_1/
      ~/.photolab/lensfun/
    """
    roots = [
        app_root(),
        os.getcwd(),
        os.path.dirname(app_root()),
        os.path.join(os.path.expanduser("~"), ".photolab"),
    ]
    # Deduplicate while preserving order
    seen = set()
    uniq_roots = []
    for r in roots:
        r = os.path.abspath(r)
        if r not in seen:
            seen.add(r)
            uniq_roots.append(r)

    subpaths = [
        ("lensfun",),
        ("lensfun", "data", "db"),
        ("lensfun", "db"),
        ("lensfun", "version_1"),
        ("data", "db"),
        ("data", "lensfun"),
    ]
    out: List[str] = []
    seen_paths = set()
    for root in uniq_roots:
        for parts in subpaths:
            p = os.path.join(root, *parts)
            if p in seen_paths:
                continue
            seen_paths.add(p)
            if _looks_like_lensfun_db(p):
                out.append(p)
            # If version_1 is itself the xml folder, parent is the db root lensfunpy prefers
            parent = os.path.dirname(p)
            if (
                parts[-1] == "version_1"
                and _looks_like_lensfun_db(p)
                and parent not in seen_paths
            ):
                seen_paths.add(parent)
                if _looks_like_lensfun_db(parent) or os.path.isdir(parent):
                    out.append(parent)
    return out


def primary_lensfun_db() -> Optional[str]:
    """First discoverable Lensfun DB path, or None."""
    paths = lensfun_db_paths()
    return paths[0] if paths else None


def user_data_dir() -> str:
    """Per-user PhotoLab data dir (catalog DB, logs, cache). Created if missing."""
    d = os.path.join(os.path.expanduser("~"), ".photolab")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        log.debug("user_data_dir: non-critical failure, continuing", exc_info=True)
    return d


def log_dir() -> str:
    """Directory for PhotoLab's rotating log files. Created if missing."""
    d = os.path.join(user_data_dir(), "logs")
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        log.debug("log_dir: non-critical failure, continuing", exc_info=True)
    return d
