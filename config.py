"""config.py — PhotoLab user configuration (INI).

Load order (later wins):
  1. Built-in defaults
  2. Optional template next to the app: photolab.ini
  3. User file: ~/.photolab/photolab.ini  (Windows: %USERPROFILE%\\.photolab\\)
  4. Environment variables PHOTOLAB_* (secrets / overrides)

Never commit real API keys or serial numbers. Keep those only in the user file.
"""
from __future__ import annotations

import configparser
import logging
import os
import sys
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

APP_DIR_NAME = ".photolab"
INI_NAME = "photolab.ini"

# Section → key → default (string form for INI)
_DEFAULTS: Dict[str, Dict[str, str]] = {
    "paths": {
        "plugin_dir": "",
        "docs_dir": "",
        "lensfun_dir": "",
        "catalog_db": "",
        "thumb_cache": "",
        "ffmpeg": "",
        "export_default_dir": "",
        "scripts_dir": "",
        "argyllcms_dir": "",
    },
    "performance": {
        "max_raw_workers": "2",
        "proxy_max_dimension": "1600",
        "use_16bit_pipeline": "false",
    },
    "ui": {
        "remember_last_folder": "true",
        "last_folder": "",
        "interface_scale": "1.0",
        "check_for_updates_url": "https://github.com/betoon/photo_lab/releases",
    },
    "licensing": {
        "serial": "",
        "customer_email": "",
    },
    "integrations": {
        "api_key": "",
        "api_endpoint": "",
    },
}

# Environment overrides: PHOTOLAB_API_KEY → integrations.api_key, etc.
_ENV_MAP = {
    "PHOTOLAB_API_KEY": ("integrations", "api_key"),
    "PHOTOLAB_API_ENDPOINT": ("integrations", "api_endpoint"),
    "PHOTOLAB_SERIAL": ("licensing", "serial"),
    "PHOTOLAB_PLUGIN_DIR": ("paths", "plugin_dir"),
    "PHOTOLAB_FFMPEG": ("paths", "ffmpeg"),
    "PHOTOLAB_CATALOG_DB": ("paths", "catalog_db"),
    "PHOTOLAB_LENSFUN_DIR": ("paths", "lensfun_dir"),
}


def user_config_dir() -> str:
    d = os.path.join(os.path.expanduser("~"), APP_DIR_NAME)
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        pass
    return d


def user_ini_path() -> str:
    return os.path.join(user_config_dir(), INI_NAME)


def app_ini_path() -> str:
    """Optional template beside main.py / frozen root."""
    try:
        from app_paths import app_root
        return os.path.join(app_root(), INI_NAME)
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), INI_NAME)


class PhotoLabConfig:
    def __init__(self):
        self._cp = configparser.ConfigParser()
        for section, keys in _DEFAULTS.items():
            self._cp[section] = dict(keys)
        self.reload()

    def reload(self) -> None:
        for section, keys in _DEFAULTS.items():
            if not self._cp.has_section(section):
                self._cp.add_section(section)
            for k, v in keys.items():
                if not self._cp.has_option(section, k):
                    self._cp.set(section, k, v)

        for path in (app_ini_path(), user_ini_path()):
            if path and os.path.isfile(path):
                try:
                    self._cp.read(path, encoding="utf-8")
                    log.debug("Loaded config: %s", path)
                except Exception:
                    log.warning("Failed to read %s", path, exc_info=True)

        for env_key, (section, key) in _ENV_MAP.items():
            val = os.environ.get(env_key)
            if val is not None and val != "":
                if not self._cp.has_section(section):
                    self._cp.add_section(section)
                self._cp.set(section, key, val)

    def get(self, section: str, key: str, fallback: str = "") -> str:
        try:
            return self._cp.get(section, key, fallback=fallback).strip()
        except Exception:
            return fallback

    def get_bool(self, section: str, key: str, fallback: bool = False) -> bool:
        raw = self.get(section, key, "true" if fallback else "false").lower()
        return raw in ("1", "true", "yes", "on")

    def get_int(self, section: str, key: str, fallback: int = 0) -> int:
        try:
            return int(float(self.get(section, key, str(fallback))))
        except Exception:
            return fallback

    def set(self, section: str, key: str, value: Any) -> None:
        if not self._cp.has_section(section):
            self._cp.add_section(section)
        self._cp.set(section, key, "" if value is None else str(value))

    def path(self, key: str) -> str:
        """Non-empty path from [paths] or empty string for 'use default'."""
        return self.get("paths", key, "")

    def save_user(self) -> str:
        """Write current config to the user INI (creates parent dir)."""
        path = user_ini_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("; PhotoLab user configuration\n")
            f.write("; Empty path values mean: use built-in / auto-discovered defaults.\n")
            f.write("; Do not commit API keys or serial numbers to git.\n\n")
            self._cp.write(f)
        return path

    def ensure_user_ini(self) -> str:
        """Create user INI from defaults if missing."""
        path = user_ini_path()
        if not os.path.isfile(path):
            self.save_user()
        return path

    def as_dict(self) -> Dict[str, Dict[str, str]]:
        return {s: dict(self._cp.items(s)) for s in self._cp.sections()}


_config: Optional[PhotoLabConfig] = None


def get_config() -> PhotoLabConfig:
    global _config
    if _config is None:
        _config = PhotoLabConfig()
    return _config


def reload_config() -> PhotoLabConfig:
    global _config
    _config = PhotoLabConfig()
    return _config
