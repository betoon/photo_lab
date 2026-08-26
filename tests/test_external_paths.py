import os
from pathlib import Path

import external_paths
from config import PhotoLabConfig


def test_config_round_trip_preserves_external_paths(tmp_path, monkeypatch):
    monkeypatch.setattr("config.user_config_dir", lambda: str(tmp_path))
    monkeypatch.setattr("config.app_ini_path", lambda: str(tmp_path / "missing-app.ini"))
    cfg = PhotoLabConfig()
    cfg.set("paths", "focus_stacker_pro", str(tmp_path / "focus.exe"))
    cfg.set("paths", "argyllcms_dir", str(tmp_path / "argyll"))
    cfg.save_user()
    loaded = PhotoLabConfig()
    assert loaded.path("focus_stacker_pro") == str(tmp_path / "focus.exe")
    assert loaded.path("argyllcms_dir") == str(tmp_path / "argyll")


def test_focus_stacker_folder_resolves_python_launcher(tmp_path, monkeypatch):
    launcher = tmp_path / "run.py"
    launcher.write_text("# launcher", encoding="utf-8")
    monkeypatch.setattr(external_paths, "configured_path", lambda key: str(tmp_path))
    assert external_paths.resolve_path("focus_stacker_pro") == str(launcher)
    command, cwd = external_paths.focus_stacker_command()
    assert command[-1] == str(launcher)
    assert cwd == str(tmp_path)


def test_invalid_configured_path_falls_back_without_crashing(tmp_path, monkeypatch):
    fallback = tmp_path / "plugin"
    fallback.mkdir()
    monkeypatch.setattr(external_paths, "configured_path", lambda key: str(tmp_path / "missing"))
    spec = external_paths._BY_KEY["plugin_dir"]
    monkeypatch.setitem(
        external_paths._BY_KEY,
        "plugin_dir",
        external_paths.PathSpec(spec.key, spec.label, spec.kind, spec.description, lambda: [str(fallback)]),
    )
    assert external_paths.resolve_path("plugin_dir") == str(fallback)


def test_argyll_validation_reports_missing_tools(tmp_path):
    ok, message = external_paths.validate_path("argyllcms_dir", str(tmp_path))
    assert not ok
    assert "dispcal" in message
