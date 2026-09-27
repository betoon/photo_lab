import os
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication
import main_window
import app_paths


def test_nested_folders_favorites_and_numeric_sort(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    root = tmp_path / "plugin"
    for folder, names in {"HDR / Portrait": [], "HDR/Portrait": ["Look 10.xmp", "Look 2.xmp"], "Other/Portrait": ["Other.xmp"]}.items():
        for name in names:
            target = root / folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("<x/>")
    (root / "Odd.xmp").write_text("<x/>")
    (root / "preset-folders.tsv").write_text("Old.xmp\tHDR/Portrait/Look 2.xmp\n")
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    settings.setValue("preset_favorites", os.path.normcase(str(root / "Old.xmp")))
    monkeypatch.setattr(main_window, "QSettings", lambda *args: settings)
    dialog = main_window.PresetBrowserDialog(None, str(root))
    assert dialog.category_combo.currentData() == "Uncategorized"
    assert dialog.list.count() == 1
    dialog.category_combo.setCurrentIndex(dialog.category_combo.findData("HDR / Portrait"))
    assert [dialog.list.item(i).text().lstrip("★ ") for i in range(dialog.list.count())] == ["Look 2.xmp", "Look 10.xmp"]
    dialog.reload()
    assert dialog.category_combo.currentData() == "HDR / Portrait"
    dialog.category_combo.setCurrentIndex(dialog.category_combo.findData("favorites"))
    assert dialog.list.count() == 1
    assert "Look 2" in dialog.list.item(0).text()
    monkeypatch.setattr(app_paths, "plugin_dir", lambda: str(root))
    assert len(app_paths.list_bundled_presets()) == 4
    dialog.close()
