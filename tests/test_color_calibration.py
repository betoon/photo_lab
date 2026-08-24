from pathlib import Path

from color_calibration import (
    build_colprof_args, build_dispcal_args, build_scanin_args, find_argyll_dir,
    validate_icc,
)


def test_dispcal_command_contains_measurement_targets(tmp_path):
    base = str(tmp_path / "display_profile")
    args = build_dispcal_args(base, display=2, whitepoint="D65", luminance=120, gamma="2.2", quality="High")
    assert args[-1] == base
    assert ["-d", "2"] == args[args.index("-d"):args.index("-d") + 2]
    assert ["-t", "6500"] == args[args.index("-t"):args.index("-t") + 2]
    assert ["-b", "120"] == args[args.index("-b"):args.index("-b") + 2]
    assert "-o" in args and base + ".icc" in args


def test_native_dispcal_omits_whitepoint_and_gamma(tmp_path):
    args = build_dispcal_args(str(tmp_path / "native"), whitepoint="Native", gamma="Native")
    assert "-t" not in args
    assert "-g" not in args


def test_camera_profile_commands_are_deterministic(tmp_path):
    base = str(tmp_path / "camera")
    assert build_scanin_args("chart.tif", "layout.cht", "reference.cie", base)[-1] == base
    args = build_colprof_args(base, "Nikon daylight")
    assert args[-1] == base
    assert "Nikon daylight" in args
    assert base + ".icc" in args


def test_argyll_detection_and_invalid_profile(tmp_path, monkeypatch):
    bindir = tmp_path / "bin"; bindir.mkdir()
    (bindir / ("dispcal.exe" if __import__("os").name == "nt" else "dispcal")).write_bytes(b"")
    assert find_argyll_dir(str(bindir)) == str(bindir)
    ok, message = validate_icc(str(tmp_path / "missing.icc"))
    assert not ok and "not found" in message.lower()
