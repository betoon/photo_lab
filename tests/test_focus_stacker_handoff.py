import json
import sys
from pathlib import Path


FOCUS_SRC = Path(__file__).resolve().parents[1] / "focus_stacker_pro" / "src"
if str(FOCUS_SRC) not in sys.path:
    sys.path.insert(0, str(FOCUS_SRC))

from focus_stacker.launch import parse_launch_args


def test_launch_args_accept_repeatable_images_and_microscope():
    args = parse_launch_args([
        "--microscope", "--image", "one.tif", "--image", "two.tif",
        "--image", "one.tif",
    ])
    assert args.microscope is True
    assert args.images == ["one.tif", "two.tif"]


def test_json_handoff_preserves_order_and_deletes_file(tmp_path):
    handoff = tmp_path / "handoff.json"
    handoff.write_text(json.dumps(["near.nef", "middle.nef", "far.nef"]), encoding="utf-8")
    args = parse_launch_args([
        "--image-list", str(handoff), "--delete-image-list",
    ])
    assert args.images == ["near.nef", "middle.nef", "far.nef"]
    assert not handoff.exists()
