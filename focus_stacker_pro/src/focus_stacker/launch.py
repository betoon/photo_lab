"""Qt-free command-line handoff parsing for Focus Stacker Pro."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path


LOG = logging.getLogger("focus_stacker")


def parse_launch_args(argv=None):
    parser = argparse.ArgumentParser(description="Focus Stacker Pro")
    parser.add_argument("--image", action="append", default=[], help="Initial source image (repeatable)")
    parser.add_argument("--image-list", help="JSON file containing initial source paths")
    parser.add_argument("--delete-image-list", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--microscope", action="store_true", help="Open the Microscope 2D workspace")
    args = parser.parse_args(argv)
    paths = list(args.image)
    if args.image_list:
        try:
            data = json.loads(Path(args.image_list).read_text(encoding="utf-8"))
            paths.extend(str(path) for path in data if path)
        except Exception as exc:
            LOG.warning("Could not read PhotoLab image handoff: %s", exc)
        finally:
            if args.delete_image_list:
                try:
                    Path(args.image_list).unlink(missing_ok=True)
                except OSError:
                    pass
    args.images = list(dict.fromkeys(paths))
    return args
