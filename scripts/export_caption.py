#!/usr/bin/env python3
"""Example PhotoLab script: write a simple caption sidecar next to the image."""
from __future__ import annotations

import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=os.environ.get("PHOTOLAB_IMAGE", ""))
    ap.add_argument("--recipe", default=os.environ.get("PHOTOLAB_RECIPE_JSON", ""))
    args = ap.parse_args()
    path = args.path
    if not path or not os.path.isfile(path):
        print("No image path", file=sys.stderr)
        sys.exit(1)
    exposure = 0.0
    if args.recipe and os.path.isfile(args.recipe):
        with open(args.recipe, "r", encoding="utf-8") as f:
            data = json.load(f)
        exposure = float(data.get("exposure", 0) or 0)
    out = path + ".caption.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(f"file: {os.path.basename(path)}\n")
        f.write(f"exposure: {exposure:+.2f} EV\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
