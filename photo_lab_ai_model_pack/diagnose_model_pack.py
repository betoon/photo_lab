"""Quick readiness report for the PhotoLab local model pack."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "tools" / "realesrgan-ncnn-vulkan" / "realesrgan-ncnn-vulkan.exe"
DDCOLOR_SOURCE = ROOT / "tools" / "ddcolor" / "ddcolor" / "__init__.py"
DDCOLOR_WEIGHTS = ROOT / "models" / "ddcolor_paper_tiny" / "pytorch_model.bin"


def main() -> int:
    manifest = ROOT / "photolab-model-pack.json"
    print("PhotoLab Local AI Model Pack diagnostics")
    print(f"Python: {sys.version.split()[0]}")
    print(f"System: {platform.platform()}")
    print(f"Manifest: {'OK' if manifest.is_file() else 'MISSING'}")
    if manifest.is_file():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            print(f"Pack: {data.get('name')} {data.get('version')}")
        except Exception as exc:
            print(f"Manifest error: {exc}")
            return 2
    print(f"Real-ESRGAN engine: {'OK' if ENGINE.is_file() else 'NOT INSTALLED'}")
    if ENGINE.is_file():
        completed = subprocess.run([str(ENGINE), "-h"], cwd=ENGINE.parent, capture_output=True, text=True)
        combined = (completed.stdout + completed.stderr).lower()
        print(f"Engine launch: {'OK' if 'usage' in combined else 'CHECK REQUIRED'}")
    else:
        print("  Run install_model_pack.ps1 to install Real-ESRGAN.")
    print(f"DDColor source: {'OK' if DDCOLOR_SOURCE.is_file() else 'NOT INSTALLED'}")
    print(f"DDColor checkpoint: {'OK' if DDCOLOR_WEIGHTS.is_file() else 'NOT INSTALLED'}")
    dependencies={name:bool(importlib.util.find_spec(name)) for name in ("torch","torchvision","cv2","numpy")}
    print("DDColor runtime: " + ", ".join(f"{name}={'OK' if ready else 'MISSING'}" for name,ready in dependencies.items()))
    if not (DDCOLOR_SOURCE.is_file() and DDCOLOR_WEIGHTS.is_file()):
        print("  Run install_ddcolor.ps1 to install local colorization.")
    print("PhotoLab setting:")
    print(f"  [paths] ai_restoration_model_pack = {ROOT}")
    return 0 if ENGINE.is_file() and DDCOLOR_SOURCE.is_file() and DDCOLOR_WEIGHTS.is_file() and all(dependencies.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
