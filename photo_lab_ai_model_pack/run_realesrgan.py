"""PhotoLab adapter for the official Real-ESRGAN NCNN/Vulkan executable."""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "tools" / "realesrgan-ncnn-vulkan" / "realesrgan-ncnn-vulkan.exe"
MODELS = ENGINE.parent / "models"


def _finish(source: Path, generated: Path, destination: Path, operation: str, strength: float) -> None:
    try:
        import cv2

        original = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        result = cv2.imread(str(generated), cv2.IMREAD_UNCHANGED)
        if original is None or result is None:
            raise RuntimeError("OpenCV could not read the provider input or output")
        if operation == "enhance":
            result = cv2.resize(result, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)
        else:
            original = cv2.resize(original, (result.shape[1], result.shape[0]), interpolation=cv2.INTER_LANCZOS4)
        finished = cv2.addWeighted(original, 1.0 - strength, result, strength, 0)
        if not cv2.imwrite(str(destination), finished):
            raise RuntimeError("OpenCV could not write the enhanced result")
    except ImportError:
        from PIL import Image

        with Image.open(source) as original, Image.open(generated) as result:
            if operation == "enhance":
                result = result.resize(original.size, Image.Resampling.LANCZOS)
            else:
                original = original.resize(result.size, Image.Resampling.LANCZOS)
            Image.blend(original.convert("RGB"), result.convert("RGB"), strength).save(destination)


def run(input_path: Path, output_path: Path, operation: str, fidelity: float, candidate: int) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if not ENGINE.is_file():
        raise RuntimeError("Real-ESRGAN is not installed. Run install_model_pack.ps1 first.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Candidate variants intentionally change restoration strength without switching
    # to the anime model, which is inappropriate for historic photographs.
    strength = max(0.1, min(1.0, 1.0 - fidelity + candidate * 0.15))
    with tempfile.TemporaryDirectory(prefix="photolab_realesrgan_") as temp_dir:
        temporary = Path(temp_dir) / "result.png"
        command = [
            str(ENGINE), "-i", str(input_path), "-o", str(temporary),
            "-m", str(MODELS), "-n", "realesrgan-x4plus",
            "-s", "4", "-f", "png", "-x", "-j", "1:2:1",
        ]
        completed = subprocess.run(command, cwd=ENGINE.parent, capture_output=True, text=True, check=False)
        if completed.returncode or not temporary.is_file():
            detail = completed.stderr or completed.stdout or "Real-ESRGAN produced no output"
            raise RuntimeError(detail[-3000:])
        _finish(input_path, temporary, output_path, operation, strength)
        print(f"Real-ESRGAN completed {operation}; AI blend strength {strength:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operation", required=True, choices=("enhance", "super_resolution"))
    parser.add_argument("--fidelity", type=float, default=0.7)
    parser.add_argument("--candidate", type=int, default=1)
    args = parser.parse_args()
    try:
        run(args.input, args.output, args.operation, args.fidelity, args.candidate)
        return 0
    except Exception as exc:
        print(f"PhotoLab provider error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
