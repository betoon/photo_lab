"""PhotoLab adapter for the official DDColor tiny colorization model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "tools" / "ddcolor"
WEIGHTS = ROOT / "models" / "ddcolor_paper_tiny" / "pytorch_model.bin"


def _apply_conservative_color(image, fidelity: float, candidate: int):
    import cv2
    import numpy as np

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    variants = (0.72, 0.90, 1.06, 1.20)
    variant = variants[max(0, min(candidate - 1, len(variants) - 1))]
    # Higher fidelity deliberately pulls chroma toward neutral while retaining
    # DDColor's luminance-preserving output.
    historical_strength = 1.0 - max(0.0, min(1.0, fidelity)) * 0.30
    lab[..., 1:] = 128.0 + (lab[..., 1:] - 128.0) * variant * historical_strength
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def run(input_path: Path, output_path: Path, fidelity: float, candidate: int) -> None:
    if not SOURCE.is_dir() or not WEIGHTS.is_file():
        raise RuntimeError("DDColor is not installed. Run install_ddcolor.ps1 first.")
    sys.path.insert(0, str(SOURCE))
    import cv2
    import torch
    from ddcolor import ColorizationPipeline, DDColor, build_ddcolor_model

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    image = cv2.imread(str(input_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read input image: {input_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_ddcolor_model(
        DDColor, model_path=str(WEIGHTS), input_size=512,
        model_size="tiny", device=device,
    )
    result = ColorizationPipeline(model, input_size=512, device=device).process(image)
    result = _apply_conservative_color(result, fidelity, candidate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), result):
        raise RuntimeError("Could not write the DDColor result")
    print(f"DDColor completed locally on {device}; candidate {candidate}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--operation", required=True, choices=("colorize",))
    parser.add_argument("--fidelity", type=float, default=0.7)
    parser.add_argument("--candidate", type=int, default=1)
    args = parser.parse_args()
    try:
        run(args.input, args.output, args.fidelity, args.candidate)
        return 0
    except Exception as exc:
        print(f"PhotoLab DDColor provider error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
