"""Export the trained checkpoint to FP32, graph-optimized ONNX."""
from __future__ import annotations

import argparse

from ner.export.onnx_export import ExportConfig, export


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True,
                   help="HF model dir produced by scripts/train.py")
    p.add_argument("--output", required=True,
                   help="Path to write model.onnx")
    p.add_argument("--opset", type=int, default=17)
    args = p.parse_args()
    out = export(ExportConfig(
        model_dir=args.model_dir, output_path=args.output, opset=args.opset,
    ))
    print(f"Wrote ONNX to {out}")


if __name__ == "__main__":
    main()
