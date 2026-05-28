"""Run inference on stdin (one record per line) or a single string."""
from __future__ import annotations

import argparse
import json
import sys
import time

from ner.infer.runtime import from_artifact_dir


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifact-dir", required=True,
                   help="Dir containing model.onnx + tokenizer files")
    p.add_argument("--text", default=None,
                   help="Single record to score; reads stdin if omitted")
    p.add_argument("--report-latency", action="store_true")
    args = p.parse_args()

    runtime = from_artifact_dir(args.artifact_dir)

    if args.text is not None:
        records = [args.text]
    else:
        records = [line.rstrip("\n") for line in sys.stdin if line.strip()]

    for text in records:
        t0 = time.perf_counter()
        spans = runtime.predict(text)
        dt_ms = (time.perf_counter() - t0) * 1000
        out = {"text": text, "entities": [e.to_dict() for e in spans]}
        if args.report_latency:
            out["latency_ms"] = round(dt_ms, 2)
        print(json.dumps(out))


if __name__ == "__main__":
    main()
