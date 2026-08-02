"""Sweep per-label confidence thresholds against the gold validation set and
write `thresholds.json` that the runtime auto-loads at startup.

Run after `scripts.train` + `scripts.export_onnx` produce an artifact dir
containing `model.onnx` and tokenizer files.

Examples:
    # Default: maximize micro F1.
    python -m scripts.tune_threshold \\
        --artifact-dir artifacts/serve \\
        --gold-jsonl data/gold_dev.jsonl

    # High-precision deployment (automated invoice processing):
    python -m scripts.tune_threshold \\
        --artifact-dir artifacts/serve \\
        --gold-jsonl data/gold_dev.jsonl \\
        --objective max_f1_at_precision_floor --precision-floor 0.95

    # High-recall deployment (analyst review queue):
    python -m scripts.tune_threshold \\
        --artifact-dir artifacts/serve \\
        --gold-jsonl data/gold_dev.jsonl \\
        --objective max_f1_at_recall_floor --recall-floor 0.90

    # Optimize one entity type only:
    python -m scripts.tune_threshold \\
        --artifact-dir artifacts/serve \\
        --gold-jsonl data/gold_dev.jsonl \\
        --objective f1_per_type:COMMODITY(NEG)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ner.eval.gold import GOLD_SPLITS, load_gold, split_gold
from ner.eval.threshold_sweep import (
    cache_probabilities,
    sweep,
    write_thresholds_json,
)


def _parse_type_floor(spec: str) -> tuple[str, float]:
    """Parse a BUCKET:FLOOR spec like 'COMMODITY(NEG):0.9'."""
    bucket, sep, floor = spec.rpartition(":")
    if not sep or not bucket:
        raise argparse.ArgumentTypeError(
            f"expected BUCKET:FLOOR (e.g. 'COMMODITY(NEG):0.9'), got {spec!r}"
        )
    try:
        return bucket, float(floor)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"bad floor in {spec!r}") from exc


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--artifact-dir", required=True,
                   help="Dir containing model.onnx + tokenizer files")
    p.add_argument("--gold-jsonl", default=None,
                   help="Path to gold JSONL; defaults to ner.eval.gold.GOLD_SEED")
    p.add_argument("--objective", default="f1_micro",
                   help="f1_micro | f1_macro | f1_per_type:<bucket> | "
                        "max_f1_at_precision_floor | max_f1_at_recall_floor")
    p.add_argument("--precision-floor", type=float, default=0.0)
    p.add_argument("--recall-floor", type=float, default=0.0)
    p.add_argument("--precision-floor-type", type=_parse_type_floor,
                   action="append", default=[], metavar="BUCKET:FLOOR",
                   help="Repeatable per-bucket precision constraint, e.g. "
                        "'COMMODITY(NEG):0.9'. Any objective becomes "
                        "infeasible unless every named bucket meets its floor.")
    p.add_argument("--eval-split", choices=GOLD_SPLITS, default="all",
                   help="Deterministic gold split to tune on. Use 'tune' here "
                        "and 'earlystop' for training so thresholds are not "
                        "tuned on the model-selection records (default: all).")
    p.add_argument("--step", type=float, default=0.01,
                   help="Threshold grid step size (default 0.01)")
    p.add_argument("--output", default=None,
                   help="Path to write thresholds.json. Defaults to "
                        "<artifact-dir>/thresholds.json")
    args = p.parse_args()

    artifact_dir = Path(args.artifact_dir)
    output = Path(args.output) if args.output else artifact_dir / "thresholds.json"

    # Lazy import to keep --help fast and importable without onnxruntime.
    from ner.infer.runtime import from_artifact_dir
    runtime = from_artifact_dir(artifact_dir)

    gold = split_gold(load_gold(args.gold_jsonl), args.eval_split)
    if not gold:
        print("ERROR: empty gold set", file=sys.stderr)
        return 2

    print(f"Caching probabilities for {len(gold)} gold records "
          f"(split={args.eval_split})...")
    cache = cache_probabilities(runtime, gold)

    # Evaluate against the preprocessed gold stored in the cache — the same
    # cleaned coordinates the cached probabilities were computed in.
    print(f"Running sweep with objective {args.objective!r} (step={args.step})...")
    result = sweep(
        cache, cache.gold, args.objective,
        step=args.step,
        precision_floor=args.precision_floor,
        recall_floor=args.recall_floor,
        per_type_precision_floors=dict(args.precision_floor_type) or None,
    )

    # Pretty-print the chosen operating point.
    print()
    print(f"Objective:  {result.objective}  (feasible={result.feasible})")
    print(f"Micro:      P={result.report.micro.precision:.4f} "
          f"R={result.report.micro.recall:.4f} "
          f"F1={result.report.micro.f1:.4f}")
    print("Per-bucket F1:")
    for bucket, m in result.report.per_type.items():
        marker = "" if m.support > 0 else "  (no support)"
        print(f"  {bucket:24s}  P={m.precision:.4f} R={m.recall:.4f} "
              f"F1={m.f1:.4f}  support={m.support}{marker}")
    print("Thresholds:")
    print(json.dumps(result.thresholds, indent=2))
    if result.notes:
        print("Notes:")
        for n in result.notes:
            print(f"  - {n}")

    if not result.feasible:
        print(f"\nERROR: objective {args.objective!r} is infeasible at any threshold. "
              "Not writing thresholds.json.", file=sys.stderr)
        return 1

    written = write_thresholds_json(
        result, output,
        gold_set_path=args.gold_jsonl,
        gold_records=len(gold),
    )
    print(f"\nWrote {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
