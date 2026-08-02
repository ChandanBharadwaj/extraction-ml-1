"""Compare commodity extractors on identical footing.

    # open-vocabulary probe (no model downloads needed for the control)
    python -m scripts.bench_commodity --gold probe --control

    # add GLiNER candidates (downloads weights on first run)
    python -m scripts.bench_commodity --gold probe --models fastino/gliner2-base-v1

    # sweep label strings and thresholds for one checkpoint
    python -m scripts.bench_commodity --gold probe --sweep fastino/gliner2-base-v1

    # YOUR data — the only number worth a go/no-go
    python -m scripts.bench_commodity --gold path/to/your_labeled.jsonl --models ...

Gold sources:
  probe   HS-derived open-vocabulary probe. Vocabulary disjoint from this repo,
          framing synthetic. Catches memorization; directional, not absolute.
  gold    The repo's 72-record fixture. Shares vocabulary with the slot-fill pool,
          so it CANNOT measure open-vocabulary extraction. Regression use only.
  <path>  Your hand-labeled JSONL. Validated on load.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ner.commodity.bench import BenchResult, format_table, per_type_table, run_benchmark
from ner.commodity.eval_sets import (
    ProbeConfig,
    build_oov_probe,
    commodity_only,
    load_jsonl_gold,
)
from ner.eval.gold import GOLD_SEED
from ner.schema import Record

DEFAULT_BUDGET_MS = 100.0


def load_gold(source: str, probe_size: int, seed: int) -> tuple[list[Record], str]:
    if source == "probe":
        records = build_oov_probe(ProbeConfig(n_records=probe_size, seed=seed))
        return records, (
            f"HS open-vocabulary probe ({len(records)} records). Vocabulary is "
            "disjoint from this repo's pool; framing is synthetic."
        )
    if source == "gold":
        records = commodity_only(list(GOLD_SEED))
        return records, (
            f"repo gold fixture ({len(records)} records). WARNING: 78% of its spans "
            "are in the repo's slot-fill pool — regression signal only, not a "
            "measure of open-vocabulary extraction."
        )
    path = Path(source)
    if not path.exists():
        print(f"ERROR: no such gold source: {source}", file=sys.stderr)
        sys.exit(2)
    records = commodity_only(load_jsonl_gold(path))
    return records, f"{path} ({len(records)} records, hand-labeled)"


def build_extractors(args: argparse.Namespace) -> list:
    extractors: list = []

    if args.control:
        try:
            from ner.commodity.gazetteer import GazetteerExtractor
            from scripts.seedgen.commodities import COMMODITIES

            extractors.append(
                GazetteerExtractor(COMMODITIES, name="CONTROL:pool-gazetteer")
            )
        except (ImportError, ValueError) as exc:  # pragma: no cover
            print(f"skipping control gazetteer: {exc}", file=sys.stderr)

    model_ids = [m.strip() for m in (args.models or "").split(",") if m.strip()]
    if model_ids or args.sweep:
        try:
            from ner.commodity.gliner_adapter import (
                GlinerCommodityExtractor,
                GlinerConfig,
                LabelSweep,
            )
        except ImportError as exc:  # pragma: no cover
            print(f"ERROR: {exc}", file=sys.stderr)
            sys.exit(3)

        for model_id in model_ids:
            extractors.append(
                GlinerCommodityExtractor(
                    GlinerConfig(
                        model_id=model_id,
                        labels=tuple(args.labels.split(",")),
                        threshold=args.threshold,
                    )
                )
            )
        if args.sweep:
            extractors.extend(LabelSweep(model_id=args.sweep).candidates())

    return extractors


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default="probe", help="probe | gold | path to JSONL")
    parser.add_argument("--probe-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20240802)
    parser.add_argument("--models", default="", help="comma-separated HF model ids")
    parser.add_argument("--sweep", default="", help="model id to sweep labels/thresholds")
    parser.add_argument("--labels", default="commodity", help="comma-separated label strings")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--control", action="store_true",
        help="include the pool-gazetteer memorization control",
    )
    parser.add_argument("--budget-ms", type=float, default=DEFAULT_BUDGET_MS)
    parser.add_argument("--no-latency", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--json", type=Path, help="write full results as JSON")
    args = parser.parse_args()

    gold, description = load_gold(args.gold, args.probe_size, args.seed)
    spans = sum(len(r.entities) for r in gold)
    print(f"gold: {description}")
    print(f"      {spans} COMMODITY spans\n")

    extractors = build_extractors(args)
    if not extractors:
        print(
            "No extractors selected. Use --control for the offline memorization "
            "control, --models/--sweep for GLiNER candidates.",
            file=sys.stderr,
        )
        sys.exit(2)

    results: list[BenchResult] = []
    for extractor in extractors:
        name = getattr(extractor, "name", type(extractor).__name__)
        print(f"running {name} ...", file=sys.stderr)
        try:
            results.append(
                run_benchmark(
                    extractor,
                    gold,
                    with_latency=not args.no_latency,
                    bootstrap_resamples=args.bootstrap,
                )
            )
        # Deliberately broad: one candidate failing to load (missing weights,
        # offline, unsupported checkpoint API) must not lose the whole comparison.
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)

    if not results:
        print("all extractors failed", file=sys.stderr)
        sys.exit(1)

    print()
    print(format_table(results, budget_ms=args.budget_ms))
    print()
    print(per_type_table(max(results, key=lambda r: r.f1)))

    if args.gold == "probe":
        print(
            "\nNote: a system scoring near the control on this probe is recalling a "
            "memorized vocabulary, not extracting commodities."
        )

    if args.json:
        args.json.write_text(
            json.dumps(
                {"gold": description, "results": [r.to_dict() for r in results]},
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
