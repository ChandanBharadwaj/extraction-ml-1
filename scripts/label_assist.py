"""Pre-annotate your own text so you can measure without sending it anywhere.

Runs entirely on your machine. Reads raw text, runs an extractor over it, and
writes JSONL you correct by hand — then feed that back to
`scripts.bench_commodity --gold <file>` for a number on your real data.

    # 1. pre-annotate (one record per line, or JSONL with a "text" field)
    python -m scripts.label_assist --input my_records.txt \
        --model fastino/gliner2-base-v1 --out to_correct.jsonl

    # 2. correct to_correct.jsonl by hand (see the format note below)

    # 3. score any candidate against it
    python -m scripts.bench_commodity --gold to_correct.jsonl --models ...

Correction format — each line is one record:

    {"text": "...", "entities": [{"type": "COMMODITY", "text": "cane sugar",
     "start": 12, "end": 22, "polarity": "POS"}], "meta": {}}

`start`/`end` are half-open character offsets, so `text[start:end]` must equal
`text`. `--verify` re-checks that for every record and reports the bad lines
rather than letting them score silently against garbage.

Sizing: aim for ~300 records. Measured on a 72-record set, the bootstrap 95% CI
on F1 was +/-0.08 — too wide to tell two candidates apart.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ner.commodity.eval_sets import load_jsonl_gold
from ner.schema import Record


def read_inputs(path: Path) -> list[str]:
    """Accept plain text (one record per line) or JSONL with a `text` field."""
    texts: list[str] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                texts.append(json.loads(line)["text"])
            except (json.JSONDecodeError, KeyError) as exc:
                print(f"{path}:{line_no}: skipping unparsable line ({exc})", file=sys.stderr)
        else:
            texts.append(line)
    return texts


def build_extractor(args: argparse.Namespace):
    if args.model:
        from ner.commodity.gliner_adapter import GlinerCommodityExtractor, GlinerConfig

        return GlinerCommodityExtractor(
            GlinerConfig(
                model_id=args.model,
                labels=tuple(args.labels.split(",")),
                threshold=args.threshold,
            )
        )
    from ner.commodity.gazetteer import GazetteerExtractor
    from scripts.seedgen.commodities import COMMODITIES

    print(
        "No --model given; falling back to the pool gazetteer. It only knows 589 "
        "hand-written strings, so expect to add many spans by hand.",
        file=sys.stderr,
    )
    return GazetteerExtractor(COMMODITIES, name="pool-gazetteer")


def verify(path: Path) -> int:
    try:
        records = load_jsonl_gold(path)
    except ValueError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1
    spans = sum(len(r.entities) for r in records)
    neg = sum(1 for r in records for e in r.entities if e.polarity == "NEG")
    print(f"OK: {len(records)} records, {spans} spans ({neg} NEG), all offsets valid")
    if len(records) < 150:
        print(
            f"WARNING: {len(records)} records is too few to compare candidates "
            "reliably — aim for ~300.",
            file=sys.stderr,
        )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, help="raw text or JSONL to pre-annotate")
    parser.add_argument("--out", type=Path, help="destination JSONL")
    parser.add_argument("--model", default="", help="HF model id (else gazetteer)")
    parser.add_argument("--labels", default="commodity")
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--limit", type=int, default=0, help="cap records processed")
    parser.add_argument("--verify", type=Path, help="validate a corrected JSONL and exit")
    args = parser.parse_args()

    if args.verify:
        sys.exit(verify(args.verify))

    if not args.input or not args.out:
        parser.error("--input and --out are required unless --verify is used")
    if not args.input.exists():
        print(f"ERROR: no such input: {args.input}", file=sys.stderr)
        sys.exit(2)

    texts = read_inputs(args.input)
    if args.limit:
        texts = texts[:args.limit]
    if not texts:
        print("ERROR: no usable records in input", file=sys.stderr)
        sys.exit(2)

    extractor = build_extractor(args)
    print(f"pre-annotating {len(texts)} records with {extractor.name} ...", file=sys.stderr)
    predictions = extractor.predict_batch(texts)

    with open(args.out, "w", encoding="utf-8") as handle:
        for text, entities in zip(texts, predictions):
            record = Record(text, list(entities), {"pre_annotated_by": extractor.name})
            record.validate()
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    total = sum(len(p) for p in predictions)
    empty = sum(1 for p in predictions if not p)
    print(f"wrote {args.out}: {len(texts)} records, {total} predicted spans")
    print(f"  {empty} records got no prediction — check those first, they are where")
    print("  recall failures hide.")
    print(f"\nNext: correct by hand, then `python -m scripts.label_assist --verify {args.out}`")


if __name__ == "__main__":
    main()
