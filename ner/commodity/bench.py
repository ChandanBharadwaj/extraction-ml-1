"""Benchmark harness: score any commodity extractor on any gold set.

Reuses `ner.eval.metrics.evaluate` unchanged — it already takes plain
`list[list[Entity]]` and knows nothing about models — and adds the three things
that were missing when choosing between extractors:

1. **An error taxonomy.** A bare F1 does not say whether a system is finding the
   wrong things, missing things, or getting the boundaries wrong. In earlier
   measurement 17 of 41 errors were boundary-off and only 1 was spurious, which
   pointed straight at span extent as the thing to fix. That is not visible in F1.
2. **Bootstrap confidence intervals.** On a 72-record set the 95% CI came out
   ±0.08 — two systems had to differ by 16 F1 points to be distinguishable. A
   comparison table without CIs invites reading noise as signal.
3. **Honest latency.** Warmup excluded, single-record (not batched, which hides
   per-request cost), reported as median and p95.
"""
from __future__ import annotations

import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from ner.commodity.protocol import CommodityExtractor
from ner.eval.metrics import EvalReport, evaluate
from ner.schema import Entity, Record


@dataclass(frozen=True, slots=True)
class ErrorTaxonomy:
    """Where an extractor's errors actually come from.

    * `boundary` — predicted a span overlapping gold but with different edges.
    * `polarity` — right span, wrong POS/NEG.
    * `spurious` — predicted a span with no gold overlap at all.
    * `missed` — gold span with no predicted overlap at all.
    """

    boundary: int = 0
    polarity: int = 0
    spurious: int = 0
    missed: int = 0
    examples: dict[str, list[str]] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return self.boundary + self.polarity + self.spurious + self.missed

    def to_dict(self) -> dict:
        return {
            "boundary": self.boundary,
            "polarity": self.polarity,
            "spurious": self.spurious,
            "missed": self.missed,
            "examples": self.examples,
        }


@dataclass(frozen=True, slots=True)
class LatencyStats:
    median_ms: float
    p95_ms: float
    n_calls: int

    def to_dict(self) -> dict:
        return {"median_ms": self.median_ms, "p95_ms": self.p95_ms, "n_calls": self.n_calls}


@dataclass(frozen=True, slots=True)
class BenchResult:
    name: str
    report: EvalReport
    taxonomy: ErrorTaxonomy
    latency: LatencyStats | None
    f1_ci: tuple[float, float] | None

    @property
    def f1(self) -> float:
        return self.report.micro.f1

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "metrics": self.report.to_dict(),
            "taxonomy": self.taxonomy.to_dict(),
            "latency": self.latency.to_dict() if self.latency else None,
            "f1_ci": list(self.f1_ci) if self.f1_ci else None,
        }


def _overlaps(a: Entity, b: Entity) -> bool:
    return not (a.end <= b.start or a.start >= b.end)


def error_taxonomy(
    predictions: Sequence[Sequence[Entity]],
    gold: Sequence[Record],
    max_examples: int = 6,
) -> ErrorTaxonomy:
    boundary = polarity = spurious = missed = 0
    examples: dict[str, list[str]] = {
        "boundary": [], "polarity": [], "spurious": [], "missed": [],
    }

    for preds, ref in zip(predictions, gold):
        gold_by_span = {(e.start, e.end): e for e in ref.entities}

        for pred in preds:
            exact = gold_by_span.get((pred.start, pred.end))
            if exact is not None:
                if exact.polarity != pred.polarity:
                    polarity += 1
                    if len(examples["polarity"]) < max_examples:
                        examples["polarity"].append(
                            f"{pred.text!r} pred={pred.polarity} gold={exact.polarity}"
                        )
                continue
            overlapping = [g for g in ref.entities if _overlaps(pred, g)]
            if overlapping:
                boundary += 1
                if len(examples["boundary"]) < max_examples:
                    examples["boundary"].append(
                        f"{pred.text!r} -> {overlapping[0].text!r}"
                    )
            else:
                spurious += 1
                if len(examples["spurious"]) < max_examples:
                    examples["spurious"].append(repr(pred.text))

        for ref_ent in ref.entities:
            if not any(_overlaps(p, ref_ent) for p in preds):
                missed += 1
                if len(examples["missed"]) < max_examples:
                    examples["missed"].append(repr(ref_ent.text))

    return ErrorTaxonomy(
        boundary=boundary, polarity=polarity, spurious=spurious, missed=missed,
        examples={k: v for k, v in examples.items() if v},
    )


def bootstrap_f1_ci(
    predictions: Sequence[Sequence[Entity]],
    gold: Sequence[Record],
    resamples: int = 1000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> tuple[float, float]:
    """Percentile bootstrap CI for micro-F1, resampling records with replacement."""
    import random

    rng = random.Random(seed)
    n = len(gold)
    if n == 0:
        return (0.0, 0.0)

    scores: list[float] = []
    for _ in range(resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        scores.append(
            evaluate([list(predictions[i]) for i in idx], [gold[i] for i in idx]).micro.f1
        )
    scores.sort()
    tail = (1.0 - confidence) / 2.0
    return (scores[int(tail * len(scores))], scores[min(len(scores) - 1, int((1 - tail) * len(scores)))])


def measure_latency(
    extractor: CommodityExtractor,
    texts: Sequence[str],
    warmup: int = 3,
    repeats: int = 1,
) -> LatencyStats:
    """Per-record latency, warmup excluded.

    Single-record calls on purpose: batching amortizes cost in a way that hides
    what a request-response deployment actually pays.
    """
    sample = list(texts)[:max(1, len(texts))]
    for text in sample[:warmup]:
        extractor.predict(text)

    timings: list[float] = []
    for _ in range(repeats):
        for text in sample:
            started = time.perf_counter()
            extractor.predict(text)
            timings.append((time.perf_counter() - started) * 1000.0)

    timings.sort()
    p95 = timings[min(len(timings) - 1, int(0.95 * len(timings)))]
    return LatencyStats(
        median_ms=statistics.median(timings), p95_ms=p95, n_calls=len(timings)
    )


def run_benchmark(
    extractor: CommodityExtractor,
    gold: Sequence[Record],
    *,
    with_latency: bool = True,
    with_ci: bool = True,
    bootstrap_resamples: int = 1000,
) -> BenchResult:
    """Score one extractor. Gold must already be COMMODITY-only."""
    texts = [r.text for r in gold]
    predictions = [
        [e for e in ents if e.type == "COMMODITY"]
        for ents in extractor.predict_batch(texts)
    ]
    if len(predictions) != len(gold):
        raise ValueError(
            f"{getattr(extractor, 'name', extractor)}: predict_batch returned "
            f"{len(predictions)} results for {len(gold)} inputs"
        )

    report = evaluate(predictions, list(gold))
    return BenchResult(
        name=getattr(extractor, "name", type(extractor).__name__),
        report=report,
        taxonomy=error_taxonomy(predictions, gold),
        latency=measure_latency(extractor, texts) if with_latency else None,
        f1_ci=bootstrap_f1_ci(predictions, gold, resamples=bootstrap_resamples)
        if with_ci
        else None,
    )


def format_table(results: Sequence[BenchResult], budget_ms: float | None = None) -> str:
    """Render a comparison table, CI-aware."""
    if not results:
        return "(no extractors ran)"

    header = (
        f"{'extractor':<28} {'P':>6} {'R':>6} {'F1':>6} {'95% CI':>16} "
        f"{'med ms':>8} {'p95 ms':>8}"
    )
    lines = [header, "-" * 84]
    for result in sorted(results, key=lambda r: r.f1, reverse=True):
        micro = result.report.micro
        ci = f"[{result.f1_ci[0]:.3f},{result.f1_ci[1]:.3f}]" if result.f1_ci else "--"
        med = f"{result.latency.median_ms:.1f}" if result.latency else "--"
        p95 = f"{result.latency.p95_ms:.1f}" if result.latency else "--"
        flag = ""
        if budget_ms and result.latency and result.latency.p95_ms > budget_ms:
            flag = "  ! over budget"
        lines.append(
            f"{result.name:<28} {micro.precision:>6.3f} {micro.recall:>6.3f} "
            f"{micro.f1:>6.3f} {ci:>16} {med:>8} {p95:>8}{flag}"
        )

    best = max(results, key=lambda r: r.f1)
    if best.f1_ci:
        width = best.f1_ci[1] - best.f1_ci[0]
        note = (
            f"CI width on the leader is {width:.3f} — treat differences smaller "
            "than that as noise."
        )
        lines += ["", note]

    lines += ["", "error taxonomy (per extractor):"]
    for result in sorted(results, key=lambda r: r.f1, reverse=True):
        tax = result.taxonomy
        lines.append(
            f"  {result.name:<26} boundary={tax.boundary:<4} missed={tax.missed:<4} "
            f"polarity={tax.polarity:<4} spurious={tax.spurious:<4}"
        )
    return "\n".join(lines)


def per_type_table(result: BenchResult) -> str:
    lines = [f"{result.name} — per-bucket:"]
    for bucket in ("COMMODITY(POS)", "COMMODITY(NEG)"):
        metrics = result.report.per_type.get(bucket)
        if metrics is None:
            continue
        lines.append(
            f"  {bucket:<16} P={metrics.precision:.3f} R={metrics.recall:.3f} "
            f"F1={metrics.f1:.3f} support={metrics.support}"
        )
    return "\n".join(lines)
