"""Python-only inference runtime: tokenizer + ONNX Runtime + BIO decoder.

No PyTorch in the serving layer. Loads the saved fast tokenizer directly via
`tokenizers.Tokenizer.from_file` (zero HuggingFace runtime dep on transformers)
when available, or falls back to `transformers.AutoTokenizer` if installed.

Threshold gating:
    If `<artifact_dir>/thresholds.json` exists (produced by
    `scripts.tune_threshold`), the runtime loads it at startup and gates each
    non-O argmax by the per-label confidence floor before BIO decoding.
    Absent that file, the runtime defaults to plain argmax decoding so
    nothing is breaking when thresholds haven't been tuned yet.

CPU-tuned threading defaults are set conservatively (intra=2, inter=1) since
records are short and we generally want batch-parallelism over thread-parallelism
when scaling to millions of records via multiprocessing.
"""
from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ner.bio import apply_threshold_gate, bio_ids_to_spans, softmax
from ner.constants import (
    MAX_INPUT_CHARS,
    MAX_SEQ_LEN,
    NUM_LABELS,
    WINDOW_OVERLAP_TOKENS,
)
from ner.preprocess import Preprocessor
from ner.schema import Entity


@dataclass
class NERRuntimeConfig:
    onnx_path: str
    tokenizer_dir: str
    max_seq_len: int = MAX_SEQ_LEN
    max_input_chars: int = MAX_INPUT_CHARS
    window_overlap: int = WINDOW_OVERLAP_TOKENS
    intra_op_threads: int = 2
    inter_op_threads: int = 1
    providers: list[str] = field(default_factory=lambda: ["CPUExecutionProvider"])
    thresholds_path: str | None = None
    preprocess_path: str | None = None  # path to preprocess.json; default config if None


class NERRuntime:
    """Single-process, thread-tuned ONNX serving wrapper."""

    def __init__(self, config: NERRuntimeConfig):
        import onnxruntime as ort

        self.config = config
        self._session = self._build_session(ort)
        self._tokenizer = self._load_tokenizer()
        self.thresholds: np.ndarray | None = self._load_thresholds()
        self.preprocessor: Preprocessor = self._load_preprocessor()

    def _build_session(self, ort: Any):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self.config.intra_op_threads
        opts.inter_op_num_threads = self.config.inter_op_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return ort.InferenceSession(
            self.config.onnx_path, opts, providers=self.config.providers,
        )

    def _load_tokenizer(self):
        tok_path = Path(self.config.tokenizer_dir) / "tokenizer.json"
        if tok_path.exists():
            from tokenizers import Tokenizer
            return Tokenizer.from_file(str(tok_path))
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.config.tokenizer_dir, use_fast=True)

    def _load_thresholds(self) -> np.ndarray | None:
        path = self.config.thresholds_path
        if path is None:
            return None
        if not Path(path).exists():
            return None
        from ner.eval.threshold_sweep import load_thresholds_json
        return load_thresholds_json(path)

    def _load_preprocessor(self) -> Preprocessor:
        path = self.config.preprocess_path
        if path is not None and Path(path).exists():
            return Preprocessor.load(path)
        return Preprocessor()  # default config — no-op for already-clean text

    def _encode(self, text: str) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        from tokenizers import Tokenizer

        if isinstance(self._tokenizer, Tokenizer):
            enc = self._tokenizer.encode(text)
            ids = np.array(enc.ids[: self.config.max_seq_len], dtype=np.int64)[None, :]
            attn = np.array(enc.attention_mask[: self.config.max_seq_len], dtype=np.int64)[None, :]
            offsets = enc.offsets[: self.config.max_seq_len]
            return ids, attn, offsets

        enc = self._tokenizer(
            text,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.config.max_seq_len,
            return_tensors="np",
        )
        return (
            enc["input_ids"].astype(np.int64),
            enc["attention_mask"].astype(np.int64),
            [tuple(o) for o in enc["offset_mapping"][0].tolist()],
        )

    def _encode_full(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        """Encode without any token-length slicing. Returns (ids, offsets)."""
        from tokenizers import Tokenizer

        if isinstance(self._tokenizer, Tokenizer):
            enc = self._tokenizer.encode(text)
            return list(enc.ids), list(enc.offsets)

        enc = self._tokenizer(text, return_offsets_mapping=True, truncation=False)
        return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]

    def _forward(self, text: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
        """Run the model over `text` of any token length.

        Fits-in-one-window inputs take the single-pass path (identical to the
        historical behavior). Longer inputs are split into overlapping
        windows of `max_seq_len` tokens, run as one batched session call
        (the exported ONNX graph has dynamic batch and seq axes), and merged
        token-wise: each token keeps the logits from the window in which it
        sits most interior (ties -> earlier window). The result is a single
        (n_tokens, n_labels) matrix decoded globally, so no span-level merge
        of overlapping entities is ever needed.
        """
        ids, offsets = self._encode_full(text)
        n = len(ids)
        w = self.config.max_seq_len
        if n <= w:
            input_ids = np.array(ids, dtype=np.int64)[None, :]
            attention_mask = np.ones((1, n), dtype=np.int64)
            logits = self._session.run(
                ["logits"],
                {"input_ids": input_ids, "attention_mask": attention_mask},
            )[0]
            return logits[0], offsets

        # Window over the content tokens, re-adding the tokenizer's special
        # tokens (first/last, offsets (0,0)) around every window so each
        # window looks like a normal sequence to the model.
        has_specials = n >= 2 and offsets[0] == (0, 0) and offsets[-1] == (0, 0)
        if has_specials:
            cls_id, sep_id = ids[0], ids[-1]
            content_ids = ids[1:-1]
            content_offsets = offsets[1:-1]
            w_content = w - 2
        else:
            cls_id = sep_id = None
            content_ids = ids
            content_offsets = offsets
            w_content = w
        m = len(content_ids)

        stride = max(1, w_content - self.config.window_overlap)
        starts = list(range(0, max(1, m - w_content) + 1, stride))
        if starts[-1] + w_content < m:
            starts.append(m - w_content)
        starts = sorted({min(s, max(0, m - w_content)) for s in starts})

        # Every start satisfies s <= m - w_content, so all windows are exactly
        # full-length — no padding needed.
        rows = []
        for s in starts:
            chunk = content_ids[s:s + w_content]
            rows.append([cls_id, *chunk, sep_id] if has_specials else list(chunk))
        batch = np.array(rows, dtype=np.int64)
        attention = np.ones_like(batch)

        logits = self._session.run(
            ["logits"],
            {"input_ids": batch, "attention_mask": attention},
        )[0]  # (n_windows, w, n_labels)

        merged = np.zeros((m, logits.shape[-1]), dtype=np.float32)
        best_interior = np.full(m, -1, dtype=np.int64)
        content_lo = 1 if has_specials else 0
        for k, s in enumerate(starts):
            length = min(w_content, m - s)
            positions = np.arange(s, s + length)
            interior = np.minimum(positions - s, (s + length - 1) - positions)
            take = interior > best_interior[positions]
            rows = logits[k, content_lo:content_lo + length, :]
            merged[positions[take]] = rows[take]
            best_interior[positions[take]] = interior[take]

        return merged, content_offsets

    def _decode_logits(
        self,
        logits: np.ndarray,
        offsets: list[tuple[int, int]],
        text: str,
    ) -> list[Entity]:
        if self.thresholds is None:
            pred_ids = np.argmax(logits, axis=-1).tolist()
        else:
            probs = softmax(logits.astype(np.float32), axis=-1)
            pred_ids = apply_threshold_gate(probs, self.thresholds).tolist()
        return bio_ids_to_spans(pred_ids, offsets, text)

    def _truncate_input(self, text: str) -> str:
        if len(text) > self.config.max_input_chars:
            warnings.warn(
                f"input of {len(text)} chars truncated to "
                f"{self.config.max_input_chars}; entities past the cap are "
                "not scored",
                stacklevel=3,
            )
            return text[: self.config.max_input_chars]
        return text

    def predict(self, text: str) -> list[Entity]:
        # Hard-cap the *raw* input first (with a warning) so we never run the
        # model on unbounded text; preprocessing can only shrink it further.
        # Within the cap, any token length is handled by windowed _forward.
        original = self._truncate_input(text)
        pre = self.preprocessor.clean(original)
        if not pre.text:
            return []
        logits, offsets = self._forward(pre.text)
        cleaned_entities = self._decode_logits(logits, offsets, pre.text)
        # Project spans back into the original-text coordinate system that
        # the caller sent us. The surface form is re-sliced from `original`
        # so `original[ent.start:ent.end] == ent.text` holds.
        return [pre.project_entity(e, original) for e in cleaned_entities]

    def predict_batch(self, texts: list[str]) -> list[list[Entity]]:
        if not texts:
            return []
        originals = [self._truncate_input(t) for t in texts]
        pre_results = [self.preprocessor.clean(t) for t in originals]
        cleaned_texts = [r.text for r in pre_results]

        # Drop records that became empty after preprocessing so we don't waste
        # a batch slot; we still need to return one list per input, so track
        # which originals fed into the batch. Inputs longer than one window
        # take the windowed _forward path individually; the rest share one
        # batched session call.
        results: list[list[Entity]] = [[] for _ in texts]
        encoded: dict[int, tuple[list[int], list[tuple[int, int]]]] = {}
        short_indices: list[int] = []
        for i, t in enumerate(cleaned_texts):
            if not t:
                continue
            ids, offsets = self._encode_full(t)
            if len(ids) <= self.config.max_seq_len:
                encoded[i] = (ids, offsets)
                short_indices.append(i)
            else:
                logits, w_offsets = self._forward(t)
                ents = self._decode_logits(logits, w_offsets, t)
                results[i] = [
                    pre_results[i].project_entity(e, originals[i]) for e in ents
                ]
        if not short_indices:
            return results

        max_len = max(len(encoded[i][0]) for i in short_indices)
        batch_ids = np.zeros((len(short_indices), max_len), dtype=np.int64)
        batch_mask = np.zeros((len(short_indices), max_len), dtype=np.int64)
        all_offsets: list[list[tuple[int, int]]] = []
        for k, i in enumerate(short_indices):
            ids, offsets = encoded[i]
            n = len(ids)
            batch_ids[k, :n] = ids
            batch_mask[k, :n] = 1
            all_offsets.append(list(offsets) + [(0, 0)] * (max_len - n))
        logits = self._session.run(
            ["logits"],
            {"input_ids": batch_ids, "attention_mask": batch_mask},
        )[0]
        for k, i in enumerate(short_indices):
            cleaned_entities = self._decode_logits(
                logits[k], all_offsets[k], cleaned_texts[i],
            )
            pre = pre_results[i]
            results[i] = [
                pre.project_entity(e, originals[i]) for e in cleaned_entities
            ]
        return results


def from_artifact_dir(artifact_dir: str | Path) -> NERRuntime:
    """Convenience loader: expects `model.onnx` and tokenizer files in
    `artifact_dir`. Auto-loads `thresholds.json` and `preprocess.json` from
    the same directory if present; either absence is a no-op."""
    artifact_dir = Path(artifact_dir)
    thresholds_path: str | None = None
    candidate = artifact_dir / "thresholds.json"
    if candidate.exists():
        thresholds_path = str(candidate)
    preprocess_path: str | None = None
    pre_candidate = artifact_dir / Preprocessor.CONFIG_FILENAME
    if pre_candidate.exists():
        preprocess_path = str(pre_candidate)
    return NERRuntime(
        NERRuntimeConfig(
            onnx_path=str(artifact_dir / "model.onnx"),
            tokenizer_dir=str(artifact_dir),
            intra_op_threads=int(os.environ.get("ORT_INTRA_OP", "2")),
            inter_op_threads=int(os.environ.get("ORT_INTER_OP", "1")),
            thresholds_path=thresholds_path,
            preprocess_path=preprocess_path,
        )
    )
