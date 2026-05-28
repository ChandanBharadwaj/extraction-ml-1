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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ner.bio import apply_threshold_gate, bio_ids_to_spans, softmax
from ner.constants import MAX_INPUT_CHARS, MAX_SEQ_LEN, NUM_LABELS
from ner.preprocess import Preprocessor
from ner.schema import Entity


@dataclass
class NERRuntimeConfig:
    onnx_path: str
    tokenizer_dir: str
    max_seq_len: int = MAX_SEQ_LEN
    max_input_chars: int = MAX_INPUT_CHARS
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

    def predict(self, text: str) -> list[Entity]:
        # Hard-truncate the *raw* input first so we never run the model on
        # arbitrarily long text; preprocessing can only shrink things further.
        original = text[: self.config.max_input_chars]
        pre = self.preprocessor.clean(original)
        if not pre.text:
            return []
        input_ids, attention_mask, offsets = self._encode(pre.text)
        logits = self._session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]
        cleaned_entities = self._decode_logits(logits[0], offsets, pre.text)
        # Project spans back into the original-text coordinate system that
        # the caller sent us. The surface form is re-sliced from `original`
        # so `original[ent.start:ent.end] == ent.text` holds.
        return [pre.project_entity(e, original) for e in cleaned_entities]

    def predict_batch(self, texts: list[str]) -> list[list[Entity]]:
        if not texts:
            return []
        originals = [t[: self.config.max_input_chars] for t in texts]
        pre_results = [self.preprocessor.clean(t) for t in originals]
        cleaned_texts = [r.text for r in pre_results]

        # Drop records that became empty after preprocessing so we don't waste
        # a batch slot; we still need to return one list per input, so track
        # which originals fed into the batch.
        active_indices = [i for i, t in enumerate(cleaned_texts) if t]
        results: list[list[Entity]] = [[] for _ in texts]
        if not active_indices:
            return results

        active_texts = [cleaned_texts[i] for i in active_indices]
        encoded = [self._encode(t) for t in active_texts]
        max_len = max(ids.shape[1] for ids, _, _ in encoded)
        batch_ids = np.zeros((len(active_texts), max_len), dtype=np.int64)
        batch_mask = np.zeros((len(active_texts), max_len), dtype=np.int64)
        all_offsets: list[list[tuple[int, int]]] = []
        for k, (ids, mask, offsets) in enumerate(encoded):
            n = ids.shape[1]
            batch_ids[k, :n] = ids[0]
            batch_mask[k, :n] = mask[0]
            padded_offsets = list(offsets) + [(0, 0)] * (max_len - n)
            all_offsets.append(padded_offsets)
        logits = self._session.run(
            ["logits"],
            {"input_ids": batch_ids, "attention_mask": batch_mask},
        )[0]
        for k, orig_idx in enumerate(active_indices):
            cleaned_entities = self._decode_logits(
                logits[k], all_offsets[k], active_texts[k],
            )
            pre = pre_results[orig_idx]
            results[orig_idx] = [
                pre.project_entity(e, originals[orig_idx]) for e in cleaned_entities
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
