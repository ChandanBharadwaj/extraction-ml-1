"""Python-only inference runtime: tokenizer + ONNX Runtime + BIO decoder.

No PyTorch in the serving layer. Loads the saved fast tokenizer directly via
`tokenizers.Tokenizer.from_file` (zero HuggingFace runtime dep on transformers)
when available, or falls back to `transformers.AutoTokenizer` if installed.

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

from ner.bio import bio_ids_to_spans
from ner.constants import MAX_INPUT_CHARS, MAX_SEQ_LEN
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


class NERRuntime:
    """Single-process, thread-tuned ONNX serving wrapper."""

    def __init__(self, config: NERRuntimeConfig):
        import onnxruntime as ort

        self.config = config
        self._session = self._build_session(ort)
        self._tokenizer = self._load_tokenizer()

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
        # Fallback to transformers if a raw tokenizer.json isn't present.
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.config.tokenizer_dir, use_fast=True)

    def _encode(self, text: str) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]]:
        from tokenizers import Tokenizer

        if isinstance(self._tokenizer, Tokenizer):
            enc = self._tokenizer.encode(text)
            ids = np.array(enc.ids[: self.config.max_seq_len], dtype=np.int64)[None, :]
            attn = np.array(enc.attention_mask[: self.config.max_seq_len], dtype=np.int64)[None, :]
            offsets = enc.offsets[: self.config.max_seq_len]
            return ids, attn, offsets

        # transformers fallback
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

    def predict(self, text: str) -> list[Entity]:
        if len(text) > self.config.max_input_chars:
            text = text[: self.config.max_input_chars]
        input_ids, attention_mask, offsets = self._encode(text)
        logits = self._session.run(
            ["logits"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )[0]
        pred_ids = np.argmax(logits[0], axis=-1).tolist()
        return bio_ids_to_spans(pred_ids, offsets, text)

    def predict_batch(self, texts: list[str]) -> list[list[Entity]]:
        """Naive batch: pad to max length and run a single session.run call.

        For very large backlogs prefer a multi-process pool; ONNX Runtime
        already parallelizes within a single inference call.
        """
        if not texts:
            return []
        truncated = [t[: self.config.max_input_chars] for t in texts]
        encoded = [self._encode(t) for t in truncated]
        max_len = max(ids.shape[1] for ids, _, _ in encoded)
        batch_ids = np.zeros((len(texts), max_len), dtype=np.int64)
        batch_mask = np.zeros((len(texts), max_len), dtype=np.int64)
        all_offsets: list[list[tuple[int, int]]] = []
        for i, (ids, mask, offsets) in enumerate(encoded):
            n = ids.shape[1]
            batch_ids[i, :n] = ids[0]
            batch_mask[i, :n] = mask[0]
            padded_offsets = list(offsets) + [(0, 0)] * (max_len - n)
            all_offsets.append(padded_offsets)
        logits = self._session.run(
            ["logits"],
            {"input_ids": batch_ids, "attention_mask": batch_mask},
        )[0]
        pred_ids = np.argmax(logits, axis=-1)
        return [
            bio_ids_to_spans(pred_ids[i].tolist(), all_offsets[i], truncated[i])
            for i in range(len(texts))
        ]


def from_artifact_dir(artifact_dir: str | Path) -> NERRuntime:
    """Convenience loader: expects `model.onnx` and tokenizer files in `artifact_dir`."""
    artifact_dir = Path(artifact_dir)
    return NERRuntime(
        NERRuntimeConfig(
            onnx_path=str(artifact_dir / "model.onnx"),
            tokenizer_dir=str(artifact_dir),
            intra_op_threads=int(os.environ.get("ORT_INTRA_OP", "2")),
            inter_op_threads=int(os.environ.get("ORT_INTER_OP", "1")),
        )
    )
