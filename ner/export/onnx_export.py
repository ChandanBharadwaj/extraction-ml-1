"""Export a fine-tuned token-classification model to FP32 ONNX with graph
optimizations (constant folding, node fusion). Quantization is intentionally
NOT applied — the TDD prohibits INT8 because it degrades logits at token
boundaries.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ner.constants import MAX_SEQ_LEN


@dataclass
class ExportConfig:
    model_dir: str
    output_path: str
    opset: int = 17
    max_seq_len: int = MAX_SEQ_LEN


def export(config: ExportConfig) -> Path:
    """Trace the model and write `model.onnx` to `output_path`."""
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(config.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(config.model_dir)
    model.eval()

    dummy = tokenizer(
        "dummy text for tracing",
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=config.max_seq_len,
    )
    inputs = (dummy["input_ids"], dummy["attention_mask"])
    input_names = ["input_ids", "attention_mask"]
    output_names = ["logits"]
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "seq"},
        "attention_mask": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"},
    }

    out_path = Path(config.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    class Wrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(input_ids=input_ids, attention_mask=attention_mask).logits

    torch.onnx.export(
        Wrapper(model),
        inputs,
        str(out_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=config.opset,
        do_constant_folding=True,
    )

    _optimize_graph(out_path)
    return out_path


def _optimize_graph(path: Path) -> None:
    """Apply onnxruntime graph optimizations and save back in-place (FP32)."""
    import onnxruntime as ort

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.optimized_model_filepath = str(path)
    # Build a session purely to write the optimized graph back; we discard it.
    _ = ort.InferenceSession(str(path), sess_opts, providers=["CPUExecutionProvider"])
