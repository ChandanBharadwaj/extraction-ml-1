"""Fine-tune DeBERTa-v3-base for BIO token classification.

Run via `python -m scripts.train --train-jsonl ... --output-dir ...`.

Design notes:
  - Gold validation records (real, hand-labeled) drive early stopping; the
    synthetic training set never leaks into eval.
  - We use the fast tokenizer's offset_mapping to build labels; this is the
    same machinery used at inference, so char-offset semantics stay aligned
    end to end.
  - No mixed precision on the export head — INT8 is prohibited by the TDD.
    BF16 *training* is fine because the exported ONNX graph is cast to FP32.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ner.constants import BASE_MODEL, MAX_SEQ_LEN, NUM_LABELS, ID2LABEL, LABEL2ID


@dataclass
class TrainConfig:
    train_jsonl: str
    output_dir: str
    base_model: str = BASE_MODEL
    gold_jsonl: str | None = None  # None -> use built-in GOLD_SEED
    epochs: int = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_len: int = MAX_SEQ_LEN
    seed: int = 42
    early_stopping_patience: int = 2


def train(config: TrainConfig) -> Path:
    """Run training. Imports torch/transformers lazily so the package can be
    installed in `inference` mode without these deps."""
    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    from ner.data.assembler import read_jsonl
    from ner.eval.gold import load_gold
    from ner.train.dataset import encode_records
    from ner.train.metrics import SpanF1Metric

    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(
        config.base_model,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_records = read_jsonl(config.train_jsonl)
    gold_records = load_gold(config.gold_jsonl)

    train_ds = Dataset.from_generator(
        encode_records,
        gen_kwargs={
            "records": train_records,
            "tokenizer": tokenizer,
            "max_length": config.max_seq_len,
        },
    )
    eval_ds = Dataset.from_generator(
        encode_records,
        gen_kwargs={
            "records": gold_records,
            "tokenizer": tokenizer,
            "max_length": config.max_seq_len,
        },
    )

    args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=config.seed,
        report_to=[],
        bf16=False,
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=SpanF1Metric(tokenizer, gold_records),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    return Path(config.output_dir)
