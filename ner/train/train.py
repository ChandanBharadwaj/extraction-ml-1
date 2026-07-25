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
    # Path to preprocess.json (the same config the assembler used). If None,
    # we look next to train_jsonl, then fall back to default Preprocessor().
    preprocess_path: str | None = None
    epochs: int = 3
    per_device_train_batch_size: int = 32
    per_device_eval_batch_size: int = 32
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_seq_len: int = MAX_SEQ_LEN
    seed: int = 42
    early_stopping_patience: int = 2
    # Deterministic gold split for early stopping ("all" | "earlystop" |
    # "tune"). Use "earlystop" here and tune thresholds with --eval-split
    # tune so the two consumers don't share records (ner.eval.gold.split_gold).
    gold_split: str = "all"
    # Checkpoint-selection metric. "f1" is micro span-F1; SpanF1Metric also
    # emits per-bucket keys like "f1_COMMODITY(NEG)" for cargo-focused runs.
    metric_for_best_model: str = "f1"
    # "none" | "neg_boost": weight the rare B-/I-NEG_COMMODITY labels in the
    # cross-entropy loss. Data-side rebalancing is the primary lever; this is
    # an opt-in knob, not a default.
    class_weights: str = "none"


NEG_BOOST_WEIGHT: float = 3.0


def class_weight_values(scheme: str) -> list[float] | None:
    """Per-label loss weights for a scheme; None means unweighted."""
    if scheme == "none":
        return None
    if scheme == "neg_boost":
        weights = [1.0] * NUM_LABELS
        for label in ("B-NEG_COMMODITY", "I-NEG_COMMODITY"):
            weights[LABEL2ID[label]] = NEG_BOOST_WEIGHT
        return weights
    raise ValueError(f"unknown class_weights scheme {scheme!r}")


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
    from ner.eval.gold import load_gold, split_gold
    from ner.preprocess import Preprocessor
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
    gold_records = split_gold(load_gold(config.gold_jsonl), config.gold_split)

    # Resolve the preprocess config: explicit arg > sibling of train_jsonl >
    # default. The same config is later saved into the model output dir so
    # ONNX export + serving inherit it automatically.
    preprocess_path = config.preprocess_path
    if preprocess_path is None:
        sibling = Path(config.train_jsonl).with_name(Preprocessor.CONFIG_FILENAME)
        if sibling.exists():
            preprocess_path = str(sibling)
    preprocessor = (
        Preprocessor.load(preprocess_path) if preprocess_path else Preprocessor()
    )

    train_ds = Dataset.from_generator(
        encode_records,
        gen_kwargs={
            "records": train_records,
            "tokenizer": tokenizer,
            "max_length": config.max_seq_len,
        },
    )
    # Project gold into the same cleaned coordinate system the training
    # records live in. This *is* the eval set used for early stopping.
    preprocessed_gold = preprocessor.apply_to_records(gold_records)
    eval_ds = Dataset.from_generator(
        encode_records,
        gen_kwargs={
            "records": preprocessed_gold,
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
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=True,
        seed=config.seed,
        report_to=[],
        bf16=False,
        fp16=False,
    )

    weight_values = class_weight_values(config.class_weights)
    trainer_cls = Trainer
    trainer_extra: dict = {}
    if weight_values is not None:
        import torch

        class WeightedLossTrainer(Trainer):
            """Stock Trainer with per-label CrossEntropyLoss weights, so the
            rare NEG_COMMODITY labels can be boosted without oversampling."""

            def __init__(self, *t_args, class_weight_tensor=None, **t_kwargs):
                super().__init__(*t_args, **t_kwargs)
                self._class_weight_tensor = class_weight_tensor

            def compute_loss(self, model, inputs, return_outputs=False, **_):
                labels = inputs.pop("labels")
                outputs = model(**inputs)
                logits = outputs.logits
                loss_fct = torch.nn.CrossEntropyLoss(
                    weight=self._class_weight_tensor.to(logits.device),
                    ignore_index=-100,
                )
                loss = loss_fct(
                    logits.view(-1, logits.shape[-1]), labels.view(-1)
                )
                return (loss, outputs) if return_outputs else loss

        trainer_cls = WeightedLossTrainer
        trainer_extra["class_weight_tensor"] = torch.tensor(
            weight_values, dtype=torch.float32
        )

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
        compute_metrics=SpanF1Metric(
            tokenizer, gold_records, preprocessor,
            max_length=config.max_seq_len,
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)],
        **trainer_extra,
    )

    trainer.train()
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    # Persist the preprocess config alongside the model so export + serving
    # pick it up automatically (symmetric with how `thresholds.json` flows).
    preprocessor.save(Path(config.output_dir) / Preprocessor.CONFIG_FILENAME)
    return Path(config.output_dir)
