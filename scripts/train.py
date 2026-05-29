"""Train DeBERTa-v3-base on the synthetic JSONL; evaluate against gold."""
from __future__ import annotations

import argparse

from ner.constants import BASE_MODEL
from ner.train.train import TrainConfig, train


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gold-jsonl", default=None,
                   help="Optional path to a real-labeled gold JSONL; "
                        "defaults to ner.eval.gold.GOLD_SEED")
    p.add_argument("--base-model", default=BASE_MODEL,
                   help="HF model id; e.g. microsoft/deberta-v3-xsmall for a "
                        "~280MB / 6GB-VRAM-friendly run")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps; effective batch = "
                        "batch-size * grad-accum (use to keep effective batch "
                        "high while fitting a small GPU)")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="Token sequence length (default from constants). "
                        "Lower (e.g. 128/192) to save GPU memory and time")
    p.add_argument("--max-train-samples", type=int, default=None,
                   help="Cap the number of synthetic training records "
                        "(useful for smoke runs)")
    p.add_argument("--fp16", action="store_true",
                   help="Enable fp16 mixed precision (halves memory; needed to "
                        "fit base models on <=6GB GPUs)")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    cfg = TrainConfig(
        train_jsonl=args.train_jsonl,
        output_dir=args.output_dir,
        base_model=args.base_model,
        gold_jsonl=args.gold_jsonl,
        epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_train_samples=args.max_train_samples,
        fp16=args.fp16,
        learning_rate=args.lr,
        seed=args.seed,
    )
    if args.max_seq_len is not None:
        cfg.max_seq_len = args.max_seq_len
    out = train(cfg)
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
