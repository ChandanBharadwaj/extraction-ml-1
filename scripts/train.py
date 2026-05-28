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
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
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
        learning_rate=args.lr,
        seed=args.seed,
    )
    out = train(cfg)
    print(f"Saved model to {out}")


if __name__ == "__main__":
    main()
