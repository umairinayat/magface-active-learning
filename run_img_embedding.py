#!/usr/bin/env python
"""
CLI utilities:
  1) Single-image embedding
  2) Fine-tune on labeled image pairs
"""
import argparse
import json
import numpy as np

from img_embedding import get_embedding, finetune_pairs


def _load_pairs(pairs_file: str):
    with open(pairs_file, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "pairs" in data:
        return data["pairs"]
    if isinstance(data, list):
        return data
    raise ValueError("pairs_file must be a list or a dict with key 'pairs'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding + fine-tune utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_embed = sub.add_parser("embed", help="Compute single-image embedding")
    p_embed.add_argument("--image", required=True, help="Path to image file")
    p_embed.add_argument("--config", default="config.yaml", help="Config file path")
    p_embed.add_argument("--device", default=None, help="cpu or cuda (optional)")
    p_embed.add_argument(
        "--assume_bgr", action="store_true", help="Treat numpy arrays as BGR (unused for file paths)"
    )

    p_ft = sub.add_parser("finetune", help="Fine-tune on labeled pairs")
    p_ft.add_argument("--pairs_file", required=True, help="JSON file with pairs list or {pairs:[...]}")
    p_ft.add_argument("--config", default="config.yaml", help="Config file path")
    p_ft.add_argument("--output_dir", default="checkpoints_feedback_pairs", help="Where to save checkpoints")
    p_ft.add_argument("--val_split", type=float, default=0.2, help="Validation split ratio")
    p_ft.add_argument("--seed", type=int, default=42, help="Random seed")
    p_ft.add_argument("--device", default=None, help="cpu or cuda (optional)")
    p_ft.add_argument("--batch_size", type=int, default=32)
    p_ft.add_argument("--lr", type=float, default=1e-5)
    p_ft.add_argument("--epochs", type=int, default=5)
    p_ft.add_argument("--margin", type=float, default=0.4)
    p_ft.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    if args.cmd == "embed":
        emb = get_embedding(
            args.image,
            config_path=args.config,
            device=args.device,
            assume_bgr=args.assume_bgr,
        )
        print("embedding.shape:", emb.shape)
        print("embedding.norm:", float(np.linalg.norm(emb)))
        print("embedding[:10]:", emb[:10])
        return

    if args.cmd == "finetune":
        pairs = _load_pairs(args.pairs_file)
        metrics = finetune_pairs(
            pairs,
            config_path=args.config,
            output_dir=args.output_dir,
            val_split=args.val_split,
            seed=args.seed,
            device=args.device,
            batch_size=args.batch_size,
            lr=args.lr,
            epochs=args.epochs,
            margin=args.margin,
            num_workers=args.num_workers,
        )
        print("baseline:", metrics["baseline"])
        if metrics["history"]:
            print("last_epoch:", metrics["history"][-1])
        print("best:", metrics["best"])
        return


if __name__ == "__main__":
    main()
