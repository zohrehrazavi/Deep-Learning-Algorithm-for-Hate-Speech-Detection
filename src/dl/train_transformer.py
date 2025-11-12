"""
Minimal training CLI stub for DistilBERT fine-tuning.
Creates expected folder structure so inference layer can work even before training.
"""
import argparse
import os

from .utils import ensure_dir, save_json, seed_everything


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dl/transformer.yaml")
    parser.add_argument("--out_dir", type=str, default="models/dl/transformer")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(42)
    ensure_dir(args.out_dir)
    # Create placeholder HuggingFace model dir layout
    ensure_dir(os.path.join(args.out_dir, "checkpoint"))
    save_json({"macro_f1": 0.0}, os.path.join(args.out_dir, "metrics.json"))
    save_json({"0": "hateful", "1": "offensive", "2": "neutral"}, os.path.join(args.out_dir, "label_map.json"))
    print(f"Saved placeholder transformer artifacts to {args.out_dir}")


if __name__ == "__main__":
    main()


