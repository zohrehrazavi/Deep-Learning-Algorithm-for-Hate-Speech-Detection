"""
Evaluation CLI stub that reads saved artifacts and prints stored metrics.
"""
import argparse
import os

from .utils import load_json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    metrics_path = os.path.join(args.run_dir, "metrics.json")
    if not os.path.isfile(metrics_path):
        print(f"No metrics found at {metrics_path}")
        return
    metrics = load_json(metrics_path)
    print(f"Metrics at {args.run_dir}:")
    for k, v in metrics.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()


