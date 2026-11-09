"""Convenience alias so `python -m src.evaluate` and `python -m src.evaluation` both work."""

from .evaluation import evaluate

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Evaluate ImageSense AI on held-out Flickr8k images.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=None, help="Optional number of test images to score.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(args.data_dir, args.limit, args.seed)

