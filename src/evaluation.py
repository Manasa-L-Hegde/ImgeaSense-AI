"""Held-out evaluation helpers for the trained ImageSense captioning model."""

import json
import re
from pathlib import Path

from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from .config import SETTINGS


def _tokens(caption: str) -> list[str]:
    return re.findall(r"[a-z]+", caption.lower())


def bleu_scores(references: list[list[str]], predictions: list[str]) -> dict[str, float]:
    """Calculate corpus BLEU-1 and BLEU-4 using one reference set per image."""
    if len(references) != len(predictions):
        raise ValueError("references and predictions must contain the same number of images.")
    if not predictions:
        raise ValueError("At least one prediction is required for evaluation.")
    tokenized_references = [[_tokens(reference) for reference in group] for group in references]
    tokenized_predictions = [_tokens(prediction) for prediction in predictions]
    smoothing = SmoothingFunction().method1
    return {
        "bleu_1": float(
            corpus_bleu(
                tokenized_references,
                tokenized_predictions,
                weights=(1, 0, 0, 0),
                smoothing_function=smoothing,
            )
        ),
        "bleu_4": float(
            corpus_bleu(tokenized_references, tokenized_predictions, smoothing_function=smoothing)
        ),
    }


def evaluate(data_dir: Path, limit: int | None = None, seed: int = 42) -> dict[str, object]:
    """Generate captions for the held-out split and save a reproducible report."""
    from .inference import CaptionGenerator
    from .train import read_flickr_captions, split_images

    image_dir = data_dir / "Flickr8k_Dataset"
    captions_file = data_dir / "Flickr8k_text" / "Flickr8k.token.txt"
    if not image_dir.exists() or not captions_file.exists():
        raise FileNotFoundError("Expected Flickr8k_Dataset and Flickr8k.token.txt under the data directory.")

    captions_by_image = read_flickr_captions(captions_file)
    names = [name for name in captions_by_image if (image_dir / name).exists()]
    if len(names) < 10:
        raise ValueError("At least 10 valid Flickr8k images are required for evaluation.")
    _, _, test_names = split_images(names, seed)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be a positive integer.")
        test_names = test_names[:limit]

    generator = CaptionGenerator()
    predictions = [generator.generate(image_dir / name) for name in test_names]
    scores = bleu_scores([captions_by_image[name] for name in test_names], predictions)
    report: dict[str, object] = {
        "seed": seed,
        "evaluated_images": len(test_names),
        **scores,
        "examples": [
            {"image": name, "prediction": prediction, "references": captions_by_image[name]}
            for name, prediction in zip(test_names[:10], predictions[:10], strict=True)
        ],
    }
    SETTINGS.model_dir.mkdir(parents=True, exist_ok=True)
    report_path = SETTINGS.model_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved evaluation report to {report_path}")
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate ImageSense AI on held-out Flickr8k images.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=None, help="Optional number of test images to score.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    evaluate(args.data_dir, args.limit, args.seed)
