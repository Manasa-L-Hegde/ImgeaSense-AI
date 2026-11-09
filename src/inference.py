from pathlib import Path

import matplotlib.cm as cm
import numpy as np
from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from .captions import END_TOKEN, START_TOKEN, caption_from_ids
from .config import SETTINGS
from .model import load_model
from .vision import extract_features, extract_spatial_features, load_image


def overlay_attention_heatmap(
    image: Image.Image,
    attention_map: np.ndarray,
    alpha: float = 0.5,
    colormap_name: str = "jet",
) -> Image.Image:
    """Overlay a 7x7 attention map onto an image using a heatmap colormap."""
    # Normalize attention weights to [0, 1]
    norm_att = attention_map.copy()
    max_val = np.max(norm_att)
    min_val = np.min(norm_att)
    if max_val > min_val:
        norm_att = (norm_att - min_val) / (max_val - min_val)
    else:
        norm_att = np.ones_like(norm_att)

    # Upsample 7x7 grid to full image size with smooth bicubic interpolation
    att_image = Image.fromarray((norm_att * 255).astype(np.uint8)).resize(
        image.size, Image.Resampling.BICUBIC
    )
    att_array = np.array(att_image) / 255.0

    # Apply colormap
    cmap = getattr(cm, colormap_name, cm.jet)
    heatmap_rgba = cmap(att_array)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)
    heatmap_image = Image.fromarray(heatmap_rgb)

    # Blend original image with heatmap
    base_image = image.convert("RGB")
    return Image.blend(base_image, heatmap_image, alpha=alpha)


class CaptionGenerator:
    def __init__(self, model_path: Path = SETTINGS.model_path, tokenizer_path: Path = SETTINGS.tokenizer_path):
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
        self.model = load_model(model_path)
        self.tokenizer = tokenizer_from_json(tokenizer_path.read_text(encoding="utf-8"))
        self.start_id = self.tokenizer.word_index.get(START_TOKEN)
        self.end_id = self.tokenizer.word_index.get(END_TOKEN)
        if not self.start_id or not self.end_id:
            raise ValueError("Tokenizer must contain <start> and <end> tokens.")
        image_input = next(item for item in self.model.inputs if item.name.startswith("image_features"))
        # Models trained before visual attention use a single pooled feature vector.
        self.uses_attention = len(image_input.shape) == 3

        # Prepare sub-model for attention score extraction if attention layer exists
        self.sub_model = None
        self.att_scale = None
        if self.uses_attention:
            try:
                self.sub_model = keras.Model(
                    inputs=self.model.inputs,
                    outputs=[
                        self.model.output,
                        self.model.get_layer("attention_query").output,
                        self.model.get_layer("visual_projection").output,
                    ],
                )
                self.att_scale = self.model.get_layer("visual_attention").get_weights()[0]
            except Exception:
                self.sub_model = None
                self.att_scale = None

    def generate(self, source: str | Path | bytes, max_length: int = SETTINGS.max_length) -> str:
        caption, _, _ = self.generate_with_attention(source, max_length=max_length)
        return caption

    def generate_with_attention(
        self, source: str | Path | bytes, max_length: int = SETTINGS.max_length
    ) -> tuple[str, list[tuple[str, np.ndarray]], Image.Image]:
        """Generate caption and return (caption, word_attentions, pil_image).

        word_attentions is a list of (word, 7x7_attention_map_ndarray).
        """
        image = load_image(source)
        features = extract_spatial_features(image) if self.uses_attention else extract_features(image)
        sequence = [self.start_id]
        word_attentions: list[tuple[str, np.ndarray]] = []

        for _ in range(max_length - 1):
            padded = sequence + [0] * (max_length - len(sequence))
            inputs = {"image_features": features, "caption_tokens": np.asarray([padded])}

            if self.sub_model is not None and self.att_scale is not None:
                logits, query, regions = self.sub_model.predict(inputs, verbose=0)
                # Compute additive attention weights
                raw_score = np.sum(self.att_scale * np.tanh(query + regions), axis=-1)  # (1, 49)
                exp_score = np.exp(raw_score - np.max(raw_score, axis=-1, keepdims=True))
                att_weights = exp_score / np.sum(exp_score, axis=-1, keepdims=True)
                att_map = att_weights[0].reshape(7, 7)
            else:
                logits = self.model.predict(inputs, verbose=0)
                att_map = np.ones((7, 7), dtype=np.float32) / 49.0

            distribution = logits[0, len(sequence) - 1] if logits.ndim == 3 else logits[0]
            next_id = int(np.argmax(distribution))
            if next_id in (0, self.end_id):
                break
            sequence.append(next_id)
            word = self.tokenizer.index_word.get(next_id, "")
            if word and word not in (START_TOKEN, END_TOKEN):
                word_attentions.append((word, att_map))

        caption = caption_from_ids(self.tokenizer, sequence) or "No caption could be generated."
        return caption, word_attentions, image

    def generate_beam_search(
        self, source: str | Path | bytes, beam_width: int = 3, max_length: int = SETTINGS.max_length
    ) -> str:
        """Generate caption using Beam Search decoding with width k."""
        image = load_image(source)
        features = extract_spatial_features(image) if self.uses_attention else extract_features(image)
        # Each beam candidate: (cumulative_log_prob, sequence)
        candidates = [(0.0, [self.start_id])]
        completed = []

        for _ in range(max_length - 1):
            all_next = []
            for score, seq in candidates:
                if seq[-1] == self.end_id:
                    completed.append((score / max(1, len(seq) - 1), seq))
                    continue
                padded = seq + [0] * (max_length - len(seq))
                inputs = {"image_features": features, "caption_tokens": np.asarray([padded])}
                logits = self.model.predict(inputs, verbose=0)
                distribution = logits[0, len(seq) - 1] if logits.ndim == 3 else logits[0]
                # Log-softmax
                max_logit = np.max(distribution)
                log_probs = distribution - max_logit - np.log(np.sum(np.exp(distribution - max_logit)) + 1e-10)
                top_indices = np.argsort(log_probs)[-beam_width:]
                for idx in top_indices:
                    all_next.append((score + float(log_probs[idx]), seq + [int(idx)]))

            if not all_next:
                break
            # Retain top beam_width candidates
            candidates = sorted(all_next, key=lambda x: x[0], reverse=True)[:beam_width]

        if not completed:
            completed = [(s / max(1, len(seq) - 1), seq) for s, seq in candidates]
        best_seq = sorted(completed, key=lambda x: x[0], reverse=True)[0][1]
        return caption_from_ids(self.tokenizer, best_seq) or "No caption could be generated."
