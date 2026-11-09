"""Train the ImageSense decoder on Flickr8k.

Expected layout:
    data/Flickr8k_Dataset/*.jpg
    data/Flickr8k_text/Flickr8k.token.txt
"""

import argparse
from pathlib import Path

from .config import SETTINGS

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from .captions import build_tokenizer, prepare_captions
from .model import build_decoder, save_model
from .vision import extract_spatial_features, load_image


def read_flickr_captions(path: Path) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "\t" not in line:
            continue
        image_name, caption = line.split("\t", 1)
        grouped.setdefault(image_name.split("#")[0], []).append(caption)
    return grouped


def make_training_arrays(
    captions_by_image: dict[str, list[str]], image_dir: Path, tokenizer, max_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features, inputs, targets = [], [], []
    for image_name, captions in captions_by_image.items():
        image_path = image_dir / image_name
        if not image_path.exists():
            continue
        image_features = extract_spatial_features(load_image(image_path))[0]
        for caption in prepare_captions(captions):
            encoded = tokenizer.texts_to_sequences([caption])[0]
            for end in range(1, len(encoded)):
                prefix = pad_sequences([encoded[:end]], maxlen=max_length, padding="post")[0]
                features.append(image_features)
                inputs.append(prefix)
                targets.append(encoded[end])
    return np.asarray(features), np.asarray(inputs), np.asarray(targets)


def make_training_dataset(
    captions_by_image: dict[str, list[str]],
    image_dir: Path,
    tokenizer,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    feature_cache: dict[str, np.ndarray] | None = None,
) -> tf.data.Dataset:
    """Stream caption examples so spatial CNN features never fill RAM.

    Every image is encoded once while its teacher-forced caption prefixes are
    yielded. This keeps the attention model practical on Flickr8k, where
    materialising all ``(49, 1280)`` feature maps would require many GB.
    """

    def examples():
        for image_name, captions in captions_by_image.items():
            image_path = image_dir / image_name
            if not image_path.exists():
                continue
            if feature_cache is not None and image_name in feature_cache:
                image_features = feature_cache[image_name]
            else:
                image_features = extract_spatial_features(load_image(image_path))[0].astype(np.float32)
                if feature_cache is not None:
                    feature_cache[image_name] = image_features
            for caption in prepare_captions(captions):
                encoded = tokenizer.texts_to_sequences([caption])[0]
                for end in range(1, len(encoded)):
                    prefix = pad_sequences([encoded[:end]], maxlen=max_length, padding="post")[0]
                    yield (
                        {"image_features": image_features, "caption_tokens": prefix.astype(np.int32)},
                        np.int32(encoded[end]),
                    )

    signature = (
        {
            "image_features": tf.TensorSpec(
                shape=(SETTINGS.feature_regions, SETTINGS.feature_dim), dtype=tf.float32
            ),
            "caption_tokens": tf.TensorSpec(shape=(max_length,), dtype=tf.int32),
        },
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    dataset = tf.data.Dataset.from_generator(examples, output_signature=signature)
    if shuffle:
        dataset = dataset.shuffle(2_048, seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def split_images(
    image_names: list[str], seed: int = 42
) -> tuple[list[str], list[str], list[str]]:
    """Split by image, never by caption, to prevent evaluation leakage."""
    train_names, held_out = train_test_split(image_names, test_size=0.2, random_state=seed)
    validation_names, test_names = train_test_split(held_out, test_size=0.5, random_state=seed)
    return train_names, validation_names, test_names


def select_captions(captions_by_image: dict[str, list[str]], names: list[str]) -> dict[str, list[str]]:
    return {name: captions_by_image[name] for name in names}


def train(
    data_dir: Path,
    epochs: int,
    batch_size: int,
    seed: int = 42,
    max_images: int | None = None,
    cache_features: bool = True,
) -> None:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must both be positive integers.")
    tf.keras.utils.set_random_seed(seed)
    image_dir = data_dir / "Flickr8k_Dataset"
    captions_file = data_dir / "Flickr8k_text" / "Flickr8k.token.txt"
    if not image_dir.exists() or not captions_file.exists():
        raise FileNotFoundError(
            f"Expected Flickr8k_Dataset and Flickr8k.token.txt under {data_dir}"
        )
    captions_by_image = read_flickr_captions(captions_file)
    image_names = [name for name in captions_by_image if (image_dir / name).exists()]
    if max_images is not None and max_images > 0:
        image_names = image_names[:max_images]
    if len(image_names) < 10:
        raise ValueError("At least 10 valid Flickr8k images are required for train/validation/test splits.")
    train_names, validation_names, test_names = split_images(image_names, seed)
    # Fit only on training captions. This keeps validation and test captions out
    # of the learned vocabulary statistics.
    tokenizer = build_tokenizer(
        caption for name in train_names for caption in captions_by_image[name]
    )
    model = build_decoder(
        len(tokenizer.word_index) + 1,
        SETTINGS.max_length,
        feature_dim=SETTINGS.feature_dim,
        feature_regions=SETTINGS.feature_regions,
    )
    feature_cache = {} if cache_features else None
    dataset = make_training_dataset(
        select_captions(captions_by_image, train_names),
        image_dir,
        tokenizer,
        SETTINGS.max_length,
        batch_size,
        shuffle=True,
        seed=seed,
        feature_cache=feature_cache,
    )
    validation_dataset = make_training_dataset(
        select_captions(captions_by_image, validation_names),
        image_dir,
        tokenizer,
        SETTINGS.max_length,
        batch_size,
        shuffle=False,
        seed=seed,
        feature_cache=feature_cache,
    )
    history = model.fit(
        dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=2, restore_best_weights=True
            )
        ],
    )
    save_model(model, SETTINGS.model_path)
    SETTINGS.tokenizer_path.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS.tokenizer_path.write_text(tokenizer.to_json(), encoding="utf-8")
    np.savez(
        SETTINGS.model_dir / "training_history.npz",
        loss=history.history["loss"],
        val_loss=history.history["val_loss"],
        train_images=np.asarray(train_names),
        validation_images=np.asarray(validation_names),
        test_images=np.asarray(test_names),
    )
    print(f"Saved model to {SETTINGS.model_path}")
    print(f"Image split: {len(train_names)} train, {len(validation_names)} validation, {len(test_names)} test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ImageSense AI on Flickr8k.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-images", type=int, default=None, help="Limit number of images for faster training/demo.")
    args = parser.parse_args()
    train(args.data_dir, args.epochs, args.batch_size, args.seed, max_images=args.max_images)
