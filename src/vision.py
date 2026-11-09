from pathlib import Path

import numpy as np
from PIL import Image

from .config import SETTINGS
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input


_feature_extractor: Model | None = None


def get_feature_extractor() -> Model:
    global _feature_extractor
    if _feature_extractor is None:
        extractor = MobileNetV2(
            include_top=False,
            weights="imagenet",
            input_shape=(*SETTINGS.image_size, 3),
        )
        extractor.trainable = False
        _feature_extractor = extractor
    return _feature_extractor


def load_image(source: str | Path | bytes) -> Image.Image:
    if isinstance(source, bytes):
        from io import BytesIO

        image = Image.open(BytesIO(source))
    else:
        image = Image.open(source)
    return image.convert("RGB")


def extract_features(image: Image.Image) -> np.ndarray:
    """Return a pooled CNN embedding for compatibility with legacy artifacts."""
    return extract_spatial_features(image).mean(axis=1)


def extract_spatial_features(image: Image.Image) -> np.ndarray:
    """Return 7×7 CNN region features for the attention-based caption decoder."""
    resized = image.resize(SETTINGS.image_size)
    array = np.asarray(resized, dtype=np.float32)
    batch = preprocess_input(np.expand_dims(array, axis=0))
    feature_map = get_feature_extractor()(batch, training=False).numpy()
    return feature_map.reshape(feature_map.shape[0], -1, feature_map.shape[-1])


