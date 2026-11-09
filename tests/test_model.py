import numpy as np

from src.model import build_decoder


def test_decoder_predicts_one_distribution_per_caption_prefix():
    model = build_decoder(
        vocab_size=12,
        max_length=5,
        feature_dim=4,
        feature_regions=3,
        embedding_dim=3,
        lstm_units=6,
    )
    predictions = model.predict(
        {
            "image_features": np.zeros((2, 3, 4), dtype=np.float32),
            "caption_tokens": np.array([[1, 2, 0, 0, 0], [1, 3, 4, 0, 0]], dtype=np.int32),
        },
        verbose=0,
    )
    assert predictions.shape == (2, 12)
