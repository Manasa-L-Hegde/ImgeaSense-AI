from pathlib import Path

from .config import SETTINGS
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def build_decoder(
    vocab_size: int,
    max_length: int,
    feature_dim: int = 1280,
    feature_regions: int = 49,
    embedding_dim: int = 256,
    lstm_units: int = 512,
) -> keras.Model:
    """Build an attention-based CNN-region + LSTM caption decoder.

    The CNN contributes 49 spatial feature regions (a 7×7 MobileNetV2 grid).
    The LSTM encodes the caption prefix, and additive attention selects the most
    relevant visual regions before predicting the next word.
    """
    image_features = keras.Input(shape=(feature_regions, feature_dim), name="image_features")
    caption_tokens = keras.Input(shape=(max_length,), dtype="int32", name="caption_tokens")

    visual_regions = layers.Dense(lstm_units, activation="relu", name="visual_projection")(
        image_features
    )
    word_embeddings = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name="word_embedding",
    )(caption_tokens)
    caption_state = layers.LSTM(lstm_units, name="caption_lstm")(word_embeddings)
    attention_query = layers.Reshape((1, lstm_units), name="attention_query")(caption_state)
    visual_context = layers.AdditiveAttention(name="visual_attention")(
        [attention_query, visual_regions]
    )
    visual_context = layers.Flatten(name="visual_context")(visual_context)
    fused_context = layers.Concatenate(name="vision_language_fusion")(
        [caption_state, visual_context]
    )
    fused_context = layers.Dropout(0.25, name="decoder_dropout")(fused_context)
    logits = layers.Dense(vocab_size, name="next_word")(fused_context)

    model = keras.Model(
        inputs={"image_features": image_features, "caption_tokens": caption_tokens},
        outputs=logits,
        name="imagesense_cnn_lstm_decoder",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model


@tf.keras.utils.register_keras_serializable(package="ImageSense")
def masked_sparse_categorical_crossentropy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    mask = tf.cast(tf.not_equal(y_true, 0), loss.dtype)
    return tf.reduce_sum(loss * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


@tf.keras.utils.register_keras_serializable(package="ImageSense")
def masked_accuracy(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    predictions = tf.argmax(y_pred, axis=-1, output_type=y_true.dtype)
    matches = tf.cast(tf.equal(y_true, predictions), tf.float32)
    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    return tf.reduce_sum(matches * mask) / tf.maximum(tf.reduce_sum(mask), 1.0)


def save_model(model: keras.Model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save(path)


def load_model(path: Path) -> keras.Model:
    if not path.exists():
        raise FileNotFoundError(f"Caption model not found at {path}")
    return keras.models.load_model(path, custom_objects={
        "masked_sparse_categorical_crossentropy": masked_sparse_categorical_crossentropy,
        "masked_accuracy": masked_accuracy,
    })
