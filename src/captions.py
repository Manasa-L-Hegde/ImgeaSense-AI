import re
from collections.abc import Iterable

from tensorflow.keras.preprocessing.text import Tokenizer


START_TOKEN = "<start>"
END_TOKEN = "<end>"


def clean_caption(caption: str) -> str:
    """Normalize a caption while preserving the words needed for training."""
    caption = caption.lower()
    caption = re.sub(r"[^a-z\s]", " ", caption)
    caption = re.sub(r"\s+", " ", caption).strip()
    return f"{START_TOKEN} {caption} {END_TOKEN}"


def prepare_captions(captions: Iterable[str]) -> list[str]:
    return [clean_caption(caption) for caption in captions if caption and caption.strip()]


def build_tokenizer(captions: Iterable[str]) -> Tokenizer:
    tokenizer = Tokenizer(
        num_words=None,
        filters="",
        lower=False,
        oov_token="<unk>",
    )
    tokenizer.fit_on_texts(prepare_captions(captions))
    return tokenizer


def caption_from_ids(tokenizer: Tokenizer, token_ids: Iterable[int]) -> str:
    words: list[str] = []
    for token_id in token_ids:
        word = tokenizer.index_word.get(int(token_id), "")
        if word in ("", START_TOKEN):
            continue
        if word == END_TOKEN:
            break
        words.append(word)
    return " ".join(words).strip().capitalize()
