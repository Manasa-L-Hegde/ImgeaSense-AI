from src.captions import END_TOKEN, START_TOKEN, build_tokenizer, caption_from_ids, clean_caption


def test_clean_caption_adds_sequence_tokens_and_normalizes_text():
    assert clean_caption("A DOG, running!") == f"{START_TOKEN} a dog running {END_TOKEN}"


def test_caption_from_ids_stops_at_end_token():
    tokenizer = build_tokenizer(["A dog runs"])
    ids = tokenizer.texts_to_sequences([f"{START_TOKEN} a dog {END_TOKEN}"])[0]
    assert caption_from_ids(tokenizer, ids) == "A dog"
