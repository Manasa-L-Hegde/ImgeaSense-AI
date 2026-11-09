import pytest

from src.evaluation import bleu_scores


def test_bleu_scores_reward_matching_captions():
    matching = bleu_scores([["a dog runs through grass"]], ["a dog runs through grass"])
    different = bleu_scores([["a dog runs through grass"]], ["a red car drives on road"])

    assert matching["bleu_1"] == 1.0
    assert matching["bleu_4"] > different["bleu_4"]


def test_bleu_scores_requires_aligned_inputs():
    with pytest.raises(ValueError, match="same number"):
        bleu_scores([["a dog"]], [])
