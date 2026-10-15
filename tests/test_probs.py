import math

import pytest

from visip.models.probs import bare_letter_tokens, char_to_token_index, letter_variant_tokens, option_distribution


def test_char_to_token_index():
    pieces = ["The", " answer", " is", " B", "."]
    prefix = [sum(len(p) for p in pieces[: i + 1]) for i in range(len(pieces))]
    text = "".join(pieces)
    assert char_to_token_index(prefix, text.index("B")) == 3
    assert char_to_token_index(prefix, 0) == 0
    assert char_to_token_index(prefix, len(text)) is None


def test_letter_variants_with_space_marker():
    vocab = {"ĠA": 10, "ĠB": 11, "ĠC": 12, "ĠD": 13, "A": 1, "B": 2}
    out = letter_variant_tokens("ĠB", "B", "ABCD", vocab.get)
    assert out == {"A": [10], "B": [11], "C": [12], "D": [13]}
    assert letter_variant_tokens("B", "B", "ABCD", vocab.get) is None  # "C" missing -> fallback


def test_bare_letter_tokens():
    enc = {"A": [1], " A": [10], "B": [2], " B": [11], "C": [3], " C": [5, 6]}
    out = bare_letter_tokens("ABC", lambda s: enc[s])
    assert out == {"A": [1, 10], "B": [2, 11], "C": [3]}


def test_option_distribution_renormalizes():
    lp = {1: math.log(0.6), 2: math.log(0.2), 3: math.log(0.1), 4: math.log(0.05)}
    probs, mass = option_distribution(lp, {"A": [1], "B": [2], "C": [3], "D": [4, 99]})
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["A"] == pytest.approx(0.6 / 0.95)
    assert mass == pytest.approx(0.95)
    probs, mass = option_distribution({}, {"A": [1]})
    assert probs is None and mass == 0.0
