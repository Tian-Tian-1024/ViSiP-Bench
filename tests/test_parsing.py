import pytest

from visip.parsing import parse_answer

L = ["A", "B", "C", "D"]
OPTS = {"A": "114", "B": "110", "C": "104", "D": "120"}


@pytest.mark.parametrize(
    "text, letter",
    [
        ("B", "B"),
        ("(B)", "B"),
        ("B.", "B"),
        ("**B**", "B"),
        ("B. 110", "B"),
        ("(B) The image shows 110.", "B"),
        ("The answer is B.", "B"),
        ("Answer: (C)", "C"),
        ("The correct option is **D**", "D"),
        ("I think A could be tempting, but the final answer is B.", "B"),
        ("Option A says 114. However, looking closely, the answer is (B).", "B"),
        ("答案是 C", "C"),
        ("我选 D。", "D"),
        ("<think>Maybe A? The answer is A.</think>\nB", "B"),
        ("The paper shows 110.", "B"),  # option text fallback
        ("I choose (A).", "A"),
    ],
)
def test_parse(text, letter):
    assert parse_answer(text, L, OPTS).letter == letter


@pytest.mark.parametrize("text", ["", "I cannot see the image.", "Either A or B is possible."])
def test_unparsable(text):
    # "A" as an English article must not be mistaken for an option
    assert parse_answer(text, L, OPTS).letter is None


def test_a_as_article_not_option():
    assert parse_answer("A boy is writing. The answer is C.", L, OPTS).letter == "C"
    assert parse_answer("A boy writes 110 on paper.", L, OPTS).letter == "B"


def test_char_pos_points_at_letter():
    for text in ["The answer is B.", "  (C) because", "Answer: D", "<think>A</think> B"]:
        p = parse_answer(text, L, OPTS)
        assert p.char_pos is not None and text[p.char_pos] == p.letter
