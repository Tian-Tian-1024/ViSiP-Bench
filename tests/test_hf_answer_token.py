"""Checks HFModel's answer-token probability logic with a fake tokenizer and tensors (no torch needed)."""

import math

import pytest

from visip.models.hf import HFModel

# BPE-style vocabulary where "Ġ" marks a leading space
VOCAB = ["The", "Ġanswer", "Ġis", "ĠA", "ĠB", "ĠC", "ĠD", "A", "B", "C", "D", ".", "Ġ(", "(", ")", "Ġ110", "<eos>"]
ID = {t: i for i, t in enumerate(VOCAB)}


class FakeTokenizer:
    def convert_tokens_to_ids(self, tok):
        return ID.get(tok)

    def convert_ids_to_tokens(self, i):
        return VOCAB[i]

    def decode(self, ids, **_):
        return "".join(VOCAB[i] for i in ids if VOCAB[i] != "<eos>").replace("Ġ", " ")

    def encode(self, s, add_special_tokens=False):
        key = s.replace(" ", "Ġ")
        return [ID[key]] if key in ID else [ID[c] for c in s if c in ID]


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def __getitem__(self, i):
        return self.values[i]

    def float(self):
        return self


class FakeTorch:
    @staticmethod
    def log_softmax(t, dim=-1):
        m = max(t.values)
        z = m + math.log(sum(math.exp(v - m) for v in t.values))
        return FakeTensor([v - z for v in t.values])


def make_model():
    m = HFModel.__new__(HFModel)
    m.tokenizer = FakeTokenizer()
    m.torch = FakeTorch()
    m._bare_tokens_cache = {}
    return m


def step(logit_by_token):
    vals = [-20.0] * len(VOCAB)
    for tok, v in logit_by_token.items():
        vals[ID[tok]] = v
    return [FakeTensor(vals)]  # batch dimension


OPTS = {"A": "114", "B": "110", "C": "104", "D": "120"}


def test_reads_distribution_at_answer_token():
    model = make_model()
    gen = [ID["The"], ID["Ġanswer"], ID["Ġis"], ID["ĠB"], ID["."]]
    logits = [step({"The": 5}), step({"Ġanswer": 5}), step({"Ġis": 5}),
              step({"ĠA": math.log(0.3), "ĠB": math.log(0.6), "ĠC": math.log(0.05), "ĠD": math.log(0.05),
                    "B": 3.0}),  # bare "B" must NOT be mixed in: the variants of "ĠB" are "ĠA".."ĠD"
              step({".": 5})]
    probs, source, parsed, meta = model._option_probs(gen, logits, model._decode(gen), list(OPTS), OPTS)
    assert parsed.letter == "B" and source == "answer_token" and meta["answer_step"] == 3
    assert probs["B"] == pytest.approx(0.6) and probs["A"] == pytest.approx(0.3)


def test_bare_letter_answer():
    model = make_model()
    gen = [ID["B"], ID["<eos>"]]
    logits = [step({"A": 1.0, "B": 2.0, "C": 0.0, "D": 0.0}), step({"<eos>": 5})]
    probs, source, parsed, _ = model._option_probs(gen, logits, model._decode(gen), list(OPTS), OPTS)
    z = math.e ** 1 + math.e ** 2 + 2
    assert source == "answer_token" and probs["B"] == pytest.approx(math.e ** 2 / z)


def test_falls_back_to_first_token_when_letter_not_located():
    model = make_model()
    gen = [ID["Ġ110"], ID["<eos>"]]  # answered with the option text, no letter in the output
    logits = [step({"ĠA": 0.0, "ĠB": 1.0, "A": 0.0, "B": 1.0}), step({"<eos>": 5})]
    probs, source, parsed, _ = model._option_probs(gen, logits, model._decode(gen), list(OPTS), OPTS)
    assert parsed.letter == "B" and parsed.rule == "option_text"
    assert source == "first_token" and max(probs, key=probs.get) == "B"
    assert sum(probs.values()) == pytest.approx(1.0)
