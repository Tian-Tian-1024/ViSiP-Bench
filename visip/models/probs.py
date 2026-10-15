"""Turn token-level scores into a distribution over option letters.

The paper locates the answer token in the model's response and reads the probability distribution over
the answer choices at that position. Here:

  1. `char_to_token_index` maps the character position of the parsed letter to a generated-token index.
  2. `letter_variant_tokens` finds, for every option letter, the token that would have been generated
     in its place (e.g. if the model wrote " B", the variants are " A", " B", " C", " D").
  3. `option_distribution` reads those tokens' probabilities and renormalizes over the options.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def char_to_token_index(prefix_lens: Sequence[int], char_pos: int) -> Optional[int]:
    """Index of the generated token whose decoded text covers character `char_pos`.

    Args:
        prefix_lens: prefix_lens[i] = len(decode(tokens[: i + 1])). Using prefix decodes (rather than
            decoding each token alone) keeps positions correct for multi-byte characters.
    """
    for i, n in enumerate(prefix_lens):
        if n > char_pos:
            return i
    return None


def _replace_single(raw: str, old: str, new: str) -> Optional[str]:
    if raw.count(old) != 1:
        return None
    return raw.replace(old, new)


def letter_variant_tokens(
    chosen_raw_token: str,
    chosen_letter: str,
    letters: Sequence[str],
    token_to_id,  # callable: raw token string -> id or None
) -> Optional[Dict[str, List[int]]]:
    """Map each letter to the token id obtained by swapping the letter inside the chosen token.

    Returns None if any variant is not a single vocabulary token; callers then fall back to bare letters.
    """
    out: Dict[str, List[int]] = {}
    for letter in letters:
        raw = _replace_single(chosen_raw_token, chosen_letter, letter)
        if raw is None:
            return None
        tid = token_to_id(raw)
        if tid is None:
            return None
        out[letter] = [tid]
    if len({ids[0] for ids in out.values()}) != len(letters):
        return None
    return out


def bare_letter_tokens(letters: Sequence[str], encode) -> Dict[str, List[int]]:
    """Fallback: single-token encodings of "A" and " A" (both are counted for each letter)."""
    out: Dict[str, List[int]] = {}
    for letter in letters:
        ids = []
        for variant in (letter, " " + letter):
            enc = encode(variant)
            if len(enc) == 1 and enc[0] not in ids:
                ids.append(enc[0])
        out[letter] = ids
    return out


def _logsumexp(values: Iterable[float]) -> float:
    values = list(values)
    m = max(values)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(v - m) for v in values))


def option_distribution(
    logprob_of: Dict[int, float], letter_tokens: Dict[str, List[int]]
) -> Tuple[Optional[Dict[str, float]], float]:
    """Renormalize token log-probabilities over the option letters.

    Args:
        logprob_of: token id -> log-probability under the full-vocabulary softmax.
        letter_tokens: letter -> candidate token ids (their probabilities are summed).
    Returns:
        (letter -> probability summing to 1, total raw probability mass on the options).
    """
    letter_lp = {}
    for letter, ids in letter_tokens.items():
        lps = [logprob_of.get(t, -math.inf) for t in ids]
        letter_lp[letter] = _logsumexp(lps) if lps else -math.inf
    total = _logsumexp(letter_lp.values())
    if total == -math.inf:
        return None, 0.0
    probs = {k: math.exp(v - total) for k, v in letter_lp.items()}
    return probs, math.exp(total)
