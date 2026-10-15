"""Extract the chosen option letter from a free-form model response.

`parse_answer` returns both the letter and its character position in the response. The position is
needed to locate the answer token in the generated sequence, where the option probability
distribution is read (paper, Sec. 3.2).

Rules are tried from most to least explicit:
  1. the whole response is just a letter        "B", "(B)", "B."
  2. an explicit answer statement (LAST one)     "The answer is B", "Answer: (B)", "答案是 B"
  3. the response starts with a letter          "B. 110 ...", "(B) because ..."
  4. a single distinct bracketed/punctuated letter anywhere   "... so I choose (B)."
  5. the response mentions exactly one option text            "110"  (no position available)
Reasoning blocks such as <think>...</think> are ignored.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

_THINK_END_RE = re.compile(r"</think>", re.IGNORECASE)


@dataclass
class ParsedAnswer:
    letter: Optional[str]
    char_pos: Optional[int]  # index of the letter in the ORIGINAL response, if known
    rule: str  # which rule matched ("none" if parsing failed)


def _visible_offset(text: str) -> int:
    """Start of the final answer area: after the last </think> if present."""
    last = None
    for m in _THINK_END_RE.finditer(text):
        last = m
    return last.end() if last else 0


def parse_answer(response: str, letters: Sequence[str], options: Optional[Dict[str, str]] = None) -> ParsedAnswer:
    if not response:
        return ParsedAnswer(None, None, "none")
    offset = _visible_offset(response)
    text = response[offset:]
    L = "".join(re.escape(x) for x in letters)  # e.g. "ABCD"

    # 1. bare letter
    m = re.fullmatch(rf"\s*[\(\[\*]*([{L}])[\)\]\*]*\s*[\.\:：。]?\s*", text)
    if m:
        return ParsedAnswer(m.group(1), offset + m.start(1), "bare")

    # 2. explicit answer statements; take the LAST one (CoT may mention several letters before concluding)
    answer_patterns = [
        rf"(?:final\s+answer|correct\s+answer|answer|correct\s+option|best\s+option|option|choice)"
        rf"\s*(?:is|would\s+be|should\s+be|will\s+be)?\s*(?:option|choice)?\s*[:：]?\s*\**\s*[\(\[]?([{L}])(?![A-Za-z])",
        rf"(?:I\s+(?:would\s+)?(?:choose|select|pick|go\s+with))\s*(?:option|choice)?\s*[\(\[]?([{L}])(?![A-Za-z])",
        rf"(?:答案|选项|正确答案|最终答案)\s*(?:是|为|应为|应该是)?\s*[:：]?\s*[\(（\[]?([{L}])",
        rf"选\s*[\(（\[]?([{L}])(?![A-Za-z])",
    ]
    last_match = None
    for pat in answer_patterns:
        for mm in re.finditer(pat, text, flags=re.IGNORECASE):
            letter = mm.group(1).upper()
            if letter in letters and (last_match is None or mm.start(1) > last_match[1]):
                last_match = (letter, mm.start(1))
    if last_match:
        return ParsedAnswer(last_match[0], offset + last_match[1], "statement")

    # 3. leading letter followed by a delimiter. A plain space is NOT a delimiter, so the English
    #    article in "A boy writes ..." is not read as option A.
    m = re.match(rf"\s*[\(\[\*]*([{L}])(?:[\)\]\.\:：、,，\*]|[ \t]*\n|$)", text)
    if m:
        return ParsedAnswer(m.group(1), offset + m.start(1), "leading")

    # 4. a single distinct letter written as an option marker anywhere: (B), [B], **B**, " B." / " B)"
    marker_re = rf"(?:[\(\[]([{L}])[\)\]])|(?:\*\*([{L}])\*\*)|(?:(?<![A-Za-z])([{L}])[\.\)](?=\s|$))"
    hits = []
    for mm in re.finditer(marker_re, text):
        idx = next(i for i in (1, 2, 3) if mm.group(i))
        hits.append((mm.group(idx), mm.start(idx)))
    if hits and len({h[0] for h in hits}) == 1:
        return ParsedAnswer(hits[-1][0], offset + hits[-1][1], "marker")

    # 5. option text match
    if options:
        low = text.lower()
        found = [k for k, v in options.items() if v and re.search(rf"(?<!\w){re.escape(v.lower())}(?!\w)", low)]
        if len(found) == 1:
            return ParsedAnswer(found[0], None, "option_text")

    return ParsedAnswer(None, None, "none")
