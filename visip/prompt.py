"""Prompt construction (paper, Appendix F).

The SAME template is used for the Blank and the Visual condition, so any difference in the model's
response comes only from the presence or absence of the image.
"""

from __future__ import annotations

import re
from typing import Optional

from .data import Sample

DEFAULT_TEMPLATE = (
    "In this picture, {question}\n"
    "{options}\n"
    "If you cannot see the image, please make your best guess among all answer choices "
    "based on your own judgment."
)

_PREFIX_RE = re.compile(r"^\s*in\s+this\s+picture\s*,\s*", re.IGNORECASE)


def format_options(sample: Sample) -> str:
    return "\n".join(f"{letter}. {text}" for letter, text in sample.options.items())


def build_prompt(sample: Sample, template: Optional[str] = None) -> str:
    """Fill the template. If the question already starts with "In this picture," the prefix is not repeated."""
    template = template or DEFAULT_TEMPLATE
    question = sample.question
    if "In this picture, {question}" in template:
        question = _PREFIX_RE.sub("", question, count=1)
    return template.format(question=question, options=format_options(sample))


def load_template(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        template = f.read()
    for key in ("{question}", "{options}"):
        if key not in template:
            raise ValueError(f"Prompt template {path} must contain {key}")
    return template
