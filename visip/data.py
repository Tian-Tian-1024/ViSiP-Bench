"""Dataset loading for ViSiP-Bench.

Expected annotation format (one JSON object per sample, in a .jsonl or a .json list):

    {
      "id": "sci_math_0001",
      "category": "Scientific & Mathematic",
      "image": "images/sci_math_0001.png",      # relative to the annotation file's folder
      "question": "What answer is written for 77+37 in the picture?",
      "options": {"A": "114", "B": "110", "C": "...", "D": "..."},
      "answer": "B"
    }

`options` may also be a list (["114", "110", ...]), in which case letters A, B, C, ... are assigned
in order. Any extra fields (e.g. "style", "image_source", "prior") are kept in `Sample.extra`.
"""

from __future__ import annotations

import glob
import json
import os
import string
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from . import CATEGORIES
from .utils import logger

REQUIRED_FIELDS = ("id", "question", "options", "answer")


@dataclass
class Sample:
    id: str
    question: str
    options: Dict[str, str]  # ordered letter -> option text
    answer: str  # ground-truth visual answer letter a*
    category: str = "Unknown"
    image_path: Optional[str] = None  # absolute path
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def letters(self) -> List[str]:
        return list(self.options.keys())


def _normalize_options(raw: Any, sample_id: str) -> Dict[str, str]:
    if isinstance(raw, dict):
        opts = {str(k).strip().upper(): str(v).strip() for k, v in raw.items()}
    elif isinstance(raw, (list, tuple)):
        if len(raw) > 26:
            raise ValueError(f"[{sample_id}] too many options ({len(raw)})")
        opts = {string.ascii_uppercase[i]: str(v).strip() for i, v in enumerate(raw)}
    else:
        raise ValueError(f"[{sample_id}] 'options' must be a dict or a list, got {type(raw).__name__}")
    for letter in opts:
        if len(letter) != 1 or letter not in string.ascii_uppercase:
            raise ValueError(f"[{sample_id}] invalid option key {letter!r}; expected single letters A-Z")
    if len(opts) < 2:
        raise ValueError(f"[{sample_id}] needs at least 2 options")
    return dict(sorted(opts.items()))


def _resolve_annotation_file(path: str) -> str:
    if os.path.isfile(path):
        return path
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Data path not found: {path}")
    candidates = sorted(glob.glob(os.path.join(path, "*.jsonl"))) or sorted(glob.glob(os.path.join(path, "*.json")))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .jsonl/.json annotation file in {path}, found {len(candidates)}: "
            f"{[os.path.basename(c) for c in candidates]}. Pass the annotation file path directly."
        )
    return candidates[0]


def _read_raw(ann_file: str) -> List[Dict[str, Any]]:
    with open(ann_file, "r", encoding="utf-8") as f:
        if ann_file.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        obj = json.load(f)
    if isinstance(obj, dict):
        for key in ("data", "samples", "annotations"):
            if isinstance(obj.get(key), list):
                return obj[key]
        raise ValueError(f"{ann_file}: JSON object must contain a 'data' list")
    if not isinstance(obj, list):
        raise ValueError(f"{ann_file}: expected a list of samples")
    return obj


def load_dataset(
    path: str,
    image_root: Optional[str] = None,
    categories: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    check_images: bool = True,
) -> List[Sample]:
    """Load and validate ViSiP-Bench samples.

    Args:
        path: annotation file, or a folder containing exactly one .jsonl/.json file.
        image_root: base folder for relative image paths (default: the annotation file's folder).
        categories: keep only these categories.
        limit: keep only the first N samples (after category filtering), for debugging.
        check_images: raise if an image file is missing.
    """
    ann_file = _resolve_annotation_file(path)
    root = image_root or os.path.dirname(os.path.abspath(ann_file))
    raw_items = _read_raw(ann_file)

    samples: List[Sample] = []
    seen_ids = set()
    unknown_categories = set()
    missing_images = []
    for idx, item in enumerate(raw_items):
        missing = [k for k in REQUIRED_FIELDS if k not in item]
        if missing:
            raise ValueError(f"Sample #{idx} is missing fields {missing}")
        sid = str(item["id"])
        if sid in seen_ids:
            raise ValueError(f"Duplicate sample id: {sid}")
        seen_ids.add(sid)

        options = _normalize_options(item["options"], sid)
        answer = str(item["answer"]).strip().upper()
        if answer not in options:
            raise ValueError(f"[{sid}] answer {answer!r} is not one of the option letters {list(options)}")

        image_path = None
        if item.get("image"):
            image_path = item["image"] if os.path.isabs(item["image"]) else os.path.join(root, item["image"])
            if check_images and not os.path.exists(image_path):
                missing_images.append(image_path)

        category = str(item.get("category", "Unknown"))
        if category not in CATEGORIES:
            unknown_categories.add(category)

        extra = {k: v for k, v in item.items() if k not in {"id", "question", "options", "answer", "category", "image"}}
        samples.append(
            Sample(
                id=sid,
                question=str(item["question"]).strip(),
                options=options,
                answer=answer,
                category=category,
                image_path=image_path,
                extra=extra,
            )
        )

    if missing_images:
        preview = "\n  ".join(missing_images[:5])
        raise FileNotFoundError(f"{len(missing_images)} image(s) not found, e.g.:\n  {preview}")
    if unknown_categories:
        logger.warning("Categories outside the six ViSiP-Bench categories: %s", sorted(unknown_categories))

    if categories:
        wanted = set(categories)
        samples = [s for s in samples if s.category in wanted]
    if limit is not None:
        samples = samples[:limit]

    logger.info("Loaded %d samples from %s", len(samples), ann_file)
    return samples
