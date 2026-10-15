"""ViSiP-Bench metrics (paper, Sec. 3.2).

Notation (per sample i):
    a_B  answer under the Blank condition (no image)    p_B  option distribution under Blank
    a_V  answer under the Visual condition (with image)  p_V  option distribution under Visual
    a*   ground-truth visual answer

    PAR  = sum_{i: a_B != a*} 1[a_V == a_B] / sum_i 1[a_B != a*]                       (Eq. 1, lower is better)
    Acc  = accuracy of a_V on items with a_B != a*  (Filtered Accuracy, higher is better)
    FC   = number of items with a_B == a*  (Filtered Count: answered correctly without the image)
    VIG  = logit(p_V(a*)) - logit(p_B(a*)),  logit(p) = ln((p + eps) / (1 - p + eps))      (Eq. 2, higher is better)

Edge cases (not specified in the paper; chosen here and reported explicitly):
    * Blank answer could not be parsed (a_B is None): the item cannot be assigned to the conflict set,
      so it is excluded from PAR / Acc / FC and counted in `n_invalid_blank`.
    * Visual answer could not be parsed (a_V is None): counted as "not adhering" for PAR and
      "wrong" for Acc, and reported in `n_invalid_visual`.
    * VIG is computed only for items where both p_B and p_V exist. By default it is averaged over the
      filtered (conflict) set, the same set used for PAR / Acc; `vig_subset="all"` uses every item.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

DEFAULT_EPS = 1e-7
# Paper, Fig. 5a: VIG histogram bins.
VIG_BINS = ((-math.inf, 8.0), (8.0, 16.0), (16.0, 24.0), (24.0, math.inf))
VIG_BIN_LABELS = ("(-33, 8)", "[8, 16)", "[16, 24)", "[24, +33)")
# Appendix A.1: a Blank-correct item whose max option probability is below this is "Chance Success".
CHANCE_SUCCESS_THRESHOLD = 0.65


@dataclass
class ItemResult:
    """Paired Blank/Visual outcome for one sample."""

    id: str
    category: str
    gt: str
    blank_pred: Optional[str]
    visual_pred: Optional[str]
    blank_probs: Optional[Dict[str, float]] = None
    visual_probs: Optional[Dict[str, float]] = None


def log_odds(p: float, eps: float = DEFAULT_EPS) -> float:
    return math.log((p + eps) / (1.0 - p + eps))


def vig_score(p_visual: float, p_blank: float, eps: float = DEFAULT_EPS) -> float:
    """Visual Information Gain for one item: log-odds shift of the ground-truth option (Eq. 2)."""
    return log_odds(p_visual, eps) - log_odds(p_blank, eps)


def _safe_div(a: float, b: float) -> Optional[float]:
    return a / b if b else None


def _pct(x: Optional[float]) -> Optional[float]:
    return None if x is None else 100.0 * x


def vig_histogram(values: Sequence[float]) -> Dict[str, float]:
    """Percentage of items in each VIG bin of Fig. 5a."""
    counts = {label: 0 for label in VIG_BIN_LABELS}
    for v in values:
        for (lo, hi), label in zip(VIG_BINS, VIG_BIN_LABELS):
            if lo <= v < hi:
                counts[label] += 1
                break
    n = len(values)
    return {k: (100.0 * c / n if n else 0.0) for k, c in counts.items()}


def compute_metrics(
    items: Sequence[ItemResult],
    eps: float = DEFAULT_EPS,
    vig_subset: str = "filtered",
) -> Dict[str, Any]:
    """Compute PAR / Filtered Acc / FC / VIG and diagnostics over a list of paired items."""
    if vig_subset not in ("filtered", "all"):
        raise ValueError("vig_subset must be 'filtered' or 'all'")

    n_total = len(items)
    valid = [it for it in items if it.blank_pred is not None]
    conflict = [it for it in valid if it.blank_pred != it.gt]  # a_B != a*
    fc = len(valid) - len(conflict)  # a_B == a*

    adhered = sum(1 for it in conflict if it.visual_pred is not None and it.visual_pred == it.blank_pred)
    correct = sum(1 for it in conflict if it.visual_pred == it.gt)

    # VIG
    vig_pool = conflict if vig_subset == "filtered" else items
    vig_values = [
        vig_score(it.visual_probs.get(it.gt, 0.0), it.blank_probs.get(it.gt, 0.0), eps)
        for it in vig_pool
        if it.blank_probs is not None and it.visual_probs is not None
    ]

    # Reference numbers (not used for the main table)
    std_acc = _safe_div(sum(1 for it in items if it.visual_pred == it.gt), n_total)
    blank_acc = _safe_div(sum(1 for it in valid if it.blank_pred == it.gt), len(valid))

    # Appendix A.1 automatic screening of Blank-correct items (Low-quality Framing needs human review)
    blank_correct = [it for it in valid if it.blank_pred == it.gt and it.blank_probs]
    chance = sum(1 for it in blank_correct if max(it.blank_probs.values()) < CHANCE_SUCCESS_THRESHOLD)

    return {
        "PAR": _pct(_safe_div(adhered, len(conflict))),
        "Acc": _pct(_safe_div(correct, len(conflict))),
        "FC": fc,
        "VIG_avg": statistics.fmean(vig_values) if vig_values else None,
        "VIG_med": statistics.median(vig_values) if vig_values else None,
        "VIG_hist": vig_histogram(vig_values) if vig_values else None,
        "n_total": n_total,
        "n_conflict": len(conflict),
        "n_vig": len(vig_values),
        "n_invalid_blank": n_total - len(valid),
        "n_invalid_visual": sum(1 for it in items if it.visual_pred is None),
        "standard_acc": _pct(std_acc),
        "blank_acc": _pct(blank_acc),
        "blank_correct_screening": {
            "n_blank_correct_with_probs": len(blank_correct),
            "chance_success": chance,
            "strong_prior_candidates": len(blank_correct) - chance,
        },
        "vig_subset": vig_subset,
    }


def compute_per_category(
    items: Sequence[ItemResult],
    categories: Optional[Iterable[str]] = None,
    eps: float = DEFAULT_EPS,
    vig_subset: str = "filtered",
) -> Dict[str, Dict[str, Any]]:
    by_cat: Dict[str, List[ItemResult]] = {}
    for it in items:
        by_cat.setdefault(it.category, []).append(it)
    order = list(categories) if categories else []
    order += sorted(c for c in by_cat if c not in order)
    return {c: compute_metrics(by_cat[c], eps, vig_subset) for c in order if c in by_cat}


SCALAR_KEYS = ("PAR", "Acc", "FC", "VIG_avg", "VIG_med", "standard_acc", "blank_acc")


def aggregate_runs(per_run: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    """Mean and sample std of scalar metrics across repeated runs (paper: 3 runs, averaged)."""
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for key in SCALAR_KEYS:
        vals = [r[key] for r in per_run if r.get(key) is not None]
        out[key] = {
            "mean": statistics.fmean(vals) if vals else None,
            "std": statistics.stdev(vals) if len(vals) > 1 else (0.0 if vals else None),
            "n_runs": len(vals),
        }
    return out


def prior_agreement(blank_answers: Mapping[str, Mapping[str, Optional[str]]]) -> Dict[str, Dict[str, Optional[float]]]:
    """Pairwise agreement of Blank-condition answers between models (paper, Fig. 4).

    Args:
        blank_answers: model name -> {sample id -> a_B}.
    Returns:
        model -> model -> fraction of shared, validly-answered items with identical a_B.
    """
    names = list(blank_answers)
    matrix: Dict[str, Dict[str, Optional[float]]] = {}
    for a in names:
        matrix[a] = {}
        for b in names:
            shared = [
                sid
                for sid, ans in blank_answers[a].items()
                if ans is not None and blank_answers[b].get(sid) is not None
            ]
            same = sum(1 for sid in shared if blank_answers[a][sid] == blank_answers[b][sid])
            matrix[a][b] = _safe_div(same, len(shared))
    return matrix
