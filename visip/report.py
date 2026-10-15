"""Human-readable summaries (console / Markdown tables)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence


def fmt(x: Optional[float], digits: int = 2) -> str:
    if x is None:
        return "–"
    if isinstance(x, int):
        return str(x)
    return f"{x:.{digits}f}"


def _digits(key: str) -> int:
    # FC is a count: print "147" (or "147.5" when averaged over runs) rather than "147.00".
    return 1 if key == "FC" else 2


def fmt_mean_std(stat: Mapping[str, Any], digits: int = 2) -> str:
    if stat.get("mean") is None:
        return "–"
    s = fmt(stat["mean"], digits)
    if stat.get("n_runs", 0) > 1 and stat.get("std"):
        s += f" ± {fmt(stat['std'], digits)}"
    return s


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join([":--"] + [":--:"] * (len(headers) - 1)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def overall_table(model_name: str, aggregate: Mapping[str, Mapping[str, Any]]) -> str:
    headers = ["Model", "PAR ↓", "Acc ↑", "FC", "VIG_avg ↑", "VIG_med ↑"]
    row = [model_name] + [fmt_mean_std(aggregate[k], _digits(k)) for k in ("PAR", "Acc", "FC", "VIG_avg", "VIG_med")]
    return markdown_table(headers, [row])


def category_table(per_category: Mapping[str, Mapping[str, Any]]) -> str:
    headers = ["Category", "PAR ↓", "Acc ↑", "FC", "VIG_avg ↑", "#Conflict"]
    rows = []
    for cat, m in per_category.items():
        rows.append([cat, fmt(m["PAR"]), fmt(m["Acc"]), str(m["FC"]), fmt(m["VIG_avg"]), str(m["n_conflict"])])
    return markdown_table(headers, rows)


def diagnostics_text(m: Mapping[str, Any]) -> str:
    lines = [
        f"samples: {m['n_total']}  |  conflict set (a_B != a*): {m['n_conflict']}  |  FC: {m['FC']}",
        f"unparsable answers: blank={m['n_invalid_blank']}, visual={m['n_invalid_visual']}",
        f"standard acc (all items, visual): {fmt(m['standard_acc'])}  |  blank acc: {fmt(m['blank_acc'])}",
    ]
    if m.get("VIG_hist"):
        hist = ", ".join(f"{k}: {v:.1f}%" for k, v in m["VIG_hist"].items())
        lines.append(f"VIG over {m['n_vig']} items ({m['vig_subset']} set): {hist}")
    else:
        lines.append("VIG: not available (backend does not expose option probabilities)")
    return "\n".join(lines)


def leaderboard_table(rows: Sequence[Dict[str, Any]]) -> str:
    headers = ["Model", "PAR ↓", "Acc ↑", "FC", "VIG_avg ↑", "VIG_med ↑", "Runs"]
    body = []
    for r in rows:
        agg = r["aggregate"]
        body.append([r["model"]] + [fmt(agg[k]["mean"], _digits(k)) for k in ("PAR", "Acc", "FC", "VIG_avg", "VIG_med")]
                    + [str(agg["PAR"]["n_runs"])])
    return markdown_table(headers, body)


def agreement_table(matrix: Mapping[str, Mapping[str, Optional[float]]]) -> str:
    names = list(matrix)
    rows = [[a] + [fmt(matrix[a][b]) for b in names] for a in names]
    return markdown_table([""] + names, rows)
