#!/usr/bin/env python
"""ViSiP-Bench evaluation entry point.

Three sub-commands:

  run      Query a model under the Blank and Visual conditions, then score it.
  score    Re-compute metrics from saved predictions (no model needed).
  compare  Build a leaderboard and the cross-model prior agreement matrix from several result folders.

Examples:
  # Open-source model (PAR / Acc / FC / VIG), 3 runs averaged as in the paper
  python evaluate.py run --model qwen3-vl-8b --data data/ViSiP-Bench --output results --num-runs 3

  # Local fine-tuned checkpoint (e.g. after MAPS)
  python evaluate.py run --model hf:/ckpt/qwen3-vl-8b-maps --name qwen3-vl-8b-maps --data data/ViSiP-Bench

  # Proprietary model (PAR / Acc / FC), 8 parallel requests
  python evaluate.py run --model gpt-5.5 --data data/ViSiP-Bench --num-workers 8

  # Any OpenAI-compatible server
  python evaluate.py run --model openai_compatible:Qwen/Qwen3-VL-8B-Instruct \
      --model-args '{"base_url": "http://localhost:8000/v1"}' --data data/ViSiP-Bench

  # Re-score and compare
  python evaluate.py score   --results results/qwen3-vl-8b --data data/ViSiP-Bench
  python evaluate.py compare --results results/* --output results/leaderboard.md
"""

from __future__ import annotations

import argparse
import datetime as _dt
import glob
import os
import sys
from typing import Any, Dict, List, Optional

from visip import CATEGORIES, CONDITIONS, __version__
from visip.data import load_dataset
from visip.metrics import DEFAULT_EPS, aggregate_runs, compute_metrics, compute_per_category, prior_agreement
from visip.models import PRESETS, build_model, parse_model_args, resolve
from visip.prompt import load_template
from visip.report import (agreement_table, category_table, diagnostics_text, leaderboard_table, overall_table)
from visip.runner import PRED_FILE, load_items, run_dir, run_inference
from visip.utils import logger, read_json, read_jsonl, set_seed, setup_logging, write_json

CONFIG_FILE = "config.json"
METRICS_FILE = "metrics.json"


# ----------------------------------------------------------------------------- scoring
def _run_ids(result_dir: str) -> List[int]:
    ids = []
    for path in glob.glob(os.path.join(result_dir, "run*", PRED_FILE)):
        name = os.path.basename(os.path.dirname(path))
        if name[3:].isdigit():
            ids.append(int(name[3:]))
    return sorted(ids)


def score(result_dir: str, samples=None, eps: float = DEFAULT_EPS, vig_subset: str = "filtered",
          model_name: Optional[str] = None) -> Dict[str, Any]:
    """Score every run in `result_dir`, write metrics.json, and print a summary."""
    runs = _run_ids(result_dir)
    if not runs:
        raise FileNotFoundError(f"No run*/{PRED_FILE} found in {result_dir}")
    cfg_path = os.path.join(result_dir, CONFIG_FILE)
    cfg = read_json(cfg_path) if os.path.exists(cfg_path) else {}
    model_name = model_name or cfg.get("name") or os.path.basename(os.path.normpath(result_dir))

    per_run, per_run_cat = [], []
    for r in runs:
        items, missing = load_items(os.path.join(run_dir(result_dir, r), PRED_FILE), samples)
        if missing:
            logger.warning("run%d: %d samples lack a Blank or Visual prediction and are skipped", r, missing)
        m = compute_metrics(items, eps, vig_subset)
        m["run"], m["n_incomplete"] = r, missing
        per_run.append(m)
        per_run_cat.append(compute_per_category(items, CATEGORIES, eps, vig_subset))

    agg = aggregate_runs(per_run)
    cats = list(dict.fromkeys(c for pc in per_run_cat for c in pc))
    agg_cat = {c: aggregate_runs([pc[c] for pc in per_run_cat if c in pc]) for c in cats}

    result = {
        "model": model_name,
        "runs": runs,
        "aggregate": agg,
        "aggregate_per_category": agg_cat,
        "per_run": per_run,
        "per_run_per_category": per_run_cat,
        "settings": {"eps": eps, "vig_subset": vig_subset},
        "scored_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "toolkit_version": __version__,
    }
    write_json(os.path.join(result_dir, METRICS_FILE), result)

    print(f"\n=== {model_name}  ({len(runs)} run{'s' if len(runs) > 1 else ''}) ===")
    print(overall_table(model_name, agg))
    print("\nPer category (run %d):" % runs[0])
    print(category_table(per_run_cat[0]))
    print("\nDiagnostics (run %d):" % runs[0])
    print(diagnostics_text(per_run[0]))
    print(f"\nSaved to {os.path.join(result_dir, METRICS_FILE)}")
    return result


# ----------------------------------------------------------------------------- sub-commands
def cmd_run(args: argparse.Namespace) -> None:
    model_args = parse_model_args(args.model_args)
    display, backend, _ = resolve(args.model)
    name = args.name or display
    out_dir = os.path.join(args.output, name)
    os.makedirs(out_dir, exist_ok=True)

    samples = load_dataset(args.data, args.image_root, args.categories, args.limit)
    template = load_template(args.prompt_template)

    config = {
        "name": name,
        "model": args.model,
        "backend": backend,
        "model_args": model_args,
        "data": os.path.abspath(args.data),
        "n_samples": len(samples),
        "categories": args.categories,
        "limit": args.limit,
        "prompt_template": args.prompt_template,
        "num_runs": args.num_runs,
        "seed": args.seed,
        "started_at": _dt.datetime.now().isoformat(timespec="seconds"),
        "toolkit_version": __version__,
        "command": " ".join(sys.argv),
    }
    cfg_path = os.path.join(out_dir, CONFIG_FILE)
    if os.path.exists(cfg_path):
        old = read_json(cfg_path)
        for key in ("model", "model_args", "prompt_template"):
            if old.get(key) != config[key] and not args.overwrite_config:
                raise SystemExit(
                    f"{out_dir} was created with a different {key!r} ({old.get(key)!r} vs {config[key]!r}). "
                    f"Use a different --name, or pass --overwrite-config to continue anyway."
                )
    write_json(cfg_path, config)

    model = build_model(args.model, model_args, name)
    if not model.supports_probs:
        logger.info("%s does not expose option probabilities: VIG will not be reported.", name)
    try:
        for r in range(1, args.num_runs + 1):
            set_seed(args.seed + r - 1)
            if hasattr(model, "seed"):
                model.seed = args.seed + r - 1
            run_inference(model, samples, out_dir, r, args.conditions, template, args.num_workers, args.max_errors)
    finally:
        model.close()

    score(out_dir, samples, args.eps, args.vig_subset, name)


def cmd_score(args: argparse.Namespace) -> None:
    samples = load_dataset(args.data, args.image_root, args.categories, check_images=False) if args.data else None
    for result_dir in args.results:
        score(result_dir, samples, args.eps, args.vig_subset)


def cmd_compare(args: argparse.Namespace) -> None:
    dirs = [d for d in args.results if os.path.isdir(d) and _run_ids(d)]
    if not dirs:
        raise SystemExit("No result folders with predictions found.")
    rows, blank_answers = [], {}
    for d in dirs:
        mpath = os.path.join(d, METRICS_FILE)
        res = read_json(mpath) if os.path.exists(mpath) and not args.rescore else score(d, eps=args.eps)
        rows.append(res)
        first = _run_ids(d)[0]
        recs = read_jsonl(os.path.join(run_dir(d, first), PRED_FILE))
        blank_answers[res["model"]] = {r["id"]: r["pred"] for r in recs if r["condition"] == "blank"}

    key = {"PAR": lambda r: (r["aggregate"]["PAR"]["mean"] is None, r["aggregate"]["PAR"]["mean"] or 0.0),
           "Acc": lambda r: (r["aggregate"]["Acc"]["mean"] is None, -(r["aggregate"]["Acc"]["mean"] or 0.0)),
           "none": None}[args.sort_by]
    if key:
        rows.sort(key=key)

    board = leaderboard_table(rows)
    agreement = prior_agreement(blank_answers)
    text = "## ViSiP-Bench Leaderboard\n\n" + board + "\n\n## Cross-model Prior Agreement (Blank condition, run 1)\n\n" \
           + agreement_table(agreement) + "\n"
    print("\n" + text)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        write_json(os.path.splitext(args.output)[0] + ".json",
                   {"leaderboard": [{"model": r["model"], "aggregate": r["aggregate"]} for r in rows],
                    "prior_agreement": agreement})
        print(f"Saved to {args.output}")


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ViSiP-Bench evaluation",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    # --log-level is accepted both before and after the sub-command
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--log-level", default=argparse.SUPPRESS, help="DEBUG / INFO / WARNING (default: INFO)")
    p.add_argument("--log-level", default="INFO")
    sub = p.add_subparsers(dest="command", required=True)

    def add_metric_args(sp):
        sp.add_argument("--eps", type=float, default=DEFAULT_EPS, help="VIG epsilon (paper: 1e-7)")
        sp.add_argument("--vig-subset", choices=["filtered", "all"], default="filtered",
                        help="items used for VIG: the filtered conflict set (default) or all items")

    def add_data_args(sp, required: bool):
        sp.add_argument("--data", required=required, help="annotation file, or folder with one .jsonl/.json")
        sp.add_argument("--image-root", default=None, help="base folder for relative image paths")
        sp.add_argument("--categories", nargs="+", default=None, choices=list(CATEGORIES), metavar="CATEGORY",
                        help="evaluate only these categories")

    r = sub.add_parser("run", help="run inference and score", parents=[common])
    r.add_argument("--model", required=True,
                   help=f"preset ({', '.join(sorted(PRESETS))}) or <backend>:<id>")
    r.add_argument("--model-args", default=None, help="JSON dict of backend arguments")
    r.add_argument("--name", default=None, help="name of the result folder (default: derived from --model)")
    add_data_args(r, required=True)
    r.add_argument("--output", default="results", help="results root folder")
    r.add_argument("--num-runs", type=int, default=1, help="repetitions to average (paper: 3)")
    r.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=list(CONDITIONS))
    r.add_argument("--prompt-template", default=None, help="text file with {question} and {options}")
    r.add_argument("--limit", type=int, default=None, help="only the first N samples (debugging)")
    r.add_argument("--num-workers", type=int, default=1, help="parallel requests (API backends only)")
    r.add_argument("--max-errors", type=int, default=50, help="abort after this many failed requests")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--overwrite-config", action="store_true",
                   help="allow resuming a folder created with different model settings")
    add_metric_args(r)
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("score", help="re-compute metrics from saved predictions", parents=[common])
    s.add_argument("--results", nargs="+", required=True, help="one or more result folders")
    add_data_args(s, required=False)
    add_metric_args(s)
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("compare", help="leaderboard + prior agreement matrix", parents=[common])
    c.add_argument("--results", nargs="+", required=True, help="result folders (globs are fine)")
    c.add_argument("--output", default=None, help="write a Markdown table here (plus a .json next to it)")
    c.add_argument("--sort-by", choices=["PAR", "Acc", "none"], default="PAR")
    c.add_argument("--rescore", action="store_true", help="ignore cached metrics.json")
    c.add_argument("--eps", type=float, default=DEFAULT_EPS)
    c.set_defaults(func=cmd_compare)
    return p


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level)
    if getattr(args, "num_runs", 1) < 1:
        raise SystemExit("--num-runs must be >= 1")
    args.func(args)


if __name__ == "__main__":
    main()
