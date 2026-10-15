"""Run Blank + Visual inference and turn predictions into paired items.

Prediction file: <output_dir>/run{k}/predictions.jsonl, one line per (sample, condition):

    {"id": ..., "category": ..., "condition": "blank" | "visual", "run": k,
     "gt": "B", "pred": "A" | null, "parse_rule": "statement", "response": "...",
     "option_probs": {"A": 0.91, ...} | null, "prob_source": "answer_token" | ..., "meta": {...}}

Lines are appended as soon as they are produced, so an interrupted run resumes where it stopped.
Failed requests are not written, so they are retried on the next launch.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Sequence, Tuple

from . import CONDITIONS
from .data import Sample
from .metrics import ItemResult
from .models.base import BaseModel
from .parsing import parse_answer
from .prompt import build_prompt
from .utils import append_jsonl, logger, progress, read_jsonl

PRED_FILE = "predictions.jsonl"


def run_dir(output_dir: str, run: int) -> str:
    return os.path.join(output_dir, f"run{run}")


def _terminate_partial_line(path: str) -> None:
    """If a crash left a half-written last line, end it so new records start on a fresh line."""
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return
    with open(path, "rb+") as f:
        f.seek(-1, os.SEEK_END)
        if f.read(1) != b"\n":
            f.write(b"\n")


def _load_done(path: str) -> Dict[Tuple[str, str], dict]:
    return {(r["id"], r["condition"]): r for r in read_jsonl(path)}


def _predict_one(model: BaseModel, sample: Sample, condition: str, run: int, template: Optional[str]) -> dict:
    prompt = build_prompt(sample, template)
    image = sample.image_path if condition == "visual" else None
    if condition == "visual" and image is None:
        raise ValueError(f"[{sample.id}] has no image for the Visual condition")
    out = model.predict(prompt, image, sample.options)
    parsed = parse_answer(out.response, sample.letters, sample.options)
    return {
        "id": sample.id,
        "category": sample.category,
        "condition": condition,
        "run": run,
        "gt": sample.answer,
        "pred": parsed.letter,
        "parse_rule": parsed.rule,
        "response": out.response,
        "option_probs": out.option_probs,
        "prob_source": out.prob_source,
        "meta": out.meta,
    }


def run_inference(
    model: BaseModel,
    samples: Sequence[Sample],
    output_dir: str,
    run: int,
    conditions: Sequence[str] = CONDITIONS,
    template: Optional[str] = None,
    num_workers: int = 1,
    max_errors: int = 50,
) -> str:
    """Run (or resume) one repetition. Returns the predictions file path."""
    rdir = run_dir(output_dir, run)
    os.makedirs(rdir, exist_ok=True)
    pred_path = os.path.join(rdir, PRED_FILE)
    done = _load_done(pred_path)

    todo = [(s, c) for s in samples for c in conditions if (s.id, c) not in done]
    logger.info("Run %d: %d done, %d to go (%s)", run, len(done), len(todo), ", ".join(conditions))
    if not todo:
        return pred_path

    _terminate_partial_line(pred_path)
    lock = threading.Lock()
    errors: List[str] = []

    def work(job):
        sample, cond = job
        try:
            rec = _predict_one(model, sample, cond, run, template)
        except Exception as e:  # noqa: BLE001 - keep going; failed items are retried on resume
            with lock:
                errors.append(f"{sample.id}/{cond}: {type(e).__name__}: {e}")
                logger.error("Failed %s/%s: %s", sample.id, cond, e)
                if len(errors) >= max_errors:
                    raise RuntimeError(f"Stopping after {len(errors)} errors; last: {errors[-1]}") from e
            return
        with lock:
            append_jsonl(pred_path, rec)

    workers = num_workers if model.thread_safe else 1
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, job) for job in todo]
            for fut in progress(as_completed(futures), total=len(futures), desc=f"{model.name} run{run}"):
                fut.result()
    else:
        for job in progress(todo, total=len(todo), desc=f"{model.name} run{run}"):
            work(job)

    if errors:
        logger.warning("Run %d finished with %d failed requests; rerun the same command to retry them.", run, len(errors))
    return pred_path


def load_items(pred_path: str, samples: Optional[Sequence[Sample]] = None) -> Tuple[List[ItemResult], int]:
    """Pair Blank and Visual records. Returns (items, number of samples missing a condition).

    If `samples` is given, ground truth and category come from the dataset (so a fixed annotation is
    picked up without rerunning inference) and only those samples are scored.
    """
    by_id: Dict[str, Dict[str, dict]] = {}
    for rec in read_jsonl(pred_path):
        by_id.setdefault(rec["id"], {})[rec["condition"]] = rec

    if samples is not None:
        ids = [s.id for s in samples]
        gt = {s.id: (s.answer, s.category) for s in samples}
    else:
        ids = list(by_id)
        gt = {}

    items: List[ItemResult] = []
    missing = 0
    for sid in ids:
        recs = by_id.get(sid, {})
        if "blank" not in recs or "visual" not in recs:
            missing += 1
            continue
        b, v = recs["blank"], recs["visual"]
        answer, category = gt.get(sid, (v["gt"], v["category"]))
        items.append(
            ItemResult(
                id=sid,
                category=category,
                gt=answer,
                blank_pred=b["pred"],
                visual_pred=v["pred"],
                blank_probs=b.get("option_probs"),
                visual_probs=v.get("option_probs"),
            )
        )
    return items, missing
