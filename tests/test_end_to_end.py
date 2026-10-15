import json
import os

import pytest
from PIL import Image

import evaluate
from visip import CATEGORIES
from visip.data import load_dataset
from visip.prompt import build_prompt
from visip.runner import PRED_FILE
from visip.utils import read_json, read_jsonl


@pytest.fixture()
def bench(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    rows = []
    for i in range(24):
        name = f"images/{i:03d}.png"
        Image.new("RGB", (8, 8), (i * 10 % 255, 0, 0)).save(tmp_path / name)
        rows.append({"id": f"s{i:03d}", "category": CATEGORIES[i % 6], "image": name,
                     "question": f"In this picture, what is item {i}?",
                     "options": {"A": f"a{i}", "B": f"b{i}", "C": f"c{i}", "D": f"d{i}"},
                     "answer": "ABCD"[i % 4], "style": "photorealistic"})
    ann = tmp_path / "visip.jsonl"
    ann.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return tmp_path


def test_load_and_prompt(bench):
    samples = load_dataset(str(bench))
    assert len(samples) == 24 and samples[0].extra["style"] == "photorealistic"
    p = build_prompt(samples[0])
    assert p.startswith("In this picture, what is item 0?\nA. a0\nB. b0")
    assert "In this picture, In this picture" not in p
    assert p.rstrip().endswith("based on your own judgment.")
    assert len(load_dataset(str(bench), categories=["Time & Space"])) == 4


def test_dataset_validation(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"id": "x", "question": "q", "options": ["1", "2"], "answer": "C"}))
    with pytest.raises(ValueError, match="not one of the option letters"):
        load_dataset(str(bad))


def test_run_resume_score_compare(bench, tmp_path, capsys):
    out = tmp_path / "results"
    base = ["run", "--model", "dummy", "--data", str(bench), "--output", str(out), "--num-runs", "2",
            "--log-level", "WARNING"]
    evaluate.main(base)
    pred = out / "dummy" / "run1" / PRED_FILE
    recs = read_jsonl(str(pred))
    assert len(recs) == 48 and {r["condition"] for r in recs} == {"blank", "visual"}

    # Blank answer is identical across runs (deterministic prior)
    r2 = {(r["id"], r["condition"]): r for r in read_jsonl(str(out / "dummy" / "run2" / PRED_FILE))}
    assert all(r2[(r["id"], "blank")]["pred"] == r["pred"] for r in recs if r["condition"] == "blank")

    m = read_json(str(out / "dummy" / "metrics.json"))
    agg = m["aggregate"]
    assert agg["PAR"]["n_runs"] == 2 and 0 <= agg["PAR"]["mean"] <= 100
    assert agg["VIG_avg"]["mean"] is not None
    run1 = m["per_run"][0]
    assert run1["n_conflict"] + run1["FC"] + run1["n_invalid_blank"] == 24

    # Resume: drop the last 5 lines and a truncated line, rerun -> back to 48 complete records
    lines = pred.read_text(encoding="utf-8").splitlines()
    pred.write_text("\n".join(lines[:-5]) + "\n{\"id\": \"trunc", encoding="utf-8")
    evaluate.main(base)
    assert len(read_jsonl(str(pred))) == 48
    assert read_json(str(out / "dummy" / "metrics.json"))["aggregate"]["PAR"] == agg["PAR"]

    # Changing the model args on an existing folder is refused
    with pytest.raises(SystemExit):
        evaluate.main(base + ["--model-args", '{"visual_strength": 0.1}'])

    # A second model, then score + compare
    evaluate.main(["run", "--model", "dummy", "--name", "dummy-weak", "--model-args", '{"visual_strength": 0.1}',
                   "--data", str(bench), "--output", str(out), "--log-level", "WARNING"])
    evaluate.main(["score", "--results", str(out / "dummy"), "--data", str(bench), "--log-level", "WARNING"])
    board = tmp_path / "board.md"
    evaluate.main(["compare", "--results", str(out / "dummy"), str(out / "dummy-weak"), "--output", str(board),
                   "--log-level", "WARNING"])
    text = board.read_text(encoding="utf-8")
    assert "dummy-weak" in text and "Prior Agreement" in text
    agreement = read_json(os.path.splitext(str(board))[0] + ".json")["prior_agreement"]
    # Same prompts -> same hashed prior -> blank answers agree completely
    assert agreement["dummy"]["dummy-weak"] == pytest.approx(1.0)
    weak = read_json(str(out / "dummy-weak" / "metrics.json"))["aggregate"]["PAR"]["mean"]
    assert weak >= agg["PAR"]["mean"]  # weaker visual signal -> more prior adherence
