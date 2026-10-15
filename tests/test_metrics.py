import math

import pytest

from visip.metrics import (ItemResult, aggregate_runs, compute_metrics, compute_per_category, log_odds,
                           prior_agreement, vig_histogram, vig_score)


def item(i, gt, b, v, bp=None, vp=None, cat="Physical Properties"):
    return ItemResult(str(i), cat, gt, b, v, bp, vp)


def test_par_acc_fc_by_hand():
    items = [
        item(1, "B", "A", "A"),  # conflict, adhered
        item(2, "B", "A", "B"),  # conflict, correct
        item(3, "B", "C", "D"),  # conflict, neither
        item(4, "B", "B", "A"),  # blank correct -> filtered (FC)
        item(5, "B", "B", "B"),  # blank correct -> filtered (FC)
    ]
    m = compute_metrics(items)
    assert m["n_conflict"] == 3
    assert m["FC"] == 2
    assert m["PAR"] == pytest.approx(100 / 3)
    assert m["Acc"] == pytest.approx(100 / 3)
    assert m["standard_acc"] == pytest.approx(40.0)  # 2 of 5 visual answers are B
    assert m["blank_acc"] == pytest.approx(40.0)


def test_invalid_answers():
    items = [
        item(1, "B", None, "B"),  # blank unparsable -> excluded
        item(2, "B", "A", None),  # visual unparsable -> not adhered, not correct
        item(3, "B", "A", "A"),
    ]
    m = compute_metrics(items)
    assert m["n_invalid_blank"] == 1 and m["n_invalid_visual"] == 1
    assert m["n_conflict"] == 2
    assert m["PAR"] == pytest.approx(50.0)
    assert m["Acc"] == pytest.approx(0.0)


def test_vig_formula_and_range():
    eps = 1e-7
    assert vig_score(0.5, 0.5) == pytest.approx(0.0)
    expected = math.log(0.9 / 0.1) - math.log(0.2 / 0.8)
    assert vig_score(0.9, 0.2, eps=0.0) == pytest.approx(expected)
    # theoretical range of about (-33, 33) with eps = 1e-7 (paper, Sec. 3.3)
    hi = vig_score(1.0, 0.0, eps)
    assert 32 < hi < 33 and vig_score(0.0, 1.0, eps) == pytest.approx(-hi)
    assert log_odds(0.0, eps) == pytest.approx(math.log(eps / (1 + eps)))


def test_vig_subset_and_missing_probs():
    probs_b = {"A": 0.8, "B": 0.1, "C": 0.05, "D": 0.05}
    probs_v = {"A": 0.1, "B": 0.8, "C": 0.05, "D": 0.05}
    items = [
        item(1, "B", "A", "B", probs_b, probs_v),
        item(2, "B", "B", "B", probs_v, probs_v),  # filtered out of the conflict set, VIG = 0
        item(3, "B", "A", "A"),  # no probabilities
    ]
    filt = compute_metrics(items, vig_subset="filtered")
    allm = compute_metrics(items, vig_subset="all")
    v1 = vig_score(0.8, 0.1)
    assert filt["n_vig"] == 1 and filt["VIG_avg"] == pytest.approx(v1)
    assert allm["n_vig"] == 2 and allm["VIG_avg"] == pytest.approx(v1 / 2)
    assert compute_metrics([item(1, "B", "A", "A")])["VIG_avg"] is None


def test_empty_and_no_conflict():
    m = compute_metrics([])
    assert m["PAR"] is None and m["Acc"] is None and m["FC"] == 0
    m = compute_metrics([item(1, "A", "A", "A")])
    assert m["PAR"] is None and m["FC"] == 1


def test_histogram_bins():
    h = vig_histogram([-5, 7.99, 8, 15, 16, 30])
    assert h == {"(-33, 8)": pytest.approx(100 / 3), "[8, 16)": pytest.approx(100 / 3),
                 "[16, 24)": pytest.approx(100 / 6), "[24, +33)": pytest.approx(100 / 6)}


def test_per_category_and_aggregate():
    items = [item(1, "B", "A", "A", cat="Time & Space"), item(2, "B", "A", "B", cat="Physical Properties")]
    pc = compute_per_category(items, ["Physical Properties", "Time & Space", "Function Cognition"])
    assert list(pc) == ["Physical Properties", "Time & Space"]
    assert pc["Time & Space"]["PAR"] == 100.0
    agg = aggregate_runs([{"PAR": 40.0, "Acc": 50.0}, {"PAR": 44.0, "Acc": None}])
    assert agg["PAR"]["mean"] == pytest.approx(42.0) and agg["PAR"]["std"] == pytest.approx(math.sqrt(8))
    assert agg["Acc"]["n_runs"] == 1 and agg["Acc"]["std"] == 0.0
    assert agg["VIG_avg"]["mean"] is None


def test_prior_agreement():
    ans = {"m1": {"1": "A", "2": "B", "3": None}, "m2": {"1": "A", "2": "C", "3": "D"}}
    mat = prior_agreement(ans)
    assert mat["m1"]["m1"] == 1.0
    assert mat["m1"]["m2"] == pytest.approx(0.5) and mat["m2"]["m1"] == pytest.approx(0.5)
