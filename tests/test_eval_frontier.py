"""Tests for the v1.3.0 honest-eval additions (D3): recall@FPR frontier,
bootstrap CIs, per-category support. Pure-function tests need no model; the
integration test skips if the runtime artifact isn't present locally."""

from __future__ import annotations

import os

import pytest

from svguardian.eval.evaluate import _bootstrap_ci, _frontier, _prf, evaluate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNTIME = os.path.join(REPO, "models", "guardian.runtime.json.gz")
TEST = os.path.join(REPO, "models", "test_split.jsonl")


def test_prf_basic():
    # 8 mal (6 caught), 2 benign (0 fp)
    prec, rec, f1, fpr = _prf(tp=6, fp=0, tn=2, fn=2)
    assert rec == 0.75 and prec == 1.0 and fpr == 0.0


def test_frontier_recall_monotonic_in_fpr():
    # higher allowed FPR can only allow >= recall
    scores = [0.1, 0.2, 0.6, 0.7, 0.85, 0.9, 0.95]
    y_mal = [False, False, True, False, True, True, True]
    fr = _frontier(scores, y_mal, targets=(0.02, 0.05, 0.10, 0.5))
    recalls = [f["recall"] for f in fr]
    assert recalls == sorted(recalls), f"recall not monotonic in FPR: {recalls}"
    for f in fr:
        assert f["fpr"] <= f["target_fpr"] + 1e-9


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    scores = [0.9, 0.8, 0.95, 0.1, 0.2, 0.85, 0.3, 0.7]
    y_mal = [True, True, True, False, False, True, False, True]
    thr = 0.5
    a = _bootstrap_ci(scores, y_mal, thr, resamples=200, seed=1337)
    b = _bootstrap_ci(scores, y_mal, thr, resamples=200, seed=1337)
    assert a == b, "bootstrap CI must be deterministic given the seed"
    point_rec = _prf(*_counts(scores, y_mal, thr))[1]
    lo, hi = a["recall"]
    assert lo <= point_rec <= hi


def _counts(scores, y_mal, thr):
    from svguardian.eval.evaluate import _binary_counts
    return _binary_counts(scores, y_mal, thr)


@pytest.mark.skipif(not (os.path.exists(RUNTIME) and os.path.exists(TEST)),
                    reason="runtime/test split not present (gitignored)")
def test_evaluate_on_pure_runtime_shape_and_determinism():
    import json

    from svguardian.model.pure_infer import PureGuardian
    rows = [json.loads(l) for l in open(TEST, encoding="utf-8")]
    g = PureGuardian.load(RUNTIME)
    r1 = evaluate(g, rows, resamples=200, seed=1337)
    r2 = evaluate(g, rows, resamples=200, seed=1337)
    assert r1 == r2, "evaluate must be deterministic given the seed"
    assert len(r1["recall_at_fpr_frontier"]) == 3
    assert set(r1["binary_ci95"]) == {"precision", "recall", "false_positive_rate"}
    # every category carries an explicit support count + reliability flag
    for c, v in r1["category"].items():
        assert "support" in v and "reliable" in v
        assert v["reliable"] == (v["support"] >= 8)
