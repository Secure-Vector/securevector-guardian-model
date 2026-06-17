"""Evaluate SecureVector Guardian on the frozen held-out (real) test split.

Reports, with honest uncertainty:
  * binary precision/recall/F1/FPR at the calibrated operating point, WITH
    95% bootstrap confidence intervals (the test set is small — point estimates
    alone are misleading);
  * the recall@FPR frontier (max recall achievable at FPR <= {0.02,0.05,0.10}),
    so the achievable trade space is visible, not just one threshold;
  * per-category recall WITH support counts inline (so single-digit-support
    categories are obviously unreliable, not silently trusted).

Evaluates the **pure-Python runtime** (what actually ships) by default; pass
--model to score the sklearn joblib instead. Deterministic given --seed.

Run: python -m svguardian.eval.evaluate --runtime models/guardian.runtime.json.gz \\
        --test models/test_split.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict


def _load(path: str) -> list:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def load_predictor(model_path: str | None, runtime_path: str | None):
    """Prefer the shipped pure runtime; fall back to the sklearn joblib."""
    if runtime_path:
        from ..model.pure_infer import PureGuardian
        return PureGuardian.load(runtime_path), "PureGuardian"
    from ..model.infer import Guardian
    return Guardian.load(model_path), "Guardian"


def _binary_counts(scores: list[float], y_mal: list[bool], thr: float):
    tp = fp = tn = fn = 0
    for s, mal in zip(scores, y_mal):
        pred = s >= thr
        if mal and pred:
            tp += 1
        elif mal and not pred:
            fn += 1
        elif not mal and pred:
            fp += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def _prf(tp, fp, tn, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1, fpr


def _frontier(scores: list[float], y_mal: list[bool], targets=(0.02, 0.05, 0.10)) -> list[dict]:
    """For each target FPR, the max recall achievable at a threshold whose FPR
    on this set is <= target (the honest 'recall @ low FPR' the model could hit)."""
    cands = sorted(set(scores)) + [max(scores) + 1.0] if scores else [0.5]
    out = []
    for t in targets:
        best = None  # (recall, threshold, fpr)
        for thr in cands:
            tp, fp, tn, fn = _binary_counts(scores, y_mal, thr)
            _, rec, _, fpr = _prf(tp, fp, tn, fn)
            if fpr <= t and (best is None or rec > best[0]):
                best = (rec, thr, fpr)
        if best:
            out.append({"target_fpr": t, "threshold": round(best[1], 4),
                        "recall": round(best[0], 4), "fpr": round(best[2], 4)})
        else:
            out.append({"target_fpr": t, "threshold": None, "recall": 0.0, "fpr": 0.0})
    return out


def _bootstrap_ci(scores: list[float], y_mal: list[bool], thr: float, *,
                  resamples: int, seed: int) -> dict:
    """95% percentile bootstrap CIs for precision/recall/FPR at the operating
    threshold. Resamples rows with replacement; the small test set means these
    intervals are wide — that is the point (don't over-read a point estimate)."""
    rng = random.Random(seed)
    n = len(scores)
    precs, recs, fprs = [], [], []
    idxs = list(range(n))
    for _ in range(resamples):
        sample = [rng.choice(idxs) for _ in range(n)]
        s = [scores[i] for i in sample]
        y = [y_mal[i] for i in sample]
        prec, rec, _, fpr = _prf(*_binary_counts(s, y, thr))
        precs.append(prec); recs.append(rec); fprs.append(fpr)

    def ci(vals):
        vals = sorted(vals)
        lo = vals[int(0.025 * len(vals))]
        hi = vals[min(len(vals) - 1, int(0.975 * len(vals)))]
        return [round(lo, 4), round(hi, 4)]

    return {"precision": ci(precs), "recall": ci(recs), "false_positive_rate": ci(fprs)}


def evaluate(predictor, rows: list, *, resamples: int = 1000, seed: int = 1337) -> dict:
    thr = float(getattr(predictor, "threshold", 0.5))
    scores, y_mal = [], []
    cat_total: Counter = Counter()
    cat_hit: Counter = Counter()
    confusion = defaultdict(Counter)

    for r in rows:
        pred = predictor.predict(r["text"])
        scores.append(float(pred["malicious_score"]))
        true_mal = r["label"] == "malicious"
        y_mal.append(true_mal)
        if true_mal:
            cat_total[r["category"]] += 1
            if pred["verdict"] == "malicious" and pred["category"] == r["category"]:
                cat_hit[r["category"]] += 1
            confusion[r["category"]][pred["category"] or "benign"] += 1

    tp, fp, tn, fn = _binary_counts(scores, y_mal, thr)
    prec, rec, f1, fpr = _prf(tp, fp, tn, fn)
    acc = (tp + tn) / len(rows) if rows else 0.0

    category = {}
    for c in sorted(cat_total):
        support = cat_total[c]
        category[c] = {"recall": round(cat_hit[c] / support, 4), "support": support,
                       "reliable": support >= 8}

    return {
        "n": len(rows),
        "operating_threshold": round(thr, 4),
        "binary": {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "accuracy": round(acc, 4),
                    "false_positive_rate": round(fpr, 4),
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "binary_ci95": _bootstrap_ci(scores, y_mal, thr, resamples=resamples, seed=seed),
        "recall_at_fpr_frontier": _frontier(scores, y_mal),
        "category": category,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/guardian.joblib", help="sklearn joblib (fallback)")
    ap.add_argument("--runtime", default=None,
                    help="pure-python runtime bundle (what ships); preferred target")
    ap.add_argument("--test", default="models/test_split.jsonl")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    predictor, kind = load_predictor(args.model, args.runtime)
    res = evaluate(predictor, _load(args.test), resamples=args.bootstrap, seed=args.seed)
    res["predictor"] = kind
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
