"""Evaluate SecureVector Guardian on the held-out (real, non-synthetic) split.

Reports binary malicious/benign precision/recall/F1 and per-category recall.
Run: python -m svguardian.eval.evaluate --model models/guardian.joblib --test models/test_split.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from ..model.infer import Guardian


def _load(path: str) -> list:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            out.append(json.loads(line))
    return out


def evaluate(model_path: str, test_path: str) -> dict:
    g = Guardian.load(model_path)
    rows = _load(test_path)

    tp = fp = tn = fn = 0
    cat_total: Counter = Counter()
    cat_hit: Counter = Counter()
    confusion = defaultdict(Counter)

    for r in rows:
        pred = g.predict(r["text"])
        true_mal = r["label"] == "malicious"
        pred_mal = pred["verdict"] == "malicious"
        if true_mal and pred_mal:
            tp += 1
        elif true_mal and not pred_mal:
            fn += 1
        elif not true_mal and pred_mal:
            fp += 1
        else:
            tn += 1
        if true_mal:
            cat_total[r["category"]] += 1
            if pred["category"] == r["category"]:
                cat_hit[r["category"]] += 1
            confusion[r["category"]][pred["category"] or "benign"] += 1

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / len(rows) if rows else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "n": len(rows),
        "binary": {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "accuracy": round(acc, 4),
                    "false_positive_rate": round(fpr, 4),
                    "tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "category_recall": {c: round(cat_hit[c] / cat_total[c], 4) for c in sorted(cat_total)},
        "category_support": dict(cat_total),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/guardian.joblib")
    ap.add_argument("--test", default="models/test_split.jsonl")
    args = ap.parse_args()
    res = evaluate(args.model, args.test)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
