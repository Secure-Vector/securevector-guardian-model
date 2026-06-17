"""Adversarial regression eval over the frozen red-team corpus (`data/redteam/`).

These are original red-team examples that are NEVER used for training. This
harness reports recall per category as a regression tripwire across releases: if
a future retrain drops red-team recall, CI/eval surfaces it. Any red-team example
that overlaps the training set (exact OR char-3gram near-dup) is DROPPED with a
reported count, so the number is always honestly held-out.

Evaluates the pure runtime (what ships) by default; --model for the joblib.
Run: python -m svguardian.eval.redteam_eval --runtime models/guardian.runtime.json.gz
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter

from ..data.loaders import Example, canonicalize, find_near_dup_leaks, key_hash

REDTEAM_DIRS = ["data/redteam", "data/redteam/round2"]


def load_redteam(dirs: list[str]) -> list[Example]:
    out: list[Example] = []
    for d in dirs:
        for p in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            for line in open(p, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                txt = r.get("text") or r.get("input") or r.get("prompt")
                if not txt:
                    continue
                label = str(r.get("label", "")).lower()
                # red-team rows label the category in either field; benign is benign.
                raw = r.get("category") or label
                if raw == "benign" or label == "benign":
                    out.append(Example(txt, "benign", "benign", "redteam"))
                else:
                    out.append(Example(txt, "malicious", canonicalize(raw), "redteam"))
    return out


def _load_jsonl(path: str) -> list[Example]:
    out = []
    for line in open(path, encoding="utf-8"):
        r = json.loads(line)
        out.append(Example(r["text"], r["label"], r["category"], r.get("source", "")))
    return out


def evaluate(predictor, redteam: list[Example], train: list[Example]) -> dict:
    # Drop any red-team example that leaked into training (exact or near-dup).
    train_keys = {key_hash(e.text) for e in train}
    held = [e for e in redteam if key_hash(e.text) not in train_keys]
    dropped_exact = len(redteam) - len(held)
    near = find_near_dup_leaks(held, train, threshold=0.8)
    near_texts = {t for t, _, _ in near}
    held = [e for e in held if e.text not in near_texts]
    dropped_near = len(near_texts)

    cat_total: Counter = Counter()
    cat_hit: Counter = Counter()
    fp = tn = 0
    for e in held:
        pred = predictor.predict(e.text)
        flagged = pred["verdict"] == "malicious"
        if e.label == "malicious":
            cat_total[e.category] += 1
            if flagged:
                cat_hit[e.category] += 1
        else:
            fp += flagged
            tn += not flagged

    total_mal = sum(cat_total.values())
    total_hit = sum(cat_hit.values())
    return {
        "n_loaded": len(redteam),
        "dropped_train_overlap": {"exact": dropped_exact, "near_dup": dropped_near},
        "n_evaluated": len(held),
        "overall_recall": round(total_hit / total_mal, 4) if total_mal else 0.0,
        "benign_fpr": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        "category_recall": {c: {"recall": round(cat_hit[c] / cat_total[c], 4), "support": cat_total[c]}
                            for c in sorted(cat_total)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/guardian.joblib")
    ap.add_argument("--runtime", default=None)
    ap.add_argument("--train", default="data/dataset.jsonl")
    args = ap.parse_args()
    if args.runtime:
        from ..model.pure_infer import PureGuardian
        predictor = PureGuardian.load(args.runtime)
    else:
        from ..model.infer import Guardian
        predictor = Guardian.load(args.model)
    res = evaluate(predictor, load_redteam(REDTEAM_DIRS), _load_jsonl(args.train))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
