"""Train the SecureVector Guardian classifier.

A single multiclass model over the 8 canonical labels (7 threat families +
``benign``). The verdict is ``malicious`` iff the predicted label != benign.

Design choices that keep the evaluation honest:
  * The held-out TEST split is drawn ONLY from real (non-synthetic) seed
    examples. Synthetic augmentation is added to TRAIN only — so test
    metrics measure generalization, never augmentation mimicry.
  * Features combine word n-grams (1-2) and character n-grams (3-5).
    Char n-grams are what give robustness to leetspeak / spacing / homoglyph
    obfuscation, where word tokens shatter.
  * LogisticRegression with balanced class weights handles the long tail.

Run:  python -m svguardian.model.train --data data/dataset.jsonl --out models/guardian.joblib
"""

from __future__ import annotations

import argparse
import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

from ..data.augment import augment
from ..data.loaders import Example

BENIGN = "benign"


def _load(path: str) -> list[Example]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out.append(Example(r["text"], r["label"], r["category"], r.get("source", "")))
    return out


def build_pipeline() -> Pipeline:
    """Original feature pipeline: word + char TF-IDF -> logistic regression."""
    word = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True,
        lowercase=True, strip_accents=None,  # keep accents: homoglyphs are signal
    )
    char = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True,
        lowercase=True,
    )
    feats = FeatureUnion([("word", word), ("char", char)])
    clf = LogisticRegression(
        max_iter=3000, C=4.0, class_weight="balanced", solver="lbfgs",
    )
    # lbfgs supports native multinomial multiclass on our 8-label space.
    return Pipeline([("features", feats), ("clf", clf)])


def _malicious_scores(pipe, texts: list, benign: str) -> list:
    """Total probability mass on non-benign classes for each text."""
    classes = list(pipe.classes_)
    bidx = classes.index(benign) if benign in classes else -1
    proba = pipe.predict_proba(texts)
    return [float(1.0 - row[bidx]) if bidx >= 0 else float(max(row)) for row in proba]


def calibrate_threshold(pipe, val: list, benign: str, target_fpr: float) -> float:
    """Choose the malicious-score threshold on the validation set.

    Objective: maximize **recall subject to FPR <= target_fpr**; if no candidate
    meets the FPR cap (small val set), fall back to the **best-F1** threshold.
    Uses BOTH benign and malicious val examples, so the operating point reflects
    where real attack scores fall — not just the benign tail."""
    texts = [e.text for e in val]
    y_mal = [e.label == "malicious" for e in val]
    if not texts or not any(y_mal):
        return 0.5
    scores = _malicious_scores(pipe, texts, benign)

    candidates = sorted(set(scores))
    best_constrained = None  # (recall, -fpr, thr)
    best_f1 = None  # (f1, thr)
    for thr in candidates:
        tp = fp = fn = tn = 0
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
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if fpr <= target_fpr:
            key = (rec, -fpr, -thr)
            if best_constrained is None or key > best_constrained[0]:
                best_constrained = (key, thr)
        if best_f1 is None or f1 > best_f1[0]:
            best_f1 = (f1, thr)

    thr = best_constrained[1] if best_constrained else best_f1[1]
    return float(min(max(thr, 0.05), 0.99))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.jsonl")
    ap.add_argument("--out", default="models/guardian.joblib")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--val-size", type=float, default=0.15)  # fraction of the non-test pool
    ap.add_argument("--target-fpr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--no-augment", action="store_true")
    args = ap.parse_args()

    real = _load(args.data)
    labels = [e.category for e in real]

    def _split(items, lbls, size):
        try:
            a, b = train_test_split(range(len(items)), test_size=size,
                                    random_state=args.seed, stratify=lbls)
        except ValueError:
            a, b = train_test_split(range(len(items)), test_size=size, random_state=args.seed)
        return a, b

    # 3-way: train / val (threshold calibration) / test (final eval, real-only).
    pool_idx, te_idx = _split(real, labels, args.test_size)
    pool = [real[i] for i in pool_idx]
    pool_lbls = [labels[i] for i in pool_idx]
    tr_idx, va_idx = _split(pool, pool_lbls, args.val_size)
    train = [pool[i] for i in tr_idx]
    val = [pool[i] for i in va_idx]
    test = [real[i] for i in te_idx]

    if not args.no_augment:
        synth = augment(train, seed=args.seed)
        train = train + synth
        print(f"train: {len(tr_idx)} real + {len(synth)} synth = {len(train)} | "
              f"val: {len(val)} real | test: {len(test)} real")
    else:
        print(f"train: {len(train)} real | val: {len(val)} | test: {len(test)} (no augmentation)")

    pipe = build_pipeline()
    pipe.fit([e.text for e in train], [e.category for e in train])

    threshold = calibrate_threshold(pipe, val, BENIGN, args.target_fpr)
    print(f"calibrated threshold (target FPR {args.target_fpr}): {threshold:.4f}")

    from ..window import WINDOW_MIN_SCORE

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump({"pipeline": pipe, "labels": sorted(set(e.category for e in train)),
                 "benign": BENIGN, "threshold": threshold, "target_fpr": args.target_fpr,
                 "window_min_score": WINDOW_MIN_SCORE}, args.out)
    print(f"saved model -> {args.out}")

    test_path = os.path.join(os.path.dirname(args.out) or ".", "test_split.jsonl")
    with open(test_path, "w", encoding="utf-8") as fh:
        for e in test:
            fh.write(json.dumps({"text": e.text, "label": e.label, "category": e.category}) + "\n")
    print(f"saved held-out test split -> {test_path}")


if __name__ == "__main__":
    main()
