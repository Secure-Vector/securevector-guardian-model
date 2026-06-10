"""Compare learning algorithms for SecureVector Guardian on the SAME data.

Linear models and tree ensembles each get their natural feature regime:
  * linear / NB  -> full sparse TF-IDF (word 1-2 + char 3-5)
  * tree ensembles -> TF-IDF -> TruncatedSVD(dense) (trees need dense, low-dim)

For each algorithm: fit on the augmented train split, calibrate a malicious-score
threshold on the validation split (max recall s.t. FPR<=target, else best F1),
then evaluate on the held-out REAL test split. Prints a comparison table.

All estimators are scikit-learn (BSD). No pretrained weights; trained from
scratch on our own data. Run:

    PYTHONPATH=src python -m svguardian.model.compare --data data/dataset.jsonl
"""

from __future__ import annotations

import argparse

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV

from ..data.augment import augment
from .train import _load

BENIGN = "benign"


def _features_sparse() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, sublinear_tf=True)),
    ])


def _dense(X):
    return X.toarray() if hasattr(X, "toarray") else X


def _make(kind: str):
    """Return a fresh Pipeline for the named algorithm."""
    sparse = _features_sparse()
    if kind == "logreg":
        return Pipeline([("f", sparse),
                         ("c", LogisticRegression(max_iter=3000, C=4.0, class_weight="balanced", solver="lbfgs"))])
    if kind == "linear_svm":
        return Pipeline([("f", sparse),
                         ("c", CalibratedClassifierCV(LinearSVC(C=1.0, class_weight="balanced"), cv=3))])
    if kind == "complement_nb":
        return Pipeline([("f", sparse), ("c", ComplementNB())])
    # Tree ensembles: reduce to dense low-dim via SVD first.
    svd = ("svd", TruncatedSVD(n_components=300, random_state=1337))
    if kind == "random_forest":
        return Pipeline([("f", sparse), svd,
                         ("c", RandomForestClassifier(n_estimators=400, class_weight="balanced", n_jobs=-1, random_state=1337))])
    if kind == "extra_trees":
        return Pipeline([("f", sparse), svd,
                         ("c", ExtraTreesClassifier(n_estimators=500, class_weight="balanced", n_jobs=-1, random_state=1337))])
    if kind == "hist_gbdt":
        return Pipeline([("f", sparse), svd, ("to_dense", FunctionTransformer(_dense, accept_sparse=True)),
                         ("c", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1, random_state=1337))])
    raise ValueError(kind)


def _malicious_scores(pipe, texts):
    classes = list(pipe.classes_)
    bidx = classes.index(BENIGN) if BENIGN in classes else -1
    proba = pipe.predict_proba(texts)
    if bidx >= 0:
        return [float(1.0 - row[bidx]) for row in proba]
    return [float(max(row)) for row in proba]


def _calibrate(scores, y_mal, target_fpr):
    best_c, best_f1 = None, None
    for thr in sorted(set(scores)):
        tp = fp = fn = tn = 0
        for s, m in zip(scores, y_mal):
            p = s >= thr
            tp += m and p; fn += m and not p; fp += (not m) and p; tn += (not m) and not p
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        if fpr <= target_fpr and (best_c is None or (rec, -fpr) > best_c[0]):
            best_c = ((rec, -fpr), thr)
        if best_f1 is None or f1 > best_f1[0]:
            best_f1 = (f1, thr)
    return float((best_c[1] if best_c else best_f1[1]))


def _evaluate(pipe, thr, test):
    texts = [e.text for e in test]
    scores = _malicious_scores(pipe, texts)
    tp = fp = fn = tn = 0
    for e, s in zip(test, scores):
        mal = e.label == "malicious"
        pred = s >= thr
        tp += mal and pred; fn += mal and not pred; fp += (not mal) and pred; tn += (not mal) and not pred
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "fpr": fpr,
            "accuracy": (tp + tn) / len(test)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dataset.jsonl")
    ap.add_argument("--target-fpr", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    real = _load(args.data)
    labels = [e.category for e in real]

    def split(items, lbls, size):
        try:
            return train_test_split(range(len(items)), test_size=size, random_state=args.seed, stratify=lbls)
        except ValueError:
            return train_test_split(range(len(items)), test_size=size, random_state=args.seed)

    pool_i, te_i = split(real, labels, 0.2)
    pool = [real[i] for i in pool_i]
    tr_i, va_i = split(pool, [labels[i] for i in pool_i], 0.15)
    train = [pool[i] for i in tr_i] + augment([pool[i] for i in tr_i], seed=args.seed)
    val = [pool[i] for i in va_i]
    test = [real[i] for i in te_i]
    print(f"train {len(train)} | val {len(val)} | test {len(test)} (real)\n")

    Xtr, ytr = [e.text for e in train], [e.category for e in train]
    y_val_mal = [e.label == "malicious" for e in val]

    algos = ["logreg", "linear_svm", "complement_nb", "random_forest", "extra_trees", "hist_gbdt"]
    rows = []
    for kind in algos:
        pipe = _make(kind)
        pipe.fit(Xtr, ytr)
        thr = _calibrate(_malicious_scores(pipe, [e.text for e in val]), y_val_mal, args.target_fpr)
        m = _evaluate(pipe, thr, test)
        rows.append((kind, thr, m))

    print(f"{'algorithm':16s} {'thr':>5s} {'prec':>6s} {'recall':>7s} {'F1':>6s} {'FPR':>6s} {'acc':>6s}")
    print("-" * 60)
    for kind, thr, m in sorted(rows, key=lambda r: r[2]["f1"], reverse=True):
        print(f"{kind:16s} {thr:5.2f} {m['precision']:6.3f} {m['recall']:7.3f} "
              f"{m['f1']:6.3f} {m['fpr']:6.3f} {m['accuracy']:6.3f}")
    best = max(rows, key=lambda r: r[2]["f1"])
    print(f"\nbest by F1: {best[0]}  (F1={best[2]['f1']:.3f}, recall={best[2]['recall']:.3f}, FPR={best[2]['fpr']:.3f})")


if __name__ == "__main__":
    main()
