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
from ..data.loaders import Example, find_near_dup_leaks, key_hash

BENIGN = "benign"


def _load(path: str) -> list[Example]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            out.append(Example(r["text"], r["label"], r["category"], r.get("source", "")))
    return out


def load_manifest(path: str) -> set[str] | None:
    """Load the frozen-test manifest (set of key hashes), or None if absent.
    Lines are ``<sha256>\\t<text>``; ``#`` lines are comments."""
    if not path or not os.path.isfile(path):
        return None
    hashes: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            hashes.add(line.split(maxsplit=1)[0])
    return hashes or None


def write_manifest(path: str, test: list[Example]) -> None:
    """Persist the held-out test set as a content-addressed manifest so it stays
    frozen across retrains (the data may grow; this set does not)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# SecureVector Guardian frozen held-out test manifest.\n")
        fh.write("# <sha256(normalized_text)>\\t<json text> — ALWAYS test, NEVER train.\n")
        for e in test:
            # JSON-encode the text so embedded newlines never split a record.
            fh.write(f"{key_hash(e.text)}\t{json.dumps(e.text)}\n")


def build_pipeline() -> Pipeline:
    """Original feature pipeline: word + char TF-IDF -> logistic regression."""
    # min_df=1 stays. B2 sweep tested min_df=2 (halves the bundle, holds obfuscation
    # recall) but it REGRESSED long-doc benign FPR (0.0 -> 0.42): pruning rare
    # features makes benign long-doc windows score above the window bar. The
    # obfuscation gate alone missed it; the long-doc eval caught it. Rejected.
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
    ap.add_argument("--test-manifest", default=None,
                    help="frozen-test manifest path (default: <out_dir>/test_manifest.sha256)")
    args = ap.parse_args()

    real = _load(args.data)
    labels = [e.category for e in real]
    out_dir = os.path.dirname(args.out) or "."
    manifest_path = args.test_manifest or os.path.join(out_dir, "test_manifest.sha256")

    def _split(items, lbls, size):
        try:
            a, b = train_test_split(range(len(items)), test_size=size,
                                    random_state=args.seed, stratify=lbls)
        except ValueError:
            a, b = train_test_split(range(len(items)), test_size=size, random_state=args.seed)
        return a, b

    # Held-out TEST: frozen by content-hash manifest so it stays identical across
    # retrains as the corpus grows (no silent re-split leakage). We still compute
    # the stratified split first, then RECONCILE it to the manifest — so when the
    # corpus is unchanged the partition (and thus the model) is bit-for-bit the
    # same, while any NEW example that landed in "test" is moved back to the pool
    # and any frozen example is guaranteed into test.
    frozen = load_manifest(manifest_path)
    pool_idx, te_idx = _split(real, labels, args.test_size)
    pool = [real[i] for i in pool_idx]
    test = [real[i] for i in te_idx]
    if frozen:
        def _in(e):
            return key_hash(e.text) in frozen
        # Reconcile: frozen -> test, everything else -> pool. Order-preserving, so
        # an UNCHANGED corpus yields the identical pool (and thus the same model);
        # any new example that the stratified split put in test moves back to pool.
        new_pool = [e for e in pool if not _in(e)] + [e for e in test if not _in(e)]
        new_test = [e for e in test if _in(e)] + [e for e in pool if _in(e)]
        pool, test = new_pool, new_test
        missing = frozen - {key_hash(e.text) for e in test}
        print(f"frozen test manifest: {manifest_path} ({len(frozen)} keys; "
              f"{len(test)} matched, {len(missing)} missing from data)")
    else:
        write_manifest(manifest_path, test)
        print(f"bootstrapped frozen test manifest -> {manifest_path} ({len(test)} keys)")

    pool_lbls = [e.category for e in pool]
    tr_idx, va_idx = _split(pool, pool_lbls, args.val_size)
    train = [pool[i] for i in tr_idx]
    val = [pool[i] for i in va_idx]

    # Leak guard: no train example may be an exact OR near-duplicate (char-3gram
    # Jaccard >= 0.8) of any frozen-test example. Exact-key exclusion above stops
    # identical rows; this stops authoring a paraphrase of a test row into train.
    leaks = find_near_dup_leaks(train + val, test)
    if leaks:
        msg = "\n".join(f"  j={j:.2f}  train={tr[:60]!r}  test={te[:60]!r}" for tr, te, j in leaks[:10])
        raise SystemExit(f"TRAIN/TEST LEAK: {len(leaks)} near-duplicate(s) of frozen test in train:\n{msg}")

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
