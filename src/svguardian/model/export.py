"""Export a trained Guardian pipeline to a pure-Python runtime bundle.

Writes a JSON the stdlib-only ``pure_infer`` runtime can load — so the local
app (which has NO ML dependencies) can run inference with zero scikit-learn /
numpy at runtime. We dump:

  * word + char TF-IDF vocabularies and their idf weights
  * the logistic-regression coefficients + intercepts + class order
  * the calibrated threshold and benign label

Run: python -m svguardian.model.export --model models/guardian.joblib --out models/guardian.runtime.json.gz
"""

from __future__ import annotations

import argparse
import gzip
import json

import joblib


def export(model_path: str, out_path: str) -> dict:
    blob = joblib.load(model_path)
    pipe = blob["pipeline"]
    feats = dict(pipe.named_steps["features"].transformer_list)
    word = feats["word"]
    char = feats["char"]
    clf = pipe.named_steps["clf"]

    def vocab_idf(vec):
        # idf_ is indexed by column; vocabulary_ maps term -> column.
        idf = vec.idf_.tolist()
        return vec.vocabulary_, idf

    word_vocab, word_idf = vocab_idf(word)
    char_vocab, char_idf = vocab_idf(char)
    n_word = len(word_vocab)

    bundle = {
        "format": "svguardian-pure-1",
        "classes": [str(c) for c in clf.classes_.tolist()],
        "benign": blob.get("benign", "benign"),
        "threshold": float(blob.get("threshold", 0.5)),
        "window_min_score": float(blob.get("window_min_score", 0.86)),
        "n_word_features": n_word,
        "word": {"vocab": {t: int(i) for t, i in word_vocab.items()},
                  "idf": word_idf,
                  "ngram_range": list(word.ngram_range),
                  "sublinear_tf": bool(word.sublinear_tf)},
        "char": {"vocab": {t: int(i) for t, i in char_vocab.items()},
                  "idf": char_idf,
                  "ngram_range": list(char.ngram_range),
                  "sublinear_tf": bool(char.sublinear_tf)},
        # coef_ is (n_classes, n_features); features are [word.. , char..].
        "coef": [row.tolist() for row in clf.coef_],
        "intercept": clf.intercept_.tolist(),
    }

    import hashlib

    raw = json.dumps(bundle).encode("utf-8")
    if out_path.endswith(".gz"):
        with gzip.open(out_path, "wb") as fh:
            fh.write(raw)
    else:
        with open(out_path, "wb") as fh:
            fh.write(raw)
    # SHA-256 of the on-disk file for integrity verification at load time.
    with open(out_path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    with open(out_path + ".sha256", "w", encoding="utf-8") as fh:
        fh.write(digest + "\n")
    return bundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/guardian.joblib")
    ap.add_argument("--out", default="models/guardian.runtime.json.gz")
    args = ap.parse_args()
    b = export(args.model, args.out)
    import os
    print(f"exported pure-Python runtime -> {args.out} "
          f"({os.path.getsize(args.out)/1e6:.2f} MB, "
          f"{len(b['word']['vocab'])} word + {len(b['char']['vocab'])} char features, "
          f"{len(b['classes'])} classes)")


if __name__ == "__main__":
    main()
