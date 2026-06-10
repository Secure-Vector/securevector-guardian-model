"""Assemble the SecureVector Guardian training dataset from our own sources.

Writes a deduplicated JSONL of {text, label, category, source} and prints a
provenance + class-balance report. Run:

    python -m svguardian.data.build_dataset --out data/dataset.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter

from .loaders import (
    dedupe,
    load_detection_corpus,
    load_rule_test_cases,
)

# Data-source locations are configured via environment variables so the
# (private) training data lives OUTSIDE this repo. No paths or sources are
# hard-coded here.
#   SV_RULES_DIR    directory of rule files carrying ``test_cases``
#   SV_CORPUS       path to a labeled detection corpus YAML
#   SV_EXTRA_RULES  (optional) directory of additional rule packs
RULES_DIR = os.environ.get("SV_RULES_DIR", "rules/community")
CORPUS = os.environ.get("SV_CORPUS", "rules/detection_corpus.yaml")
EXTRA_RULES = os.environ.get("SV_EXTRA_RULES", "")


def build() -> list:
    """Assemble the labeled dataset from the configured rule + corpus sources
    (plus any optional extra rule packs)."""
    examples = []
    examples += load_rule_test_cases(RULES_DIR, source_prefix="rules")
    examples += load_detection_corpus(CORPUS)
    if EXTRA_RULES and os.path.isdir(EXTRA_RULES):
        examples += load_rule_test_cases(EXTRA_RULES, source_prefix="extra")

    before = len(examples)
    examples = dedupe(examples)
    print(f"raw examples: {before}  ->  deduped: {len(examples)}")
    return examples


def report(examples: list) -> None:
    labels = Counter(e.label for e in examples)
    cats = Counter(e.category for e in examples)
    srcs = Counter(e.source.split(":")[0] for e in examples)
    print("\nlabel balance:")
    for k, v in labels.most_common():
        print(f"  {k:12s} {v}")
    print("\ncategory balance:")
    for k, v in cats.most_common():
        print(f"  {k:20s} {v}")
    print("\nsource provenance:")
    for k, v in srcs.most_common():
        print(f"  {k:12s} {v}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/dataset.jsonl")
    args = ap.parse_args()

    examples = build()
    report(examples)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for e in examples:
            fh.write(json.dumps({"text": e.text, "label": e.label, "category": e.category, "source": e.source}) + "\n")
    print(f"\nwrote {len(examples)} examples -> {args.out}")


if __name__ == "__main__":
    main()
