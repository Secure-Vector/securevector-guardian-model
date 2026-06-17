"""Data loaders for SecureVector Guardian.

Builds a labeled training set from SecureVector's OWN data only:

1. Rule ``test_cases`` (positive + benign examples) from configured rule dirs.
2. A curated labeled detection corpus (``detection_corpus.yaml``).
3. (Optional) additional rule packs from an extra configured directory.

Data-source locations are supplied via environment variables, so the data
itself lives outside this repository. No third-party datasets, no copied
corpora. Each example is a
``(text, label, category, source)`` record where ``label`` is
``"malicious"`` or ``"benign"`` and ``category`` is the threat family
(or ``"benign"``).

The category taxonomy is derived from the rules themselves, normalised to a
compact canonical set so long-tail rule categories collapse into the families
the local app reports on.
"""

from __future__ import annotations

import glob
import hashlib
import os
from dataclasses import dataclass

import yaml

# Canonical threat families. Raw rule categories are mapped onto these so the
# classifier predicts a stable, small label space instead of 20+ noisy strings.
CANONICAL_CATEGORIES = {
    "prompt_injection",
    "jailbreak",
    "data_exfiltration",
    "pii",
    "social_engineering",
    "harmful_content",
    "model_attack",
    "benign",
}

# Map raw rule ``category:`` values onto the canonical set above.
_CATEGORY_MAP = {
    "prompt_injection": "prompt_injection",
    "indirect_prompt_injection": "prompt_injection",
    "prompt_leaking": "prompt_injection",
    "delimiter_injection": "prompt_injection",
    "json_payload_injection": "prompt_injection",
    "code_injection": "prompt_injection",
    "sql_injection": "prompt_injection",
    "reasoning_override": "prompt_injection",
    "jailbreak_attempt": "jailbreak",
    "jailbreak_success": "jailbreak",
    "jailbreak": "jailbreak",
    "evasion_attack": "jailbreak",
    "excessive_agency": "jailbreak",
    "data_exfiltration": "data_exfiltration",
    "data_extraction": "data_exfiltration",
    "data_leakage": "data_exfiltration",
    "training_data_extraction": "data_exfiltration",
    "memory_extraction": "data_exfiltration",
    "sensitive_data_exposure": "data_exfiltration",
    "sensitive_data_disclosure": "data_exfiltration",
    "model_extraction": "model_attack",
    "model_inversion": "model_attack",
    "membership_inference": "model_attack",
    "adversarial_attack": "model_attack",
    "data_poisoning": "model_attack",
    "hallucination_exploitation": "model_attack",
    "privacy_leak": "pii",
    "pii_leakage": "pii",
    "privacy_attack": "pii",
    "social_engineering": "social_engineering",
    "phishing": "social_engineering",
    "misinformation_generation": "harmful_content",
    "inappropriate_content": "harmful_content",
    "harmful_content": "harmful_content",
    "malicious_code_generation": "harmful_content",
    "toxic_content_generation": "harmful_content",
}


def canonicalize(raw_category: str | None) -> str:
    """Collapse a raw rule category onto the canonical taxonomy. Idempotent: an
    already-canonical value (e.g. ``model_attack``, ``pii``) maps to itself."""
    if not raw_category:
        return "prompt_injection"
    key = str(raw_category).strip().lower()
    if key in CANONICAL_CATEGORIES:
        return key
    return _CATEGORY_MAP.get(key, "prompt_injection")


@dataclass(frozen=True)
class Example:
    text: str
    label: str  # "malicious" | "benign"
    category: str  # canonical category, or "benign"
    source: str  # provenance tag

    def key(self) -> str:
        """Dedup key — normalised text only, so the same string from two
        sources collapses to one example (prevents train/test leakage)."""
        return " ".join(self.text.lower().split())


def _iter_rule_files(*dirs: str):
    for d in dirs:
        if not d or not os.path.isdir(d):
            continue
        for path in sorted(glob.glob(os.path.join(d, "*.yml")) + glob.glob(os.path.join(d, "*.yaml"))):
            yield path


def _rules_from_doc(doc) -> list[dict]:
    """A rule YAML file is either a top-level list of rules or a dict with a
    ``rules:`` key. Be liberal about both shapes."""
    if isinstance(doc, list):
        return [r for r in doc if isinstance(r, dict)]
    if isinstance(doc, dict):
        if isinstance(doc.get("rules"), list):
            return [r for r in doc["rules"] if isinstance(r, dict)]
        # A single-rule document.
        if "test_cases" in doc or "patterns" in doc:
            return [doc]
    return []


def load_rule_test_cases(*dirs: str, source_prefix: str = "rule") -> list[Example]:
    """Extract labeled examples from rule ``test_cases`` blocks.

    Positive (``expected_result: match``) → label of the rule's category.
    Negative (``expected_result: no_match``) → benign.
    """
    out: list[Example] = []
    for path in _iter_rule_files(*dirs):
        try:
            with open(path, encoding="utf-8") as fh:
                doc = yaml.safe_load(fh)
        except Exception as exc:
            # Don't fail silently — a malformed rule file otherwise vanishes from
            # the dataset with no signal (the examples just never appear).
            print(f"WARNING: skipping unparseable rule file {path}: {exc}")
            continue
        fname = os.path.basename(path)
        for rule in _rules_from_doc(doc):
            cat = canonicalize(rule.get("category"))
            for tc in rule.get("test_cases", []) or []:
                if not isinstance(tc, dict):
                    continue
                text = tc.get("input")
                if not text or not isinstance(text, str):
                    continue
                expected = str(tc.get("expected_result", "")).strip().lower()
                if expected == "match":
                    out.append(Example(text, "malicious", cat, f"{source_prefix}:{fname}"))
                elif expected == "no_match":
                    out.append(Example(text, "benign", "benign", f"{source_prefix}:{fname}"))
    return out


def load_detection_corpus(path: str) -> list[Example]:
    """Load the curated hard-negative / labeled eval corpus.

    Schema: a top-level list (or dict with ``samples:``) of records with
    ``text``, ``label`` (benign|malicious) and optional ``category``.
    """
    if not path or not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh)
    if isinstance(doc, dict):
        records = doc.get("samples") or doc.get("corpus") or doc.get("cases") or []
    else:
        records = doc or []
    out: list[Example] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        text = rec.get("text") or rec.get("input")
        label = str(rec.get("label", "")).strip().lower()
        if not text or label not in {"benign", "malicious"}:
            continue
        cat = "benign" if label == "benign" else canonicalize(rec.get("category"))
        out.append(Example(text, label, cat, "corpus"))
    return out


def normalize_text(text: str) -> str:
    """Canonical normalized form used for dedup, hashing, and near-dup checks."""
    return " ".join(str(text).lower().split())


def key_hash(text: str) -> str:
    """Content-address an example by its normalized text. The frozen-test
    manifest is a set of these hashes, so the held-out set is stable across
    retrains regardless of corpus growth (see train.py)."""
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def char_ngrams(text: str, n: int = 3) -> set[str]:
    s = normalize_text(text)
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i + n] for i in range(len(s) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


def near_duplicate_index(text: str, others_ngrams: list[set], threshold: float = 0.8,
                         n: int = 3) -> int:
    """Index of the first entry in ``others_ngrams`` whose char-``n``-gram Jaccard
    with ``text`` is >= ``threshold`` (a paraphrase-level near-duplicate), else -1.
    Exact-text dedup (``key()``) cannot catch paraphrases; this can."""
    ng = char_ngrams(text, n)
    for i, ong in enumerate(others_ngrams):
        if jaccard(ng, ong) >= threshold:
            return i
    return -1


def find_near_dup_leaks(train: list[Example], test: list[Example], *,
                        threshold: float = 0.8, n: int = 3) -> list[tuple]:
    """Return (train_text, test_text, jaccard) for every train example that is a
    near-duplicate of any test example — i.e. train/test leakage that exact-key
    dedup misses. Empty list = clean."""
    test_ng = [char_ngrams(e.text, n) for e in test]
    leaks = []
    for tr in train:
        idx = near_duplicate_index(tr.text, test_ng, threshold, n)
        if idx >= 0:
            leaks.append((tr.text, test[idx].text,
                          jaccard(char_ngrams(tr.text, n), test_ng[idx])))
    return leaks


def dedupe(examples: list[Example]) -> list[Example]:
    """Drop exact-text duplicates. On a label conflict for the same text,
    keep malicious (conservative for a security detector)."""
    by_key: dict[str, Example] = {}
    for ex in examples:
        k = ex.key()
        prev = by_key.get(k)
        if prev is None:
            by_key[k] = ex
        elif prev.label == "benign" and ex.label == "malicious":
            by_key[k] = ex
    return list(by_key.values())
