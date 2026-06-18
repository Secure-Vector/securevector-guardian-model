# Changelog

All notable changes to **SecureVector Guardian** (the `securevector-guardian-model`
package) are documented here. Format follows [Keep a Changelog](https://keepachangelog.com).

Guardian is a small, fully offline, from-scratch threat-detection model that runs
as a **high-precision additive layer** above the regex rule engine. It is not the
sole line of defense.

## [1.4.0] — 2026-06-17

Reduces false positives on benign **technical content** — source code, product
and security documentation, and security-agent system prompts — that earlier
versions could misflag as attacks because they share vocabulary with real
threats (`secret`, `key`, `severity`, `model`, `injection`, `exfiltration`).

### Fixed
- **Benign source code, docs, and security system prompts no longer
  false-positive.** Reading a source file (e.g. `def get_api_key(...)`,
  `import React`), a product README/API reference, or a SOC-analyst system
  prompt ("monitor for prompt injection and data exfiltration") previously
  scored as an attack. New original benign training examples for these
  content classes correct the over-defense.

### Changed
- Retrained on the original corpus with the added benign-technical examples.
  **Held-out precision and false-positive rate are unchanged** from v1.3.0
  (precision ≈ 0.97, held-out FPR ≈ 0.02; long-document benign FPR 0.0).
  Obfuscation (leetspeak / homoglyph / spacing), buried-in-document, and
  base64 / hex / URL-encoded robustness are maintained, and common direct
  attacks (instruction-override, persona-jailbreak, credential exfiltration)
  remain caught.

### Notes
- This is a precision improvement for the **additive** layer; it does not change
  the regex rule engine. Content that quotes a **literal** attack payload
  verbatim (e.g. a changelog that prints an injection string as an example) is
  still flagged — that text is genuinely attack-shaped, and is best handled by
  scoping enforcement to what an agent *executes* vs. what it merely *reads*.
- Data & legal posture unchanged: 100% original training data, no third-party
  datasets or pretrained weights; zero-dependency pure-Python runtime, byte-exact
  to the trained model (parity Δ = 0).

### Verification
- Full test suite green (behavioral + sklearn↔pure-runtime parity, Δ = 0);
  held-out FPR 0.02, long-document benign FPR 0.0; validated end-to-end in the
  local app.

## [1.3.0] — 2026-06-16

Adds encoded-payload and agent-era injection coverage, and hardens the model's
evaluation and data-provenance guarantees. All training data remains 100%
SecureVector-original — now enforced by automated tests.

### Added
- **URL / percent-encoding decode-and-rescan.** The model now percent-decodes
  inline `%xx` payloads and rescans the plaintext, catching encoded injections
  (e.g. `ignore%20all%20previous%20instructions`) the model previously missed.
  Gated so it only activates when `%xx` is present and decoding changes the text —
  benign prose and benign encoded URLs produce no false positives.
- **Broadened agent-era injection coverage** via original training templates:
  tool/plugin misuse, RAG / retrieved-document indirect injection, and
  memory/conversation poisoning. (Concepts from OWASP LLM06/LLM08 and MITRE ATLAS;
  all example text authored by SecureVector.)
- **Honest, leak-proof evaluation.** A content-hash–frozen held-out test set that
  stays identical across retrains, with a train/test near-duplicate (paraphrase)
  leak guard that fails the build on any overlap. Reported metrics now include a
  recall-at-FPR frontier, 95% bootstrap confidence intervals, and per-category
  support counts (so small-sample categories are flagged as unreliable rather than
  silently trusted).
- **Adversarial red-team regression eval** over a frozen 1,955-example corpus
  (held out of training, verified by the leak guard) as a cross-release tripwire.
- **Provenance enforcement.** Automated tests enforce that every training example
  comes from a SecureVector-internal source and that no public-dataset marker
  appears in the training data — run during training, where the corpus lives — plus
  a static guard, run in CI, that the package never imports a public-dataset loader.

### Changed
- Retrained on the original corpus. **Precision held** (held-out false-positive
  rate ≈ 0.02; long-document benign false-positive rate 0.0); obfuscation
  (leetspeak / homoglyph / spacing), buried-in-long-document, and base64/hex
  robustness maintained.
- `canonicalize()` is now idempotent (already-canonical category names map to
  themselves) — fixes mis-bucketing of `model_attack` / `pii` labels in tooling.
- Malformed rule files now emit a warning instead of being silently skipped.

### Fixed
- Eliminated a class of evaluation leakage in which growing the dataset silently
  reshuffled the held-out split, making release-over-release metrics incomparable.

### Data & legal posture (unchanged, now enforced)
- **100% original training data.** No third-party datasets, prompts, rules, code,
  or model weights are used or distributed. No pretrained checkpoints — every
  weight is trained from scratch on SecureVector's own data. This is now enforced
  by automated provenance tests, not just convention.
- **Public benchmarks are evaluation-only** and are not part of this release; no
  public dataset is used for training.
- **Attack-taxonomy alignment** (OWASP LLM Top 10, MITRE ATLAS) uses public
  concepts/names only — facts/ideas, not copyrightable expression. Every example
  string is authored by SecureVector.
- **Dependencies are permissive open source** used as libraries, not vendored:
  scikit-learn / NumPy / SciPy (BSD), PyYAML / joblib (MIT). No GPL/AGPL.
- Ships a **zero-dependency pure-Python runtime** that is byte-exact to the
  trained model (parity Δ = 0); inference is offline, deterministic, and CPU-only.

### Notes
- Guardian is a high-precision **additive** layer that runs alongside the regex
  rule engine and any enabled cloud review — not a standalone or exhaustive detector.
- Scope threat-scanning to agent inputs and actions. Very large documents that
  *quote* attack strings as examples (e.g. a security tool's own README or API
  docs) can be flagged, since that content is genuinely attack-shaped — enforce on
  what an agent executes and monitor what it merely reads.

### Verification
- Full test suite green (behavioral + sklearn↔pure-runtime parity, Δ = 0);
  long-document benign FPR 0.0; validated end-to-end in the local app.

## [1.2.0] — earlier
- History-cleaned release; runtime unchanged from 1.1.0.

## [1.1.0] — earlier
- Model weights fetched on first use and cached per-user.
