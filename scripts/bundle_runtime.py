#!/usr/bin/env python3
"""Copy the trained model runtime into the package so the wheel ships it.

Run this once, immediately before ``python -m build``. It resolves the runtime
bundle and copies it (plus its ``.sha256`` sidecar) into
``src/svguardian/_runtime/`` — the in-wheel home declared as package-data in
``pyproject.toml``. The result is a self-contained wheel: ``pip install`` works
offline and ``pip install -U`` updates the model with the code.

Where the runtime comes from (first hit wins — reuses the package's own
resolver, so the rules match what end-users get):

1. ``SV_GUARDIAN_RUNTIME`` — explicit path (CI can point at a downloaded asset).
2. A dev checkout's ``models/guardian.runtime.json.gz`` — the local build path.
3. The GitHub release asset — downloaded + SHA-verified (the CI path, where the
   source checkout has no ``models/``). Attach the new runtime to the GitHub
   Release before publishing so the release build embeds the right weights.

Stdlib only; no third-party deps. Safe to re-run (overwrites the bundle).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO, "src"))

from svguardian import _bundle  # noqa: E402  (after sys.path setup)

_DEST_DIR = os.path.join(_REPO, "src", "svguardian", "_runtime")
_BUNDLE_NAME = "guardian.runtime.json.gz"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    # download=True so a CI checkout (no models/) fetches the release asset.
    src = _bundle.resolve_runtime(download=True, verbose=True)
    if not os.path.exists(src):
        print(f"bundle_runtime: runtime not found at {src}", file=sys.stderr)
        return 1

    os.makedirs(_DEST_DIR, exist_ok=True)
    dest = os.path.join(_DEST_DIR, _BUNDLE_NAME)
    shutil.copyfile(src, dest)

    # Sidecar: copy if it travels with the source, else compute one so the
    # in-wheel bundle keeps PureGuardian's tamper check working.
    dest_sha = dest + ".sha256"
    src_sha = src + ".sha256"
    if os.path.exists(src_sha):
        shutil.copyfile(src_sha, dest_sha)
    else:
        with open(dest_sha, "w", encoding="utf-8") as fh:
            fh.write(_sha256(dest) + "\n")

    size_mb = os.path.getsize(dest) / (1024 * 1024)
    print(f"bundle_runtime: bundled {dest} ({size_mb:.2f} MB) for the wheel")
    print(f"bundle_runtime: sidecar  {dest_sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
