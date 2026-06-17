"""In-wheel home for the trained Guardian runtime bundle.

This package (``svguardian._runtime``) ships in source control, but the weights
it holds — ``guardian.runtime.json.gz`` and its ``.sha256`` sidecar — do NOT.
They are a build-time artifact: gitignored, and copied in by
``scripts/bundle_runtime.py`` just before the wheel is built. A published wheel
therefore carries the model *inside* it, so ``pip install`` /
``pip install -U`` ship code and weights together — fully offline, and a new
model version is simply a new wheel (``pip -U`` updates both at once).

A plain source checkout (no build step run) has no bundle here, so
``bundled_path()`` returns ``None`` and Guardian falls back to the dev
``models/`` checkout, the per-user cache, or the GitHub-release download — see
``svguardian._bundle.resolve_runtime``.
"""

from __future__ import annotations

import os

_BUNDLE_NAME = "guardian.runtime.json.gz"


def bundled_path() -> str | None:
    """Absolute path to the in-wheel runtime bundle, or ``None`` if not bundled.

    Resolves relative to this file so it works identically whether imported from
    a source checkout or an installed wheel (wheels unpack into site-packages as
    regular directories, so ``__file__`` is a real path and the ``.sha256``
    sidecar sitting beside the bundle stays loadable by ``PureGuardian.load``).
    """
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, _BUNDLE_NAME)
    return path if os.path.exists(path) else None
