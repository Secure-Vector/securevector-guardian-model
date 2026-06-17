"""Locate (and, only as a last resort, fetch) the trained model bundle.

The published wheel bundles the model (``guardian.runtime.json.gz`` + its
``.sha256`` sidecar) inside ``svguardian/_runtime/`` — so a ``pip install`` is
self-contained and fully offline, and ``pip install -U`` updates code and
weights together. The GitHub-release download remains only as a fallback for
source checkouts that were never built into a wheel.

Resolution order (first hit wins):

1. ``SV_GUARDIAN_RUNTIME`` env var — an explicit path you provide. Never
   downloads; ideal for air-gapped installs (pre-place the file, point here).
2. A dev checkout's ``models/guardian.runtime.json.gz`` (repo-relative) — so a
   freshly trained local model is used ahead of whatever a wheel bundled.
3. The in-wheel bundle (``svguardian/_runtime/``) — present in every published
   wheel; never touches the network.
4. The per-user cache — populated by a previous release download.
5. The GitHub release asset — downloaded once and cached (fallback for an
   unbuilt source checkout).

Stdlib only (urllib / hashlib / gzip) so the runtime keeps zero dependencies
and works identically on macOS, Linux, and Windows.
"""

from __future__ import annotations

import hashlib
import os
import sys
import urllib.request

from . import _runtime

_BUNDLE_NAME = "guardian.runtime.json.gz"

# Public release assets. "latest" lets a newer model ship without a code
# release; override the whole URL with SV_GUARDIAN_BUNDLE_URL if you mirror it.
_RELEASE_BASE = (
    "https://github.com/Secure-Vector/securevector-guardian-model"
    "/releases/latest/download"
)
_BUNDLE_URL = os.environ.get("SV_GUARDIAN_BUNDLE_URL", f"{_RELEASE_BASE}/{_BUNDLE_NAME}")
_SHA_URL = os.environ.get("SV_GUARDIAN_BUNDLE_SHA_URL", f"{_BUNDLE_URL}.sha256")

_DOWNLOAD_TIMEOUT = 60  # seconds


def cache_dir() -> str:
    """Per-user cache directory, following each OS's convention.

    Honours ``SV_GUARDIAN_CACHE`` then ``XDG_CACHE_HOME``; otherwise:
    Windows ``%LOCALAPPDATA%``, macOS ``~/Library/Caches``, Linux ``~/.cache``.
    """
    override = os.environ.get("SV_GUARDIAN_CACHE")
    if override:
        return override
    base = os.environ.get("XDG_CACHE_HOME")
    if not base:
        if sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(
                r"~\AppData\Local"
            )
        elif sys.platform == "darwin":
            base = os.path.expanduser("~/Library/Caches")
        else:
            base = os.path.expanduser("~/.cache")
    return os.path.join(base, "svguardian")


def _dev_models_path() -> str:
    # <repo>/models/guardian.runtime.json.gz relative to this file
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), "models", _BUNDLE_NAME)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "svguardian"})
    with urllib.request.urlopen(req, timeout=_DOWNLOAD_TIMEOUT) as resp:  # noqa: S310 (https only)
        return resp.read()


def _download(dest: str, *, verbose: bool) -> None:
    """Fetch bundle + sidecar to ``dest``, verifying SHA-256. Atomic on success."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if verbose:
        print(f"svguardian: downloading model bundle (~1.8 MB) to {dest}", file=sys.stderr)
    data = _get(_BUNDLE_URL)
    # Sidecar is "<hex>" or "<hex>  filename"; take the first token.
    try:
        expected = _get(_SHA_URL).decode().split()[0].strip().lower()
    except Exception:  # noqa: BLE001 — sidecar optional; fall back to no-verify
        expected = ""
    got = hashlib.sha256(data).hexdigest()
    if expected and got != expected:
        raise RuntimeError(
            f"model bundle SHA-256 mismatch (expected {expected}, got {got}) — refusing to cache"
        )
    tmp = f"{dest}.{os.getpid()}.part"
    with open(tmp, "wb") as fh:
        fh.write(data)
    if expected:
        with open(f"{dest}.sha256", "w") as fh:
            fh.write(f"{expected}\n")
    os.replace(tmp, dest)  # atomic on every platform
    if verbose:
        print("svguardian: model ready (cached for offline use)", file=sys.stderr)


def resolve_runtime(*, download: bool = True, verbose: bool = False) -> str:
    """Return a path to the model bundle, fetching+caching it if needed.

    Raises ``RuntimeError`` with actionable guidance if it isn't present and
    can't be downloaded (e.g. air-gapped first run with no override set).
    """
    env = os.environ.get("SV_GUARDIAN_RUNTIME")
    if env:
        return env

    dev = _dev_models_path()
    if os.path.exists(dev):
        return dev

    packaged = _runtime.bundled_path()
    if packaged:
        return packaged

    cached = os.path.join(cache_dir(), _BUNDLE_NAME)
    if os.path.exists(cached):
        return cached

    if not download:
        return cached  # caller will report "not found"

    try:
        _download(cached, verbose=verbose)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"could not download the model bundle: {exc}\n"
            f"  source : {_BUNDLE_URL}\n"
            "  offline/air-gapped? download guardian.runtime.json.gz from\n"
            "  https://github.com/Secure-Vector/securevector-guardian-model/releases\n"
            "  then: export SV_GUARDIAN_RUNTIME=/path/to/guardian.runtime.json.gz"
        ) from exc
    return cached
