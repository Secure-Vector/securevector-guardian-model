"""SecureVector Guardian — lightweight offline prompt/AI-attack detector.

The trained model bundle ships inside the wheel (``svguardian/_runtime/``), so
an install is self-contained and offline and ``pip install -U`` updates the
model with the code. ``resolve_runtime()`` returns its path, preferring (in
order) an explicit override, a dev ``models/`` checkout, the in-wheel bundle,
the per-user cache, and — only for an unbuilt source checkout — a one-time
GitHub-release download. Callers should not hard-code a location.
"""

from ._bundle import resolve_runtime

__all__ = ["resolve_runtime"]
