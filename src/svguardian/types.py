"""Shared structural types for SecureVector Guardian.

A ``Predictor`` is anything that classifies text — both the scikit-learn
``Guardian`` and the zero-dependency ``PureGuardian`` satisfy it. Using the
Protocol keeps the public inference contract (``serve.analyze`` /
``window.predict_windowed``) explicit without importing either concrete class.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Predictor(Protocol):
    threshold: float
    window_min_score: float

    def predict(self, text: str) -> dict:
        """Return at least ``{"verdict", "category", "malicious_score", ...}``."""
        ...
