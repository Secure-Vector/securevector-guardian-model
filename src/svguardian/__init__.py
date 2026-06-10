"""SecureVector Guardian — lightweight offline prompt/AI-attack detector.

The trained model bundle is not shipped in the wheel; it's fetched from the
GitHub release on first use and cached per-user. ``resolve_runtime()`` returns
its path (downloading if needed) so callers don't hard-code a location.
"""

from ._bundle import resolve_runtime

__all__ = ["resolve_runtime"]
