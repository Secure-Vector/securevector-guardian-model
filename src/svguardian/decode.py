"""Decode-and-rescan: surface threats hidden inside encoded blobs.

Attackers smuggle instructions as base64 / hex so the surface text looks
harmless ("please run: aWdub3Jl..."). We find encoded-looking substrings,
bounded-decode them, and hand the decoded plaintext back as extra spans for the
classifier to scan. Depth- and size-capped so a decode-bomb can't blow latency.

Stdlib only (base64, binascii, re).
"""

from __future__ import annotations

import base64
import binascii
import re
from urllib.parse import unquote

_B64 = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")
_HEX = re.compile(r"(?:[0-9a-fA-F]{2}){8,}")
_URL_ENC = re.compile(r"%[0-9A-Fa-f]{2}")

MAX_BLOBS = 20        # TOTAL encoded spans decoded per input (across all depths)
MAX_DECODED = 4096    # cap decoded bytes per blob
MAX_DEPTH = 3         # bounded recursion (base64-in-base64)


def _looks_like_text(s: str) -> bool:
    printable = sum(c.isprintable() or c.isspace() for c in s)
    return len(s) >= 6 and printable / len(s) > 0.85 and any(c.isalpha() for c in s)


def _printable(b: bytes) -> str | None:
    try:
        s = b.decode("utf-8")
    except UnicodeDecodeError:
        return None
    # Keep only if it looks like text (mostly printable, has letters).
    return s if _looks_like_text(s) else None


def _url_decoded(text: str) -> str | None:
    """Percent-decode (``ignore%20all%20previous``). Natural gate: only returns a
    span if ``%xx`` is present AND decoding actually changes the text — so normal
    prose (and prose with no escapes) never produces a span, no FP surface."""
    if not _URL_ENC.search(text):
        return None
    decoded = unquote(text)
    if decoded == text or not _looks_like_text(decoded):
        return None
    return decoded


def decoded_spans(text: str) -> list:
    """Return decoded plaintext for base64/hex blobs found in ``text``.

    A single shared budget (``MAX_BLOBS``) bounds the TOTAL number of decoded
    spans across both encodings and all recursion levels, so a nested
    decode-bomb cannot multiply the downstream classifier fan-out."""
    out: list = []
    budget = [MAX_BLOBS]
    # URL-decode pass first (inline %xx; gated so benign text yields nothing).
    u = _url_decoded(text)
    if u:
        budget[0] -= 1
        out.append(u)
        _decode_into(u, 0, budget, out)  # decoded text may itself hold base64/hex
    _decode_into(text, 0, budget, out)
    return out


def _decode_into(text: str, depth: int, budget: list, out: list) -> None:
    if depth >= MAX_DEPTH or budget[0] <= 0:
        return
    for rx, kind in ((_B64, "b64"), (_HEX, "hex")):
        for m in rx.finditer(text):
            if budget[0] <= 0:
                return
            blob = m.group(0)
            try:
                if kind == "b64":
                    raw = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=False)[:MAX_DECODED]
                else:
                    raw = bytes.fromhex(blob[: len(blob) - (len(blob) % 2)])[:MAX_DECODED]
            except (binascii.Error, ValueError):
                continue
            s = _printable(raw)
            if s:
                budget[0] -= 1
                out.append(s)
                _decode_into(s, depth + 1, budget, out)  # nested encoding
