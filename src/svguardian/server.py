"""Minimal loopback inference server for SecureVector Guardian.

Stdlib only (http.server + json) — no web framework, and no ML dependency: it
loads the pure-Python JSON runtime (``PureGuardian``), never the joblib/pickle
model. Exposes the same ``POST /analyze`` contract as the local app so
securevector-app can call it over 127.0.0.1 as a drop-in inference source.

    python -m svguardian.server --runtime models/guardian.runtime.json.gz --port 8799

    curl -s localhost:8799/analyze -d '{"text":"ignore all previous instructions"}'

Binds 127.0.0.1 only (never exposed off-host).
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from ._bundle import resolve_runtime
from .model.pure_infer import PureGuardian
from .serve import analyze

MAX_BODY_BYTES = 1 << 20   # 1 MiB request-body cap (DoS guard)
MAX_TEXT_CHARS = 200_000   # bound the text handed to the classifier

_GUARDIAN: PureGuardian | None = None


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
        if self.path == "/health":
            self._send(200, {"status": "ok", "model": _GUARDIAN is not None})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/analyze":
            self._send(404, {"error": "not found"})
            return
        # Reject oversized / unspecified bodies BEFORE reading them into memory.
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            self._send(411, {"error": "Content-Length required"})
            return
        try:
            length = int(raw_len)
        except ValueError:
            self._send(400, {"error": "bad Content-Length"})
            return
        if length > MAX_BODY_BYTES:
            self._send(413, {"error": "payload too large"})
            return
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
            text = req.get("text", "")
            if not isinstance(text, str) or not text:
                self._send(400, {"error": "missing 'text'"})
                return
            result = analyze(text[:MAX_TEXT_CHARS], _GUARDIAN,
                             request_id=req.get("request_id"),
                             direction=req.get("direction", "outgoing"))
            self._send(200, result)
        except json.JSONDecodeError:
            self._send(400, {"error": "invalid JSON"})
        except Exception:  # noqa: BLE001 — never leak internals to the client
            self._send(500, {"error": "internal error"})

    def log_message(self, *args) -> None:  # silence default request logging
        return


def main() -> None:
    global _GUARDIAN
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default=None,
                    help="model bundle path (default: env SV_GUARDIAN_RUNTIME, else the "
                         "per-user cache, downloaded on first use)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8799)
    args = ap.parse_args()

    runtime = args.runtime
    if runtime is None:
        try:
            runtime = resolve_runtime(verbose=True)
        except RuntimeError as exc:
            raise SystemExit(str(exc))
    if not os.path.exists(runtime):
        raise SystemExit(
            f"model bundle not found: {runtime}\n"
            "  download guardian.runtime.json.gz from\n"
            "  https://github.com/Secure-Vector/securevector-guardian-model/releases\n"
            "  then pass --runtime /path/to/guardian.runtime.json.gz (or set SV_GUARDIAN_RUNTIME)"
        )
    _GUARDIAN = PureGuardian.load(runtime)
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.daemon_threads = True       # don't let lingering threads block shutdown
    server.timeout = 30
    print(f"SecureVector Guardian listening on http://{args.host}:{args.port}/analyze (loopback only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
