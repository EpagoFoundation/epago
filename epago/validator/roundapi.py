"""API-key round trigger for the scoring validator.

The owner opens a competition by POSTing to this endpoint with a shared API
key, instead of publishing a signed ``er1`` on chain. This is a deliberate
move of the trigger *off* chain, and it changes one property: auditors can no
longer independently verify when a round opened or that the minimum interval
was respected — those checks now live inside the one scoring validator.

What is preserved is the property that actually protects miners. The round's
exam is still minted from a **chain block hash the owner cannot choose**: on a
valid request the validator stamps the round with the current chain block and
seeds the exam from that block's hash. A trigger cannot leak the questions,
because the questions do not exist until a block the owner did not pick is
already sealed — exactly as with the on-chain trigger.

The server is stdlib-only (no new dependency), single-threaded, and does
nothing but flip a flag: the validator's own tick loop reads the flag, enforces
the interval, builds the round, and runs it. So a request never touches the
duel path directly, and a flood of requests collapses to "one round, when the
interval allows".
"""

from __future__ import annotations

import hmac
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger(__name__)

#: Header carrying the shared key. Compared in constant time.
API_KEY_HEADER = "X-Epago-Round-Key"


class RoundTrigger:
    """Thread-safe latch between the HTTP endpoint and the validator tick.

    ``request()`` (called from the server thread) raises the latch;
    ``take()`` (called from the tick thread) lowers it and reports whether it
    was raised. A burst of requests between two ticks is one pending round, not
    many — the latch is a boolean, not a counter.
    """

    def __init__(self) -> None:
        self._pending = False
        self._lock = threading.Lock()
        self._requests = 0  # lifetime count, for status/telemetry only

    def request(self) -> None:
        with self._lock:
            self._pending = True
            self._requests += 1

    def take(self) -> bool:
        with self._lock:
            was = self._pending
            self._pending = False
            return was

    @property
    def total_requests(self) -> int:
        with self._lock:
            return self._requests


def _make_handler(api_key: str, trigger: RoundTrigger):
    class Handler(BaseHTTPRequestHandler):
        # Silence the default stderr access log; failures still log below.
        def log_message(self, *_args) -> None:  # noqa: D401
            pass

        def _reply(self, code: int, body: str) -> None:
            payload = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:  # noqa: N802 - stdlib name
            if self.path.rstrip("/") != "/round/start":
                self._reply(404, "not found")
                return
            provided = self.headers.get(API_KEY_HEADER, "")
            # Constant-time compare so a wrong key cannot be found by timing.
            if not hmac.compare_digest(provided, api_key):
                logger.warning("round trigger: rejected request with bad/absent API key")
                self._reply(401, "unauthorized")
                return
            trigger.request()
            logger.info("round trigger: accepted request; a round opens next tick")
            self._reply(202, "round requested")

        def do_GET(self) -> None:  # noqa: N802 - liveness probe, no key required
            if self.path.rstrip("/") == "/healthz":
                self._reply(200, "ok")
            else:
                self._reply(404, "not found")

    return Handler


def serve_round_trigger(bind: str, api_key: str, trigger: RoundTrigger) -> ThreadingHTTPServer:
    """Start the trigger server on ``bind`` (``host:port``) in a daemon thread.

    Returns the server so the caller can ``shutdown()`` it. Raises if the key
    is empty — an unauthenticated trigger would let anyone open rounds.
    """
    if not api_key:
        raise ValueError("round trigger API key is empty; refusing to serve an open trigger")
    host, _, port = bind.partition(":")
    server = ThreadingHTTPServer((host or "127.0.0.1", int(port or 8799)),
                                 _make_handler(api_key, trigger))
    thread = threading.Thread(target=server.serve_forever, name="epago-round-trigger", daemon=True)
    thread.start()
    logger.info("round trigger listening on %s (POST /round/start)", bind)
    return server
