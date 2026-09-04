"""One vLLM replica, pinned to one GPU, hosted in its own process.

Why a subprocess and not just a device keyword
----------------------------------------------

``CUDA_VISIBLE_DEVICES`` is read once, when CUDA initializes; a second engine
constructed later in the same process cannot be steered onto a different card
by re-setting it. Handing vLLM an explicit device index instead would work only
for the parts of the stack that honour it, and would leave the replica running
in a process that can *see* the other GPUs — the exact condition under which a
future vLLM version might decide to use them.

A child process with ``CUDA_VISIBLE_DEVICES=<one device>`` removes the whole
class of problem: the engine sees exactly one GPU, enumerated as device 0, and
is constructed by the same :class:`~epago.eval.backend.VllmBackend` a
single-GPU validator constructs, from the same environment variables. That is
the strongest available statement of the design's central claim — a replica is
not "a model on GPU 5", it is "a single-GPU validator's engine, which happens
to live on GPU 5". Two further properties fall out for free:

* **eviction actually frees the card.** Releasing a vLLM engine in-process is
  famously incomplete; killing the process is not. A pool that must swap a
  17 GB checkpoint whenever a new king is crowned needs the reliable version.
* **replicas do not contend.** Each has its own interpreter, so N replicas are
  N independent decode loops rather than N threads trading one GIL.

The protocol
------------

Length-prefixed JSON over an ``AF_UNIX`` socketpair, one request at a time.
The socket is used rather than stdin/stdout because vLLM (and torch, and NCCL)
print banners on stdout, which would corrupt a stdout-framed protocol; the
child redirects its stdout onto stderr for exactly that reason, so engine logs
still reach the operator's journal interleaved with everything else.

Only text crosses the boundary: prompts in, completions out. Tool sessions,
judging, and the episode state machine all stay in the parent, so the harness
loop that produces a scored result is the same object graph it has always been
and nothing about scoring semantics depends on where the weights sit.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["WorkerBackend", "WorkerError", "main"]

#: fd number of the child's end of the protocol socket, passed through the env.
FD_ENV = "EPAGO_WORKER_FD"
#: Interpreter for replica processes; defaults to the parent's. A validator
#: whose eval extras live in a second virtualenv points this at that python.
PYTHON_ENV = "EPAGO_EVAL_WORKER_PYTHON"
#: Seconds to wait for a replica to finish loading its weights. 17 GB of 4-bit
#: MoE weights off a cold page cache is minutes, so the default is generous;
#: it exists to turn a wedged load into an error instead of a hung validator.
LOAD_TIMEOUT_ENV = "EPAGO_EVAL_WORKER_LOAD_TIMEOUT"
#: Seconds to wait for one batched generation step.
CALL_TIMEOUT_ENV = "EPAGO_EVAL_WORKER_TIMEOUT"

_HEADER = struct.Struct("!I")
_MAX_FRAME = 512 * 1024 * 1024


class WorkerError(RuntimeError):
    """The replica process failed, died, or timed out."""


# --- framing ------------------------------------------------------------------


def _send(sock: socket.socket, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    sock.sendall(_HEADER.pack(len(body)) + body)


def _recv_exactly(sock: socket.socket, n: int) -> bytes | None:
    chunks: list[bytes] = []
    got = 0
    while got < n:
        chunk = sock.recv(min(n - got, 1 << 20))
        if not chunk:
            return None
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


def _recv(sock: socket.socket) -> dict | None:
    header = _recv_exactly(sock, _HEADER.size)
    if header is None:
        return None
    (size,) = _HEADER.unpack(header)
    if size > _MAX_FRAME:
        raise WorkerError(f"oversized frame: {size} bytes")
    body = _recv_exactly(sock, size)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))


# --- parent side --------------------------------------------------------------


def _package_root() -> str:
    """Directory that must be on ``PYTHONPATH`` for the child to import epago.

    A validator commonly runs from a checkout rather than an installed wheel,
    so the child cannot be assumed to inherit an import path that works — it is
    started with an explicit cwd-independent one.
    """
    import epago

    return str(Path(epago.__file__).resolve().parent.parent)


class WorkerBackend:
    """A :class:`~epago.eval.backend.ModelBackend` living on one pinned device.

    Satisfies the same generate / generate_many / close protocol the harness
    talks to, so :func:`~epago.eval.harness.run_rollouts_batched` cannot tell
    the difference between this and an in-process engine.
    """

    def __init__(
        self,
        model_dir: Path,
        device: str,
        *,
        python: str | None = None,
        load_timeout: float | None = None,
        call_timeout: float | None = None,
    ) -> None:
        self.device = str(device)
        self.model_dir = Path(model_dir)
        self._call_timeout = (
            call_timeout
            if call_timeout is not None
            else float(os.environ.get(CALL_TIMEOUT_ENV, "3600"))
        )
        load_timeout = (
            load_timeout
            if load_timeout is not None
            else float(os.environ.get(LOAD_TIMEOUT_ENV, "1800"))
        )
        self._lock = threading.Lock()
        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
        env = dict(os.environ)
        # The child's value is resolved against the *physical* enumeration, so
        # this must be the identifier resolve_devices() forwarded, never a
        # re-derived index. Everything else (determinism flags, memory caps) is
        # inherited unchanged: a replica is configured exactly like the
        # single-GPU engine it stands in for.
        env["CUDA_VISIBLE_DEVICES"] = self.device
        env[FD_ENV] = str(child_sock.fileno())
        root = _package_root()
        env["PYTHONPATH"] = (
            root if not env.get("PYTHONPATH") else root + os.pathsep + env["PYTHONPATH"]
        )
        interpreter = python or os.environ.get(PYTHON_ENV) or sys.executable
        # Put the interpreter's own bin directory on PATH, which is all that
        # activating its virtualenv would have done. Without it the replica
        # inherits whatever PATH the parent was launched with, and vLLM's JIT
        # sampling kernels shell out to `ninja` at engine start — a validator
        # started from an unactivated venv would see every replica die during
        # load with a bare FileNotFoundError.
        # Deliberately not resolve(): a venv's bin/python is a symlink to the
        # system interpreter, and following it would put /usr/bin on PATH and
        # leave the venv's own tools exactly as unreachable as before.
        bindir = os.path.dirname(os.path.abspath(shutil.which(interpreter) or interpreter))
        env["PATH"] = (
            bindir if not env.get("PATH") else bindir + os.pathsep + env["PATH"]
        )
        try:
            self._proc = subprocess.Popen(  # noqa: S603 - fixed argv, no shell
                [interpreter, "-m", "epago.eval.worker"],
                env=env,
                pass_fds=(child_sock.fileno(),),
                stdin=subprocess.DEVNULL,
                # Own session: vLLM starts an EngineCore child of its own, and
                # a process-group kill is the only reliable way to take the
                # whole replica down when it stops answering.
                start_new_session=True,
            )
        finally:
            child_sock.close()
        self._sock = parent_sock
        try:
            self._sock.settimeout(load_timeout)
            self._call({"op": "load", "model_dir": str(self.model_dir)})
        except BaseException:
            self.close()
            raise
        self._sock.settimeout(self._call_timeout)

    # -- protocol --------------------------------------------------------------

    def _call(self, request: dict) -> dict:
        try:
            _send(self._sock, request)
            reply = _recv(self._sock)
        except socket.timeout as exc:
            raise WorkerError(
                f"replica on device {self.device} timed out during {request['op']}"
            ) from exc
        except OSError as exc:
            raise WorkerError(
                f"replica on device {self.device} broke during {request['op']}: {exc}"
            ) from exc
        if reply is None:
            code = self._proc.poll()
            raise WorkerError(
                f"replica on device {self.device} exited during {request['op']} "
                f"(returncode {code})"
            )
        if not reply.get("ok"):
            raise WorkerError(
                f"replica on device {self.device} failed {request['op']}: "
                f"{reply.get('error')}"
            )
        return reply

    # -- ModelBackend ----------------------------------------------------------

    def generate(self, prompt: str, max_tokens: int, stop: list[str]) -> str:
        return self.generate_many([prompt], max_tokens, stop)[0]

    def generate_many(self, prompts: list[str], max_tokens: int, stop: list[str]) -> list[str]:
        with self._lock:
            reply = self._call(
                {
                    "op": "generate",
                    "prompts": list(prompts),
                    "max_tokens": int(max_tokens),
                    "stop": list(stop),
                }
            )
        texts = reply.get("texts")
        if not isinstance(texts, list) or len(texts) != len(prompts):
            raise WorkerError(
                f"replica on device {self.device} returned {len(texts or ())} completions "
                f"for {len(prompts)} prompts"
            )
        return texts

    def close(self) -> None:
        """Shut the replica down and free the card. Idempotent, never raises."""
        sock, self._sock = getattr(self, "_sock", None), None
        if sock is not None:
            try:
                sock.settimeout(30)
                _send(sock, {"op": "close"})
                _recv(sock)
            except Exception:  # noqa: BLE001 - a dead replica is already closed
                pass
            finally:
                sock.close()
        proc = getattr(self, "_proc", None)
        if proc is None:
            return
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            logger.warning("replica on device %s did not exit; killing", self.device)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:  # pragma: no cover - unkillable child
                logger.error("replica on device %s is unkillable", self.device)


# --- child side ---------------------------------------------------------------


def _teardown(code: int) -> None:
    """Exit now, and take the engine's own child processes with us.

    Returning from :func:`main` is not enough for two reasons, both measured:
    vLLM leaves non-daemon threads that stall interpreter shutdown, and it runs
    its engine in a *further* child process which would otherwise be reparented
    and keep the card allocated. A replica that outlives its parent is 17 GB of
    VRAM nobody can reclaim without an operator noticing and killing it by
    hand — the exact failure the pool exists to make impossible.

    The group kill is guarded on actually being the group leader, which
    ``start_new_session=True`` guarantees for a pool-spawned replica. Run by
    hand from a shell, that guard is false and this degrades to exiting only
    itself, rather than killing the operator's terminal.
    """
    try:
        if os.getpgid(0) == os.getpid():
            os.killpg(os.getpgid(0), signal.SIGKILL)
    except Exception:  # noqa: BLE001 - fall through to the plain exit
        pass
    os._exit(code)


def main() -> int:
    """Replica entry point: ``python -m epago.eval.worker`` with :data:`FD_ENV` set."""
    fd = os.environ.get(FD_ENV)
    if fd is None:
        print(f"{FD_ENV} is not set; this module is spawned by the eval pool", file=sys.stderr)
        return 2
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM, fileno=int(fd))
    # vLLM, torch and NCCL all write to stdout. The protocol does not live
    # there, but anything the parent's stdout is wired to might, so send it to
    # stderr where an operator reads engine logs anyway.
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    backend = None
    while True:
        try:
            message = _recv(sock)
        except Exception:  # noqa: BLE001 - a broken pipe is the parent going away
            _teardown(0)
        if message is None:
            # EOF: the parent closed the socket or died. Either way this replica
            # has no one left to serve, so it must not keep holding a card.
            _teardown(0)
        op = message.get("op")
        try:
            if op == "load":
                from epago.eval.backend import VllmBackend

                backend = VllmBackend(Path(message["model_dir"]))
                _send(sock, {"ok": True})
            elif op == "generate":
                if backend is None:
                    raise RuntimeError("no model loaded")
                texts = backend.generate_many(
                    list(message["prompts"]), int(message["max_tokens"]), list(message["stop"])
                )
                _send(sock, {"ok": True, "texts": list(texts)})
            elif op == "close":
                _send(sock, {"ok": True})
                _teardown(0)
            else:
                _send(sock, {"ok": False, "error": f"unknown op {op!r}"})
        except BaseException as exc:  # noqa: BLE001 - report, never die silently
            try:
                _send(sock, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            except Exception:  # noqa: BLE001 - parent gone
                _teardown(1)


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
