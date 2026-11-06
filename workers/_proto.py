"""Tiny JSON-lines protocol shared by the long-lived engine workers.

The server talks to a worker over stdin/stdout. Libraries love to print to
stdout, so we move the real stdout to a private file descriptor and point
fd 1 at stderr before anything else is imported.
"""

import json
import os
import queue
import sys
import threading

_proto_fd = os.dup(1)
os.dup2(2, 1)
sys.stdout = sys.stderr
_out = os.fdopen(_proto_fd, "w", buffering=1)
_out_lock = threading.Lock()

cancelled: set[str] = set()
_jobs: "queue.Queue[dict]" = queue.Queue()


def send(**msg) -> None:
    with _out_lock:
        _out.write(json.dumps(msg, ensure_ascii=False) + "\n")
        _out.flush()


def _reader() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        if msg.get("cmd") == "cancel":
            cancelled.add(msg.get("target", ""))
        else:
            _jobs.put(msg)
    _jobs.put({"cmd": "exit"})


def serve(handlers: dict) -> None:
    """Run handlers[cmd](msg) for each request; handlers return a result dict."""
    threading.Thread(target=_reader, daemon=True).start()
    send(type="ready", pid=os.getpid())
    while True:
        msg = _jobs.get()
        cmd, rid = msg.get("cmd"), msg.get("id", "")
        if cmd == "exit":
            return
        try:
            result = handlers[cmd](msg)
            send(id=rid, type="result", **(result or {}))
        except Exception as e:  # report and keep serving
            import traceback

            traceback.print_exc()
            send(id=rid, type="error", error=f"{type(e).__name__}: {e}")
        finally:
            cancelled.discard(rid)
