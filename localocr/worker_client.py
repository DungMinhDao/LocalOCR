"""Client for the long-lived engine worker processes (see workers/_proto.py)."""

import json
import os
import queue
import subprocess
import threading
import time
import uuid
from collections import deque

from .paths import ROOT, WORKERS


class WorkerError(RuntimeError):
    pass


class WorkerClient:
    def __init__(self, name: str, script: str, env: dict | None = None):
        self.name = name
        self.script = WORKERS / script
        self.env = env or {}
        self.proc: subprocess.Popen | None = None
        self.state = "stopped"  # stopped | starting | ready | busy | error
        self.error = ""
        self.loaded: list[str] = []
        self.log = deque(maxlen=200)
        self._pending: dict[str, queue.Queue] = {}
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._busy = 0

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self.proc and self.proc.poll() is None:
                return
            self._ready.clear()
            self.state, self.error = "starting", ""
            env = {**os.environ, "PYTHONUNBUFFERED": "1", **self.env}
            self.proc = subprocess.Popen(
                ["uv", "run", "--quiet", "--script", str(self.script)],
                cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, env=env,
            )
            threading.Thread(target=self._read_stdout, args=(self.proc,), daemon=True).start()
            threading.Thread(target=self._read_stderr, args=(self.proc,), daemon=True).start()

    def stop(self) -> None:
        with self._lock:
            p, self.proc = self.proc, None
        if p and p.poll() is None:
            p.terminate()
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()
        self.state, self.loaded = "stopped", []

    def _read_stderr(self, proc) -> None:
        for line in proc.stderr:
            line = line.rstrip()
            if line:
                self.log.append(line)

    def _read_stdout(self, proc) -> None:
        for line in proc.stdout:
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "ready":
                self.state = "ready"
                self._ready.set()
                continue
            q = self._pending.get(msg.get("id", ""))
            if q:
                q.put(msg)
        # process exited
        code = proc.wait()
        if proc is self.proc or self.proc is None:
            tail = "\n".join(list(self.log)[-15:])
            self.state = "error" if code else "stopped"
            self.error = f"worker exited with code {code}\n{tail}" if code else ""
            self.loaded = []
        self._ready.set()
        for q in list(self._pending.values()):
            q.put({"type": "error", "error": self.error or "worker stopped"})

    # -- requests --------------------------------------------------------
    def request(self, cmd: str, on_event=None, rid: str | None = None, timeout: float = 3600, **payload) -> dict:
        self.start()
        if not self._ready.wait(900) or self.state not in ("ready", "busy"):
            raise WorkerError(self.error or f"{self.name} worker failed to start")
        rid = rid or uuid.uuid4().hex
        q: queue.Queue = queue.Queue()
        self._pending[rid] = q
        self._busy += 1
        self.state = "busy"
        try:
            self.proc.stdin.write(json.dumps({"cmd": cmd, "id": rid, **payload}) + "\n")
            self.proc.stdin.flush()
            deadline = time.time() + timeout
            while True:
                msg = q.get(timeout=max(1, deadline - time.time()))
                if msg.get("type") == "result":
                    if "loaded" in msg:
                        self.loaded = msg["loaded"]
                    return msg
                if msg.get("type") == "error":
                    raise WorkerError(msg.get("error", "unknown worker error"))
                if on_event:
                    on_event(msg)
        finally:
            self._pending.pop(rid, None)
            self._busy -= 1
            if self.state == "busy" and self._busy == 0:
                self.state = "ready"

    def cancel(self, rid: str) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write(json.dumps({"cmd": "cancel", "target": rid}) + "\n")
                self.proc.stdin.flush()
            except (BrokenPipeError, OSError):
                pass

    def info(self) -> dict:
        return {"state": self.state, "error": self.error, "loaded": self.loaded,
                "pid": self.proc.pid if self.proc and self.proc.poll() is None else None,
                "log": list(self.log)[-30:]}
