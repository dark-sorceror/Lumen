"""Cross-thread control channel. Engine runs in a worker thread; the server posts here."""
from __future__ import annotations

import queue
from enum import Enum


class Control(str, Enum):
    PAUSE = "pause"
    RESUME = "resume"
    ABORT = "abort"


class ControlQueue:
    def __init__(self):
        self._q: queue.Queue[Control] = queue.Queue()

    def post(self, msg: Control) -> None:
        self._q.put(msg)

    def discard_pending_aborts(self) -> int:
        """Non-blocking: remove every ABORT currently queued, re-queuing any
        non-ABORT (PAUSE/RESUME) messages in order. Returns the count removed.

        Used at tool-round boundaries (app.py) to drop a stale
        watchdog ABORT that the engine self-terminated past without consuming,
        so it can't wrongly abort the next round. Only messages already in the
        queue at call time are affected -- a USER abort posted afterwards (e.g.
        during tool execution) is untouched and still stops the turn."""
        kept: list[Control] = []
        removed = 0
        while True:
            try:
                msg = self._q.get_nowait()
            except queue.Empty:
                break
            if msg == Control.ABORT:
                removed += 1
            else:
                kept.append(msg)
        for msg in kept:
            self._q.put(msg)
        return removed

    def checkpoint(self) -> str:
        """Engine calls this once per token step. Non-blocking unless paused."""
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            return "continue"
        if msg == Control.ABORT:
            return "abort"
        if msg == Control.PAUSE:
            while True:  # block, keeping cache intact, until resume/abort
                nxt = self._q.get()
                if nxt == Control.ABORT:
                    return "abort"
                if nxt == Control.RESUME:
                    return "continue"
        return "continue"
