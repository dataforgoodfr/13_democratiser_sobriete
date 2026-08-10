"""Append-only key-event log for pipeline monitoring.

Jean Zay compute nodes have no internet, so events are written as JSONL to a
directory on the shared filesystem and shipped to Logfire afterwards by
`logfire_relay.py` running on a login node. Stdlib-only so it is safe to
import in the GPU hot path, and emit() never raises — monitoring must not
kill a 10 h run.

Events land in $PIPELINE_EVENTS_DIR (default: data/events), one file per
process named <job>_<array_task>_<pid>.jsonl.

Usage:
    from _events import emit
    emit("chunk_done", done=8000, total=645000, fails=12)
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path

_fh = None


def _context() -> dict:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }


def emit(event: str, **attrs) -> None:
    """Append one event record; swallows every error."""
    global _fh
    try:
        if _fh is None:
            root = Path(os.environ.get("PIPELINE_EVENTS_DIR", "data/events"))
            root.mkdir(parents=True, exist_ok=True)
            ctx = _context()
            name = f"{ctx['job_id'] or 'local'}_{ctx['array_task_id'] or 0}_{ctx['pid']}"
            _fh = (root / f"{name}.jsonl").open("a")
        rec = {"ts": time.time(), "event": event, **_context(), **attrs}
        _fh.write(json.dumps(rec, default=str) + "\n")
        _fh.flush()
    except Exception as e:
        print(f"[events] emit({event}) failed: {e}")
