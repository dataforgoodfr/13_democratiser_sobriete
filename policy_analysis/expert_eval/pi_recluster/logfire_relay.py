"""Ship pipeline key events from the shared filesystem into Logfire.

Compute nodes are offline, so scripts write key events as JSONL via
`_events.py`; this relay runs where there IS internet (a Jean Zay login
node, or your laptop over sshfs/rsync) and forwards them. A byte-offset
cursor per source file (kept in <events-dir>/.relay_cursors.json) makes the
relay idempotent — each event ships once, and it is safe to rerun or run in
--watch mode alongside the jobs.

Without LOGFIRE_TOKEN set it degrades to printing the events it would ship
(send_to_logfire="if-token-present"), so the plumbing can be tested before
the token exists.

Usage (login node):
    export LOGFIRE_TOKEN=...   # never commit this
    python logfire_relay.py --events-dir $WORK/pi_recluster/events --watch 300
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import logfire

# Everything else ships as info.
LEVELS = {
    "stage_failed": "error",
    "structured_output_degraded": "warn",
}


def ship(events_dir: Path) -> int:
    cursor_path = events_dir / ".relay_cursors.json"
    cursors = json.loads(cursor_path.read_text()) if cursor_path.exists() else {}
    shipped = 0
    for src in sorted(events_dir.glob("*.jsonl")):
        pos = cursors.get(src.name, 0)
        with src.open() as fh:
            fh.seek(pos)
            while True:
                line = fh.readline()
                if not line or not line.endswith("\n"):
                    break  # EOF or half-written line — picked up next pass
                pos = fh.tell()
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                event = rec.pop("event", "event")
                level = LEVELS.get(event, "info")
                status = rec.get("status")
                if event == "stage_end" and status == "failed":
                    level = "error"
                # Original wall-clock time rides along as an attribute; the
                # span timestamp is relay time (we can't backdate logs).
                logfire.log(level, event, attributes=rec)
                shipped += 1
        cursors[src.name] = pos
    cursor_path.write_text(json.dumps(cursors, indent=0))
    return shipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events-dir", required=True)
    ap.add_argument("--watch", type=float, default=None,
                    help="poll interval in seconds; omit for a single pass")
    ap.add_argument("--service-name", default="sufficiency-pipeline")
    args = ap.parse_args()

    logfire.configure(
        service_name=args.service_name,
        send_to_logfire="if-token-present",
        console=False,
    )

    events_dir = Path(args.events_dir)
    while True:
        n = ship(events_dir)
        print(f"[relay] shipped {n} event(s) from {events_dir}")
        if args.watch is None:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
