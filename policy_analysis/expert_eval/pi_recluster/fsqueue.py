"""Filesystem-backed work queue for parallel SLURM workers.

Jean Zay compute nodes have no route to Postgres, so queue state lives
entirely in atomic renames on the shared filesystem (GPFS rename is atomic,
so exactly one claimant wins). Layout under a queue root:

    todo/<unit>.jsonl                    unclaimed work units
    claimed/<unit>.jsonl.claim-<owner>   units being processed
    done/<unit>.jsonl                    finished units

Workers killed mid-unit leave their file in claimed/; `requeue-stale`
returns claims whose mtime is older than a TTL to todo/ (claims are touched
on claim and re-touched by the worker per chunk as a heartbeat). Output
files, not this queue, are the source of truth for what was computed:
re-processing a requeued unit overwrites the same output idempotently.

CLI (used by the SLURM scripts):
    python fsqueue.py stats --root <root>
    python fsqueue.py requeue-stale --root <root> --ttl 7200
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

CLAIM_SEP = ".claim-"
_DIRS = ("todo", "claimed", "done")


def init(root: Path) -> None:
    for d in _DIRS:
        (root / d).mkdir(parents=True, exist_ok=True)


def claim(root: Path, owner: str) -> Path | None:
    """Atomically move one todo unit to claimed/; None when the queue is empty."""
    for path in sorted((root / "todo").glob("*.jsonl")):
        target = root / "claimed" / f"{path.name}{CLAIM_SEP}{owner}"
        try:
            os.rename(path, target)
        except OSError:
            continue  # another worker won the rename race
        os.utime(target)  # rename preserves mtime; stamp the claim time
        return target
    return None


def unit_name(claimed_path: Path) -> str:
    return claimed_path.name.split(CLAIM_SEP, 1)[0]


def heartbeat(claimed_path: Path) -> None:
    try:
        os.utime(claimed_path)
    except OSError:
        pass  # claim was requeued as stale; the unit will be redone


def mark_done(root: Path, claimed_path: Path) -> bool:
    try:
        os.replace(claimed_path, root / "done" / unit_name(claimed_path))
        return True
    except OSError:
        return False  # requeued as stale meanwhile — redo is idempotent


def requeue_stale(root: Path, ttl_seconds: float) -> int:
    now = time.time()
    n = 0
    for path in (root / "claimed").glob(f"*{CLAIM_SEP}*"):
        try:
            if now - path.stat().st_mtime > ttl_seconds:
                os.rename(path, root / "todo" / unit_name(path))
                n += 1
        except OSError:
            continue  # concurrent requeue/done — either way it's handled
    return n


def stats(root: Path) -> dict[str, int]:
    return {d: sum(1 for _ in (root / d).glob("*")) for d in _DIRS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stats")
    p.add_argument("--root", required=True)
    p = sub.add_parser("requeue-stale")
    p.add_argument("--root", required=True)
    p.add_argument("--ttl", type=float, default=7200, help="seconds before a claim is stale")
    args = ap.parse_args()

    root = Path(args.root)
    init(root)
    if args.cmd == "stats":
        print(stats(root))
    elif args.cmd == "requeue-stale":
        n = requeue_stale(root, args.ttl)
        print(f"requeued {n} stale unit(s); queue now {stats(root)}")


if __name__ == "__main__":
    main()
