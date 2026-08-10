"""Split an items JSONL into fixed-size work units in an fsqueue todo/.

Run on the login node after `build_classify_input.py` (or
`build_pass2_input.py`). Idempotent restarts: done-ness is derived from the
outputs directory (--exclude-done-from), never from the queue itself —
rebuilding after a crash enqueues only rows that have no successful record
yet.

Usage:
    python build_queue.py \\
        --items to_classify_full.jsonl \\
        --queue-dir $WORK/pi_recluster/queue \\
        --exclude-done-from $WORK/pi_recluster/outputs \\
        --rebuild
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import fsqueue
from _events import emit
from classify_vllm import _load_done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--queue-dir", required=True)
    ap.add_argument("--unit-size", type=int, default=8_000)
    ap.add_argument("--exclude-done-from", default=None,
                    help="directory of output *.jsonl — rows already classified are skipped")
    ap.add_argument("--rebuild", action="store_true",
                    help="clear an existing queue before building")
    args = ap.parse_args()

    root = Path(args.queue_dir)
    if root.exists() and any(root.rglob("*.jsonl")):
        if not args.rebuild:
            raise SystemExit(f"{root} is not empty — pass --rebuild to replace it")
        shutil.rmtree(root)
    fsqueue.init(root)

    done: set = set()
    if args.exclude_done_from:
        for out in Path(args.exclude_done_from).glob("*.jsonl"):
            done |= _load_done(out)

    items = [
        json.loads(l) for l in Path(args.items).read_text().splitlines() if l.strip()
    ]
    todo = [it for it in items if (it["policy_uid"], it.get("cluster_uid")) not in done]

    units = 0
    for start in range(0, len(todo), args.unit_size):
        chunk = todo[start : start + args.unit_size]
        name = f"unit_{units:05d}.jsonl"
        tmp = root / f".{name}.tmp"
        tmp.write_text("".join(json.dumps(it) + "\n" for it in chunk))
        tmp.rename(root / "todo" / name)
        units += 1

    skipped = len(items) - len(todo)
    print(f"{len(items):,} items, {skipped:,} already done, "
          f"{len(todo):,} enqueued in {units} unit(s) of {args.unit_size:,}")
    emit("queue_built", queue=str(root), units=units, items=len(todo), skipped_done=skipped)


if __name__ == "__main__":
    main()
