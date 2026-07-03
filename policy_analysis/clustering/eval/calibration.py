"""Calibration subsample for the LLM-as-judge intrusion task.

Goal: before running the judge on the full set, pick a small per-sector
subsample (default 5 per sector → ~55 items), have a human reference
labeller (Claude in this case) pick the intruder *blind*, and compare:

  - reference vs gold-by-construction → is the task well-formed?
  - judge vs gold-by-construction → judge accuracy
  - judge vs reference → inter-rater agreement

We emit two files for the calibration set:
  * items_blind.jsonl — no gold_intruder_index, no source_cluster_uid in
    candidate entries. Safe to show the reference labeller.
  * items_gold.jsonl  — full items with gold; consumed by the judge and
    by the scoring step.
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def split_calibration(
    items: list[dict], per_sector: int = 5, seed: int = 0
) -> tuple[list[dict], list[dict]]:
    """Return (calibration_items_full, remainder).

    Items in the calibration set retain their gold; ``items_blind.jsonl`` is
    derived from this list separately by ``write_blind``.
    """
    rng = random.Random(seed)
    by_sector: dict[str, list[dict]] = {}
    for it in items:
        by_sector.setdefault(it["sector"], []).append(it)

    cal: list[dict] = []
    keep_uids: set[str] = set()
    for sector, sec_items in by_sector.items():
        rng.shuffle(sec_items)
        picks = sec_items[:per_sector]
        cal.extend(picks)
        keep_uids.update(p["cluster_uid"] for p in picks)

    remainder = [it for it in items if it["cluster_uid"] not in keep_uids]
    return cal, remainder


def write_blind(items: list[dict], path: str | Path) -> Path:
    """Strip gold from each item so a reference labeller can label without
    peeking. We keep ``cluster_uid`` so we can re-join on scoring."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            blind = {
                "cluster_uid": it["cluster_uid"],
                "sector": it["sector"],
                "items": [{"text": c["text"]} for c in it["items"]],
            }
            f.write(json.dumps(blind) + "\n")
    return path


def write_gold(items: list[dict], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return path
