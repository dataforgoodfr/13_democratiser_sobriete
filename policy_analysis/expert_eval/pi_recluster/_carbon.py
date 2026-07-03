"""Small helper: one context manager that instruments each script with codecarbon.

All stages of the pipeline log to `data/carbon/emissions.csv` with a
`project_name` field distinguishing the stage (download, classify, filter,
recluster, ...). Aggregating them across a run gives the whole-pipeline CO2
budget for this iteration.

Reasoning: the codecarbon SDK auto-detects the country / grid mix of the
cloud box on start. If detection fails (e.g. no internet during a run), it
falls back to a conservative global average — so a value always comes out.

Usage:
    from _carbon import track
    with track("stage-name"):
        ...
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path


@contextlib.contextmanager
def track(stage: str, out_dir: str | os.PathLike = "data/carbon"):
    """Wrap a block in a codecarbon EmissionsTracker.

    On stop, the tracker appends a row to <out_dir>/emissions.csv. Import
    happens lazily so scripts still work if codecarbon is missing.
    """
    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        print("[codecarbon] not installed — skipping emissions tracking")
        yield
        return

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    tracker = EmissionsTracker(
        project_name=f"pi_recluster:{stage}",
        output_dir=str(out_dir),
        output_file="emissions.csv",
        log_level="warning",
        allow_multiple_runs=True,
    )
    tracker.start()
    try:
        yield tracker
    finally:
        emissions = tracker.stop()
        try:
            kwh = tracker.final_emissions_data.energy_consumed
        except Exception:
            kwh = None
        if kwh is not None:
            print(f"[codecarbon] stage={stage}  emissions={emissions:.4g} kg CO2eq  energy={kwh:.4g} kWh")
        else:
            print(f"[codecarbon] stage={stage}  emissions={emissions:.4g} kg CO2eq")
