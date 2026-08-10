"""Small helper: one context manager that instruments each script with codecarbon.

All stages of the pipeline log to `data/carbon/emissions.csv` with a
`project_name` field distinguishing the stage (download, classify, filter,
recluster, ...). Aggregating them across a run gives the whole-pipeline CO2
budget for this iteration.

Reasoning: the codecarbon SDK auto-detects the country / grid mix of the
cloud box on start. If detection fails (e.g. no internet during a run), it
falls back to a conservative global average — so a value always comes out.

Every stage also emits `stage_start` / `stage_end` key events (see
`_events.py`) so a monitoring relay can follow the run without parsing
stdout; energy/CO2 figures ride along on `stage_end`.

Usage:
    from _carbon import track
    with track("stage-name"):
        ...
"""
from __future__ import annotations

import contextlib
import os
from pathlib import Path

from _events import emit


@contextlib.contextmanager
def track(stage: str, out_dir: str | os.PathLike = "data/carbon"):
    """Wrap a block in a codecarbon EmissionsTracker.

    On stop, the tracker appends a row to <out_dir>/emissions.csv. Import
    happens lazily so scripts still work if codecarbon is missing.
    """
    emit("stage_start", stage=stage)
    tracker = None
    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        print("[codecarbon] not installed — skipping emissions tracking")
    else:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        tracker = EmissionsTracker(
            project_name=f"pi_recluster:{stage}",
            output_dir=str(out_dir),
            output_file="emissions.csv",
            log_level="warning",
            allow_multiple_runs=True,
        )
        tracker.start()

    status = "ok"
    try:
        yield tracker
    except BaseException:
        status = "failed"
        raise
    finally:
        attrs: dict = {}
        if tracker is not None:
            emissions = tracker.stop()
            try:
                kwh = tracker.final_emissions_data.energy_consumed
            except Exception:
                kwh = None
            attrs = {"emissions_kg": emissions, "energy_kwh": kwh}
            if kwh is not None:
                print(f"[codecarbon] stage={stage}  emissions={emissions:.4g} kg CO2eq"
                      f"  energy={kwh:.4g} kWh")
            else:
                print(f"[codecarbon] stage={stage}  emissions={emissions:.4g} kg CO2eq")
        emit("stage_end", stage=stage, status=status, **attrs)
