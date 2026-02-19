from __future__ import annotations

import os
from pathlib import Path


def default_output_dir() -> Path:
    """Default output directory: stream3_visualization/Well-being/output."""
    return Path(__file__).resolve().parent.parent / "output"


def get_output_dir() -> Path:
    """Get the output directory for the currently running pipeline.

    Override with environment variable:
      - EWBI_OUTPUT_DIR
    """
    raw = os.getenv("EWBI_OUTPUT_DIR")
    if raw:
        # `strict=False` keeps this safe even if the path doesn't exist yet.
        return Path(raw).expanduser().resolve(strict=False)
    return default_output_dir()


_TRUE = {"1", "true", "t", "yes", "y", "on"}
_FALSE = {"0", "false", "f", "no", "n", "off"}


def env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    return default


def env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw


def env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default
