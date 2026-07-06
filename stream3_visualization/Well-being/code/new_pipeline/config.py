"""
Pipeline Configuration
======================
Edit EXTERNAL_DATA_ROOT to point to your raw data directory.
All other paths are derived automatically from the repo structure.
"""
from pathlib import Path

# ── Raw data location (update this to match your environment) ─────────────────
EXTERNAL_DATA_ROOT = Path(
    r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI"
)

# ── Derived raw-data paths ────────────────────────────────────────────────────
SILC_RAW_DIR  = EXTERNAL_DATA_ROOT / "0_data" / "EU-SILC" / "_Cross_2004-2023_full_set" / "_Cross_2004-2023_full_set"
HBS_RAW_DIR   = EXTERNAL_DATA_ROOT / "0_data" / "HBS"
LFS_RAW_DIR   = EXTERNAL_DATA_ROOT / "0_data" / "LFS" / "LFS_1983-2023_YEARLY_full_set-002" / "LFS_1983-2023_YEARLY_full_set"
EHIS_RAW_DIR  = EXTERNAL_DATA_ROOT / "0_data" / "EHIS"

# ── Pipeline output paths ─────────────────────────────────────────────────────
_HERE         = Path(__file__).parent          # new_pipeline/
INDICATORS_DIR = _HERE / "output" / "indicators"   # one CSV per indicator code
CACHE_DIR      = _HERE / "output" / "cache"        # merged raw data (speed cache)
QUALITY_DIR    = _HERE / "output" / "quality"      # quality report outputs

# ── Countries in scope ────────────────────────────────────────────────────────
# Standard EU-27 + Norway, Switzerland, Iceland, UK, Serbia
EU_COUNTRIES = [
    "AT", "BE", "BG", "CH", "CY", "CZ", "DE", "DK", "EE", "EL",
    "ES", "FI", "FR", "HR", "HU", "IE", "IS", "IT", "LT", "LU",
    "LV", "MT", "NL", "NO", "PL", "PT", "RO", "RS", "SE", "SI",
    "SK", "UK",
]

# The 27 EU member states (excludes CH, IS, NO, RS, UK)
EU27 = [
    "AT", "BE", "BG", "CY", "CZ", "DE", "DK", "EE", "EL", "ES",
    "FI", "FR", "HR", "HU", "IE", "IT", "LT", "LU", "LV", "MT",
    "NL", "PL", "PT", "RO", "SE", "SI", "SK",
]

# ── EU-SILC years ─────────────────────────────────────────────────────────────
SILC_YEARS = list(range(2004, 2024))   # 2004–2023

# ── HBS waves ─────────────────────────────────────────────────────────────────
HBS_YEARS = [2010, 2015, 2020]

# ── LFS years ─────────────────────────────────────────────────────────────────
LFS_YEARS = list(range(2005, 2024))

# ── EHIS waves ────────────────────────────────────────────────────────────────
EHIS_WAVES = {
    1: (2006, 2009),
    2: (2013, 2015),
    3: (2018, 2020),
}

def ensure_dirs():
    """Create all output directories if they don't exist."""
    for d in (INDICATORS_DIR, CACHE_DIR, QUALITY_DIR):
        d.mkdir(parents=True, exist_ok=True)
