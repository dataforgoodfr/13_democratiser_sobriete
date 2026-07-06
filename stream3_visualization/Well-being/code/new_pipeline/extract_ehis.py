"""
EHIS Extractor
==============
Reads raw European Health Interview Survey (EHIS) files across waves 1-3 and
computes one CSV per indicator. No zero/NaN filtering.

Wave aggregation
----------------
Each wave is treated as a single observation point. All respondents in a wave
are pooled (across all years in the wave) and assigned a single representative
year (the midpoint of the wave's year range):
  Wave 1 (2006–2009) → 2007
  Wave 2 (2013–2015) → 2014
  Wave 3 (2018–2020) → 2019

Income quintile → decile mapping
---------------------------------
EHIS uses income quintiles (1-5). These are expanded to deciles 1-10 so that
EHIS output has the same 'decile' column as all other sources:
  Quintile 1 → Deciles 1 & 2   (same value applied to both)
  Quintile 2 → Deciles 3 & 4
  Quintile 3 → Deciles 5 & 6
  Quintile 4 → Deciles 7 & 8
  Quintile 5 → Deciles 9 & 10

Output per indicator: output/indicators/{CODE}.csv
  country, year, decile (1-10 + "All"), value, n_obs, n_weighted

File layout
-----------
Wave 1 (2006-2009): {EHIS_DIR}/EHIS all waves/EHIS wave 1/Data EHIS/EHIS1.csv
  separator=',', weight=PWGT, year from YEAR or IP04 (ddmmyyyy)
  column renames: PWGT->WGT, IP01->COUNTRY, HA01A->HA1A, HA01B->HA1B,
                  FV01->FV1, PE06->PE6, SK01->SK1, AL01->AL1, DH03->DH3

Wave 2 (2013-2015): {EHIS_DIR}/EHIS all waves/EHIS wave 2/{CC}_Anonymisation.csv
  separator=';', weight=WGT, year from REFYEAR

Wave 3 (2018-2020): {EHIS_DIR}/EHIS all waves/EHIS wave 3/{CC}_Anonymisation.csv
  separator=';', weight=WGT, year from YEAR or REFDATE (YYYYMMDD)

Usage
-----
    python extract_ehis.py                   # compute missing indicators
    python extract_ehis.py --force AH-EHIS-1
    python extract_ehis.py --force-all
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config
from indicators import BY_SOURCE

config.ensure_dirs()

# ─── Parquet cache dir ────────────────────────────────────────────────────────
CACHE_EHIS = config.CACHE_DIR / "ehis"
CACHE_EHIS.mkdir(parents=True, exist_ok=True)

EHIS_ALL_WAVES = config.EHIS_RAW_DIR / "EHIS all waves"

WAVE_YEAR_RANGE = {1: (2006, 2009), 2: (2013, 2015), 3: (2018, 2020)}

# Wave 1 column renames
WAVE1_RENAME = {
    "PWGT": "WGT", "IP01": "COUNTRY",
    "HA01A": "HA1A", "HA01B": "HA1B",
    "FV01": "FV1",   "PE06": "PE6",
    "SK01": "SK1",   "AL01": "AL1",
    "DH03": "DH3",
    "BMI01": "BM1",  # measured height in cm (wave 1 raw name)
    "BMI02": "BM2",  # measured weight in kg (wave 1 raw name)
}

# Columns to keep
EHIS_COLS = [
    "COUNTRY", "REFYEAR", "WGT",
    "HHINCOME", "IN04",
    "HA1A", "HA1B",                 # health status
    "FV1",                          # fruit/veg consumption
    "DH3",                          # dental hygiene
    "BM1",                          # measured height without shoes (cm); wave 1 = BMI01
    "BM2",                          # measured weight without clothes (kg); wave 1 = BMI02
    "BMI",                          # body mass index (categorical, IT wave 3 only: 1-4)
    "SS1",                          # stress
    "PE6",                          # physical exercise
    "UN2C",                         # unmet medical need
    "SK1",                          # smoking
    "AL1",                          # alcohol
    "AC1A",                         # accessibility
]


# ─── Flag conditions ──────────────────────────────────────────────────────────

def _flag(df: pd.DataFrame, code: str) -> pd.Series:
    nan_s = pd.Series(np.nan, index=df.index)

    def col(c):
        return df[c] if c in df.columns else nan_s.copy()

    def binary(c, vals, missing=None):
        s = pd.to_numeric(col(c), errors="coerce")
        if missing:
            s = s.where(~s.isin(missing), other=np.nan)
        valid = s.notna()
        f = nan_s.copy()
        f[valid] = s[valid].isin(vals).astype(float)
        return f

    def cmp(c, op, v, missing=None):
        s = pd.to_numeric(col(c), errors="coerce")
        if missing:
            s = s.where(~s.isin(missing), other=np.nan)
        valid = s.notna()
        f = nan_s.copy()
        if op == ">=":
            f[valid] = (s[valid] >= v).astype(float)
        elif op == "<=":
            f[valid] = (s[valid] <= v).astype(float)
        elif op == ">":
            f[valid] = (s[valid] > v).astype(float)
        elif op == "<":
            f[valid] = (s[valid] < v).astype(float)
        return f

    # EHIS uses negative values (-1=don't know, -2=not applicable, -3=not asked)
    # as well as 8/9/98/99 for refusals — all treated as missing.
    MISSING_EHIS = {-3, -2, -1, 8, 9, 98, 99}
    # BM1/BM2-specific missing codes per EHIS documentation:
    #   1 = Not stated,  -3 = Proxy (already in MISSING_EHIS)
    # NOTE: do NOT add 1 to MISSING_EHIS globally — it is a valid response in other variables.
    BM_MISSING = {1, -3}

    # Codes must match indicators.py exactly.
    # HA1A: perceived health (1=Very good … 5=Very bad); bad health = [3,4,5]
    # HA1B: chronic illness (1=Yes, 2=No)
    # FV1:  fruit freq (1=Every day … 4=Less than once/never); low = [4]
    # DH3:  dental care (1=In last 12 months … 5=Never); long ago/never = [4,5]
    # BM1:  measured height without shoes (cm). Wave 1 raw = BMI01, renamed in WAVE1_RENAME.
    # BM2:  measured weight without clothes (kg). Wave 1 raw = BMI02, renamed in WAVE1_RENAME.
    # BMI:  categorical 1-4 (IT wave 3 only; 4=Obese ≥30). All other wave 3 countries use
    #       continuous values — handled via BM1/BM2 above.
    # SS1:  social support (1=Always … 5=Never); no support = [4,5]
    # PE6:  physical activity outside work (1=Vigorous … 4=None); none = [4]
    # UN2C: unmet dental care (1=Yes, 2=No)
    # SK1:  smoking (1=Daily, 2=Occasional, 3=Ex, 4=Never); current = [1,2]
    # AL1:  alcohol frequency (1=Daily … 6=Never); daily/almost = [1]
    # AC1A: access to healthcare (1=Yes difficulties, 2=No); difficulty = [1]

    def bmi_obesity():
        """Obesity flag from measured height (BM1 cm) and weight (BM2 kg).

        Primary: BMI = BM2 / (BM1/100)^2 >= 30  (WHO threshold; consistent with
        EHIS categorical code 4 = obese, i.e. BMI >= 30).
        Fallback: categorical BMI column == 4 (used where BM1/BM2 absent,
        e.g. IT wave 3 which kept the 1-4 ordinal scale).
        Missing codes for BM1/BM2: 1 = Not stated, -3 = Proxy.
        """
        h = pd.to_numeric(col("BM1"), errors="coerce")
        w = pd.to_numeric(col("BM2"), errors="coerce")
        h = h.where(~h.isin(BM_MISSING), other=np.nan)
        w = w.where(~w.isin(BM_MISSING), other=np.nan)
        h_m = (h / 100).replace(0, np.nan)   # convert cm → m, guard /0
        bmi_cont = w / (h_m ** 2)

        # Categorical fallback (IT wave 3: values {-1,1,2,3,4})
        bmi_cat = pd.to_numeric(col("BMI"), errors="coerce")
        bmi_cat = bmi_cat.where(~bmi_cat.isin(MISSING_EHIS), other=np.nan)

        f = nan_s.copy()
        has_cont = bmi_cont.notna()
        f[has_cont] = (bmi_cont[has_cont] >= 30).astype(float)
        # Use categorical only where no continuous measurement is available
        cat_mask = ~has_cont & bmi_cat.notna()
        f[cat_mask] = (bmi_cat[cat_mask] == 4).astype(float)
        return f

    dispatch = {
        "AN-EHIS-1": lambda: binary("HA1A",  [3, 4, 5], missing=MISSING_EHIS),  # poor health
        "AE-EHIS-1": lambda: binary("FV1",   [4],       missing=MISSING_EHIS),  # low fruit intake
        "AE-EHIS-2": lambda: binary("DH3",   [4, 5],    missing=MISSING_EHIS),  # no recent dental
        "AN-EHIS-2": bmi_obesity,  # obesity: BMI>=30 from BM1/BM2; fallback categorical BMI==4
        "EC-EHIS-1": lambda: binary("SS1",   [4, 5],    missing=MISSING_EHIS),  # no social support
        "ED-EHIS-1": lambda: binary("HA1B",  [1],       missing=MISSING_EHIS),  # chronic illness
        "AH-EHIS-2": lambda: binary("PE6",   [0, 4],    missing=MISSING_EHIS),  # no physical activity (0=none wave 1-2, 4=none wave 3)
        "AC-EHIS-1": lambda: binary("UN2C",  [1],       missing=MISSING_EHIS),  # unmet dental need
        "AB-EHIS-1": lambda: binary("SK1",   [1, 2],    missing=MISSING_EHIS),  # current smoker
        "AB-EHIS-2": lambda: binary("AL1",   [1],       missing=MISSING_EHIS),  # daily alcohol
        "AB-EHIS-3": lambda: binary("AC1A",  [1],       missing=MISSING_EHIS),  # healthcare access
    }

    fn = dispatch.get(code)
    return fn() if fn else nan_s


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _weighted_share(flag: np.ndarray, weight: np.ndarray):
    avail = ~np.isnan(flag) & ~np.isnan(weight) & (weight > 0)
    n = int(avail.sum())
    if n == 0:
        return np.nan, 0, 0.0
    f, w = flag[avail], weight[avail]
    n_w = float(w.sum())
    if n_w == 0:
        return np.nan, n, 0.0
    return float((f * w).sum() / n_w * 100), n, n_w


def _extract_year_wave(df: pd.DataFrame, wave: int) -> pd.Series:
    """Return integer year series for a given wave."""
    if wave == 1:
        if "YEAR" in df.columns:
            return pd.to_numeric(df["YEAR"], errors="coerce")
        elif "IP04" in df.columns:
            return pd.to_numeric(
                df["IP04"].astype(str).str.strip().str[-4:], errors="coerce"
            )
    elif wave == 2:
        if "REFYEAR" in df.columns:
            return pd.to_numeric(df["REFYEAR"], errors="coerce")
    elif wave == 3:
        if "YEAR" in df.columns:
            return pd.to_numeric(df["YEAR"], errors="coerce")
        elif "REFDATE" in df.columns:
            return pd.to_numeric(
                df["REFDATE"].astype(str).str.strip().str[:4], errors="coerce"
            )
    return pd.Series(np.nan, index=df.index)


def _parse_quintile(df: pd.DataFrame) -> pd.Series:
    """
    Return quintile (1-5) from HHINCOME, falling back to IN04.
    Values outside 1-5 are set to NaN.
    """
    q = pd.to_numeric(df.get("HHINCOME", pd.Series(np.nan, index=df.index)),
                      errors="coerce")
    # Fallback: IN04 → quintile = ceil(IN04_numeric / 2)
    if "IN04" in df.columns:
        in04 = pd.to_numeric(df["IN04"].astype(str).str.strip(), errors="coerce")
        mask = q.isna() & in04.notna()
        q[mask] = ((in04[mask] + 1) // 2)
    q = q.where(q.between(1, 5), other=np.nan)
    return q


def _load_wave_frames() -> list[pd.DataFrame]:
    """Load all EHIS waves and return a list of standardised DataFrames."""
    frames: list[pd.DataFrame] = []
    wave_dir = EHIS_ALL_WAVES

    if not wave_dir.exists():
        print(f"  ⚠ EHIS raw data not found: {wave_dir}")
        return frames

    for wave in (1, 2, 3):
        yr_min, yr_max = WAVE_YEAR_RANGE[wave]

        if wave == 1:
            fpath = wave_dir / f"EHIS wave 1" / "Data EHIS" / "EHIS1.csv"
            if not fpath.exists():
                print(f"  ⚠ Wave 1 file not found: {fpath}")
                continue
            try:
                df = pd.read_csv(fpath, sep=",", low_memory=False)
                df = df.rename(columns=WAVE1_RENAME)
                df["REFYEAR"] = _extract_year_wave(df, 1)
                if "COUNTRY" not in df.columns:
                    if "IP01" in df.columns:
                        df["COUNTRY"] = df["IP01"]
                # Validate year range
                df = df[df["REFYEAR"].between(yr_min, yr_max)]
                df["wave"] = 1
                frames.append(df)
                print(f"  Wave 1: {len(df):,} rows, {df['COUNTRY'].nunique()} countries")
            except Exception as exc:
                print(f"  ⚠ Wave 1 load error: {exc}")

        else:
            folder = wave_dir / f"EHIS wave {wave}"
            if not folder.exists():
                print(f"  ⚠ Wave {wave} folder not found: {folder}")
                continue
            wave_files = list(folder.glob("*_Anonymisation.csv"))
            if not wave_files:
                print(f"  ⚠ Wave {wave}: no *_Anonymisation.csv files in {folder}")
                continue
            for fpath in tqdm(wave_files, desc=f"EHIS wave {wave}", leave=False):
                try:
                    df = pd.read_csv(fpath, sep=";", low_memory=False)
                    df["REFYEAR"] = _extract_year_wave(df, wave)
                    ctry = fpath.stem.split("_")[0].upper()
                    if "COUNTRY" not in df.columns:
                        df["COUNTRY"] = ctry
                    df = df[df["REFYEAR"].between(yr_min, yr_max)]
                    df["wave"] = wave
                    frames.append(df)
                except Exception as exc:
                    tqdm.write(f"    ⚠ {fpath.name}: {exc}")

    return frames


# ─── Main extraction ───────────────────────────────────────────────────────────

def extract_ehis(force_codes: set[str] | None = None, force_all: bool = False):
    ehis_indicators = BY_SOURCE.get("EHIS", [])
    all_codes = [ind["code"] for ind in ehis_indicators]

    codes_to_run = set()
    if force_all:
        codes_to_run = set(all_codes)
    else:
        for code in all_codes:
            out = config.INDICATORS_DIR / f"{code}.csv"
            if not out.exists() or (force_codes and code in force_codes):
                codes_to_run.add(code)

    if not codes_to_run:
        print("EHIS: all indicator CSVs are up to date.")
        return

    print(f"EHIS: computing {len(codes_to_run)} indicators …")

    # ── Load / build combined DataFrame, with Parquet caching ──
    cache_path = CACHE_EHIS / "EHIS_all_waves.parquet"
    if cache_path.exists():
        print("Loading EHIS waves from Parquet cache …")
        all_df = pd.read_parquet(cache_path)
    else:
        print("Loading EHIS waves from raw files …")
        frames = _load_wave_frames()
        if not frames:
            print("  ERROR: No EHIS data loaded.")
            return
        print(f"Combining {len(frames)} wave/country chunks …")
        all_df = pd.concat(frames, ignore_index=True)
        # Normalise COUNTRY column before caching
        all_df["COUNTRY"] = all_df["COUNTRY"].astype(str).str.strip().str.upper()
        try:
            all_df.to_parquet(cache_path, index=False)
            print(f"  Parquet cache written → {cache_path.name}")
        except Exception as exc:
            print(f"  ⚠ Could not write EHIS cache: {exc}")

    # Normalise COUNTRY (in case loaded from cache without re-normalising)
    all_df["COUNTRY"] = all_df["COUNTRY"].astype(str).str.strip().str.upper()

    # Income quintile — build as a new column on a copy to avoid fragmentation
    all_df = all_df.copy()
    all_df["quintile"] = _parse_quintile(all_df)

    # Weight
    wgt_col = "WGT"
    if wgt_col not in all_df.columns:
        all_df[wgt_col] = np.nan

    weight_arr   = pd.to_numeric(all_df[wgt_col], errors="coerce").to_numpy(dtype=float)
    quintile_arr = all_df["quintile"].to_numpy(dtype=float)

    accum: dict[str, list] = {c: [] for c in codes_to_run}

    # Group by (country, wave) — all years within a wave are pooled together.
    # valid_mask: wave column must be known (non-NaN)
    valid_mask = all_df["wave"].notna().to_numpy()
    groups = all_df.loc[valid_mask].groupby(["COUNTRY", "wave"]).groups

    for code in tqdm(codes_to_run, desc="EHIS indicators"):
        flag_arr = _flag(all_df, code).to_numpy(dtype=float)

        for (ctry, wave_num), grp_idx in groups.items():
            if ctry not in config.EU_COUNTRIES:
                continue
            # Use the actual REFYEAR from the raw data (one year per country×wave)
            yr_series = all_df.loc[grp_idx, "REFYEAR"].dropna()
            if yr_series.empty:
                continue
            wave_yr = int(yr_series.mode().iloc[0])
            idx = grp_idx
            f_g = flag_arr[all_df.index.get_indexer(idx)]
            w_g = weight_arr[all_df.index.get_indexer(idx)]
            q_g = quintile_arr[all_df.index.get_indexer(idx)]

            # Each quintile expands to two consecutive deciles (same value).
            for q in range(1, 6):
                mask = q_g == q
                val, n_obs, n_w = _weighted_share(f_g[mask], w_g[mask])
                d_low = 2 * q - 1  # Q1→D1, Q2→D3, Q3→D5, Q4→D7, Q5→D9
                d_high = 2 * q     # Q1→D2, Q2→D4, Q3→D6, Q4→D8, Q5→D10
                accum[code].append((ctry, wave_yr, d_low,  val, n_obs, n_w))
                accum[code].append((ctry, wave_yr, d_high, val, n_obs, n_w))

            # "All" aggregate — all respondents with a valid quintile
            all_q_mask = ~np.isnan(q_g)
            val_all, n_obs_all, n_w_all = _weighted_share(f_g[all_q_mask], w_g[all_q_mask])
            accum[code].append((ctry, wave_yr, "All", val_all, n_obs_all, n_w_all))

    print("\nSaving EHIS indicator CSV files …")
    for code in sorted(codes_to_run):
        rows = accum.get(code, [])
        if not rows:
            print(f"  [SKIP] {code}: no data")
            continue
        out_df = pd.DataFrame(rows, columns=["country", "year", "decile",
                                              "value", "n_obs", "n_weighted"])
        out_path = config.INDICATORS_DIR / f"{code}.csv"
        out_df.to_csv(out_path, index=False)
        print(f"  [OK] {code}: {len(out_df)} rows → {out_path.name}")

    print("\nEHIS done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract EHIS indicators")
    parser.add_argument("codes", nargs="*")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-all", dest="force_all", action="store_true")
    args = parser.parse_args()
    extract_ehis(force_codes=set(args.codes) if args.codes else None,
                 force_all=args.force_all)
