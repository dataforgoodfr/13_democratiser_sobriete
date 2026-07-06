# EWBI New Pipeline

Clean, modular data pipeline for the European Well-Being Index (EWBI).
One file per indicator, no zero/NaN filtering during extraction.

---

## Structure

```
new_pipeline/
├── indicators.py       # Registry of all 69 EWBI indicators (single source of truth)
├── config.py           # Paths + scope settings (countries, years, waves)
├── extract_eusilc.py   # EU-SILC extractor  (~40 indicators)
├── extract_hbs.py      # HBS extractor       (~18 indicators)
├── extract_lfs.py      # LFS extractor       (~10 indicators)
├── extract_ehis.py     # EHIS extractor      (~11 indicators)
├── quality_check.py    # Quality checker
├── run_all.py          # Pipeline orchestrator
└── output/
    ├── indicators/     # One CSV per indicator: {CODE}.csv
    └── quality/        # quality_summary.csv + quality_report.xlsx
```

---

## Running the pipeline

```bash
# Run everything (skips indicators whose CSV already exists)
python run_all.py

# Recompute all indicators from scratch
python run_all.py --force-all

# Run only specific sources
python run_all.py --source eusilc lfs

# Only refresh quality report (extraction already done)
python run_all.py --quality-only

# Run a single extractor manually
python extract_lfs.py --force-all
python extract_eusilc.py HQ-SILC-1 HQ-SILC-2 --force
```

---

## Output format

Each indicator CSV (`output/indicators/{CODE}.csv`) has these columns:

| Column | Type | Description |
|---|---|---|
| `country` | str | ISO 2-letter country code (e.g. `FR`) |
| `year` | int | Reference year |
| `decile` | int or "All" | Income decile 1-10 (or quintile 1-5 for EHIS), `All` = all deciles combined |
| `value` | float | Share of population in condition (0-100 %) or NaN |
| `n_obs` | int | Number of observations |
| `n_weighted` | float | Sum of weights |

---

## Indicator registry

`indicators.py` is the single source of truth.
Each indicator has:

```python
{
    "code":       "HQ-SILC-1",
    "name":       "Inadequate housing quality",
    "source":     "EU-SILC",           # EU-SILC | HBS | LFS | EHIS
    "variable":   "HH040",             # raw variable name
    "condition":  "HH040 == 1",        # plain-text condition
    "level":      "household",         # household | personal
    "dimension":  "Housing quality",
    "in_ewbi":    True,                # included in published EWBI
}
```

### Adding a new indicator

1. Add a dict entry to `INDICATORS` in `indicators.py`.
2. Add the flag logic to the relevant extractor's `_flag()` / `_build_hh_flags()` / `_build_pers_flags()` function.
3. Re-run the extractor: `python extract_eusilc.py NEW-CODE`.

### Removing an indicator

Delete the entry from `INDICATORS` in `indicators.py` and remove the CSV from `output/indicators/`.

---

## Quality check

`quality_check.py` classifies each (code, country, year) as:

| Status | Meaning |
|---|---|
| `OK` | ≥3 valid non-zero decile values |
| `ABSENT` | All decile values are NaN – no source data |
| `ZERO_ALL` | ≥3 valid values, all exactly 0 – likely a data gap |
| `ZERO_PARTIAL` | Some zeros, not all – may be valid for upper-decile indicators |
| `NAN_PARTIAL` | Some NaN but not all – possible computation issue |
| `SPARSE` | Fewer than 3 valid decile values |

Output: `output/quality/quality_summary.csv` + `output/quality/quality_report.xlsx`

---

## Key design decisions

- **No zero-suppression**: If a source column is all-zeros, the indicator value is 0, not NaN. The quality checker flags these.
- **Incremental runs**: Each extractor skips codes whose CSV already exists unless `--force` is used.
- **EHIS uses quintiles**: The `decile` column stores 1-5 for EHIS indicators.
- **LFS income decile**: Uses `INCDECIL` (already in raw LFS files).
- **HBS income decile**: Computed from equivalised income (`EUR_HH095 / HB061`) with `HA10` weights.
