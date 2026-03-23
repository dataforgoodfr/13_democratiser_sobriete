#!/usr/bin/env python3
"""Export weighted LFS workers by REGION, NACE3D and SIZEFIRM."""

from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd
import numpy as np


DEFAULT_DATA_ROOT = Path(
	r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI/0_data/LFS/"
	r"LFS_1983-2023_YEARLY_full_set-002/LFS_1983-2023_YEARLY_full_set"
)
DEFAULT_COUNTRY = "FR"
DEFAULT_YEAR = 2021


SIZEFIRM_CODE_ORDER = [
	"1-9",
	"01", "02", "03", "04", "05", "06", "07", "08", "09",
	"10", "11", "12", "13", "14", "15", "99", "BLANK",
]


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
	"""Resolve dataset-specific columns to canonical REGION/NACE3D/SIZEFIRM/COEFFY."""
	candidates = {
		"REGION": ["REGION", "REGION_3D", "REGION_2D", "REGION_2DW"],
		"NACE3D": ["NACE3D", "NACE2_3D", "NACE2_2D", "NACE2_1D", "NACE1_1D"],
		"SIZEFIRM": ["SIZEFIRM"],
		"COEFFY": ["COEFFY", "COEFF"],
		"ILOSTAT": ["ILOSTAT"],
	}

	resolved: dict[str, str] = {}
	for key, options in candidates.items():
		for col in options:
			if col in df.columns:
				resolved[key] = col
				break

	missing = [k for k in ["REGION", "NACE3D", "SIZEFIRM", "COEFFY"] if k not in resolved]
	if missing:
		raise ValueError(f"Missing required source columns for: {missing}")

	return resolved


def load_lfs_data(country_code: str, year: int, data_root: Path) -> pd.DataFrame:
	"""Load one country-year yearly LFS file."""
	year_file = data_root / f"{country_code}_YEAR" / f"{country_code}{year}_y.csv"
	if not year_file.exists():
		raise FileNotFoundError(f"LFS file not found: {year_file}")

	df = pd.read_csv(year_file)
	print(f"Loaded {year_file.name}: {df.shape[0]:,} rows, {df.shape[1]} columns")
	return df


def normalize_sizefirm(value: object) -> str:
	"""Normalize SIZEFIRM into canonical codes, preserving blanks and unknowns."""
	if pd.isna(value):
		return "BLANK"

	text = str(value).strip()
	if text == "":
		return "BLANK"

	upper_text = text.upper()
	if upper_text in {"1-9", "01-09", "01–09", "01—09"}:
		return "1-9"

	numeric = pd.to_numeric(text, errors="coerce")
	if pd.isna(numeric):
		return upper_text

	code_int = int(numeric)
	if 1 <= code_int <= 99:
		return f"{code_int:02d}"
	return str(code_int)


def build_weighted_region_nace_sizefirm(df: pd.DataFrame) -> pd.DataFrame:
	"""Build weighted workers table by REGION + NACE3D with SIZEFIRM columns."""
	resolved = resolve_columns(df)
	region_col = resolved["REGION"]
	nace_col = resolved["NACE3D"]
	sizefirm_col = resolved["SIZEFIRM"]
	weight_col = resolved["COEFFY"]
	ilostat_col = resolved.get("ILOSTAT")

	print("Column mapping used:")
	print(f"  REGION  <- {region_col}")
	print(f"  NACE3D  <- {nace_col}")
	print(f"  SIZEFIRM<- {sizefirm_col}")
	print(f"  COEFFY  <- {weight_col}")

	work = df.copy()

	if ilostat_col is not None:
		work[ilostat_col] = pd.to_numeric(work[ilostat_col], errors="coerce")
		work = work[work[ilostat_col] == 1].copy()

	work["REGION"] = work[region_col].astype(str).str.strip()
	work["NACE3D"] = work[nace_col].astype(str).str.strip()
	work["COEFFY"] = pd.to_numeric(work[weight_col], errors="coerce")
	work = work.dropna(subset=["COEFFY"]).copy()
	work = work[(work["REGION"] != "") & (work["NACE3D"] != "")].copy()

	work["SIZEFIRM_norm"] = work[sizefirm_col].apply(normalize_sizefirm)

	weighted_total = (
		work.groupby(["REGION", "NACE3D"], as_index=False)["COEFFY"]
		.sum()
		.rename(columns={"COEFFY": "workers_weighted_total"})
	)

	weighted_by_size = (
		work.groupby(["REGION", "NACE3D", "SIZEFIRM_norm"], as_index=False)["COEFFY"]
		.sum()
		.pivot(index=["REGION", "NACE3D"], columns="SIZEFIRM_norm", values="COEFFY")
		.fillna(0.0)
	)

	ordered_codes = [code for code in SIZEFIRM_CODE_ORDER if code in weighted_by_size.columns]
	other_codes = sorted([code for code in weighted_by_size.columns if code not in ordered_codes])
	weighted_by_size = weighted_by_size[ordered_codes + other_codes]
	weighted_by_size = weighted_by_size.rename(
		columns={code: f"workers_weighted_sizefirm_{code}" for code in weighted_by_size.columns}
	).reset_index()

	result = weighted_total.merge(weighted_by_size, on=["REGION", "NACE3D"], how="left")
	result = result.sort_values(["REGION", "NACE3D"]).reset_index(drop=True)
	return result


def verify_presence(df: pd.DataFrame) -> None:
	"""Print quick checks to verify key columns and data are present."""
	for col in ["REGION", "NACE3D", "SIZEFIRM", "COEFFY", "REGION_2D", "NACE2_1D"]:
		non_null = df[col].notna().sum() if col in df.columns else 0
		print(f"{col}: present={col in df.columns}, non_null={non_null:,}")

	try:
		resolved = resolve_columns(df)
		print("Resolved columns:")
		print(resolved)
	except ValueError as error:
		print(f"Column resolution error: {error}")

	region_col = "REGION" if "REGION" in df.columns else ("REGION_2D" if "REGION_2D" in df.columns else None)
	nace_col = "NACE3D" if "NACE3D" in df.columns else ("NACE2_1D" if "NACE2_1D" in df.columns else None)
	if region_col is not None:
		print(f"Distinct {region_col}: {df[region_col].nunique(dropna=True):,}")
	if nace_col is not None:
		print(f"Distinct {nace_col}: {df[nace_col].nunique(dropna=True):,}")
	if "SIZEFIRM" in df.columns:
		sizefirm_norm = df["SIZEFIRM"].apply(normalize_sizefirm)
		size_counts = sizefirm_norm.value_counts(dropna=False).head(15)
		print("Top SIZEFIRM normalized values:")
		print(size_counts.to_string())


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description=(
			"Build weighted workers export by REGION and NACE3D with SIZEFIRM columns"
		)
	)
	parser.add_argument("--country", default=DEFAULT_COUNTRY, help="Country code, e.g. FR")
	parser.add_argument("--year", type=int, default=DEFAULT_YEAR, help="Year, e.g. 2021")
	parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Root LFS yearly folder")
	parser.add_argument(
		"--output",
		type=Path,
		default=Path(__file__).parent / "lfs_weighted_workers_by_region_nace3d_sizefirm_2021.csv",
		help="Output CSV path",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	df = load_lfs_data(args.country, args.year, args.data_root)
	print("\nData verification (source columns):")
	verify_presence(df)

	result = build_weighted_region_nace_sizefirm(df)
	args.output.parent.mkdir(parents=True, exist_ok=True)
	result.to_csv(args.output, index=False)

	print("\nExport complete")
	print(f"Output file: {args.output}")
	print(f"Output shape: {result.shape[0]:,} rows, {result.shape[1]} columns")
	print("Sample columns:")
	print(result.columns[:20].tolist())
	print("\nHead:")
	print(result.head(5).to_string(index=False))


if __name__ == "__main__":
	main()
