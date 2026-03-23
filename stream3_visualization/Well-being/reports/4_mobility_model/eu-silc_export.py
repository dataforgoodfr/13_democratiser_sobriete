#!/usr/bin/env python3
"""EU-SILC export: weighted population by region, activity status and NACE."""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


DEFAULT_BASE_DATA_PATH = Path(
	r"C:/Users/valentin.stuhlfauth/OneDrive - univ-lyon2.fr/1_WSL/1_EWBI/0_data/"
	r"EU-SILC/_Cross_2004-2023_full_set/_Cross_2004-2023_full_set"
)
DEFAULT_COUNTRY = "FR"
DEFAULT_YEAR = 2021


RB211_LABELS = {
	1: "Employed",
	2: "Unemployed",
	3: "Retired",
	4: "Unable to work due to long-standing health problems",
	5: "Student, pupil",
	6: "Fulfilling domestic tasks",
	7: "Compulsory military or civilian service",
	8: "Other",
}


def build_year_folder(base_data_path: Path, country: str, year: int) -> Path:
	return base_data_path / country / str(year)


def read_csv_required(file_path: Path, usecols: list[str]) -> pd.DataFrame:
	if not file_path.exists():
		raise FileNotFoundError(f"Missing file: {file_path}")
	return pd.read_csv(file_path, usecols=usecols, on_bad_lines="skip")


def normalize_person_id(series: pd.Series) -> pd.Series:
	return series.fillna(0).astype("int64").astype(str)


def normalize_household_id(series: pd.Series) -> pd.Series:
	return series.fillna(0).astype("int64").astype(str)


def normalize_nace(value: object) -> str | None:
	if pd.isna(value):
		return None

	text = str(value).strip()
	if text == "" or text.lower() == "nan":
		return None

	numeric = pd.to_numeric(text, errors="coerce")
	if pd.isna(numeric):
		return text

	return f"{int(numeric):02d}"


def load_dpr_data(base_data_path: Path, country: str, year: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	year_suffix = str(year)[-2:]
	year_folder = build_year_folder(base_data_path, country, year)

	d_file = year_folder / f"UDB_c{country}{year_suffix}D.csv"
	p_file = year_folder / f"UDB_c{country}{year_suffix}P.csv"
	r_file = year_folder / f"UDB_c{country}{year_suffix}R.csv"

	d_df = read_csv_required(d_file, ["DB010", "DB020", "DB030", "DB040"])
	p_df = read_csv_required(p_file, ["PB010", "PB020", "PB030", "PB040", "PL111A"])
	r_df = read_csv_required(r_file, ["RB010", "RB020", "RB030", "RB050", "RB211"])

	return d_df, p_df, r_df


def build_export_dataframe(d_df: pd.DataFrame, p_df: pd.DataFrame, r_df: pd.DataFrame) -> pd.DataFrame:
	p_df = p_df.copy()
	r_df = r_df.copy()
	d_df = d_df.copy()

	p_df["PB030"] = normalize_person_id(p_df["PB030"])
	r_df["RB030"] = normalize_person_id(r_df["RB030"])
	d_df["DB030"] = normalize_household_id(d_df["DB030"])

	pr_df = p_df.merge(
		r_df,
		left_on=["PB010", "PB020", "PB030"],
		right_on=["RB010", "RB020", "RB030"],
		how="inner",
	)
	pr_df["household_id"] = pr_df["PB030"].str[:-2]

	d_subset = d_df[["DB010", "DB020", "DB030", "DB040"]].drop_duplicates()
	merged = pr_df.merge(
		d_subset,
		left_on=["PB010", "PB020", "household_id"],
		right_on=["DB010", "DB020", "DB030"],
		how="left",
	)

	merged["region"] = merged["DB040"].where(merged["DB040"].notna(), pd.NA)
	merged["region"] = merged["region"].astype("string").str.strip()
	merged.loc[merged["region"].isin(["", "nan", "NaN", "<NA>"]), "region"] = pd.NA
	merged["nace"] = merged["PL111A"].apply(normalize_nace)
	merged["activity_status_code"] = pd.to_numeric(merged["RB211"], errors="coerce")
	merged["activity_status"] = merged["activity_status_code"].map(RB211_LABELS)

	merged["weight"] = pd.to_numeric(merged["RB050"], errors="coerce")
	missing_weight_mask = merged["weight"].isna()
	if missing_weight_mask.any():
		merged.loc[missing_weight_mask, "weight"] = pd.to_numeric(
			merged.loc[missing_weight_mask, "PB040"], errors="coerce"
		)

	export_df = merged.dropna(subset=["activity_status", "weight", "region"]).copy()

	result = (
		export_df.groupby(["region", "activity_status", "nace"], as_index=False, dropna=False)["weight"]
		.sum()
		.rename(columns={"weight": "number"})
		.sort_values(["region", "activity_status", "nace"]) 
		.reset_index(drop=True)
	)

	return result


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Export EU-SILC weighted population by region, activity status and NACE"
	)
	parser.add_argument("--country", default=DEFAULT_COUNTRY, help="Country code (e.g. FR)")
	parser.add_argument("--year", type=int, default=DEFAULT_YEAR, help="Year (e.g. 2021)")
	parser.add_argument(
		"--base-data-path",
		type=Path,
		default=DEFAULT_BASE_DATA_PATH,
		help="EU-SILC base directory",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path(__file__).parent / "eu_silc_region_activity_nace_population.csv",
		help="Output CSV path",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	d_df, p_df, r_df = load_dpr_data(args.base_data_path, args.country, args.year)
	print(f"Loaded D: {d_df.shape}, P: {p_df.shape}, R: {r_df.shape}")

	result = build_export_dataframe(d_df, p_df, r_df)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	result.to_csv(args.output, index=False)

	print("Export complete")
	print(f"Output: {args.output}")
	print(f"Shape: {result.shape}")
	print("Columns:", result.columns.tolist())
	print("Head:")
	print(result.head(10).to_string(index=False))


if __name__ == "__main__":
	main()
