"""CH decomposition visual export.

Generates a standalone visual based on the Decomposition dashboard data
(`Decomposition/Output/unified_decomposition_data.csv`).

Requested visual:
- Zone: Switzerland
- Scenario: Scenario Zer0 C
- Chart: "Share of Planned CO2 Reduction by Lever"
- Legend: Sector (instead of Scenario) so all sectors are visible on one chart.

Outputs are written to: Decomposition/reports/CH/decomposition_visual/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px


LEVER_ORDER = [
	"Population",
	"Sufficiency",
	"Energy Efficiency",
	"Supply Side Decarbonation",
]


def _project_paths() -> tuple[Path, Path]:
	ch_dir = Path(__file__).resolve().parent
	decomposition_dir = ch_dir.parent.parent  # .../Decomposition
	data_path = decomposition_dir / "Output" / "unified_decomposition_data.csv"
	out_dir = ch_dir / "decomposition_visual"
	return data_path, out_dir


def get_chart_data(data: pd.DataFrame) -> pd.DataFrame:
	required = {"Zone", "Sector", "Scenario", "Lever", "Contrib_2015_2050_pct"}
	missing = required.difference(data.columns)
	if missing:
		raise ValueError(f"Missing required columns in unified data: {sorted(missing)}")

	data = data.copy()
	data["Scenario"] = data["Scenario"].astype(str).str.strip()
	data["Zone"] = data["Zone"].astype(str).str.strip()
	data["Sector"] = data["Sector"].astype(str).str.strip()
	data["Lever"] = data["Lever"].astype(str).str.strip()

	chart_data = data[
		(data["Zone"] == "Switzerland")
		& (data["Scenario"] == "Scenario Zer0 C")
		& (data["Lever"] != "Total")
	].copy()

	if chart_data.empty:
		zones = sorted(data["Zone"].dropna().unique().tolist())
		scenarios = sorted(data["Scenario"].dropna().unique().tolist())
		raise ValueError(
			"No rows after filtering for Switzerland / Scenario Zer0 C. "
			f"Available zones include: {zones[:10]}{'...' if len(zones) > 10 else ''}. "
			f"Available scenarios include: {scenarios[:10]}{'...' if len(scenarios) > 10 else ''}."
		)

	chart_data["Lever"] = pd.Categorical(chart_data["Lever"], categories=LEVER_ORDER, ordered=True)
	chart_data = chart_data.sort_values(["Lever", "Sector"], kind="stable")

	return chart_data


def build_figure(chart_data: pd.DataFrame) -> "px.Figure":
	if chart_data.empty:
		raise ValueError("chart_data is empty")

	y_col = "Contrib_2015_2050_pct"
	fig = px.bar(
		chart_data,
		x="Lever",
		y=y_col,
		color="Sector",
		barmode="group",
		title="Share of Planned CO2 Reduction by Lever - All Sectors (Switzerland) - Scenario Zer0 C",
		labels={
			y_col: "Contribution (%)",
			"Lever": "Lever",
			"Sector": "Sector",
		},
	)

	# Match the dashboard styling as closely as possible
	y_min = float(chart_data[y_col].min())
	y_max = float(chart_data[y_col].max())
	yaxis_range = [min(-20.0, y_min - 5.0), max(140.0, y_max + 5.0)]

	fig.update_layout(
		title=dict(font=dict(size=16, color="#f4d03f"), x=0.5, y=0.95),
		xaxis_title="Levers",
		yaxis_title="Contribution to CO2 Change (%)",
		height=600,
		showlegend=True,
		plot_bgcolor="white",
		font=dict(family="Arial, sans-serif", size=14),
		margin=dict(t=90, b=50, l=60, r=60),
		yaxis=dict(range=yaxis_range),
		legend_title_text="Sector",
	)
	fig.update_xaxes(showgrid=False)
	fig.update_yaxes(showgrid=False)

	# Data labels (can get busy with many sectors, but keeps the app’s style)
	fig.update_traces(texttemplate="%{y:.0f}%", textposition="outside", cliponaxis=False)

	fig.update_traces(
		hovertemplate=(
			"<b>%{x}</b><br>"
			+ "<b>Sector:</b> %{fullData.name}<br>"
			+ "<b>Contribution:</b> %{y:.0f}%<br>"
			+ "<extra></extra>"
		)
	)

	return fig


def build_export_table(chart_data: pd.DataFrame, title: str) -> pd.DataFrame:
	export_df = chart_data[["Sector", "Lever", "Contrib_2015_2050_pct"]].copy()
	export_df = export_df.rename(columns={"Contrib_2015_2050_pct": "value"})
	export_df.insert(0, "visual_number", np.nan)
	export_df.insert(1, "visual_name", title)
	export_df.insert(2, "year", np.nan)
	export_df.insert(5, "decile", np.nan)
	export_df["unit"] = "%"
	export_df = export_df[
		[
			"visual_number",
			"visual_name",
			"year",
			"Sector",
			"Lever",
			"decile",
			"value",
			"unit",
		]
	]
	return export_df


def main() -> None:
	data_path, out_dir = _project_paths()
	out_dir.mkdir(parents=True, exist_ok=True)

	if not data_path.exists():
		raise FileNotFoundError(f"Could not find unified data at: {data_path}")

	print(f"Loading: {data_path}")
	data = pd.read_csv(data_path)

	chart_data = get_chart_data(data)
	fig = build_figure(chart_data)

	png_path = out_dir / "CH_Switzerland_ScenarioZer0C_share_CO2_reduction_by_lever_by_sector.png"
	try:
		fig.write_image(png_path, width=1600, height=800, scale=2)
		print(f"[OK] Wrote PNG:  {png_path}")
	except Exception as exc:  # noqa: BLE001
		raise RuntimeError(
			"PNG export failed. Install plotly's image engine with 'pip install -U kaleido'. "
			f"Original error: {exc}"
		) from exc

	title = fig.layout.title.text or ""
	export_df = build_export_table(chart_data, title)
	xlsx_path = out_dir / "CH_Switzerland_ScenarioZer0C_share_CO2_reduction_by_lever_by_sector.xlsx"
	export_df.to_excel(xlsx_path, index=False)
	print(f"[OK] Wrote Excel: {xlsx_path}")


if __name__ == "__main__":
	main()

