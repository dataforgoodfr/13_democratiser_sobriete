import pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the CSV
csv_path = 'stream3_visualization/Well-being/output/unified_all_levels_1_to_5.csv'
df = pd.read_csv(csv_path)

# Filter for Decile == 'All', Level == 1, exclude 'All countries'
ewbi_rows = df[(df['Decile'] == 'All') & (df['Level'] == 1) & (df['Country'] != 'All Countries')]

# Get all countries
countries = ewbi_rows['Country'].unique()

# Get all years
years = ewbi_rows['Year'].unique()

# Get all Level 5 indicators
level5_rows = df[(df['Level'] == 5) & (df['Country'] != 'All Countries') & (df['Decile'] == 'All')]
indicators = level5_rows['Primary and raw data'].unique()

# Build a dictionary: country -> year -> {indicator: value}
country_year_indicator = defaultdict(lambda: defaultdict(dict))
for _, row in level5_rows.iterrows():
	country = row['Country']
	year = row['Year']
	indicator = row['Primary and raw data']
	value = row['Value']
	country_year_indicator[country][year][indicator] = value

# Build PCA input matrices per country (rows: years, columns: indicators)
pca_inputs = {}
for country in countries:
	data = []
	year_list = []
	for year in sorted(country_year_indicator[country].keys()):
		row = [country_year_indicator[country][year].get(ind, float('nan')) for ind in indicators]
		if not all(pd.isna(row)):
			data.append(row)
			year_list.append(year)
	if data:
		pca_inputs[country] = pd.DataFrame(data, columns=indicators, index=year_list)

# --- PCA computation ---
pca_results = {}
for country, df_input in pca_inputs.items():
	# Drop columns with all NaN and rows with any NaN
	df_clean = df_input.dropna(axis=1, how='all').dropna(axis=0)
	if df_clean.shape[0] < 2 or df_clean.shape[1] < 2:
		continue  # Not enough data for PCA
	pca = PCA()
	X = df_clean.values
	pca.fit(X)
	pca_results[country] = {
		'explained_variance_ratio': pca.explained_variance_ratio_,
		'components': pca.components_,
		'feature_names': df_clean.columns.tolist(),
		'years': df_clean.index.tolist()
	}



# --- ANALYSIS 1: PCA per country-year ---
import os
output_dir = 'stream3_visualization/Well-being/output/2_data_validation'
os.makedirs(output_dir, exist_ok=True)

print("\nPCA Analysis 1: Per Country-Year")
level1_rows = df[(df['Level'] == 1) & (df['Decile'] == 'All') & (df['Country'] != 'All Countries')]
level5_rows_full = df[(df['Level'] == 5) & (df['Country'] != 'All Countries')]

results1 = []
for country in countries:
	for year in sorted(level1_rows[level1_rows['Country'] == country]['Year'].unique()):
		y = level1_rows[(level1_rows['Country'] == country) & (level1_rows['Year'] == year)]['Value'].values
		X_df = level5_rows_full[(level5_rows_full['Country'] == country) & (level5_rows_full['Year'] == year)]
		if X_df.empty or len(y) == 0:
			continue
		X = X_df.pivot(index=['Decile','Quintile'], columns='Primary and raw data', values='Value').fillna(0)
		if X.shape[0] < 2 or X.shape[1] < 2:
			continue
		pca = PCA()
		scores = pca.fit_transform(X)
		results1.append({
			'Country': country,
			'Year': year,
			'ExplainedVariance': list(pca.explained_variance_ratio_),
			'Scores': scores.tolist(),
			'Index': X.index.tolist(),
			'FeatureNames': X.columns.tolist()
		})


# Save results1 to CSV
import csv
with open(os.path.join(output_dir, 'pca_country_year_results.csv'), 'w', newline='', encoding='utf-8') as f:
	writer = csv.writer(f)
	writer.writerow(['Country', 'Year', 'ExplainedVariance', 'Index', 'PC1', 'PC2'])
	for res in results1:
		for i, idx in enumerate(res['Index']):
			pc1 = res['Scores'][i][0] if len(res['Scores'][i]) > 0 else ''
			pc2 = res['Scores'][i][1] if len(res['Scores'][i]) > 1 else ''
			writer.writerow([
				res['Country'],
				res['Year'],
				'|'.join(map(str, res['ExplainedVariance'])),
				str(idx),
				pc1,
				pc2
			])



# --- ANALYSIS 2: PCA per country (all years stacked) ---
print("\nPCA Analysis 2: Per Country (All Years)")
results2 = []
for country in countries:
	y = level1_rows[level1_rows['Country'] == country]['Value'].values
	X_df = level5_rows_full[level5_rows_full['Country'] == country]
	if X_df.empty or len(y) == 0:
		continue
	X_df['Year_Decile_Quintile'] = X_df['Year'].astype(str) + '_' + X_df['Decile'].astype(str) + '_' + X_df['Quintile'].astype(str)
	X = X_df.pivot(index='Year_Decile_Quintile', columns='Primary and raw data', values='Value').fillna(0)
	if X.shape[0] < 2 or X.shape[1] < 2:
		continue
	pca = PCA()
	scores = pca.fit_transform(X)
	results2.append({
		'Country': country,
		'ExplainedVariance': list(pca.explained_variance_ratio_),
		'Scores': scores.tolist(),
		'Index': X.index.tolist(),
		'FeatureNames': X.columns.tolist()
	})


# Save results2 to CSV
with open(os.path.join(output_dir, 'pca_country_all_years_results.csv'), 'w', newline='', encoding='utf-8') as f:
	writer = csv.writer(f)
	writer.writerow(['Country', 'ExplainedVariance', 'Index', 'PC1', 'PC2'])
	for res in results2:
		for i, idx in enumerate(res['Index']):
			pc1 = res['Scores'][i][0] if len(res['Scores'][i]) > 0 else ''
			pc2 = res['Scores'][i][1] if len(res['Scores'][i]) > 1 else ''
			writer.writerow([
				res['Country'],
				'|'.join(map(str, res['ExplainedVariance'])),
				str(idx),
				pc1,
				pc2
			])

# --- Visualize PCA scores for analysis 2 (first two components) ---
import matplotlib.pyplot as plt
for res in results2:
	scores = res['Scores']
	idx = res['Index']
	country = res['Country']
	if len(scores[0]) < 2:
		continue
	plt.figure(figsize=(8,6))
	xs = [s[0] for s in scores]
	ys = [s[1] for s in scores]
	plt.scatter(xs, ys, alpha=0.7)
	for i, label in enumerate(idx):
		plt.text(xs[i], ys[i], label, fontsize=7, alpha=0.6)
	plt.xlabel('PCA Component 1')
	plt.ylabel('PCA Component 2')
	plt.title(f'PCA Scores: {country} (All Years)')
	plt.tight_layout()
	plt.savefig(os.path.join(output_dir, f'pca_scores_{country}.png'))
	plt.close()
import pandas as pd
from collections import defaultdict
from sklearn.decomposition import PCA

# Load the CSV
csv_path = 'stream3_visualization/Well-being/output/unified_all_levels_1_to_5.csv'
df = pd.read_csv(csv_path)

# Filter for Decile == 'All', Level == 1, exclude 'All countries'
ewbi_rows = df[(df['Decile'] == 'All') & (df['Level'] == 1) & (df['Country'] != 'All Countries')]

# Get all countries
countries = ewbi_rows['Country'].unique()

# Get all years
years = ewbi_rows['Year'].unique()

# Get all Level 5 indicators
level5_rows = df[(df['Level'] == 5) & (df['Country'] != 'All Countries') & (df['Decile'] == 'All')]
indicators = level5_rows['Primary and raw data'].unique()

# Build a dictionary: country -> year -> {indicator: value}
country_year_indicator = defaultdict(lambda: defaultdict(dict))
for _, row in level5_rows.iterrows():
	country = row['Country']
	year = row['Year']
	indicator = row['Primary and raw data']
	value = row['Value']
	country_year_indicator[country][year][indicator] = value

# Build PCA input matrices per country (rows: years, columns: indicators)
pca_inputs = {}
for country in countries:
	data = []
	year_list = []
	for year in sorted(country_year_indicator[country].keys()):
		row = [country_year_indicator[country][year].get(ind, float('nan')) for ind in indicators]
		if not all(pd.isna(row)):
			data.append(row)
			year_list.append(year)
	if data:
		pca_inputs[country] = pd.DataFrame(data, columns=indicators, index=year_list)

# --- PCA computation ---
pca_results = {}
for country, df_input in pca_inputs.items():
	# Drop columns with all NaN and rows with any NaN
	df_clean = df_input.dropna(axis=1, how='all').dropna(axis=0)
	if df_clean.shape[0] < 2 or df_clean.shape[1] < 2:
		continue  # Not enough data for PCA
	pca = PCA()
	X = df_clean.values
	pca.fit(X)
	pca_results[country] = {
		'explained_variance_ratio': pca.explained_variance_ratio_,
		'components': pca.components_,
		'feature_names': df_clean.columns.tolist(),
		'years': df_clean.index.tolist()
	}


# Example: print explained variance for each country
for country, res in pca_results.items():
	print(f'Country: {country}')
	print('Explained variance ratio:', res['explained_variance_ratio'])
	print('---')
