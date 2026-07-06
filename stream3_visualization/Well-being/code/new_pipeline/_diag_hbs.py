"""
Diagnostic: check TS-HBS and EC-HBS threshold values for suspicious cases.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')
import config, pandas as pd, numpy as np

for code in ["TS-HBS-1", "TS-HBS-2", "EC-HBS-1", "EC-HBS-2"]:
    csv = config.INDICATORS_DIR / f"{code}.csv"
    if not csv.exists():
        print(f"{code}: file not found"); continue
    df = pd.read_csv(csv)
    print(f"\n{'='*60}")
    print(f"{code}")

    # Show countries with 100% or 0% values (suspicious)
    extreme = df[(df["value"] >= 99.9) | (df["value"] <= 0.1)]
    if not extreme.empty:
        print("  Extreme values (>=99.9% or <=0.1%):")
        print(extreme[["country","year","decile","value","n_obs"]].to_string(index=False))

    # Show BE 2020 specifically
    be = df[(df["country"]=="BE") & (df["year"]==2020)].sort_values("decile")
    if not be.empty:
        print("  BE 2020:")
        print(be[["decile","value","n_obs"]].to_string(index=False))

# Now check the actual threshold for TS-HBS in BE 2020
print("\n" + "="*60)
print("DEEP DIVE: TS-HBS threshold for BE 2020")
from extract_hbs import load_wave, HBS_SRC_COL, HBS_USE_MEAN, decile_balanced_weights
from extract_hbs import weighted_quantile

df_wave = load_wave(2020)
be_df = df_wave[df_wave["COUNTRY"] == "BE"].copy()

src_col = HBS_SRC_COL["TS-HBS-2"]   # EUR_HJ90
equiv_col = f"equiv_{src_col}"

print(f"  Source column: {src_col}, equiv: {equiv_col}")
print(f"  Total BE 2020 rows: {len(be_df)}")

if equiv_col in be_df.columns:
    e = pd.to_numeric(be_df[equiv_col], errors="coerce").to_numpy(dtype=float)
    w = be_df.get("HA10", pd.Series(np.nan)).to_numpy(dtype=float)
    dec = be_df["decile"].to_numpy(dtype=float)

    # Replicate the threshold computation (HBS_USE_MEAN path)
    bal_w = decile_balanced_weights(w, dec)
    valid_m = ~np.isnan(e) & (bal_w > 0)
    m = float(np.average(e[valid_m], weights=bal_w[valid_m])) if valid_m.sum() > 0 else np.nan

    print(f"  Decile-balanced weighted mean (m): {m:.4f}")
    print(f"  Threshold for TS-HBS-1 (>2×m): {2*m:.4f}")
    print(f"  Threshold for TS-HBS-2 (<0.5×m): {0.5*m:.4f}")
    print(f"  % of BE households with expense=0: {(e[~np.isnan(e)]==0).mean()*100:.1f}%")
    print(f"  Simple stats of equiv expense:")
    valid = ~np.isnan(e)
    vals = e[valid]
    print(f"    n_valid={valid.sum()}, min={vals.min():.2f}, p10={np.percentile(vals,10):.2f}, p25={np.percentile(vals,25):.2f}, p50={np.percentile(vals,50):.2f}, p75={np.percentile(vals,75):.2f}, max={vals.max():.2f}")
    print(f"    % below 0.5×m: {(vals < 0.5*m).mean()*100:.1f}%")
    print(f"    % above 2.0×m: {(vals > 2.0*m).mean()*100:.1f}%")

    # By decile
    print("\n  By decile:")
    for d in range(1, 11):
        mask = dec == d
        e_d = e[mask]
        valid_d = ~np.isnan(e_d)
        if valid_d.sum() == 0:
            continue
        pct_below = (e_d[valid_d] < 0.5*m).mean() * 100
        pct_above = (e_d[valid_d] > 2.0*m).mean() * 100
        print(f"    D{d}: n={valid_d.sum()}, % <0.5m={pct_below:.1f}%, % >2m={pct_above:.1f}%, sum={pct_below+pct_above:.1f}%, gap={100-pct_below-pct_above:.1f}%")
else:
    print(f"  Column {equiv_col} not found in wave. Available: {[c for c in be_df.columns if 'equiv' in c]}")
