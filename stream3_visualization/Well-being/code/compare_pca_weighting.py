#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparaison des poids : PCA globale vs PCA par EU Priority

Deux approches de pondération sont comparées :
  A) GLOBALE (actuelle)  : PCA calculée sur les 38 indicateurs ensemble
  B) PAR PRIORITY        : PCA calculée séparément pour chaque EU priority

Les deux utilisent la même logique de pondération JRC :
  - Sélection JRC (eigenvalue > 1, variance > 10%, cumulé ≥ 75%)
  - Rotation Varimax si > 1 composante
  - Poids intra-composite = squared loading normalisé
  - Poids composite = proportion de variance expliquée
  - Poids final = produit des deux, normalisé à 1

Usage : python compare_pca_weighting.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ─── Chemins ────────────────────────────────────────────────────────────────
CODE_DIR = Path(__file__).parent
DATA_DIR = CODE_DIR.parent / "data"
OUTPUT_DIR = CODE_DIR.parent / "output"

PCA_RESULTS_PATH = OUTPUT_DIR / "2_multivariate_analysis_output" / "pca_results_full.json"
EWBI_CONFIG_PATH = DATA_DIR / "ewbi_indicators.json"
RAW_DATA_PATH = OUTPUT_DIR / "1_missing_data_output" / "raw_data_break_adjusted.csv"

BAR = "=" * 72
THIN = "─" * 72


# ─── Chargement des données ───────────────────────────────────────────────────

def load_ewbi_config():
    with open(EWBI_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    eu_priorities, descriptions = {}, {}
    for priority in config["EWBI"]:
        pname = priority["name"]
        codes = []
        for comp in priority["components"]:
            for ind in comp["indicators"]:
                codes.append(ind["code"])
                descriptions[ind["code"]] = ind.get("description", "")
        eu_priorities[pname] = codes
    return eu_priorities, descriptions


def load_global_pca():
    with open(PCA_RESULTS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    first_key = next(iter(raw))
    entry = raw[first_key]
    print(f"[INFO] PCA globale de référence : {first_key}")
    print(f"       {entry['n_components']} composantes | "
          f"variance totale : {entry['total_variance_explained']*100:.1f}%")
    return entry


def load_raw_data():
    print(f"[INFO] Chargement des données brutes : {RAW_DATA_PATH.name}")
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"       {len(df):,} lignes — colonnes : {list(df.columns)}")
    return df


# ─── Logique PCA ──────────────────────────────────────────────────────────────

def jrc_factor_selection(eigenvalues, variance_ratios):
    """Sélection JRC : eigenvalue > 1 ET variance > 10% ET cumulé ≥ 75%."""
    selected, cumulative = [], 0.0
    for i, (ev, vr) in enumerate(zip(eigenvalues, variance_ratios)):
        if ev > 1.0 and vr * 100 > 10.0:
            selected.append(i)
            cumulative += vr * 100
            if cumulative >= 75.0:
                break
    if not selected:
        # Fallback : premier facteur avec eigenvalue > 1
        for i, ev in enumerate(eigenvalues):
            if ev > 1.0:
                selected = [i]
                break
        if not selected:
            selected = [0]
    return selected


def varimax(loadings, max_iter=1000, tol=1e-10):
    """Rotation Varimax."""
    p, k = loadings.shape
    if k == 1:
        return loadings, np.eye(1)
    R = np.eye(k)
    d = 0
    for _ in range(max_iter):
        d_old = d
        L = np.dot(loadings, R)
        u, s, vh = np.linalg.svd(
            np.dot(loadings.T,
                   np.asarray(L) ** 3 - (1.0 / p) * L * np.dot(np.ones((p, 1)),
                   np.dot(np.ones((1, p)), L ** 2))))
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return np.dot(loadings, R), R


def run_priority_pca(raw_df, indicators):
    """
    Exécute une PCA JRC-compliant sur un sous-ensemble d'indicateurs.

    Retourne un dict compatible avec compute_jrc_weights(), ou None si
    les données sont insuffisantes.
    """
    # Filtrer les données
    df_filtered = raw_df[
        (raw_df['Country'] != 'All Countries') &
        (raw_df['Decile'] != 'All') &
        (raw_df['Primary and raw data'].isin(indicators))
    ].copy()

    if df_filtered.empty:
        return None

    # Pivot : lignes = (Country, Year, Decile), colonnes = indicateurs
    pivot = df_filtered.pivot_table(
        index=['Country', 'Year', 'Decile'],
        columns='Primary and raw data',
        values='Value',
        aggfunc='first'
    )
    pivot = pivot.dropna(axis=1, how='all').dropna(axis=0, how='all').dropna()

    available_indicators = pivot.columns.tolist()
    if pivot.shape[0] < 3 or pivot.shape[1] < 2:
        return None

    # Standardisation
    scaler = StandardScaler()
    X = scaler.fit_transform(pivot)

    # PCA complète
    pca_full = PCA()
    pca_full.fit(X)

    selected = jrc_factor_selection(pca_full.explained_variance_,
                                    pca_full.explained_variance_ratio_)
    n_comp = len(selected)

    # PCA avec n_comp composantes
    pca = PCA(n_components=n_comp)
    pca.fit(X)

    # Loadings : shape (n_indicators, n_components)
    loadings = pca.components_.T

    # Varimax si > 1 composante
    if n_comp > 1:
        rotated_loadings, _ = varimax(loadings)
    else:
        rotated_loadings = loadings

    return {
        "indicator_names": available_indicators,
        "n_components": n_comp,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "eigenvalues": pca.explained_variance_.tolist(),
        "rotated_loadings": rotated_loadings.T.tolist(),   # (n_comp, n_ind)
        "varimax_applied": n_comp > 1,
        "full_eigenvalues": pca_full.explained_variance_.tolist(),
        "full_variance_ratio": pca_full.explained_variance_ratio_.tolist(),
    }


# ─── Calcul des poids JRC ────────────────────────────────────────────────────

def compute_jrc_weights(pca_entry, available_indicators):
    """
    Poids JRC à partir d'une entrée PCA (globale ou par priority).
    Retourne {indicator: weight} ou None.
    """
    rotated_loadings = np.array(
        pca_entry.get("rotated_loadings") or pca_entry.get("component_loadings", [])
    )
    indicator_names = pca_entry["indicator_names"]
    evr = np.array(pca_entry["explained_variance_ratio"])

    if rotated_loadings.size == 0:
        return None

    sq = rotated_loadings ** 2  # (n_comp, n_ind)

    composites = {}
    for ind in available_indicators:
        if ind not in indicator_names:
            continue
        col = indicator_names.index(ind)
        best_f = int(np.argmax(sq[:, col]))
        cid = best_f
        if cid not in composites:
            composites[cid] = {"indicators": [], "sq_loadings": [], "variance": float(evr[best_f])}
        composites[cid]["indicators"].append(ind)
        composites[cid]["sq_loadings"].append(float(sq[best_f, col]))

    if not composites:
        return None

    # Poids intra-composite
    for cid in composites:
        total = sum(composites[cid]["sq_loadings"])
        composites[cid]["intra"] = [s / total if total > 0 else 1.0 / len(composites[cid]["sq_loadings"])
                                    for s in composites[cid]["sq_loadings"]]

    # Importance des composites
    total_var = sum(c["variance"] for c in composites.values())
    for cid in composites:
        composites[cid]["importance"] = (composites[cid]["variance"] / total_var
                                         if total_var > 0 else 1.0 / len(composites))

    # Poids final
    final = {}
    for cid, comp in composites.items():
        for ind, w in zip(comp["indicators"], comp["intra"]):
            final[ind] = w * comp["importance"]

    total_w = sum(final.values())
    if total_w > 0:
        final = {k: v / total_w for k, v in final.items()}

    return final, composites


# ─── Affichage ────────────────────────────────────────────────────────────────

def fmt_bar(value, width=20):
    filled = round(value * width)
    return "█" * filled + "░" * (width - filled)


def delta_arrow(diff):
    if diff > 0.005:
        return f"+{diff:.4f} ▲"
    elif diff < -0.005:
        return f"{diff:.4f} ▼"
    return f"{diff:.4f}  "


def print_comparison(priority_name, global_weights, local_weights, local_pca, descriptions):
    print(f"\n{BAR}")
    print(f"  EU PRIORITY : {priority_name.upper()}")
    print(BAR)

    if global_weights is None and local_weights is None:
        print("  [SKIP] Données insuffisantes pour les deux approches.")
        return

    # Indicateurs présents dans au moins une des deux
    all_inds = sorted(set(list(global_weights or {}) + list(local_weights or {})))

    if local_pca:
        n_comp_local = local_pca["n_components"]
        var_local = sum(local_pca["explained_variance_ratio"]) * 100
        varimax_str = "Oui" if local_pca.get("varimax_applied") else "Non"
        print(f"  PCA par priority  : {n_comp_local} composante(s)  |  "
              f"variance expliquée : {var_local:.1f}%  |  Varimax : {varimax_str}")
    else:
        print("  PCA par priority  : impossible (données insuffisantes)")

    print(f"  PCA globale       : 4 composantes  |  variance totale : 74.8%")
    print()

    # En-tête
    print(f"  {'Indicateur':<14} {'Description':<38} "
          f"{'Global':>8} {'Local':>8} {'Écart':>12}  {'Local':25}  {'Global'}")
    print(f"  {'─'*14} {'─'*38} {'─'*8} {'─'*8} {'─'*12}  {'─'*25}  {'─'*25}")

    for ind in all_inds:
        desc = descriptions.get(ind, "")[:38]
        w_global = (global_weights or {}).get(ind, float("nan"))
        w_local = (local_weights or {}).get(ind, float("nan"))

        if np.isnan(w_global) or np.isnan(w_local):
            diff_str = "  n/a"
        else:
            diff = w_local - w_global
            diff_str = delta_arrow(diff)

        bar_local = fmt_bar(w_local) if not np.isnan(w_local) else " " * 20
        bar_global = fmt_bar(w_global) if not np.isnan(w_global) else " " * 20

        w_g_str = f"{w_global:.4f}" if not np.isnan(w_global) else "  n/a "
        w_l_str = f"{w_local:.4f}" if not np.isnan(w_local) else "  n/a "

        print(f"  {ind:<14} {desc:<38} {w_g_str:>8} {w_l_str:>8} {diff_str:>12}  "
              f"{bar_local}  {bar_global}")

    # Résumé
    if global_weights and local_weights:
        common = set(global_weights) & set(local_weights)
        max_shift = max((abs(local_weights[i] - global_weights[i]) for i in common), default=0)
        winner = max(common, key=lambda i: abs(local_weights[i] - global_weights[i]))
        print()
        print(f"  → Écart max : {max_shift:.4f} ({winner}) | "
              f"Somme local={sum(local_weights.values()):.6f} | "
              f"Somme global={sum(global_weights.values()):.6f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(BAR)
    print("  COMPARAISON : PCA GLOBALE vs PCA PAR EU PRIORITY")
    print("  Légende barres : Local (gauche)  |  Global (droite)")
    print(BAR)

    eu_priorities, descriptions = load_ewbi_config()
    global_pca = load_global_pca()
    raw_df = load_raw_data()

    # Vérifier la colonne qui contient le code indicateur
    indicator_col = None
    for candidate in ['Primary and raw data', 'indicator', 'code', 'variable']:
        if candidate in raw_df.columns:
            indicator_col = candidate
            break
    if indicator_col is None:
        raise ValueError(f"Colonne indicateur introuvable. Colonnes : {list(raw_df.columns)}")
    if indicator_col != 'Primary and raw data':
        raw_df = raw_df.rename(columns={indicator_col: 'Primary and raw data'})

    print(f"\n[INFO] Colonne indicateur : '{indicator_col}'")
    print(f"[INFO] EU Priorities : {list(eu_priorities.keys())}")

    summary_rows = []

    for priority_name, indicators in eu_priorities.items():
        if not indicators:
            continue

        # ── Poids GLOBAUX ──
        try:
            g_result = compute_jrc_weights(global_pca, indicators)
            global_weights, global_composites = g_result if g_result else (None, None)
        except Exception as e:
            print(f"[WARN] Poids globaux impossibles pour {priority_name}: {e}")
            global_weights, global_composites = None, None

        # ── PCA et poids PAR PRIORITY ──
        local_pca = run_priority_pca(raw_df, indicators)
        if local_pca:
            try:
                l_result = compute_jrc_weights(local_pca, local_pca["indicator_names"])
                local_weights, local_composites = l_result if l_result else (None, None)
            except Exception as e:
                print(f"[WARN] Poids locaux impossibles pour {priority_name}: {e}")
                local_weights, local_composites = None, None
        else:
            local_weights, local_composites = None, None

        # ── Affichage ──
        print_comparison(priority_name, global_weights, local_weights,
                         local_pca, descriptions)

        # Collecte pour le résumé final
        for ind in indicators:
            w_g = (global_weights or {}).get(ind, float("nan"))
            w_l = (local_weights or {}).get(ind, float("nan"))
            summary_rows.append({
                "EU Priority": priority_name,
                "Indicator": ind,
                "Description": descriptions.get(ind, ""),
                "Weight_Global": w_g,
                "Weight_Local": w_l,
                "Delta": w_l - w_g if not (np.isnan(w_g) or np.isnan(w_l)) else float("nan"),
            })

    # ── Résumé synthétique ──
    summary_df = pd.DataFrame(summary_rows)
    summary_df["AbsDelta"] = summary_df["Delta"].abs()

    print(f"\n{BAR}")
    print("  RÉSUMÉ SYNTHÉTIQUE — TOP 10 INDICATEURS LES PLUS AFFECTÉS")
    print(BAR)
    print(f"\n  {'Indicateur':<14} {'EU Priority':<20} {'Description':<38} "
          f"{'Global':>8} {'Local':>8} {'Écart':>8}")
    print(f"  {'─'*14} {'─'*20} {'─'*38} {'─'*8} {'─'*8} {'─'*8}")

    top10 = summary_df.dropna(subset=["AbsDelta"]).nlargest(10, "AbsDelta")
    for _, row in top10.iterrows():
        sign = "▲" if row["Delta"] > 0 else "▼"
        print(f"  {row['Indicator']:<14} {row['EU Priority']:<20} "
              f"{row['Description'][:38]:<38} "
              f"{row['Weight_Global']:>8.4f} {row['Weight_Local']:>8.4f} "
              f"{row['Delta']:>+7.4f} {sign}")

    print(f"\n{BAR}")
    print("  INTERPRÉTATION")
    print(BAR)
    n_global_composites = summary_df.dropna(subset=["Weight_Global"])["EU Priority"].nunique()
    avg_delta = summary_df["AbsDelta"].mean()
    print(f"""
  PCA GLOBALE (actuelle)
  • Un seul jeu de facteurs pour les 38 indicateurs → Factor 1 = "précarité générale"
  • L'indicateur qui charge sur Factor 1 dans chaque priority hérite de ~38-63% du poids
  • Résultat : poids très inégaux, fortement influencés par la corrélation inter-priorities

  PCA PAR PRIORITY
  • Facteurs calculés sur 4-11 indicateurs homogènes → facteurs thématiques
  • Tous les indicateurs d'une priority "compétent" sur le même terrain
  • Résultat : poids plus équilibrés, reflétant la structure interne de chaque priority
  • Avec 1 seule composante (Energy, Equality…) → poids = squared loadings normalisés

  Écart moyen absolu entre les deux approches : {avg_delta:.4f}
  """)

    print(f"\n{BAR}")
    print("  FIN DU RAPPORT")
    print(BAR)


if __name__ == "__main__":
    main()
