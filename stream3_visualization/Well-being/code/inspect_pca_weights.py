#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCA Weight Inspector — EWBI

Pour chaque EU Priority, affiche :
  - Nombre de composites PCA retenus
  - Variance expliquée par composite
  - Indicateurs associés à chaque composite (loading le plus fort)
  - Chargement au carré, poids intra-composite, poids final par indicateur

Utilise uniquement les fichiers déjà calculés :
  - output/2_multivariate_analysis_output/pca_results_full.json
  - data/ewbi_indicators.json

Usage : python inspect_pca_weights.py
"""

import json
import numpy as np
from pathlib import Path

# ─── Chemins ────────────────────────────────────────────────────────────────
CODE_DIR = Path(__file__).parent
DATA_DIR = CODE_DIR.parent / "data"
OUTPUT_DIR = CODE_DIR.parent / "output" / "2_multivariate_analysis_output"

PCA_RESULTS_PATH = OUTPUT_DIR / "pca_results_full.json"
EWBI_CONFIG_PATH = DATA_DIR / "ewbi_indicators.json"


# ─── Chargement ──────────────────────────────────────────────────────────────

def load_pca_reference():
    """
    Charge pca_results_full.json et retourne le premier enregistrement.
    Toutes les entrées partagent la même PCA globale ; une seule suffit.
    """
    with open(PCA_RESULTS_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    first_key = next(iter(raw))
    entry = raw[first_key]
    print(f"[INFO] PCA de référence : {first_key}")
    print(f"       {entry['n_components']} composantes retenues  |  "
          f"Variance totale expliquée : {entry['total_variance_explained']*100:.1f}%")
    print(f"       JRC criteria applied : {entry.get('jrc_criteria_applied', '?')}")
    return entry


def load_ewbi_config():
    """Retourne un dict {priority_name: [code1, code2, ...]} et les descriptions."""
    with open(EWBI_CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)

    eu_priorities = {}
    descriptions = {}
    for priority in config["EWBI"]:
        pname = priority["name"]
        codes = []
        for component in priority["components"]:
            for ind in component["indicators"]:
                codes.append(ind["code"])
                descriptions[ind["code"]] = ind.get("description", "")
        eu_priorities[pname] = codes

    return eu_priorities, descriptions


# ─── Logique de pondération (miroir de get_jrc_pca_weights_for_country_year) ─

def compute_pca_weights_for_priority(pca_entry, priority_indicators):
    """
    Reproduit la logique de get_jrc_pca_weights_for_country_year() de Stage 4
    pour un sous-ensemble d'indicateurs (une EU priority).

    Retourne un dict structuré avec :
      composites : liste de composites avec indicateurs, loadings, poids
      final_weights : {indicator: poids_final}
    """
    rotated_loadings = np.array(pca_entry.get("rotated_loadings") or
                                pca_entry.get("component_loadings"))
    # Shape attendue : (n_components, n_indicators)
    indicator_names = pca_entry["indicator_names"]
    explained_variance_ratio = np.array(pca_entry["explained_variance_ratio"])

    # Filtre sur les indicateurs de la priority présents dans la PCA
    available = [ind for ind in priority_indicators if ind in indicator_names]
    ind_indices = {ind: indicator_names.index(ind) for ind in available}

    squared_loadings = rotated_loadings ** 2  # (n_components, n_indicators)

    # ── Étape 1 : Associer chaque indicateur au facteur avec le plus fort loading² ──
    composites = {}  # composite_id -> {...}

    for ind in available:
        col_idx = ind_indices[ind]
        factor_sq_loadings = squared_loadings[:, col_idx]
        best_factor = int(np.argmax(factor_sq_loadings))
        cid = f"composite_{best_factor}"

        if cid not in composites:
            composites[cid] = {
                "factor_idx": best_factor,
                "variance_explained_pct": float(explained_variance_ratio[best_factor]) * 100,
                "indicators": [],
                "loadings_rotated": [],      # loading brut (signe inclus)
                "squared_loadings": [],
                "intra_weights": [],         # rempli après normalisation
            }

        composites[cid]["indicators"].append(ind)
        composites[cid]["loadings_rotated"].append(
            float(rotated_loadings[best_factor, col_idx])
        )
        composites[cid]["squared_loadings"].append(float(factor_sq_loadings[best_factor]))

    if not composites:
        return None

    # ── Étape 2 : Poids intra-composite (squared loadings normalisés → somme = 1) ──
    for cid, comp in composites.items():
        total_sq = sum(comp["squared_loadings"])
        if total_sq > 0:
            comp["intra_weights"] = [sq / total_sq for sq in comp["squared_loadings"]]
        else:
            n = len(comp["indicators"])
            comp["intra_weights"] = [1.0 / n] * n

    # ── Étape 3 : Importance de chaque composite (part de variance expliquée) ──
    total_var = sum(comp["variance_explained_pct"] for comp in composites.values())
    for comp in composites.items():
        comp[1]["composite_importance"] = (
            comp[1]["variance_explained_pct"] / total_var if total_var > 0 else
            1.0 / len(composites)
        )

    # ── Étape 4 : Poids final = intra_weight × importance_composite ──
    final_weights = {}
    for cid, comp in composites.items():
        for ind, w_intra in zip(comp["indicators"], comp["intra_weights"]):
            final_weights[ind] = w_intra * comp["composite_importance"]

    # Normalisation finale (somme = 1)
    total_w = sum(final_weights.values())
    if total_w > 0:
        final_weights = {k: v / total_w for k, v in final_weights.items()}

    return {
        "composites": composites,
        "final_weights": final_weights,
        "n_composites": len(composites),
        "indicators_in_pca": available,
        "indicators_missing_from_pca": [i for i in priority_indicators if i not in available],
    }


# ─── Affichage ───────────────────────────────────────────────────────────────

BAR = "=" * 72
THIN = "─" * 72


def fmt_bar(value, max_width=25):
    """Mini barre ASCII proportionnelle."""
    filled = round(value * max_width)
    return "█" * filled + "░" * (max_width - filled)


def print_priority_report(priority_name, result, descriptions):
    print(f"\n{BAR}")
    print(f"  EU PRIORITY : {priority_name.upper()}")
    print(BAR)

    if result is None:
        print("  [SKIP] Aucun indicateur de cette priority trouvé dans la PCA.")
        return

    n_comp = result["n_composites"]
    missing = result["indicators_missing_from_pca"]

    print(f"  Indicateurs dans PCA       : {len(result['indicators_in_pca'])}")
    if missing:
        print(f"  ⚠ Absents de la PCA       : {', '.join(missing)}")
    print(f"  Composites PCA retenus     : {n_comp}")
    print()

    composites = result["composites"]
    final_weights = result["final_weights"]

    # Trier les composites par numéro de facteur
    sorted_composites = sorted(composites.items(), key=lambda x: x[1]["factor_idx"])

    for cid, comp in sorted_composites:
        f_idx = comp["factor_idx"] + 1   # 1-indexé pour la lisibilité
        var_pct = comp["variance_explained_pct"]
        importance = comp["composite_importance"] * 100
        n_ind = len(comp["indicators"])

        print(f"  {'─'*68}")
        print(f"  COMPOSITE {cid.replace('composite_', 'C')}  "
              f"(Facteur PCA #{f_idx})")
        print(f"  Variance expliquée par ce facteur : {var_pct:5.1f}%  "
              f"{fmt_bar(var_pct/100)}")
        print(f"  Poids du composite dans la priority : {importance:5.1f}%")
        print(f"  Indicateurs associés : {n_ind}")
        print()

        # En-tête du tableau
        print(f"  {'Indicateur':<14} {'Description':<42} "
              f"{'Loading':<9} {'Loading²':<9} {'w_intra':<9} {'w_final':<9}")
        print(f"  {'─'*14} {'─'*42} {'─'*9} {'─'*9} {'─'*9} {'─'*9}")

        # Trier par loading² décroissant (indicateur le plus représentatif en premier)
        rows = sorted(
            zip(comp["indicators"],
                comp["loadings_rotated"],
                comp["squared_loadings"],
                comp["intra_weights"]),
            key=lambda x: -x[2]
        )

        for ind, loading, sq, w_intra in rows:
            w_final = final_weights.get(ind, float("nan"))
            desc = descriptions.get(ind, "")[:42]
            sign = "▲" if loading >= 0 else "▼"
            print(f"  {ind:<14} {desc:<42} "
                  f"{loading:+.4f} {sign}  "
                  f"{sq:.4f}    "
                  f"{w_intra:.4f}    "
                  f"{w_final:.4f}")

        print()

    # ── Récapitulatif des poids finaux triés ──
    print(f"  {THIN}")
    print(f"  POIDS FINAUX PAR INDICATEUR (triés par importance)")
    print(f"  {'─'*14} {'─'*42} {'─'*9} {'barre'}")
    for ind, w in sorted(final_weights.items(), key=lambda x: -x[1]):
        desc = descriptions.get(ind, "")[:42]
        bar = fmt_bar(w)
        print(f"  {ind:<14} {desc:<42} {w:.4f}    {bar}")

    print()
    print(f"  Somme des poids finaux : {sum(final_weights.values()):.6f}  (doit = 1.000000)")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(BAR)
    print("  EWBI — INSPECTION DES POIDS PCA PAR EU PRIORITY")
    print(BAR)

    pca_entry = load_pca_reference()
    eu_priorities, descriptions = load_ewbi_config()

    print(f"\n[INFO] EU Priorities chargées : {list(eu_priorities.keys())}")
    print(f"[INFO] Indicateurs totaux dans la PCA : {len(pca_entry['indicator_names'])}")

    for priority_name, indicators in eu_priorities.items():
        if not indicators:
            continue
        result = compute_pca_weights_for_priority(pca_entry, indicators)
        print_priority_report(priority_name, result, descriptions)

    print(f"\n{BAR}")
    print("  FIN DU RAPPORT")
    print(BAR)


if __name__ == "__main__":
    main()
