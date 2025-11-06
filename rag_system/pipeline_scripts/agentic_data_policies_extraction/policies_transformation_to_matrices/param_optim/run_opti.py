# optuna_optimization/run_optuna_study.py
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from optimize_clustering import run_optuna_study
from plotly.io import write_image
from preprocess import preprocess_data
from utils import save_optuna_plots

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# PARAMS
N_TRIALS = 500
LIMIT = 5000
# DONT FORGET TO SETUP HYPERPARAMETERS IN OPTIMIZE_CLUSTERING


def main():
    # Load preprocessed data
    flattened_df, pivot_df = preprocess_data(limit=LIMIT)

    logger.info("preprocess_data loaded.")

    # Run opti
    logger.info("Starting Optuna study...")
    study, best_clustered_df = run_optuna_study(flattened_df, N_TRIALS=N_TRIALS)
    logger.info("Optuna study completed.")

    # Best params
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value (silhouette score): {trial.value}")

    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Sauvegarder le nombre de clusters
    n_clusters_KNN = len(best_clustered_df["cluster_id"].unique())
    n_clusters_Kmean = len(best_clustered_df["cluster_id_KMEAN"].unique())
    logger.info(
        f"Nombre de clusters dans le meilleur essai : KNN {n_clusters_KNN}, KMEAN {n_clusters_Kmean}"
    )

    # Save data
    current_dir = os.path.dirname(os.path.abspath(__file__))

    flattened_df.to_csv(os.path.join(current_dir, "flattened_df.csv"), index=False)

    pivot_df.to_csv(os.path.join(current_dir, "pivot_df.csv"))

    best_clustered_df.to_csv(
        os.path.join(current_dir, "best_clustered_results.csv"), index=False
    )

    # Save best params
    best_params = trial.params

    with open(os.path.join(current_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=4)

    # Create and save plots
    os.makedirs(os.path.join(current_dir, "plots"), exist_ok=True)
    save_optuna_plots(study, current_dir, logger)


if __name__ == "__main__":
    main()
