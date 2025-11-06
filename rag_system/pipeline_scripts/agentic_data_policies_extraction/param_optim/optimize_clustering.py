#!/usr/bin/env python3
"""
This code isolates the clustering step (from `complete_pipeline.py`) to optimize its hyper-parameters
"""

import logging
import traceback

# optuna_optimization/optimize_clustering.py
import optuna
from ..policies_transformation_to_matrices.merge_policies_kmean_2 import (merge_policies_kmeans_2,
                                    prepare_evaluation_features)
from ..policies_transformation_to_matrices.merge_policies_knn import CFG, merge_policies_semantic_medoid
from optuna.samplers import RandomSampler
from sklearn.metrics import silhouette_score

logger = logging.getLogger(__name__)


def objective(trial, preprocessed_df):
    """
    Optuna objective function for clustering optimization.
    Uses preprocessed data for efficiency.
    """

    try:
        # Draw Hyperparameters for KNN
        CFG.max_neighbors = trial.suggest_int("max_neighbors", 5, 100)

        CFG.sim_threshold = trial.suggest_float("sim_threshold", 0.75, 0.80)

        CFG.medoid_exact_max_cluster = trial.suggest_int(
            "medoid_exact_max_cluster", 100, 800, step=2
        )

        # Draw Hyperparameters for KMEAN
        """
        CFG_Kmean.n_words = trial.suggest_int("n_words", 
                                              3, 
                                              5
                                              )"""
        """
        CFG_Kmean.max_clusters = trial.suggest_int("max_clusters", 
                                              200, 
                                              500
                                              )"""

        logger.info(
            f"max_neighbors={CFG.max_neighbors}, "
            f"sim_threshold={CFG.sim_threshold}, medoid_exact_max_cluster={CFG.medoid_exact_max_cluster}"
        )

        # Run clustering with the updated config
        clustered_df = merge_policies_semantic_medoid(preprocessed_df.copy(), text_col="policy")

        clustered_df = merge_policies_kmeans_2(
            clustered_df, clustered_df["policy_canonical"], max_clusters=50
        )

        logger.info("Extracting features for evaluation...")

        X, labels = prepare_evaluation_features(clustered_df, text_column="policy")

        # Calculate and return silhouette score
        score = silhouette_score(X.toarray(), labels)

        logger.info(f"Trial completed with silhouette score: {score}")

        # return le score seulement
        return score

    except Exception as e:
        logger.error(f"Optuna trial failed: {e}")
        traceback.print_exc()
        return 0.0


def get_best_clustered_df(study, preprocessed_df):
    """
    Re-run the clustering with the best parameters and return the clustered_df.
    """
    best_trial = study.best_trial
    best_params = best_trial.params

    # Appliquer les meilleurs paramètres
    CFG.max_neighbors = best_params["max_neighbors"]
    CFG.sim_threshold = best_params["sim_threshold"]
    CFG.medoid_exact_max_cluster = best_params["medoid_exact_max_cluster"]

    # Exécuter le clustering avec les meilleurs paramètres
    clustered_df = merge_policies_semantic_medoid(preprocessed_df.copy(), text_col="policy")

    clustered_df = merge_policies_kmeans_2(clustered_df, clustered_df["policy_canonical"])

    return clustered_df


def run_optuna_study(preprocessed_df, N_TRIALS):
    """
    Run the Optuna study using preprocessed data.
    """
    study = optuna.create_study(
        direction="maximize",
        sampler=RandomSampler(),
    )
    study.optimize(
        lambda trial: objective(trial, preprocessed_df), n_trials=N_TRIALS, timeout=10000
    )

    return study, get_best_clustered_df(study, preprocessed_df)
