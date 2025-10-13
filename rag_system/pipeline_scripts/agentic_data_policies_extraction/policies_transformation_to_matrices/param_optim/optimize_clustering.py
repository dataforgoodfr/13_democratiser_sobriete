#!/usr/bin/env python3
"""
This code isolates the clustering step (from `complete_pipeline.py`) to optimize its hyper-parameters
"""

# optuna_optimization/optimize_clustering.py
import optuna
from optuna.samplers import RandomSampler

import logging
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from merge_policies_knn import merge_policies_semantic_medoid, Config, CFG

import traceback

logger = logging.getLogger(__name__)

def objective(trial, preprocessed_df):
    """
    Optuna objective function for clustering optimization.
    Uses preprocessed data for efficiency.
    """

    try:
        # Suggest hyperparameters
        CFG.batch_size = trial.suggest_int("batch_size", 
                                           16, 
                                           128
                                           )
        
        CFG.max_neighbors = trial.suggest_int("max_neighbors", 
                                              5, 
                                              100
                                              )
        
        CFG.sim_threshold = trial.suggest_float("sim_threshold", 
                                                0.75,
                                                0.85
                                                )
        
        CFG.medoid_exact_max_cluster = trial.suggest_int("medoid_exact_max_cluster", 
                                                         5, 
                                                         500, 
                                                         step=2
                                                         )

        logger.info(f"Running Optuna trial with batch_size={CFG.batch_size}, max_neighbors={CFG.max_neighbors}, "
                    f"sim_threshold={CFG.sim_threshold}, medoid_exact_max_cluster={CFG.medoid_exact_max_cluster}")

        # Run clustering with the updated config
        clustered_df = merge_policies_semantic_medoid(
            preprocessed_df.copy(),
            text_col="policy"
            )

        # Extract features for evaluation
        logger.info("Extracting features for evaluation...")
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform(clustered_df['policy'])
        labels = clustered_df["cluster_id"].values

        # Calculate and return silhouette score
        score = silhouette_score(X.toarray(), 
                                 labels
                                 )
        
        # calculate the number of clusters
        n_clusters = len(clustered_df["cluster_id"].unique())

        logger.info(f"Trial completed with silhouette score: {score}")
        
        
        # Stocker resultats dans user_attrs
        #trial.set_user_attr("n_clusters", n_clusters)
        #trial.set_user_attr("sim_threshold", CFG.sim_threshold)

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
    CFG.batch_size = best_params["batch_size"]
    CFG.max_neighbors = best_params["max_neighbors"]
    CFG.sim_threshold = best_params["sim_threshold"]
    CFG.medoid_exact_max_cluster = best_params["medoid_exact_max_cluster"]

    # Exécuter le clustering avec les meilleurs paramètres
    clustered_df = merge_policies_semantic_medoid(
        preprocessed_df.copy(),
        text_col="policy"
    )

    return clustered_df

def run_optuna_study(preprocessed_df, N_TRIALS):
    """
    Run the Optuna study using preprocessed data.
    """
    study = optuna.create_study(direction="maximize",
                                sampler = RandomSampler(), 
                                )
    study.optimize(lambda trial: objective(trial, preprocessed_df), 
                   n_trials=N_TRIALS, 
                   timeout=3600
                   )
    
    return study, get_best_clustered_df(study, preprocessed_df)