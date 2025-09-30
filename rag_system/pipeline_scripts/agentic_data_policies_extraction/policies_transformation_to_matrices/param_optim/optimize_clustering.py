#!/usr/bin/env python3
"""
This code isolates the clustering step (from `complete_pipeline.py`) to optimize its hyper-parameters
"""

# optuna_optimization/optimize_clustering.py
import optuna
import logging
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer
from merge_policies_knn import merge_policies_semantic_medoid, Config, CFG

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
                                              2, 
                                              20
                                              )
        
        CFG.sim_threshold = trial.suggest_float("sim_threshold", 
                                                0.75, 
                                                0.85
                                                )
        
        CFG.medoid_exact_max_cluster = trial.suggest_int("medoid_exact_max_cluster", 
                                                         50, 
                                                         500, 
                                                         step=10
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
        logger.info(f"Trial completed with silhouette score: {score}")
        return score

    except Exception as e:
        logger.error(f"Optuna trial failed: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


def run_optuna_study(preprocessed_df, n_trials):
    """
    Run the Optuna study using preprocessed data.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, preprocessed_df), n_trials=n_trials, timeout=3600)
    return study