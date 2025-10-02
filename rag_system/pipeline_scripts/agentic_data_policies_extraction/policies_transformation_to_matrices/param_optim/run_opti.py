# optuna_optimization/run_optuna_study.py
import sys
from pathlib import Path
import pandas as pd
import logging
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_contour,
    plot_parallel_coordinate
)
from plotly.io import write_image
from preprocess import preprocess_data
import os
import json
import traceback


from optimize_clustering import run_optuna_study

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True 
)
logger = logging.getLogger(__name__)


current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

# PARAMS
N_TRIALS = 5
LIMIT = 15

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
    n_clusters = len(best_clustered_df["cluster_id"].unique())
    logger.info(f"Nombre de clusters dans le meilleur essai : {n_clusters}")

    # Save data
    current_dir = os.path.dirname(os.path.abspath(__file__))

    flattened_df.to_csv(os.path.join(current_dir, 
                                     "flattened_df.csv"), 
                                     index=False
                                     )
    
    pivot_df.to_csv(os.path.join(current_dir, 
                                 "pivot_df.csv")
                                 )

    best_clustered_df.to_csv(os.path.join(current_dir, 
                                          "best_clustered_results.csv"), 
                             index=False
                             )


    # Save best params
    best_params = trial.params
    
    with open(os.path.join(current_dir, 
                           "best_params.json"), 
                           "w") as f:
        
        json.dump(best_params, 
                  f, 
                  indent=4
                  )

    # Create and save plots
    os.makedirs(os.path.join(current_dir, 
                             "plots"), 
                             exist_ok=True
                             )

    try:
        fig1 = plot_optimization_history(study)
        write_image(fig1, 
                    os.path.join(current_dir, 
                                 "plots", 
                                 "optimization_history.png")
                                 )

        fig2 = plot_param_importances(study)
        write_image(fig2, 
                    os.path.join(current_dir, 
                                 "plots", 
                                 "param_importances.png"
                                 ))
        
        params = list(trial.params.keys())
        for param in params:
            fig = plot_slice(study, 
                             params=[param]
                             )
            write_image(fig, 
                        os.path.join(current_dir, 
                                     f"plots/slice_{param}.png")
                                     )
        
        for i in range(len(params)):
            for j in range(i + 1, 
                           len(params)):
                fig = plot_contour(study, 
                                   params=[params[i], 
                                           params[j]]
                                           )
                write_image(fig, 
                            os.path.join(current_dir, 
                                         f"plots/contour_{params[i]}_vs_{params[j]}.png"))
        
        fig5 = plot_parallel_coordinate(study)
        write_image(fig5, 
                    os.path.join(current_dir, 
                                 "plots", 
                                 "parallel_coordinate.png")
                                 )
        
        logger.info(f"All plots saved to: {os.path.join(current_dir, 'plots')}")
    except Exception as e:

        logger.error(f"Error saving images: {e}")

        traceback.print_exc()

if __name__ == "__main__":
    main()