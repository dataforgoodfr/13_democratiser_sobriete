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

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



# Add the parent directory to the path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from optimize_clustering import run_optuna_study

def main():
    # Load preprocessed data
    flattened_df, pivot_df = preprocess_data()
    
    logger.info("preprocess_data loaded.")

    # Run Optuna study
    logger.info("Starting Optuna study...")
    study = run_optuna_study(flattened_df, 
                             n_trials=500
                             )
    logger.info("Optuna study completed.")

    # Print best hyperparameters
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (silhouette score): {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    

    
    params = list(trial.params.keys()) # Liste tous les paramètres optimisés

    try:
        fig1 = plot_optimization_history(study)
        write_image(fig1, "graphics/optimization_history.png")

        fig2 = plot_param_importances(study)
        write_image(fig2, "graphics/param_importances.png")

        for param in params:
            fig = plot_slice(study, params=[param])
            write_image(fig, f"graphics/slice_{param}.png")

        for i in range(len(params)):
            for j in range(i + 1, len(params)):
                fig = plot_contour(study, params=[params[i], params[j]])
                write_image(fig, f"graphics/contour_{params[i]}_vs_{params[j]}.png")

        fig5 = plot_parallel_coordinate(study)
        write_image(fig5, "graphics/parallel_coordinate.png")

        logger.info(f"All plots saved to: {os.path.abspath('graphics')}")

    except Exception as e:
        logger.error(f"Error saving images: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
