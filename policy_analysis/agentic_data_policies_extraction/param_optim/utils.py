import os
import traceback

import numpy as np
import plotly.graph_objects as go
from optuna.study import Study
from optuna.visualization import (plot_contour, plot_optimization_history,
                                  plot_parallel_coordinate,
                                  plot_param_importances, plot_slice)
from scipy.stats import pearsonr


def save_optuna_plots(study: Study, output_dir: str, logger=None) -> None:
    """
    Save all Optuna visualization plots to the specified directory.

    Args:
        study: Optuna study object containing the optimization results.
        output_dir: Directory path where plots will be saved.
        logger: Optional logger for logging messages (default: print).
    """
    try:
        # Create plots directory if it doesn't exist
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)

        log = logger.info if logger else print

        # 1. Optimization history plot
        fig1 = plot_optimization_history(study)
        fig1.write_image(os.path.join(plots_dir, "optimization_history.png"))
        log("Saved optimization history plot")

        # 2. Parameter importances plot
        fig2 = plot_param_importances(study)
        fig2.write_image(os.path.join(plots_dir, "param_importances.png"))
        log("Saved parameter importances plot")

        # Get parameter names
        params = list(study.best_trial.params.keys())

        # 3. Slice plots for each parameter
        for param in params:
            fig = plot_slice(study, params=[param])
            fig.write_image(os.path.join(plots_dir, f"slice_{param}.png"))
            log(f"Saved slice plot for {param}")

        # 4. Contour plots for parameter pairs
        for i in range(len(params)):
            for j in range(i + 1, len(params)):
                fig = plot_contour(study, params=[params[i], params[j]])
                fig.write_image(
                    os.path.join(plots_dir, f"contour_{params[i]}_vs_{params[j]}.png")
                )
                log(f"Saved contour plot for {params[i]} vs {params[j]}")

        # 5. Parallel coordinate plot
        fig5 = plot_parallel_coordinate(study)
        fig5.write_image(os.path.join(plots_dir, "parallel_coordinate.png"))
        log("Saved parallel coordinate plot")

        log(f"All plots saved to: {plots_dir}")

    except Exception as e:
        error_msg = f"Error saving images: {e}"
        if logger:
            logger.error(error_msg)
        else:
            print(error_msg)
        traceback.print_exc()


# Example usage:
# save_optuna_plots(study, current_dir, logger)


def nb_cluster_analysis(study):
    # Extraire les données des essais Optuna
    sim_thresholds = []
    n_clusters_list = []
    scores = []

    for trial in study.trials:
        if (
            trial.value is not None
            and "n_clusters" in trial.user_attrs
            and "sim_threshold" in trial.user_attrs
            and "score" in trial.user_attrs
        ):
            sim_thresholds.append(trial.user_attrs["sim_threshold"])
            n_clusters_list.append(trial.user_attrs["n_clusters"])
            scores.append(trial.user_attrs["score"])

    # Conversion en tableaux numpy
    sim_thresholds = np.array(sim_thresholds)
    n_clusters_list = np.array(n_clusters_list)
    scores = np.array(scores)

    # ----- Plot 3D : sim_threshold vs n_clusters vs score -----
    fig_3d = go.Figure(
        data=[
            go.Scatter3d(
                x=sim_thresholds,
                y=n_clusters_list,
                z=scores,
                mode="markers",
                marker=dict(
                    size=5,
                    color=scores,
                    colorscale="Blues",
                    opacity=0.7,
                    colorbar=dict(title="calinski harabasz Score"),
                ),
            )
        ]
    )

    fig_3d.update_layout(
        scene=dict(
            xaxis_title="sim_threshold",
            yaxis_title="Nombre de clusters",
            zaxis_title="calinski harabasz Score",
        ),
        title="sim_threshold vs nombre de clusters vs calinski harabasz score",
        margin=dict(l=0, r=0, b=0, t=30),
    )

    # ----- Scatter plot : sim_threshold vs score -----
    corr_1, p_value_1 = pearsonr(sim_thresholds, scores)

    fig_1 = go.Figure(
        data=[
            go.Scatter(
                x=sim_thresholds,
                y=scores,
                mode="markers",
                marker=dict(
                    size=8,
                    color=n_clusters_list,
                    colorscale="Blues",
                    opacity=0.7,
                    colorbar=dict(title="Nombre de clusters"),
                ),
            )
        ]
    )

    fig_1.update_layout(
        title="sim_threshold vs calinski harabasz Score",
        xaxis_title="sim_threshold",
        yaxis_title="calinski harabasz Score",
        annotations=[
            dict(
                x=0.95,
                y=0.05,
                xref="paper",
                yref="paper",
                text=f"Corrélation : {corr_1:.3f}<br>p-value : {p_value_1:.3f}",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
            )
        ],
    )

    # ----- Scatter plot : n_clusters vs score -----
    corr_2, p_value_2 = pearsonr(n_clusters_list, scores)

    fig_2 = go.Figure(
        data=[
            go.Scatter(
                x=n_clusters_list,
                y=scores,
                mode="markers",
                marker=dict(
                    size=8,
                    color=sim_thresholds,
                    colorscale="Blues",
                    opacity=0.7,
                    colorbar=dict(title="sim_threshold"),
                ),
            )
        ]
    )

    fig_2.update_layout(
        title="Nombre de clusters vs calinski harabasz Score",
        xaxis_title="Nombre de clusters",
        yaxis_title="calinski harabasz Score",
        annotations=[
            dict(
                x=0.95,
                y=0.05,
                xref="paper",
                yref="paper",
                text=f"Corrélation : {corr_2:.3f}<br>p-value : {p_value_2:.3f}",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
            )
        ],
    )

    # ----- Scatter plot : sim_threshold vs n_clusters -----
    corr_3, p_value_3 = pearsonr(sim_thresholds, n_clusters_list)

    fig_3 = go.Figure(
        data=[
            go.Scatter(
                x=sim_thresholds,
                y=n_clusters_list,
                mode="markers",
                marker=dict(
                    size=8,
                    color=scores,
                    colorscale="Blues",
                    opacity=0.7,
                    colorbar=dict(title="calinski harabasz Score"),
                ),
            )
        ]
    )

    fig_3.update_layout(
        title="sim_threshold vs Nombre de clusters",
        xaxis_title="sim_threshold",
        yaxis_title="Nombre de clusters",
        annotations=[
            dict(
                x=0.95,
                y=0.05,
                xref="paper",
                yref="paper",
                text=f"Corrélation : {corr_3:.3f}<br>p-value : {p_value_3:.3f}",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
            )
        ],
    )

    return fig_3d, fig_1, fig_2, fig_3
