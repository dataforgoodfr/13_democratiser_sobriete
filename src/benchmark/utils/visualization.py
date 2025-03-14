import argparse
import matplotlib.pyplot as plt
import pandas as pd
import math
import os


def read_args():
    """Read command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluation Visualization")

    parser.add_argument(
        "--eval_file_name",
        type=str,
        required=True,
        help="Path to the evaluation file (CSV format)",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the output image (default: False, just displays it)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="metrics_visualization.png",
        help="Output image file name",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["png", "jpg", "pdf"],
        default="png",
        help="Output file format",
    )
    parser.add_argument("--limit_k", type=int, default=None, help="Define a @k limit")

    return parser.parse_args()


def display_metrics(
    df: pd.DataFrame,
    output_file: str = "metrics_visualization.png",
    save: bool = False,
    limit_k: int = None,
):
    """Generate and save all metric@k plots in a single image file."""

    config_names = df.loc[:, ("retriever", "@k")].to_list()

    # Drop the 'retriever' column since it's just an identifier
    df = df.drop(columns=[("retriever", "@k")])

    # Extract unique metric names
    metrics = sorted(set(metric for metric, _ in df.columns))

    # Determine grid size (rows, cols) dynamically based on the number of metrics
    num_metrics = len(metrics)
    num_cols = min(3, num_metrics)  # Max 3 columns for better readability
    num_rows = math.ceil(num_metrics / num_cols)

    # Create subplots for all metrics in a single figure
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 5, num_rows * 4))
    axes = axes.flatten()  # Flatten to easily iterate, even if 1 row

    for i, metric in enumerate(metrics):
        ax = axes[i]  # Select the subplot

        ax.set_title(f"{metric} @ k")

        # Extract k values and sort them
        k_values = sorted([int(k) for m, k in df.columns if m == metric])
        if limit_k:
            k_values = filter(lambda k: k <= limit_k, k_values)
        k_values = [str(k) for k in k_values]  # Ensure k values are strings

        # Plot each retriever's scores
        for index, row in df.iterrows():
            retriever_name = config_names[index]  # Retrieve configuration name
            scores = row[[(metric, k) for k in k_values]]  # Get scores
            ax.plot(k_values, scores, marker="o", linestyle="-", label=retriever_name)

        # Formatting
        ax.set_xlabel("k")
        ax.set_ylabel("Metric Score")
        ax.legend()
        ax.grid(True)

    # Hide empty subplots if there are fewer metrics than grid spaces
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout
    plt.tight_layout()

    if save:
        plt.savefig(output_file, dpi=300)
        print(f"All plots saved to {output_file}")
    else:
        plt.show()

    plt.close()


def main():
    """Main function to run the script."""
    args = read_args()

    # Check if the file exists
    if not os.path.exists(args.eval_file_name):
        print(f"Error: File '{args.eval_file_name}' not found!")
        return

    # Read CSV file with MultiIndex columns
    try:
        df = pd.read_csv(args.eval_file_name, header=[0, 1])  # Expecting MultiIndex
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    # Construct output file with chosen format
    output_file = f"{os.path.splitext(args.output_file)[0]}.{args.format}"

    # Display metrics (and save if requested)
    display_metrics(df, output_file=output_file, save=args.save, limit_k=args.limit_k)


if __name__ == "__main__":
    main()
