#!/usr/bin/env python3
"""
Test script for merge_policies_knn.py with actual database data
"""

import sys
from pathlib import Path

import pandas as pd

# Add the parent directory to the path to import the merge_policies_knn module
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from agentic_data_policies_extraction.policies_transformation_to_matrices.merge_policies_knn import merge_policies_semantic_medoid


def test_with_database_data():
    """Test the clustering with actual data from the flattened policies CSV"""
    try:
        # Load the flattened data we created earlier
        input_file = "./data/flattened_policies.csv"

        if not Path(input_file).exists():
            print(f"Error: {input_file} not found. Please run db_to_csv.py first.")
            return

        print(f"Loading data from {input_file}...")
        df = pd.read_csv(input_file)

        if "policy" not in df.columns:
            print("Error: 'policy' column not found in the data")
            print(f"Available columns: {df.columns.tolist()}")
            return

        print(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
        print(f"Sample policies:")
        for i, policy in enumerate(df["policy"].head(5)):
            print(f"  {i + 1}. {policy[:100]}...")

        # Test clustering with a subset to avoid memory issues
        test_size = min(50, len(df))
        test_df = df.head(test_size).copy()

        print(f"\nTesting clustering with {test_size} policies...")

        # Run the clustering
        result_df = merge_policies_semantic_medoid(
            test_df,
            text_col="policy",
            batch_size=16,  # Small batch size for safety
            max_neighbors=8,  # Reduced neighbors
            sim_threshold=0.75,  # Slightly lower threshold
        )

        # Show results
        print("\nClustering Results:")
        print(f"Total policies: {len(result_df)}")
        print(f"Number of clusters: {result_df['cluster_id'].nunique()}")

        # Show cluster summary
        cluster_summary = (
            result_df.groupby(["cluster_id", "policy_canonical"])
            .size()
            .reset_index(name="count")
            .sort_values(["cluster_id", "count"], ascending=[True, False])
        )

        print("\nCluster Summary:")
        print(cluster_summary)

        # Show some example clusters
        print("\nExample Clusters:")
        for cluster_id in sorted(result_df["cluster_id"].unique())[:5]:
            cluster_policies = result_df[result_df["cluster_id"] == cluster_id]
            canonical = cluster_policies["policy_canonical"].iloc[0]
            count = len(cluster_policies)

            print(f"\nCluster {cluster_id} ({count} policies):")
            print(f"  Canonical: {canonical[:80]}...")

            if count > 1:
                print("  Members:")
                for _, row in cluster_policies.iterrows():
                    policy = row["policy"]
                    if policy != canonical:
                        print(f"    - {policy[:60]}...")

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    test_with_database_data()
