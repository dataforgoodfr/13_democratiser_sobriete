# optuna_optimization/preprocess_for_optuna.py
import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Add the parent directory to the path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir))

from db_to_csv import (assign_factor_to_related_taxonomies,
                       assign_policy_to_related_taxonomies,
                       flatten_extracted_data, get_dataframe_with_filters)


def preprocess_data(limit: int = 10):
    print("preprocess_data")
    """
    Preprocess data for Optuna optimization.
    Returns a tuple of (flattened_df, pivot_df) just like run_complete_pipeline.
    """
    try:
        logger.info("=== Starting Data Preprocessing for Optuna ===")

        # Step 1: Extract data from database
        logger.info("Step 1: Extracting data from database...")
        df = get_dataframe_with_filters(limit=limit)
        if df is None or len(df) == 0:
            logger.error("No data extracted from database")
            return None, None

        logger.info(f"Extracted {len(df)} records from database")

        # Step 2: Flatten the extracted data
        logger.info("Step 2: Flattening extracted data...")
        flattened_df = flatten_extracted_data(df)
        if flattened_df is None or len(flattened_df) == 0:
            logger.error("No data to flatten")
            return None, None

        logger.info(f"Flattened to {len(flattened_df)} policy-factor combinations")

        # Step 3: Assign taxonomies
        logger.info("Step 3: Assigning taxonomies...")
        try:
            # Process in smaller batches to avoid memory issues
            batch_size = 100
            total_rows = len(flattened_df)

            # Initialize columns
            flattened_df["related_studied_policy_area"] = "Unknown"
            flattened_df["related_studied_sector"] = "Unknown"

            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)

                logger.info(
                    f"Processing taxonomy batch {start_idx // batch_size + 1}/{(total_rows + batch_size - 1) // batch_size}"
                )

                # Process factors
                for idx in range(start_idx, end_idx):
                    if idx < len(flattened_df):
                        factor = flattened_df.loc[idx, "factor"]
                        policy = flattened_df.loc[idx, "policy"]

                        try:
                            flattened_df.loc[idx, "related_studied_policy_area"] = (
                                assign_factor_to_related_taxonomies(factor)
                            )
                            flattened_df.loc[idx, "related_studied_sector"] = (
                                assign_policy_to_related_taxonomies(policy)
                            )
                        except Exception as e:
                            logger.warning(f"Error processing row {idx}: {e}")
                            continue
        except Exception as e:
            logger.error(f"Error during taxonomy assignment: {e}")
            # Fallback: assign all to "Unknown"
            flattened_df["related_studied_policy_area"] = "Unknown"
            flattened_df["related_studied_sector"] = "Unknown"

        # Step 4: Transform correlation values
        logger.info("Step 4: Transforming correlation values...")
        correlation_mapping = {"decreasing": -1, "increasing": 1}
        flattened_df["correlation_numeric"] = flattened_df["correlation"].map(
            correlation_mapping
        )
        flattened_df["correlation_numeric"] = flattened_df["correlation_numeric"].fillna(0)

        # Step 5: Create pivot table
        logger.info("Step 5: Creating policy-sector correlation matrix...")
        pivot_df = flattened_df.pivot_table(
            index="related_studied_policy_area",
            columns="related_studied_sector",
            values="correlation_numeric",
            aggfunc="mean",
            fill_value=0,
        )

        logger.info("=== Data Preprocessing Completed ===")
        logger.info(f"Final dataset size: {len(flattened_df)}")
        return flattened_df, pivot_df

    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        import traceback

        traceback.print_exc()
        return None, None
