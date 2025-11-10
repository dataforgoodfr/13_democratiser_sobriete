import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Add the rag_system path to sys.path to access taxonomy
current_dir = Path(__file__).parent.resolve()
rag_system_dir = current_dir.parent.parent.parent
sys.path.append(str(rag_system_dir))

from taxonomy.taxonomy.themes_taxonomy import (Studied_policy_area,
                                               Studied_sector)

# Global model variable - will be initialized when needed
_model = None

# Add the parent directory to the path to import the DatabaseClient
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from clients.database_client import DatabaseClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_dataframe_with_filters(where_clause: str = None, limit: int = None) -> None:
    """
    Export policies with custom filters.

    Args:
        output_file: Name of the output CSV file
        where_clause: SQL WHERE clause (without the WHERE keyword)
        limit: Maximum number of records to export
    """
    try:
        db_client = DatabaseClient()

        # Build the query
        query = "SELECT * FROM public.policies_abstracts_all"
        params = {}

        if where_clause:
            query += f" WHERE {where_clause}"

        query += " ORDER BY openalex_id"

        if limit:
            query += " LIMIT :limit"
            params["limit"] = limit

        logger.info(f"Executing query: {query}")
        policies = db_client.execute_query(query, params)

        if not policies:
            logger.warning("No records found matching the criteria")
            return

        # Convert to DataFrame and export
        df = pd.DataFrame(policies)
        return df

    except Exception as e:
        logger.error(f"Error exporting filtered policies: {e}")
        raise


def flatten_extracted_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the extracted_data JSON column into a flattened DataFrame.

    Args:
        df: DataFrame containing extracted_data column with JSON data

    Returns:
        pd.DataFrame: Flattened DataFrame with columns: policy, actor, population, factor, correlation
    """
    flattened_rows = []

    for idx, row in df.iterrows():
        try:
            # Parse the JSON data from extracted_data column
            extracted_data_raw = row["extracted_data"]
            if extracted_data_raw is None or (
                isinstance(extracted_data_raw, float) and pd.isna(extracted_data_raw)
            ):
                logger.warning(f"No extracted_data found for row {idx}")
                continue

            # Handle both string and dict types
            if isinstance(extracted_data_raw, str):
                extracted_data = json.loads(extracted_data_raw)
            else:
                extracted_data = extracted_data_raw

            # Handle different data structures
            policies_to_process = []

            if isinstance(extracted_data, list):
                # If it's a list, iterate through each item in the list
                for item in extracted_data:
                    if isinstance(item, dict):
                        policies_to_process.append(item)
            elif isinstance(extracted_data, dict):
                # If it's a dict, add it directly
                policies_to_process.append(extracted_data)
            else:
                logger.warning(
                    f"Row {idx}: extracted_data is not a dict or list, type: {type(extracted_data)}"
                )
                continue

            # Process all policy dictionaries
            for policy_dict in policies_to_process:
                # Skip the GEOGRAPHIC key and process policy keys
                for policy_name, policy_data in policy_dict.items():
                    if policy_name == "GEOGRAPHIC":
                        continue

                    if not isinstance(policy_data, dict):
                        continue

                    # Extract basic policy information
                    actor = policy_data.get("ACTOR", "None")
                    population = policy_data.get("POPULATION", "None")

                    # Process factors
                    factors = policy_data.get("FACTOR", {})
                    if isinstance(factors, dict):
                        for factor_name, factor_data in factors.items():
                            if isinstance(factor_data, dict):
                                correlation = factor_data.get("CORRELATION", "None")

                                flattened_rows.append(
                                    {
                                        "policy": policy_name,
                                        "actor": actor,
                                        "population": population,
                                        "factor": factor_name,
                                        "correlation": correlation,
                                    }
                                )

        except (json.JSONDecodeError, TypeError, KeyError) as e:
            logger.error(f"Error processing row {idx}: {e}")
            continue

    if not flattened_rows:
        logger.warning("No valid data found to flatten")
        return pd.DataFrame(columns=["policy", "actor", "population", "factor", "correlation"])

    flattened_df = pd.DataFrame(flattened_rows)
    logger.info(f"Successfully flattened {len(flattened_rows)} policy-factor combinations")

    return flattened_df


def get_model():
    """Get or initialize the sentence transformer model with error handling."""
    global _model
    try:
        if _model is None:
            logger.info("Initializing sentence transformer model...")
            _model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
            logger.info("Model initialized successfully")
        return _model
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        return None


def assign_factor_to_related_taxonomies(factor_name: str):
    """
    Assign the factor to the related taxonomies.
    """
    if not factor_name or factor_name == "None":
        return "Unknown"

    try:
        model = get_model()
        if model is None:
            logger.warning("Model not available, returning 'Unknown'")
            return "Unknown"

        # encodage des catégories une seule fois
        cat_texts = [item.value for item in Studied_policy_area]
        cat_vectors = model.encode(cat_texts, normalize_embeddings=True)

        # encodage du facteur
        factor_vector = model.encode([factor_name], normalize_embeddings=True)

        # calculer les similarités
        sims = cosine_similarity(factor_vector, cat_vectors)[
            0
        ]  # similarité avec chaque catégorie
        best = int(np.argmax(sims))  # indice de la + proche
        return list(Studied_policy_area)[best].name
    except Exception as e:
        logger.error(f"Error in assign_factor_to_related_taxonomies: {e}")
        return "Unknown"


def assign_policy_to_related_taxonomies(policy_name: str):
    """
    Assign the factor to the related taxonomies.
    """
    if not policy_name or policy_name == "None":
        return "Unknown"

    try:
        model = get_model()
        if model is None:
            logger.warning("Model not available, returning 'Unknown'")
            return "Unknown"

        # encodage des catégories une seule fois
        cat_texts = [item.value for item in Studied_sector]
        cat_vectors = model.encode(cat_texts, normalize_embeddings=True)

        # encodage du facteur
        factor_vector = model.encode([policy_name], normalize_embeddings=True)

        # calculer les similarités
        sims = cosine_similarity(factor_vector, cat_vectors)[
            0
        ]  # similarité avec chaque catégorie
        best = int(np.argmax(sims))  # indice de la + proche
        return list(Studied_policy_area)[best].name
    except Exception as e:
        logger.error(f"Error in assign_policy_to_related_taxonomies: {e}")
        return "Unknown"


def main():
    """
    Main function to run the export process.
    """
    try:
        # Test model initialization first
        logger.info("Testing model initialization...")
        test_model = get_model()
        if test_model is None:
            logger.error("Failed to initialize model. Exiting.")
            return

        df = get_dataframe_with_filters(
            limit=5  # Start with a smaller limit to test
        )
        print("Original DataFrame:")
        print(df.head())
        print(f"\nColumns: {df.columns.tolist()}")

        # Transform the extracted_data into flattened format
        flattened_df = flatten_extracted_data(df)

        print("\nFlattened DataFrame:")
        print(flattened_df)

        # Apply the function to each factor and create new column
        print("\nApplying factor taxonomy assignment...")
        try:
            # Process in smaller batches to avoid memory issues
            batch_size = 100
            total_rows = len(flattened_df)

            # Initialize columns
            flattened_df["related_studied_policy_area"] = "Unknown"
            flattened_df["related_studied_sector"] = "Unknown"

            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                batch_df = flattened_df.iloc[start_idx:end_idx]

                logger.info(
                    f"Processing batch {start_idx // batch_size + 1}/{(total_rows + batch_size - 1) // batch_size}"
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

        # Transform correlation values: 'decreasing' to -1, 'increasing' to 1
        print("\nTransforming correlation values...")
        correlation_mapping = {"decreasing": -1, "increasing": 1}
        flattened_df["correlation_numeric"] = flattened_df["correlation"].map(
            correlation_mapping
        )

        # Fill NaN values with 0 for any unmapped correlations
        flattened_df["correlation_numeric"] = flattened_df["correlation_numeric"].fillna(0)

        print(
            "\nDataFrame with related_studied_sector, related_studied_policy_area, and correlation_numeric columns:"
        )
        print(flattened_df)

        # Create pivot table: rows=policy_areas, columns=sectors, values=mean correlation
        print("\nCreating pivot table...")
        pivot_df = flattened_df.pivot_table(
            index="related_studied_policy_area",
            columns="related_studied_sector",
            values="correlation_numeric",
            aggfunc="mean",
            fill_value=0,
        )

        print("\nPivot DataFrame (Policy Areas x Sectors):")
        print(pivot_df)

        # Save pivot table to CSV
        pivot_output_file = "policy_sector_correlation_matrix.csv"
        pivot_df.to_csv(pivot_output_file)
        logger.info(f"Pivot table saved to {pivot_output_file}")

        # Optionally save to CSV
        output_file = "flattened_policies.csv"
        flattened_df.to_csv(output_file, index=False)
        logger.info(f"Flattened data saved to {output_file}")

        logger.info("Script completed successfully")

    except Exception as e:
        logger.error(f"Script failed: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Clean up model to free memory
        global _model
        if _model is not None:
            try:
                del _model
                _model = None
                logger.info("Model cleaned up")
            except:
                pass


if __name__ == "__main__":
    main()
