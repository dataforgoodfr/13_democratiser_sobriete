#!/usr/bin/env python3
"""
Complete pipeline: Database extraction → Flattening → Taxonomy → Clustering → Analysis
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the current directory to the path to import our modules
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from db_to_csv import (
    get_dataframe_with_filters, 
    flatten_extracted_data, 
    assign_factor_to_related_taxonomies, 
    assign_policy_to_related_taxonomies
)
from merge_policies_knn import merge_policies_semantic_medoid

def run_complete_pipeline(limit: int = 100, sim_threshold: float = 0.75):
    """
    Run the complete pipeline from database to final analysis
    """
    try:
        logger.info("=== Starting Complete Pipeline ===")
        
        # Step 1: Extract data from database
        logger.info("Step 1: Extracting data from database...")
        df = get_dataframe_with_filters(limit=limit)
        if df is None or len(df) == 0:
            logger.error("No data extracted from database")
            return None
            
        logger.info(f"Extracted {len(df)} records from database")
        
        # Step 2: Flatten the extracted data
        logger.info("Step 2: Flattening extracted data...")
        flattened_df = flatten_extracted_data(df)
        if flattened_df is None or len(flattened_df) == 0:
            logger.error("No data to flatten")
            return None
            
        logger.info(f"Flattened to {len(flattened_df)} policy-factor combinations")
        
        # Step 3: Assign taxonomies
        logger.info("Step 3: Assigning taxonomies...")
        try:
            # Process in smaller batches to avoid memory issues
            batch_size = 50
            total_rows = len(flattened_df)
            
            # Initialize columns
            flattened_df['related_studied_policy_area'] = "Unknown"
            flattened_df['related_studied_sector'] = "Unknown"
            
            for start_idx in range(0, total_rows, batch_size):
                end_idx = min(start_idx + batch_size, total_rows)
                
                logger.info(f"Processing taxonomy batch {start_idx//batch_size + 1}/{(total_rows + batch_size - 1)//batch_size}")
                
                # Process factors
                for idx in range(start_idx, end_idx):
                    if idx < len(flattened_df):
                        factor = flattened_df.loc[idx, 'factor']
                        policy = flattened_df.loc[idx, 'policy']
                        
                        try:
                            flattened_df.loc[idx, 'related_studied_policy_area'] = assign_factor_to_related_taxonomies(factor)
                            flattened_df.loc[idx, 'related_studied_sector'] = assign_policy_to_related_taxonomies(policy)
                        except Exception as e:
                            logger.warning(f"Error processing row {idx}: {e}")
                            continue

        except Exception as e:
            logger.error(f"Error during taxonomy assignment: {e}")
            # Fallback: assign all to "Unknown"
            flattened_df['related_studied_policy_area'] = "Unknown"
            flattened_df['related_studied_sector'] = "Unknown"
        
        # Step 4: Transform correlation values
        logger.info("Step 4: Transforming correlation values...")
        correlation_mapping = {'decreasing': -1, 'increasing': 1}
        flattened_df['correlation_numeric'] = flattened_df['correlation'].map(correlation_mapping)
        flattened_df['correlation_numeric'] = flattened_df['correlation_numeric'].fillna(0)
        
        # Step 5: Create pivot table
        logger.info("Step 5: Creating policy-sector correlation matrix...")
        pivot_df = flattened_df.pivot_table(
            index='related_studied_policy_area',
            columns='related_studied_sector',
            values='correlation_numeric',
            aggfunc='mean',
            fill_value=0
        )
        
        # Step 6: Cluster similar policies
        logger.info("Step 6: Clustering similar policies...")
        try:            
            flattened_df = merge_policies_semantic_medoid(
                flattened_df,
                text_col="policy",
                batch_size=256,
                max_neighbors=8,
                sim_threshold=sim_threshold
            )

        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            flattened_df['cluster_id'] = -1
            flattened_df['policy_canonical'] = flattened_df['policy']
        
        # Step 7: Save results
        logger.info("Step 7: Saving results...")
        
        # Save flattened data with all annotations
        flattened_output = "complete_flattened_policies.csv"
        flattened_df.to_csv(flattened_output, index=False)
        logger.info(f"Complete flattened data saved to {flattened_output}")
        
        # Save pivot table
        pivot_output = "complete_policy_sector_matrix.csv"
        pivot_df.to_csv(pivot_output)
        logger.info(f"Policy-sector matrix saved to {pivot_output}")
        
        # Save clustering results if available
        if 'cluster_id' in flattened_df.columns and flattened_df['cluster_id'].nunique() > 1:
            cluster_output = "policy_clusters.csv"
            cluster_summary = (flattened_df.groupby(['cluster_id', 'policy_canonical'])
                              .size()
                              .reset_index(name='count')
                              .sort_values(['cluster_id', 'count'], ascending=[True, False]))
            cluster_summary.to_csv(cluster_output, index=False)
            logger.info(f"Policy clusters saved to {cluster_output}")
        
        # Step 8: Generate summary report
        logger.info("Step 8: Generating summary report...")
        
        summary_report = {
            "total_records": len(df),
            "total_policy_factors": len(flattened_df),
            "unique_policies": flattened_df['policy'].nunique(),
            "unique_factors": flattened_df['factor'].nunique(),
            "policy_areas": flattened_df['related_studied_policy_area'].nunique(),
            "sectors": flattened_df['related_studied_sector'].nunique(),
            "clusters": flattened_df['cluster_id'].nunique() if 'cluster_id' in flattened_df.columns else 0,
            "correlation_distribution": flattened_df['correlation_numeric'].value_counts().to_dict()
        }
        
        logger.info("=== Pipeline Summary ===")
        for key, value in summary_report.items():
            logger.info(f"{key}: {value}")
        
        logger.info("=== Pipeline Completed Successfully ===")
        return flattened_df, pivot_df
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    result_df, pivot_df = run_complete_pipeline(limit=50000, sim_threshold=0.75)
    
    if result_df is not None:
        print("\nPipeline completed successfully!")
        print(f"Results saved to: complete_flattened_policies.csv, complete_policy_sector_matrix.csv")
    else:
        print("Pipeline failed. Check logs for details.") 