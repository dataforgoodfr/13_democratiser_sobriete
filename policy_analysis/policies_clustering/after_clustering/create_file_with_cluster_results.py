import numpy as np
from datasets import load_dataset

# --- Configuration ---
HF_DATASET_ID = "EdouardCallet/wsl-policy-10k"
LABELS_PATH = 'src/policy_analysis/policies_clustering/results/clustering_experiment/labels.npy'
OUTPUT_FILENAME = "wsl_policies_with_clusters.csv"

def main():
    print(f"Loading dataset: {HF_DATASET_ID}...")
    ds = load_dataset(HF_DATASET_ID, split="train")
    
    print(f"Loading labels from: {LABELS_PATH}...")
    try:
        labels = np.load(LABELS_PATH, allow_pickle=True)
    except Exception as e:
        print(f"❌ Error loading .npy file: {e}")
        return

    # --- Verification ---
    print("\n--- Checking Dimensions ---")
    n_rows = len(ds)
    n_labels = len(labels)
    print(f"Dataset rows: {n_rows}")
    print(f"Labels count: {n_labels}")

    if n_rows != n_labels:
        print("\n⚠️  MISMATCH DETECTED ⚠️")
        print(f"You have {n_rows} documents but {n_labels} labels.")
        print("Reason: Your clustering likely ran on 'chunks' (sentences) rather than full documents.")
        print("To fix this, we need to map the chunks back to the parent documents.")
        return

    # --- Merging ---
    print("\nMerging data...")
    df = ds.to_pandas()
    df['cluster_label'] = labels

    # --- Inspecting Result ---
    print("\nTop 5 rows with new labels:")
    print(df[['cluster_label']].head(5)) 

    # --- Saving ---
    print(f"\nSaving to {OUTPUT_FILENAME}...")
    df.to_csv(OUTPUT_FILENAME, index=False)
    print("✅ Success! File saved locally.")

    # Optional: Push to Hub
    # print("Pushing to Hub...")
    # new_dataset = Dataset.from_pandas(df)
    # new_dataset.push_to_hub("EdouardCallet/wsl_10k_policy_and_taxonomy_clustered")

# --- EXECUTION BLOCK (This was likely missing) ---
if __name__ == "__main__":
    print("Script started...")
    main()