import pandas as pd
import openai
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

# --- Configuration ---
INPUT_CSV = "wsl_policies_with_clusters.csv"
OUTPUT_CSV = "wsl_policies_clustered_and_named.csv"
MODEL_NAME = "gpt-4o-mini"
SAMPLES_PER_CLUSTER = 40     

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_cluster_name(cluster_id, texts):
    """Sends a list of policy texts to the LLM and asks for a category name."""
    text_preview = "\n- ".join(texts)
    
    prompt = (
        f"You are a policy analyst. Below is a list of policy excerpts that belong to the same cluster (Cluster ID: {cluster_id}).\n\n"
        f"POLICIES:\n{text_preview}\n\n"
        f"TASK: Provide a short, specific, and professional name for this cluster (max 5-7 words). "
        f"Do not use quotes. Just the name."
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a helpful taxonomist for public policy."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error naming cluster {cluster_id}: {e}")
        return f"Cluster {cluster_id}"

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"❌ Could not find {INPUT_CSV}. Run the linking script first!")
        return

    print(f"Loading {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)

    text_col = 'single_policy_item' 
    if text_col not in df.columns:
        possible_cols = [c for c in df.columns if df[c].dtype == 'object']
        print(f"⚠️ Column '{text_col}' not found. Using '{possible_cols[0]}' instead.")
        text_col = possible_cols[0]

    unique_clusters = df['cluster_label'].unique()
    unique_clusters = sorted([c for c in unique_clusters if pd.notna(c)])
    
    print(f"Found {len(unique_clusters)} unique clusters.")
    print("Generating names (this may take a moment)...")

    cluster_names_map = {}

    for cluster_id in tqdm(unique_clusters, desc="Naming Clusters"):
        # Filter for current cluster
        cluster_data = df[df['cluster_label'] == cluster_id]
        
        # Get valid texts only
        valid_texts = cluster_data[text_col].dropna()
        
        # Skip if no text available
        if len(valid_texts) == 0:
            cluster_names_map[cluster_id] = f"Cluster {cluster_id}"
            continue

        # Calculate safe sample size based on VALID texts count
        n_samples = min(SAMPLES_PER_CLUSTER, len(valid_texts))
        
        # Sample
        sample_texts = valid_texts.sample(n=n_samples, random_state=42).tolist()
        
        # Call LLM
        name = get_cluster_name(cluster_id, sample_texts)
        cluster_names_map[cluster_id] = name

    print("\nApplying names to dataset...")
    df['cluster_name'] = df['cluster_label'].map(cluster_names_map)

    print("\n--- Sample of Generated Names ---")
    print(df[['cluster_label', 'cluster_name']].drop_duplicates().head(10))

    print(f"\nSaving to {OUTPUT_CSV}...")
    df.to_csv(OUTPUT_CSV, index=False)
    print("✅ Done!")

if __name__ == "__main__":
    main()