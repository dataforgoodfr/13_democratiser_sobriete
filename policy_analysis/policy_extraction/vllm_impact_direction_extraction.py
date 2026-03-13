"""
WARNING: this script HEAVILY relies on prefix caching, or it would be impossibly slow.
The system prompt is repeated across all prompts, but each chunk text is also repeated many times (~20 on average) across policy clusters and impacts, and each policy cluster across impacts.
So don't change the structure of the prompts without understanding how prefix caching works.
Another optimization is that the labels are single-token, so the model has a single token to generate per prediction.
"""

import json
import os

import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


OFFSET = 0
if "SLURM_ARRAY_TASK_ID" in os.environ:
    OFFSET = int(os.environ["SLURM_ARRAY_TASK_ID"])
NJOBS = 10
BATCH_SIZE = 10_000
print(f"Running with OFFSET={OFFSET}, NJOBS={NJOBS}, BATCH_SIZE={BATCH_SIZE}")

INPUT_FILE = "chunked_conclusions_557k_2026-01-26.parquet"
CLUSTER_FILE = 'cluster_representatives_2026-03-10.csv'
MAPPING_FILE = 'chunk_cluster_impacts_map_2026-03-11.parquet'
PROMPT_FILE = "IMPACT_DIRECTION_EXTRACTION_PROMPT.txt"
OUTPUT_FILE = f"impactdir_extraction_results_{OFFSET}.jsonl"
MODEL_NAME = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
NGPUS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

clusters = pd.read_csv(CLUSTER_FILE)
clusters = clusters.rename(columns={'text': 'cluster_name'}).set_index('cluster_id')['cluster_name']
print('Loaded clusters:', len(clusters))

mapping = pd.read_parquet(MAPPING_FILE)
print('Loaded mapping:', len(mapping))

model_path = f"{os.environ['DSDIR']}/HuggingFace_Models/{MODEL_NAME}"

print("Loading model", MODEL_NAME)
if "mistral" in MODEL_NAME:
    llm = LLM(
        model_path,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        tensor_parallel_size=NGPUS,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
    )
else:
    llm = LLM(
        model_path,
        tensor_parallel_size=NGPUS,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.9,
    )
print("Model loaded successfully")

with open(PROMPT_FILE) as f:
    system_prompt = f.read().strip()


choices = ['positive', 'negative', 'neutral', 'unknown']
sampling_params = SamplingParams(
    temperature=0,
    max_tokens=1,
    structured_outputs=StructuredOutputsParams(
        choice=choices
    ),
)

def preprocess_batch(batch: pd.DataFrame):
    print('Chunks in batch:', len(batch))
    batch = batch.merge(mapping, on=['openalex_id', 'chunk_idx'], how='inner')
    print('Of which have policies:', len(batch))

    if len(batch) == 0:
        return None
    
    rename_dict = {
        'Resource_pred': 'Natural Resources',
        'wellbeing_pred': 'Wellbeing',
        'Need_pred': 'Human Needs',
        'PlanetaryBoundary_pred': 'Planetary Boundaries',
        'Justice_pred': 'Justice',
    }
    batch = batch.rename(columns=rename_dict)
    batch["impact_dim"] = batch.apply(
        lambda row: [
            f"{col} - {dim.replace('_', ' ').capitalize()}"
            for col in rename_dict.values()
            for dim in row[col]
        ],
        axis=1,
    )
    batch = batch[['openalex_id', 'chunk_idx', 'text', 'cluster_id', 'impact_dim']]
    batch = batch.explode('cluster_id').reset_index(drop=True).join(clusters, on='cluster_id')
    batch = batch[['openalex_id', 'chunk_idx', 'text', 'cluster_id', 'cluster_name', 'impact_dim']]
    batch = batch.explode('impact_dim').reset_index(drop=True)
    print('Impact dimensions to classify:', len(batch))
    return batch


def get_prompts_metadata(batch: pd.DataFrame):
    all_prompts = []
    all_metadata = []
    for row in batch.itertuples():
        prompt = f"{system_prompt}\n\nReference text:\n{row.text}\n\nPolicy: {row.cluster_name}\nImpact Dimension: {row.impact_dim}\nImpact direction:"
        all_prompts.append(prompt)
        all_metadata.append({
            'openalex_id': row.openalex_id,
            'chunk_idx': int(row.chunk_idx),
            'cluster_id': int(row.cluster_id),
            'impact_dim': row.impact_dim,
        })
    return all_prompts, all_metadata


def batch_extract_impact_directions(batch: pd.DataFrame):
    batch = preprocess_batch(batch)
    
    if batch is None:
        return []
    
    prompts, metadata = get_prompts_metadata(batch)
    outputs = llm.generate(prompts, sampling_params)
    results = []
    for output, meta in zip(outputs, metadata, strict=True):
        try:
            label = output.outputs[0].text.strip()
            r = meta | {"impact_direction": label}
        except Exception as e:
            print(f"Error parsing output: {e}")
            r = meta | {"error": str(e)}
        finally:
            results.append(r)
    return results


parquet_file = pq.ParquetFile(INPUT_FILE)
total_rows = parquet_file.metadata.num_rows
n_batches = (total_rows + BATCH_SIZE - 1) // BATCH_SIZE
print(f"File contains {total_rows} rows, amounting to {n_batches} batches of {BATCH_SIZE}")

for batch in tqdm(
    parquet_file.iter_batches(BATCH_SIZE),
    desc="Processing batches",
    total=n_batches,
):
    batch_df = batch.to_pandas().reset_index().iloc[OFFSET::NJOBS]
    results = batch_extract_impact_directions(batch_df)

    with open(OUTPUT_FILE, "a") as f_out:
        for r in results:
            f_out.write(json.dumps(r) + "\n")
