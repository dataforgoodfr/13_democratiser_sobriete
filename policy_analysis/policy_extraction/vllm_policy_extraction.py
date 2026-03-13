import json
import os

import pyarrow.parquet as pq
from pydantic import BaseModel, Field
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams


OFFSET = int(os.environ["SLURM_ARRAY_TASK_ID"])
NJOBS = 10
BATCH_SIZE = 10_000
print(f"Running with OFFSET={OFFSET}, NJOBS={NJOBS}, BATCH_SIZE={BATCH_SIZE}")

INPUT_FILE = "chunked_conclusions_557k_2026-01-26.parquet"
OUTPUT_FILE = f"policy_extraction_results_{OFFSET}.jsonl"
MODEL_NAME = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
NGPUS = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

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


class PolicyExtractionResponse(BaseModel):
    contains_policies: bool = Field(
        ..., description="Whether the text mentions at least one policy."
    )
    policies: list[str] = Field(
        ...,
        description="A list of policies mentioned in the text. If contains_policies is False, this list should be empty.",
    )


sampling_params = SamplingParams(
    temperature=0.01,
    max_tokens=1024,
    structured_outputs=StructuredOutputsParams(
        json=PolicyExtractionResponse.model_json_schema(),
    ),
)

with open("POLICIES_EXTRACTION_PROMPT.txt", "r") as f:
    prompt = f.read()


def extract_policies(texts: list[str]) -> list[PolicyExtractionResponse]:
    inputs = [
        [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ]
        for text in texts
    ]
    outputs = llm.chat(inputs, sampling_params)
    results = []
    for output in outputs:
        try:
            result = PolicyExtractionResponse.model_validate_json(output.outputs[0].text)
        except Exception as e:
            print(f"Error parsing output: {e}")
            result = PolicyExtractionResponse(contains_policies=False, policies=["error"])
        results.append(result)
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
    texts = batch_df.text.values.tolist()
    preds = extract_policies(texts)

    with open(OUTPUT_FILE, "a") as f_out:
        for i, pred in enumerate(preds):
            row = batch_df.iloc[i]
            d = pred.model_dump() | {
                "openalex_id": str(row.openalex_id),
                "chunk_idx": str(row.chunk_idx),
            }
            f_out.write(json.dumps(d) + "\n")
