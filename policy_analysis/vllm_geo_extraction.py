import json
import os
from typing import Literal

import pyarrow.parquet as pq
from pydantic import BaseModel
from tqdm.auto import tqdm
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

from dspy_policies_and_taxonomy_extraction.taxonomy_definition.geographical_taxonomy import (
    Geographical_scope,
    Studied_country,
    Regional_group,
)


OFFSET = int(os.environ["SLURM_ARRAY_TASK_ID"])
NJOBS = 10
BATCH_SIZE = 10_000
print(f"Running with OFFSET={OFFSET}, NJOBS={NJOBS}, BATCH_SIZE={BATCH_SIZE}")

INPUT_FILE = "chunked_conclusions_557k_2026-01-26.parquet"
OUTPUT_FILE = f"geo_extraction_results_{OFFSET}.jsonl"
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


# prompt from Edouard's DSPY optimization
system_prompt = """
Read the provided paragraph from the conclusion section of a scientific study.
Step 2: Identify all explicit location mentions (e.g., countries, cities, regions) in the paragraph.
Step 3: For each location mention, map them to specific countries if possible.
Step 4: Map the identified countries to broader regional groups and geographical scopes.
Step 5: Return the following output fields:
   - **Regional Group**: A single value indicating the regional group mentioned (e.g., "Asia-Pacific States"). Return empty if no regional group is explicitly mentioned.
   - **Geographical Scopes**: A list of values indicating the geographical scopes mentioned (e.g., "Countries", "Local or subnational"). Return an empty list if none are explicitly mentioned.
   - **Main Country Focus**: A list of values indicating the primary countries mentioned in the text (e.g., "Singapore", "Italy"). Return an empty list if no countries are explicitly mentioned.
Step 6: Ensure that the outputs are standardized and consistent with predefined values for regional groups, geographical scopes, and countries.
Step 7: Do not output any intermediate steps or reasoning; only provide the final outputs in the specified format.
"""


class GeoExtractionResponse(BaseModel):
    regional_group: Literal[*[e.value for e in Regional_group]]
    geographical_scopes: list[Literal[*[e.value for e in Geographical_scope]]]
    main_country_focus: list[Literal[*[e.value for e in Studied_country]]]


sampling_params = SamplingParams(
    temperature=0.01,
    max_tokens=512,
    structured_outputs=StructuredOutputsParams(
        json=GeoExtractionResponse.model_json_schema(),
    ),
)


def extract_geographies(texts: list[str]) -> list[GeoExtractionResponse]:
    inputs = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ]
        for text in texts
    ]
    outputs = llm.chat(inputs, sampling_params)
    results = []
    for output in outputs:
        try:
            result = GeoExtractionResponse.model_validate_json(output.outputs[0].text)
        except Exception as e:
            print(f"Error parsing output: {e}")
            result = e
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
    preds = extract_geographies(texts)

    with open(OUTPUT_FILE, "a") as f_out:
        for i, pred in enumerate(preds):
            row = batch_df.iloc[i]
            if isinstance(pred, Exception):
                d = {"error": str(pred)}
            else:
                d = pred.model_dump()

            d |= {
                "openalex_id": str(row.openalex_id),
                "chunk_idx": str(row.chunk_idx),
            }
            f_out.write(json.dumps(d) + "\n")
