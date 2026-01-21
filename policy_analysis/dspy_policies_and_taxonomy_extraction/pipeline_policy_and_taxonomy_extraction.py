import dspy
import os
import pickle
import pyarrow.parquet as pq
from dotenv import load_dotenv
from tqdm import tqdm
import json
import contextlib
import sys

load_dotenv()

# --------------------------------------------------
# Helper to suppress ALL output (stdout and stderr)
# --------------------------------------------------
@contextlib.contextmanager
def suppress_output():
    """
    Context manager to redirect both stdout and stderr to devnull.
    This effectively silences dspy's internal print statements and progress bars.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

# --------------------------------------------------
# DSPy setup
# --------------------------------------------------
"""lm = dspy.LM(
    model="mistral/mistral-small-3.2-24b-instruct-2506:fp8",
    api_key=os.getenv("SCALEWAY_API_KEY"),
    api_base="https://37d2b07c-4a0b-4b15-aa11-0fb2fb89c078.ifr.fr-par.scaleway.com"
)"""

lm = dspy.LM(
        model="mistral/mistral-small-3.2-24b-instruct-2506:fp8",
        api_key=os.getenv('SCALEWAY_API_KEY'),
        api_base="https://api.scaleway.ai/a2dc0d31-c47f-47f1-b0b9-9877dd4eb2b5"
        )

dspy.configure(lm=lm)

# Load programs
with open("saved_dspy_model/best_policy_extraction_model/program.pkl", "rb") as f:
    policy_program = pickle.load(f)

with open("saved_dspy_model/best_geo_impact_extraction_model/program.pkl", "rb") as f:
    geo_program = pickle.load(f)

# --------------------------------------------------
# Parameters
# --------------------------------------------------
BATCH_SIZE = 128
MAX_ROWS = 100 
OUTPUT_PATH = "policy_extraction_outputs.jsonl"
WRITE_BUFFER_SIZE = 200

parquet_path = "C:/Users/calle/Downloads/chunked_results_conclusions_585k_cs1024_ov100_qw3-06B.parquet"
parquet_file = pq.ParquetFile(parquet_path)

rows_seen = 0
# Initialize main progress bar
pbar = tqdm(total=MAX_ROWS, desc="Processing rows", unit="row")

# --------------------------------------------------
# Main loop
# --------------------------------------------------
buffered_records = []

with open(OUTPUT_PATH, "a", encoding="utf-8") as out_f:

    for rg_idx in range(parquet_file.num_row_groups):
        table = parquet_file.read_row_group(rg_idx, columns=["openalex_id","text", "chunk_idx"])
        df = table.to_pandas()
        df.reset_index(inplace=True)

        batch_texts = []
        batch_meta = []

        for row in df.itertuples(index=False):
            if rows_seen >= MAX_ROWS:
                break
            
            openalex_id = row.openalex_id
            text = row.text
            chunk_idx = row.chunk_idx

            batch_texts.append(text)
            batch_meta.append((openalex_id, chunk_idx))

            # Run models in batch when size is reached
            if len(batch_texts) == BATCH_SIZE:
                policy_examples = [dspy.Example(question=t).with_inputs("question") for t in batch_texts]
                geo_examples = [dspy.Example(question=t).with_inputs("question") for t in batch_texts]

                # --- CHANGE HERE: SUPPRESS BOTH STDOUT AND STDERR ---
                with suppress_output():
                    policy_outputs = policy_program.batch(policy_examples)
                    geo_outputs = geo_program.batch(geo_examples)
                # ----------------------------------------------------

                for (text, (openalex_id, chunk_idx), p_out, g_out) in zip(
                    batch_texts, batch_meta, policy_outputs, geo_outputs, strict=False
                ):
                    policy_dict = p_out.toDict() if hasattr(p_out, "toDict") else p_out
                    geo_dict = g_out.toDict() if hasattr(g_out, "toDict") else g_out

                    policy_list = []
                    if isinstance(policy_dict.get("response"), str):
                        policy_list = [p.strip() for p in policy_dict["response"].split(";") if p.strip()]

                    record = {
                        "openalex_id": openalex_id,
                        "chunk_id": chunk_idx,
                        "input_text": text,
                        "policy_list": policy_list,
                        "regional_group": geo_dict.get("regional_group"),
                        "geographical_scopes": geo_dict.get("geographical_scopes", []),
                        "main_country_focus": geo_dict.get("main_country_focus", []),
                        "human_needs": geo_dict.get("human_needs", []),
                        "natural_resources": geo_dict.get("natural_ressource", []),
                        "wellbeing": geo_dict.get("wellbeing", []),
                        "justice": geo_dict.get("justice_consideration", []),
                        "planetary_boundaries": geo_dict.get("planetary_boundaries", []),
                    }

                    buffered_records.append(record)
                    rows_seen += 1
                    pbar.update(1)

                    if len(buffered_records) >= WRITE_BUFFER_SIZE:
                        out_f.write("\n".join(json.dumps(r) for r in buffered_records) + "\n")
                        buffered_records.clear()

                batch_texts.clear()
                batch_meta.clear()

        if rows_seen >= MAX_ROWS:
            break

    if buffered_records:
        out_f.write("\n".join(json.dumps(r) for r in buffered_records) + "\n")
        buffered_records.clear()

pbar.close()
print(f"Finished {rows_seen} rows")