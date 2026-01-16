import os
import re
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from dotenv import load_dotenv

# ============================================================
# ⚙️ CONFIGURATION
# ============================================================

TEST_RUN = True  # Set to False to process the full 585k rows
INPUT_PATH = "C:/Users/calle/Downloads/chunked_results_conclusions_585k_cs1024_ov100_qw3-06B.parquet"
OUTPUT_PATH = "cleaned_results_scaleway.parquet"
load_dotenv()
# Scaleway Configuration
# Must be set as SCALEWAY_API_KEY in your environment
OPENAI_API_KEY = os.getenv("SCALEWAY_API_KEY")
SCALEWAY_API_URL = "https://api.scaleway.ai/a2dc0d31-c47f-47f1-b0b9-9877dd4eb2b5/v1"

# Model Choice
MODEL_ID = "pixtral-12b-2409"

# ============================================================
# 1️⃣ REGEX PRE-PROCESSING
# ============================================================

RE_PAGE_NUM = re.compile(r'^\s*\d+\s*$')
RE_URL_DOI = re.compile(r'^(https?:\/\/|doi:)', re.IGNORECASE)
RE_FIGURE_TABLE = re.compile(r'\b(?:Figure|Fig\.?|Table)\s+[A-Z]?\d+(?:-\d+)?(?:\s*\([a-z]\))?\b', re.IGNORECASE)
RE_SECTION_HEADER = re.compile(r'^\s*(#+\s*)?(RESULTS?|DISCUSSION|CONCLUSIONS?|SUMMARY|METHODS?|REFERENCES?)\s*$', re.IGNORECASE)
RE_CITATION = re.compile(r'\[\s*\d+(?:\s*,\s*\d+)*\s*\]')


def regex_pre_clean(text):
    """Fast CPU cleaning to remove obvious garbage."""
    if not isinstance(text, str) or not text.strip():
        return ""

    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if RE_PAGE_NUM.match(stripped) or RE_SECTION_HEADER.match(stripped):
            continue
        if RE_FIGURE_TABLE.match(stripped):
            continue

        line = RE_CITATION.sub("", line)
        if len(stripped) > 5:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


# ============================================================
# 2️⃣ SYNC API CLIENT
# ============================================================

client = OpenAI(base_url=SCALEWAY_API_URL, api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """
You are a text analysis and cleaning agent.

Objective
Analyze the input text and output a cleaned version optimized for downstream model analysis, while preserving the original information verbatim.

Hard constraints (must follow exactly):

Do NOT rewrite, paraphrase, translate, correct, or add content.

Any retained text must be word-for-word identical to the input.

Only delete content that does not meet the relevance criteria below.

Relevance criteria (keep ONLY content that fits at least one):

Policies: rules, laws, regulations, guidelines, standards, government actions, policy recommendations.

Taxonomy: definitions, classifications, categories, hierarchies, conceptual groupings.

Cleaning operations to apply:

Remove tables and tabular structures.

Remove formatting artifacts and non-essential punctuation (keep punctuation necessary for meaning).

Remove incomplete sentences at the beginning of the text.

Remove truncated or unfinished sentences at the end of the text.

Remove references to external materials that do not contain substantive policy or taxonomy information (citations, bibliographies, cross-references, notes, links, mentions of other documents).

Remove any content that does not contribute to identifying or understanding policies or taxonomy.

Output rules:

Output only the cleaned text.

No introductions, explanations, headings, markdown, or comments.

If no content remains after cleaning that matches the relevance criteria, output exactly:
NO_DATA
"""


def clean_text_with_api(text):
    """Send a single text chunk to Scaleway API synchronously."""
    text = str(text)  # ensure it's a string
    if not text.strip():  # check if empty
        return ""

    try:
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Text to clean:\n{text}"}
            ],
            temperature=0.0,
            max_tokens=512
        )
        result = response.choices[0].message.content.strip()
        return "" if result.strip() == "NO_DATA" else result
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        return text  # Return original on fail


# ============================================================
# 3️⃣ PROCESS DATAFRAME BATCH
# ============================================================

def process_batch(df_chunk):
    """Processes a dataframe batch: Regex first, then API calls."""
    target_col = "input_text" if "input_text" in df_chunk.columns else "text"

    # 1. CPU Pre-cleaning
    df_chunk["pre_cleaned"] = df_chunk[target_col].apply(regex_pre_clean)
    df_chunk["pre_cleaned"] = df_chunk["pre_cleaned"].astype(str)  # ensure string

    # 2. Filter rows with content
    mask_has_content = df_chunk["pre_cleaned"].str.len() > 15
    indices_to_process = df_chunk.index[mask_has_content]

    print(f"  -> Sending {mask_has_content.sum()} valid rows to Scaleway API...")

    # 3. Synchronous API calls
    for idx in tqdm(indices_to_process, desc="API Calls"):
        text = df_chunk.at[idx, "pre_cleaned"]
        df_chunk.at[idx, target_col] = clean_text_with_api(text)

    df_chunk[target_col].fillna("", inplace=True)
    return df_chunk.drop(columns=["pre_cleaned"])


# ============================================================
# 4️⃣ MAIN EXECUTION
# ============================================================

def main():
    if not OPENAI_API_KEY:
        raise ValueError(
            "❌ Please set your Scaleway API Key in the environment as SCALEWAY_API_KEY!"
        )

    print(f"🚀 Starting Scaleway API Pipeline (Model: {MODEL_ID})")

    parquet_file = pq.ParquetFile(INPUT_PATH)
    schema = parquet_file.schema.to_arrow_schema()

    with pq.ParquetWriter(OUTPUT_PATH, schema) as writer:
        for i, batch in enumerate(parquet_file.iter_batches(batch_size=200)):
            print(f"\n📦 Batch {i+1}...")

            df_chunk = batch.to_pandas()
            df_processed = process_batch(df_chunk)

            table = pa.Table.from_pandas(df_processed, schema=schema)
            writer.write_table(table)

            if TEST_RUN:
                print("🛑 Test Run enabled: Stopping after first batch.")
                break

    print(f"✅ Done! Output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
