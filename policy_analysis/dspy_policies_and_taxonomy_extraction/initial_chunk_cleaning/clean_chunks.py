import pyarrow.parquet as pq
import re
import json

# ============================================================
# CONFIGURATION
# ============================================================

TEST_RUN = True  
INPUT_PATH = "C:/Users/calle/Downloads/chunked_results_conclusions_585k_cs1024_ov100_qw3-06B.parquet"
OUTPUT_JSON_PATH = "cleaned_results_preview.jsonl"
SAMPLE_SIZE = 100 

# ============================================================
# REGEX DEFINITIONS
# ============================================================

REGEX_RULES = {
    # --- GLOBAL RULES (Applied to whole text) ---
    
    # 1. PUA / Garbage Symbols: Removes  etc.
    "remove_pua_symbols": re.compile(r'[\uE000-\uF8FF]+'),

    # 2. Trim Garbage Start: Stops at the first "dot + space + Capital"
    "trim_garbage_start": re.compile(r'^(?![A-Z]).*?[.,][\s]*(?=[A-Z])', re.DOTALL),
    
    # 3. Results/Discussion Header Cleaning
    "results_discussions": re.compile(r'(?:^|(?<=[.!?]))[\s#]*(?:RESULTS|DISCUSSION)\b[:\s]*', re.IGNORECASE),

    # 4. SUPER TABLE SOUP
    "regex_table_soup": re.compile(
        r'(?:\b(?:[A-Z0-9]{1,5}|[A-Z][a-z]{0,3}\.|omponen|Loading|value|Komponen|N/A|Ave|Max|Min|Tot)(?:\s+|$)){3,}', 
        re.MULTILINE
    ),

    # 5. CITATION REMOVER (NEW)
    #    Handles: (Smith et al. 2019), (2015-2016), (e.g. Doe, 2020; Lee 2021)
    "remove_citations": re.compile(
        r'\((?:e\.g\.,?\s*)?(?:[A-Za-z\s.&-]+(?:et\s+al\.?)?,?\s*\d{4}[a-z]?(?:[-–]\d{2,4})?|(?:\d{4}[a-z]?(?:[-–]\d{2,4})?)(?:,\s*\d{4}[a-z]?(?:[-–]\d{2,4})?)*)(?:[;,]\s*(?:[A-Za-z\s.&-]+(?:et\s+al\.?)?,?\s*\d{4}[a-z]?(?:[-–]\d{2,4})?|(?:\d{4}[a-z]?(?:[-–]\d{2,4})?)(?:,\s*\d{4}[a-z]?(?:[-–]\d{2,4})?)*))*\)',
        re.IGNORECASE
    ),

    # --- LINE-BY-LINE RULES ---
    "url_or_doi": re.compile(r'(?:https?://\s*\S+(?:\s+\S+)?)|(?:\bdoi:\s*10\.\S+(?:\s+\S+)?)', re.IGNORECASE),
    "chapter_titles": re.compile(r'^\s*\d+(?:\.\d+)*\.?\s+.*$', re.MULTILINE),
    "fig_table_ref": re.compile(r'\s*[\(\[]?(?:see\s+)?(?:Figures?|Figs?\.?|Tables?|Tabs?\.?)\s+(?:S-?)?\d+[A-Za-z]*(?:[-–,]\s*(?:S-?)?\d+[A-Za-z]*)*[\)\]]?', re.IGNORECASE),
    "encoding_errors": re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufffd]+'),
    "figure_or_table": re.compile(r'\b(?:Figure|Fig\.?|Table)\s+[A-Z]?\d+(?:-\d+)?(?:\s*\([a-z]\))?\b', re.IGNORECASE),
}

# ============================================================
# CLEANING FUNCTION
# ============================================================

def apply_regex_cleaning(text, regex_rules):
    if not isinstance(text, str):
        return "", 0, []

    deleted_count = 0
    triggered_rules = []

    # --- PHASE 1: Global Rules (Before splitting) ---
    global_keys = ["remove_pua_symbols", "trim_garbage_start", "results_discussions", "regex_table_soup", "remove_citations"]
    
    for key in global_keys:
        if key in regex_rules:
            regex = regex_rules[key]
            if regex.search(text):
                deleted_count += len(regex.findall(text))
                if key not in triggered_rules:
                    triggered_rules.append(key)
                text = regex.sub(" ", text) # Replace with space

    # --- PHASE 2: Line-by-Line Processing ---
    lines = text.split("\n")
    cleaned_lines = []

    for line in lines:
        line_deleted = 0
        
        # 1. Check for full-line deletion
        if regex_rules.get("figure_or_table") and regex_rules["figure_or_table"].match(line.strip()):
            deleted_count += 1
            if "figure_or_table" not in triggered_rules:
                triggered_rules.append("figure_or_table")
            continue 

        # 2. Apply inline regexes
        for name, regex in regex_rules.items():
            if name in global_keys or name == "figure_or_table":
                continue
            
            matches = regex.findall(line)
            if matches:
                line_deleted += len(matches)
                line = regex.sub("", line)
                if name not in triggered_rules:
                    triggered_rules.append(name)

        if line.strip():
            cleaned_lines.append(line.strip())
        
        deleted_count += line_deleted

    # Final cleanup
    final_text = " ".join(cleaned_lines)
    final_text = re.sub(r'\s+', ' ', final_text).strip()
    
    return final_text, deleted_count, triggered_rules

# ============================================================
# MAIN EXECUTION
# ============================================================

parquet_file = pq.ParquetFile(INPUT_PATH)
target_col = "input_text" if "input_text" in parquet_file.schema.names else "text"

all_results = []

for batch in parquet_file.iter_batches(batch_size=10000):
    df = batch.to_pandas()
    if target_col not in df.columns:
        continue

    if TEST_RUN:
        df = df.head(SAMPLE_SIZE)

    for text in df[target_col]:
        cleaned_text, deleted_count, triggered_rules = apply_regex_cleaning(text, REGEX_RULES)
        
        result = {
            "cleaned_text": cleaned_text,
            "deleted_count": deleted_count,
            "triggered_rules": triggered_rules
        }
        all_results.append(result)

    if TEST_RUN:
        break

with open(OUTPUT_JSON_PATH, "w", encoding="utf-8") as f:
    for entry in all_results:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"✅ Nettoyage terminé. Résultat JSONL sauvegardé dans {OUTPUT_JSON_PATH}")