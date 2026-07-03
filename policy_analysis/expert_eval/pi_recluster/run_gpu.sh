#!/usr/bin/env bash
# One-command wrapper for the full re-clustering iteration on a GPU host.
#
# Assumes the repo is at $REPO_ROOT (default /home/ubuntu/13_democratiser_sobriete).
# All paths are anchored to that root — no `../../..` counting.
#
# Modes (env vars, mutually exclusive):
#   USE_VLLM=1      Option C (default on a GPU box): full 1.47M via local vLLM.
#                   Set VLLM_MODEL=<hf-repo-id> to override google/gemma-4-12B-it.
#   FULL_CLASSIFY=1 Option B: full 1.47M via DeepSeek V4 API (needs DEEPSEEK_API_KEY).
#   (neither)       Option A: re-cluster an existing classifications.jsonl
#                   sitting at $CLASSIFICATIONS (default data/classifications.jsonl).
#
# Default on a GPU host is Option C. Trip the flag when you actually want it.
set -euo pipefail

cd "$(dirname "$0")"

REPO_ROOT="${REPO_ROOT:-/home/ubuntu/13_democratiser_sobriete}"
DATA_ROOT="${DATA_ROOT:-data}"

CLASSIFICATIONS="${CLASSIFICATIONS:-$DATA_ROOT/classifications.jsonl}"
CLUSTERS_DIR="${CLUSTERS_DIR:-$DATA_ROOT/hf/clusters_2026-03-18}"
EMBEDDINGS_PATH="${EMBEDDINGS_PATH:-$DATA_ROOT/hf/embeddings_policies_Qwen3-4B_2026-03-05.parquet}"
FILTERED="${FILTERED:-$DATA_ROOT/filtered.parquet}"
OUT_CLUSTERS="${OUT_CLUSTERS:-$DATA_ROOT/reclustered}"
FEW_SHOT="${FEW_SHOT:-$REPO_ROOT/runs/expert_gold/few_shot_v1.jsonl}"
VLLM_MODEL="${VLLM_MODEL:-google/gemma-4-12B-it}"

mkdir -p "$DATA_ROOT/carbon"

echo "=== step 1: download data (clusters + embeddings) ==="
uv run python download_data.py --dest "$DATA_ROOT/hf"

if [[ "${USE_VLLM:-0}" == "1" || "${FULL_CLASSIFY:-0}" == "1" ]]; then
    echo "=== step 2a: build full 1.47M to_classify.jsonl ==="
    uv run python build_classify_input.py \
        --clusters "$CLUSTERS_DIR" \
        --out "$DATA_ROOT/to_classify_full.jsonl"

    if [[ "${USE_VLLM:-0}" == "1" ]]; then
        echo "=== step 2b: classify via local vLLM ($VLLM_MODEL) ==="
        uv run --extra gpu python classify_vllm.py \
            --items "$DATA_ROOT/to_classify_full.jsonl" \
            --out   "$DATA_ROOT/classifications_full.jsonl" \
            --few-shot "$FEW_SHOT" \
            --model "$VLLM_MODEL"
    else
        echo "=== step 2b: classify via DeepSeek V4 API ==="
        if [[ -z "${DEEPSEEK_API_KEY:-}" ]]; then
            echo "  DEEPSEEK_API_KEY not set — aborting"; exit 1
        fi
        HERE="$(pwd)"
        (cd "$REPO_ROOT" && uv run python -m policy_analysis.expert_eval.classify_v1 \
            --items "$HERE/$DATA_ROOT/to_classify_full.jsonl" \
            --out   "$HERE/$DATA_ROOT/classifications_full.jsonl" \
            --few-shot "$FEW_SHOT")
    fi
    CLASSIFICATIONS="$DATA_ROOT/classifications_full.jsonl"
fi

echo "=== step 3: filter + join to embeddings ==="
uv run python filter_and_stratify.py \
    --classifications "$CLASSIFICATIONS" \
    --clusters "$CLUSTERS_DIR" \
    --embeddings "$EMBEDDINGS_PATH" \
    --out "$FILTERED"

echo "=== step 4: stratified re-clustering (Leiden per sector, sub_code) ==="
uv run python recluster.py \
    --filtered "$FILTERED" \
    --out-dir "$OUT_CLUSTERS"

echo "=== done ==="
echo "outputs in $OUT_CLUSTERS/"
echo "CO2 log in $DATA_ROOT/carbon/emissions.csv"
if [[ -f "$DATA_ROOT/carbon/emissions.csv" ]]; then
    echo
    echo "--- CO2 summary (kg CO2eq per stage) ---"
    uv run python -c "
import pandas as pd
d = pd.read_csv('$DATA_ROOT/carbon/emissions.csv')
if 'project_name' in d.columns:
    s = d.groupby('project_name')[['duration','energy_consumed','emissions']].sum()
    print(s.round(4).to_string())
    print(f'\nTOTAL emissions: {d[\"emissions\"].sum():.4f} kg CO2eq')
    print(f'TOTAL energy:    {d[\"energy_consumed\"].sum():.4f} kWh')
"
fi
