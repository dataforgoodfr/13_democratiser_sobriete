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
# Few-shot lives bundled in gold/ so it's present on a fresh clone; fall back to
# the parent-repo copy if someone points REPO_ROOT at a full checkout.
if [[ -f "gold/few_shot_v1.jsonl" ]]; then
    FEW_SHOT="${FEW_SHOT:-gold/few_shot_v1.jsonl}"
else
    FEW_SHOT="${FEW_SHOT:-$REPO_ROOT/runs/expert_gold/few_shot_v1.jsonl}"
fi
VLLM_MODEL="${VLLM_MODEL:-google/gemma-4-12B-it}"
# Prompt version fed to classify_vllm.py. v1 shipped: it validated best on Gemma
# (the six-pillar v2/v3 variants hurt the smaller model — see RESEARCH_NOTES
# iteration 5). Override with PROMPTS_VERSION=v2 etc. to re-test a variant.
export PROMPTS_VERSION="${PROMPTS_VERSION:-v1}"

mkdir -p "$DATA_ROOT/carbon"

# Preflight: if we're going to use vLLM, make sure nvcc is reachable. vLLM's
# Triton JIT needs it at engine start-up; a missing nvcc surfaces very late
# (after model weights load) with a `Could not find nvcc` error.
if [[ "${USE_VLLM:-0}" == "1" ]]; then
    if [[ -z "${CUDA_HOME:-}" ]]; then
        # Auto-detect: prefer versioned /usr/local/cuda-XX.Y (newest first),
        # then /usr/local/cuda, then /opt/cuda, then /usr (apt install path).
        for candidate in $(ls -d /usr/local/cuda-* 2>/dev/null | sort -Vr) /usr/local/cuda /opt/cuda /usr; do
            if [[ -x "$candidate/bin/nvcc" ]]; then
                export CUDA_HOME="$candidate"
                break
            fi
        done
    fi
    if [[ -n "${CUDA_HOME:-}" ]]; then
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

        # Persist to ~/.bashrc on first detection so future shells (post-SSH
        # reconnect, box reboot, tmux respawn) pick up nvcc automatically.
        _bashrc="${HOME}/.bashrc"
        _marker="# pi_recluster: CUDA toolkit auto-configured"
        if [[ -w "$_bashrc" ]] && ! grep -qF "$_marker" "$_bashrc" 2>/dev/null; then
            {
                echo ""
                echo "$_marker"
                echo "export CUDA_HOME=\"$CUDA_HOME\""
                echo 'export PATH="$CUDA_HOME/bin:$PATH"'
                echo 'export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"'
            } >> "$_bashrc"
            echo "  → persisted CUDA env to $_bashrc (future shells will pick it up)"
        fi
    fi
    if ! command -v nvcc >/dev/null 2>&1; then
        cat <<EOF >&2
ERROR: nvcc not found. vLLM needs the CUDA toolkit at runtime.

Install the version matching your torch build (check with:
    uv run python -c 'import torch; print(torch.version.cuda)'
).

Ubuntu 22.04 example (CUDA 12.4):
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update && sudo apt-get install -y cuda-toolkit-12-4

Then re-run:  USE_VLLM=1 bash run_gpu.sh
EOF
        exit 1
    fi
    # flashinfer, xgrammar, and vLLM's inductor path all invoke nvcc which
    # shells out to `gcc -x c++` as the host C++ compiler. Ubuntu installs
    # can leave gcc's C++ backend broken while g++ works fine (mismatched
    # major versions, missing cc1plus for gcc but present for g++). Test
    # both; if only g++ works, point nvcc at g++ via NVCC_CCBIN.
    _probe=/tmp/_pi_cxx_probe
    _gxx_ok=0
    _gcc_cxx_ok=0
    if echo 'int main(){}' | g++ -x c++ - -o "$_probe" 2>/dev/null; then _gxx_ok=1; fi
    if echo 'int main(){}' | gcc -x c++ - -o "$_probe" 2>/dev/null; then _gcc_cxx_ok=1; fi
    rm -f "$_probe"

    if [[ "$_gxx_ok" == "0" && "$_gcc_cxx_ok" == "0" ]]; then
        cat <<EOF >&2
ERROR: neither g++ nor gcc-x-c++ can compile — cc1plus is unavailable.

Fix:
    sudo apt-get update
    sudo apt-get install -y --reinstall build-essential g++ gcc

Then clear the flashinfer JIT cache and re-run:
    rm -rf ~/.cache/flashinfer
    USE_VLLM=1 bash run_gpu.sh
EOF
        exit 1
    elif [[ "$_gxx_ok" == "1" && "$_gcc_cxx_ok" == "0" ]]; then
        # g++ works but gcc's C++ frontend doesn't — force nvcc to use g++.
        export NVCC_CCBIN="$(command -v g++)"
        echo "  → gcc-x-c++ broken; nvcc pinned to \$(g++)=$NVCC_CCBIN"
        # Persist for future shells
        if [[ -w "${HOME}/.bashrc" ]] && ! grep -qF "# pi_recluster: NVCC_CCBIN" "${HOME}/.bashrc" 2>/dev/null; then
            {
                echo ""
                echo "# pi_recluster: NVCC_CCBIN auto-set (gcc-x-c++ was broken)"
                echo "export NVCC_CCBIN=\"$NVCC_CCBIN\""
            } >> "${HOME}/.bashrc"
            echo "  → persisted NVCC_CCBIN to ~/.bashrc"
        fi
        # Bust the cached failed build so it retries with g++
        if [[ -d "$HOME/.cache/flashinfer" ]]; then
            rm -rf "$HOME/.cache/flashinfer"
            echo "  → cleared ~/.cache/flashinfer (was built against broken gcc)"
        fi
    fi
    echo "  cuda toolkit: $(nvcc --version | tail -1)"
    echo "  host cxx    : $(g++ --version | head -1)"
    echo "  CUDA_HOME   : $CUDA_HOME"
fi

# Optional preflight: validate the prompt on the 760-row expert-gold set through
# the ACTUAL model (Gemma) before committing to the 1.47M run and the 7 GB
# download. The gold validation earlier was on DeepSeek; this confirms the prompt
# transfers to Gemma. Runs fast (only loads the model + 760 rows), prints a
# SHIP/BORDERLINE/DO_NOT_SCALE verdict, then STOPS so you can review.
#   PREFLIGHT_GOLD=1 USE_VLLM=1 bash run_gpu.sh
if [[ "${PREFLIGHT_GOLD:-0}" == "1" ]]; then
    if [[ "${USE_VLLM:-0}" != "1" ]]; then
        echo "PREFLIGHT_GOLD needs USE_VLLM=1 (it validates the local Gemma classifier)." >&2
        exit 1
    fi
    echo "=== PREFLIGHT: validate prompt '$PROMPTS_VERSION' on expert gold via $VLLM_MODEL ==="
    rm -f "$DATA_ROOT/gold_predictions.jsonl"
    uv run --extra gpu python classify_vllm.py \
        --items "gold/to_classify.jsonl" \
        --out   "$DATA_ROOT/gold_predictions.jsonl" \
        --few-shot "$FEW_SHOT" \
        --exclude-ids "gold/few_shot_v1_ids.txt" \
        --model "$VLLM_MODEL"
    echo
    set +e
    uv run python validate_gold.py \
        --classifications "$DATA_ROOT/gold_predictions.jsonl" \
        --gold "gold/gold.parquet"
    _verdict=$?
    set -e
    echo
    echo "Preflight complete (predictions in $DATA_ROOT/gold_predictions.jsonl)."
    echo "If the verdict is SHIP, launch the full run with:"
    echo "    USE_VLLM=1 bash run_gpu.sh"
    exit $_verdict
fi

echo "=== step 1: download data (clusters + embeddings) ==="
uv run python download_data.py --dest "$DATA_ROOT/hf"

if [[ "${USE_VLLM:-0}" == "1" || "${FULL_CLASSIFY:-0}" == "1" ]]; then
    echo "=== step 2a: build full 1.47M to_classify.jsonl ==="
    uv run python build_classify_input.py \
        --clusters "$CLUSTERS_DIR" \
        --out "$DATA_ROOT/to_classify_full.jsonl"

    if [[ "${USE_VLLM:-0}" == "1" ]]; then
        echo "=== step 2b: classify via local vLLM ($VLLM_MODEL, prompt $PROMPTS_VERSION) ==="
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
