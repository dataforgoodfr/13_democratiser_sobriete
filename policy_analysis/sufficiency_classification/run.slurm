#!/bin/bash
#SBATCH --job-name=suff-classify
#SBATCH --output=logs/suff-classify-%A_%a.out
#SBATCH --error=logs/suff-classify-%A_%a.err
#SBATCH --array=0-9                  # 10 shards; tune to match --num-shards
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1                 # single A100; bump to 4-8 for big MoE
#SBATCH --time=20:00:00              # raise if a shard takes longer; 1.5M/10 = 150k rows/shard
#SBATCH --hint=nomultithread

# --- Jean-Zay specifics: set these for your project --------------------------
# #SBATCH --account=<YOUR_PROJECT>@a100
# #SBATCH --partition=gpu_p5        # A100 80GB partition; check `idrcontrib`
# #SBATCH --qos=qos_gpu-t3          # 20h limit; use qos_gpu-t4 for longer runs

set -euo pipefail
mkdir -p logs

# --- Environment -------------------------------------------------------------
module purge
module load cpuarch/amd                # required on some Jean-Zay A100 nodes
module load arch/a100
module load python/3.12
# If you maintain a venv on $WORK:
#   source $WORK/venvs/sufficiency/bin/activate
# Otherwise, with uv (recommended; share the project lockfile):
export UV_PROJECT_ENVIRONMENT="$WORK/venvs/sufficiency"
cd "$WORK/13_democratiser_sobriete"   # adjust to your checkout

# --- Knobs (override via `sbatch --export=ALL,MODEL=...,QUANTIZATION=fp8 run.slurm`) ---
: "${MODEL:=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"   # fits A100 80GB BF16
: "${QUANTIZATION:=}"                                    # set to fp8 / awq if needed
: "${TP:=1}"                                             # tensor-parallel size
: "${NUM_SHARDS:=10}"
: "${DATA_ROOT:=$WORK/data/clusters_2026-03-18}"
: "${OUT_DIR:=$SCRATCH/sufficiency_runs/v0}"
: "${EXTRA_ARGS:=}"

echo "==[ shard ${SLURM_ARRAY_TASK_ID}/${NUM_SHARDS} | model=${MODEL} tp=${TP} quant=${QUANTIZATION:-none} ]=="
nvidia-smi || true

# vLLM env tweaks: enable prefix caching (it's on by default since 0.5.4 but
# pin in case of older builds), set HF cache to $SCRATCH to avoid quota issues.
export HF_HOME="$SCRATCH/hf_cache"
export VLLM_USE_V1=1                  # opt into the v1 engine if available
export TRANSFORMERS_OFFLINE=0

uv run --no-sync python -m policy_analysis.sufficiency_classification.classify \
    --data "$DATA_ROOT" \
    --out  "$OUT_DIR" \
    --model "$MODEL" \
    ${QUANTIZATION:+--quantization "$QUANTIZATION"} \
    --tp "$TP" \
    --num-shards "$NUM_SHARDS" \
    $EXTRA_ARGS

# SLURM_ARRAY_TASK_ID is picked up automatically inside the python entrypoint.
