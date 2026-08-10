#!/bin/bash
#SBATCH --job-name=policy-extract
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=utx@a100
#SBATCH --array=0-9

# Policy extraction as a pool of queue workers — successor to
# policy_extraction/jz_policy_extraction.sh with idempotent resubmit
# (no duplicate rows on restart) and monitoring events.
#
# Prerequisites (login node, in pi_recluster/):
#   python chunk_corpus.py --corpus "$ROOT/corpus/*.parquet" \
#       --out "$ROOT/chunked_corpus.parquet" \
#       --tokenizer "$DSDIR/HuggingFace_Models/Qwen/Qwen3-Embedding-0.6B"
#   python build_extract_input.py --chunks "$ROOT/chunked_corpus.parquet" \
#       --out "$ROOT/extract/items.jsonl"
#   python build_queue.py --items "$ROOT/extract/items.jsonl" \
#       --queue-dir "$ROOT/extract/queue" \
#       --exclude-done-from "$ROOT/extract/outputs" --rebuild
#
# Afterwards (login node): python merge_extraction.py \
#       --outputs "$ROOT/extract/outputs" --out "$ROOT/policies.parquet"

module purge
conda deactivate
module load arch/a100
module load miniforge/25.9.1
conda activate vllmenv
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Compute nodes are offline — fail fast on any accidental hub call.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

ROOT=${ROOT:-$WORK/pi_recluster}
MODEL_DIR=${MODEL_DIR:-$DSDIR/HuggingFace_Models/mistralai/Mistral-Small-3.2-24B-Instruct-2506}
PROMPT=${PROMPT:-../../policy_extraction/POLICIES_EXTRACTION_PROMPT.txt}
export PIPELINE_EVENTS_DIR=${PIPELINE_EVENTS_DIR:-$ROOT/events}

set -x
cd "$WORK/13_democratiser_sobriete/policy_analysis/expert_eval/pi_recluster"

python fsqueue.py requeue-stale --root "$ROOT/extract/queue" --ttl "${STALE_TTL:-7200}"

srun python extract_vllm.py \
    --queue-dir "$ROOT/extract/queue" \
    --out-dir "$ROOT/extract/outputs" \
    --prompt "$PROMPT" \
    --model "$MODEL_DIR" \
    ${EXTRACT_EXTRA_ARGS:-}
