#!/bin/bash
#SBATCH --job-name=suff-prescreen
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

# Gemma-based abstract prescreening — replaces the SetFit stage-2 classifier
# (library/prescreening/stage2/predict_sufficiency.py). Same queue-worker
# mechanics as jz_classify.slurm; the prompt (PROMPTS_VERSION=prescreen)
# applies the expert sufficiency definition to title+abstract, any language.
#
# Prerequisites (login node):
#   stage-1 parquets landed in $ROOT/prescreen/stage1/  (scaleway_update.sh)
#   python build_prescreen_input.py --parquet "$ROOT/prescreen/stage1/chunk_*.parquet" \
#       --out "$ROOT/prescreen/items.jsonl"
#   python build_queue.py --items "$ROOT/prescreen/items.jsonl" \
#       --queue-dir "$ROOT/prescreen/queue" \
#       --exclude-done-from "$ROOT/prescreen/outputs" --rebuild
#
# Keep-list afterwards (login node): filter outputs on category/confidence,
# send the id list back to the scraping box (scaleway_update.sh pull-keep-list).

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
MODEL_DIR=${MODEL_DIR:-$WORK/models/gemma-4-12B-it}
export PROMPTS_VERSION=prescreen
export PIPELINE_EVENTS_DIR=${PIPELINE_EVENTS_DIR:-$ROOT/events}

set -x
cd "$WORK/13_democratiser_sobriete/policy_analysis/expert_eval/pi_recluster"

python fsqueue.py requeue-stale --root "$ROOT/prescreen/queue" --ttl "${STALE_TTL:-7200}"

srun python classify_vllm.py \
    --queue-dir "$ROOT/prescreen/queue" \
    --out-dir "$ROOT/prescreen/outputs" \
    --model "$MODEL_DIR" \
    ${CLASSIFY_EXTRA_ARGS:-}
