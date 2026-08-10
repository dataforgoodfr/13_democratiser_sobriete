#!/bin/bash
#SBATCH --job-name=suff-gold-gate
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=utx@a100

# Gold preflight: classify the 770 expert-gold rows, then validate_gold.py
# decides SHIP / BORDERLINE / DO_NOT_SCALE via its exit code. Submit the full
# run gated on it:
#     gold=$(sbatch --parsable jz_gold_gate.slurm)
#     sbatch --dependency=afterok:$gold jz_classify.slurm
#
# Prerequisite: bash jz_prefetch.sh on a login node (data + model weights).
# For pass-2: export PROMPTS_VERSION=pass2 before sbatch (it is forwarded).

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
export PROMPTS_VERSION=${PROMPTS_VERSION:-v1}
export PIPELINE_EVENTS_DIR=${PIPELINE_EVENTS_DIR:-$ROOT/events}

set -x
cd "$WORK/13_democratiser_sobriete/policy_analysis/expert_eval/pi_recluster"

# Idempotent: reruns resume from the append-mode output.
srun python classify_vllm.py \
    --items gold/to_classify.jsonl \
    --out "$ROOT/gold_predictions_${PROMPTS_VERSION}.jsonl" \
    --few-shot gold/few_shot_v1.jsonl \
    --exclude-ids gold/few_shot_v1_ids.txt \
    --model "$MODEL_DIR" \
    ${CLASSIFY_EXTRA_ARGS:-}

python validate_gold.py \
    --classifications "$ROOT/gold_predictions_${PROMPTS_VERSION}.jsonl" \
    --gold gold/gold.parquet
