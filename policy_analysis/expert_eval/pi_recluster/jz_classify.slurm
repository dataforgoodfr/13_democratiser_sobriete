#!/bin/bash
#SBATCH --job-name=suff-classify
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=utx@a100
#SBATCH --array=0-9

# Full-corpus classification as a pool of queue workers. Each array task
# loads the model once, then pulls 8k-row work units from the filesystem
# queue until it is drained — so the job is:
#   - parallel: any array width works; workers never overlap on a unit
#     (claims are atomic renames on the shared FS, see fsqueue.py);
#   - idempotent: resubmitting after a crash or walltime kill first returns
#     stale claims to the queue, and unit outputs are written tmp-then-rename,
#     so nothing is double-counted and nothing is lost but the in-flight unit.
#
# Prerequisites: bash jz_prefetch.sh on a login node (data, model weights,
# queue), and jz_gold_gate.slurm SHIPped:
#     gold=$(sbatch --parsable jz_gold_gate.slurm)
#     sbatch --dependency=afterok:$gold jz_classify.slurm
#
# Monitoring: key events land in $PIPELINE_EVENTS_DIR as JSONL; ship them to
# Logfire from a login node with logfire_relay.py (compute nodes are offline).

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

# Return units abandoned by killed workers (safe to run from every task:
# rename races resolve to exactly one winner).
python fsqueue.py requeue-stale --root "$ROOT/queue" --ttl "${STALE_TTL:-7200}"

srun python classify_vllm.py \
    --queue-dir "$ROOT/queue" \
    --out-dir "$ROOT/outputs" \
    --few-shot gold/few_shot_v1.jsonl \
    --model "$MODEL_DIR" \
    ${CLASSIFY_EXTRA_ARGS:-}
