#!/bin/bash
#SBATCH --job-name=policy-embed
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --constraint=a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --hint=nomultithread
#SBATCH --account=utx@a100

# Policy embeddings for clustering — an INTERMEDIATE artefact this run:
# written to $SCRATCH (purged by JZ policy), never uploaded to HF/S3/Qdrant.
# Library-index embeddings are iteration-2 work and are NOT computed here.
#
# No resume: a killed job recomputes from scratch (single-GPU, a few hours
# at full corpus scale — accepted trade-off, the output stays consistent).
#
# Prerequisites: $ROOT/policies.parquet (merge_extraction.py) and the model
# under $DSDIR/HuggingFace_Models (or prefetched — set MODEL to a local path).
# Submit after extraction drains:
#     sbatch jz_embed.slurm

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
MODEL=${MODEL:-Qwen/Qwen3-Embedding-4B}
INPUT=${INPUT:-$ROOT/policies.parquet}
OUTPUT=${OUTPUT:-$SCRATCH/pi_recluster/policy_embeddings.parquet}
export PIPELINE_EVENTS_DIR=${PIPELINE_EVENTS_DIR:-$ROOT/events}

set -x
cd "$WORK/13_democratiser_sobriete/policy_analysis/expert_eval/pi_recluster"
mkdir -p "$(dirname "$OUTPUT")"

python -c "from _events import emit; emit('stage_start', stage='policy_embed',
input='$INPUT', model='$MODEL')"

srun python ../../compute_embeddings.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --text-column policy_text \
    --model "$MODEL"
rc=$?

python -c "from _events import emit; emit('stage_end', stage='policy_embed',
status='$([ $rc -eq 0 ] && echo ok || echo failed)', output='$OUTPUT')"
exit $rc
