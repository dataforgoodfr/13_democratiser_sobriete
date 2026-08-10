#!/bin/bash
# Login-node prefetch for the Jean Zay classification run.
#
# Compute nodes have no internet: everything the GPU jobs touch must already
# be on the shared filesystem. Run this on a LOGIN node, inside the project
# env (e.g. `conda activate vllmenv`), from this directory:
#
#     bash jz_prefetch.sh
#
# Then submit:
#     gold=$(sbatch --parsable jz_gold_gate.slurm)
#     sbatch --dependency=afterok:$gold jz_classify.slurm
#
# For the pass-2 replay, export PROMPTS_VERSION=pass2 before submitting and
# build the queue from build_pass2_input.py output instead (see below).
set -euo pipefail

ROOT=${ROOT:-$WORK/pi_recluster}                 # campaign data root (shared FS)
VLLM_MODEL=${VLLM_MODEL:-google/gemma-4-12B-it}
MODEL_DIR=${MODEL_DIR:-$WORK/models/$(basename "$VLLM_MODEL")}
UNIT_SIZE=${UNIT_SIZE:-8000}

mkdir -p "$ROOT"

echo "== 1/4 data artefacts → $ROOT/data/hf (add --classifications + HF_TOKEN for the"
echo "        private checkpoint; pass-2 also works from it instead of re-running pass 1)"
python download_data.py --dest "$ROOT/data/hf"

echo "== 2/4 model weights → $MODEL_DIR (check \$DSDIR/HuggingFace_Models first)"
if [ -d "$DSDIR/HuggingFace_Models/$VLLM_MODEL" ]; then
    echo "   found in \$DSDIR — jobs can use MODEL_DIR=$DSDIR/HuggingFace_Models/$VLLM_MODEL"
else
    python -c "from huggingface_hub import snapshot_download; \
snapshot_download(repo_id='$VLLM_MODEL', local_dir='$MODEL_DIR')"
fi

echo "== 3/4 classify input → $ROOT/to_classify_full.jsonl"
# Pass-2 variant: python build_pass2_input.py --clustering <reclustered dir> \
#                     --out "$ROOT/to_classify_full.jsonl"
python build_classify_input.py \
    --clusters "$ROOT/data/hf/clusters_2026-03-18" \
    --out "$ROOT/to_classify_full.jsonl"

echo "== 4/4 work queue → $ROOT/queue"
python build_queue.py \
    --items "$ROOT/to_classify_full.jsonl" \
    --queue-dir "$ROOT/queue" \
    --unit-size "$UNIT_SIZE" \
    --exclude-done-from "$ROOT/outputs" \
    --rebuild

echo "prefetch done."
