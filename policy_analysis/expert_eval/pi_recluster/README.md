# Prime Intellect re-clustering package

Self-contained pipeline to run stratified `(sector, sub_code)` Leiden re-clustering on the sufficiency policy library, using a rented CPU box.

**Default mode (Option A, no new API spend)**: reuse the 100k policies already classified in `runs/api_100k/classifications.jsonl`. Filter → ~34k survive → Leiden per cell → new per-sector cluster parquets. Total cost: **PI box only (~$3–8)**.

**Optional mode (Option B, `FULL_CLASSIFY=1`)**: classify all 1.47M policies via DeepSeek V4 first (~$280 API), then re-cluster all ~490k survivors. Skip unless credits allow.

All compute is instrumented with **codecarbon** — an `emissions.csv` covering every stage lands in `data/carbon/` and a summary is printed at the end.

## What the box needs to look like

| Resource | Minimum | Recommended |
|---|---|---|
| vCPU | 8 | 16–32 (AMD Epyc Genoa or Intel Sapphire Rapids) |
| RAM | 32 GB | 64 GB |
| Disk | 40 GB SSD | 100 GB SSD |
| GPU | none | none |
| OS | Ubuntu 22.04 / Debian 12 | same |
| Python | 3.10+ | 3.12 |
| Network | any | egress to HuggingFace |

Prime Intellect usually rents this class at €0.30–1.50/hour. A single Option A run takes ~2 hours end-to-end; Option B (with the full classifier stage) adds ~1.5–2 hours of API-bound wait.

## Execution plan (Option A — recommended)

```bash
# 1. On the box: install uv, git-clone the repo, cd into this dir
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone <this-repo-url>
cd 13_democratiser_sobriete/policy_analysis/expert_eval/pi_recluster

# 2. Install deps for this iteration only (isolated env)
uv sync

# 3. On the box: make the data dir
mkdir -p data

# 4. On your LAPTOP (from the repo root), upload the 100k classifications:
#    scp runs/api_100k/classifications.jsonl \
#        root@<PI-HOST>:/root/13_democratiser_sobriete/policy_analysis/expert_eval/pi_recluster/data/classifications.jsonl

# 5. Back on the box: one-shot run
CLASSIFICATIONS=data/classifications.jsonl bash run.sh
```

That single `run.sh` invocation does:

1. **Download** the 11 cluster parquets + the 7.15 GB embeddings parquet from HuggingFace (`sufficiencylab/sufficiency-library`). `hf_transfer` gets ~200 MB/s → ~40 s. Cached across re-runs.
2. **Filter + join** — apply the classifier verdicts, keep `sufficiency + ambiguous`, subset the embeddings by surviving `policy_uid`. Emits `data/filtered.parquet`.
3. **Stratified Leiden** — group by `(sector, sub_code)`, run Leiden per cell with size-adaptive resolution, pick medoids. Emits `data/reclustered/<SECTOR>_reclustered_2026-06-13.parquet`.
4. **Summary** — prints per-sector new cluster counts + the codecarbon emissions table.

## Execution plan (Option B — full 1.47M classification via DeepSeek API)

```bash
# same setup as above, plus:
export DEEPSEEK_API_KEY=sk-…
FULL_CLASSIFY=1 bash run.sh
```

This will:
1. Build the full `to_classify_full.jsonl` from the cluster parquets (1.47M rows).
2. Run the v1 prompt classifier via DeepSeek V4 API (~$280, ~2 hours at 48 concurrency).
3. Continue with filter + Leiden as in Option A.

Note: Option B currently invokes the classifier by reaching into the repo root's `policy_analysis.expert_eval.classify_v1` module. That means the whole repo has to be cloned (which it is) AND the repo root's `pyproject.toml` needs to be installed so imports resolve. From the repo root: `uv sync`. Or run the classifier stage manually as documented in `../classify_v1.py`.

## Execution plan (Option C — full 1.47M classification + re-clustering on a GPU host)

**Recommended small-model path.** Runs Gemma 4 12B on a single H100 via vLLM, then Leiden clustering on the same box (GPU idle during Leiden — acceptable given the box is already rented). Expected cost: **$10–20** compute + $0 API. Expected wall clock: **3–5 h** classifier + **~10 min** Leiden.

Assumes the repo lives at `/home/ubuntu/13_democratiser_sobriete` on the GPU host (typical Prime Intellect Ubuntu image). Override with `REPO_ROOT=<path>` if yours differs.

Use `run_gpu.sh` — the GPU-host variant of the runner. It defaults to Option A (no classification) so `USE_VLLM=1` is the flag to trip.

### On the box, first-time setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

git clone <this-repo-url> /home/ubuntu/13_democratiser_sobriete
cd /home/ubuntu/13_democratiser_sobriete/policy_analysis/expert_eval/pi_recluster
uv sync --extra gpu     # installs vllm + xgrammar
```

### From your LAPTOP: upload the two files this pipeline needs (repo root)

```bash
mkdir -p runs/expert_gold        # skip if you already have these locally
scp runs/expert_gold/few_shot_v1.jsonl \
    ubuntu@<GPU-HOST>:/home/ubuntu/13_democratiser_sobriete/runs/expert_gold/few_shot_v1.jsonl
scp runs/expert_gold/gold.parquet \
    ubuntu@<GPU-HOST>:/home/ubuntu/13_democratiser_sobriete/runs/expert_gold/gold.parquet
```

### On the box: validate first (770 gold rows, ~15 min, essentially free)

```bash
cd /home/ubuntu/13_democratiser_sobriete/policy_analysis/expert_eval/pi_recluster

# Build a gold to_classify.jsonl from the parquet
uv run python -c "
import json, pandas as pd
df = pd.read_parquet('/home/ubuntu/13_democratiser_sobriete/runs/expert_gold/gold.parquet')
with open('data/gold_to_classify.jsonl', 'w') as f:
    for _, r in df.iterrows():
        f.write(json.dumps({
            'policy_uid': r['policy_id'],
            'sector': r['sector'],
            'policy_text': r['policy_text'],
        }) + '\n')
print('wrote', len(df), 'rows')
"

# Classify the gold rows
uv run --extra gpu python classify_vllm.py \
    --items data/gold_to_classify.jsonl \
    --out   data/gold_predictions.jsonl \
    --few-shot /home/ubuntu/13_democratiser_sobriete/runs/expert_gold/few_shot_v1.jsonl \
    --model google/gemma-4-12B-it

# Score against the gold — prints VERDICT: SHIP | BORDERLINE | DO NOT SCALE
uv run python validate_gold.py \
    --classifications data/gold_predictions.jsonl \
    --gold /home/ubuntu/13_democratiser_sobriete/runs/expert_gold/gold.parquet
```

If the verdict is SHIP, continue. If BORDERLINE, try `VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct` (needs TP=1 on H100 in FP8 or BF16) and re-validate. If DO NOT SCALE, stop and reconsider.

### On the box: full 1.47M classification + re-clustering

```bash
cd /home/ubuntu/13_democratiser_sobriete/policy_analysis/expert_eval/pi_recluster
USE_VLLM=1 bash run_gpu.sh
```

This runs, in order:
1. **download** — HF pull of the 11 sector parquets + 7.15 GB embeddings file.
2. **build_classify_input** — writes `data/to_classify_full.jsonl` (1.47M rows).
3. **classify_vllm** — Gemma 4 12B on the local GPU, guided-JSON, prefix-cached. Writes `data/classifications_full.jsonl`.
4. **filter_and_stratify** — filters to `sufficiency + ambiguous`, joins with embeddings.
5. **recluster** — Leiden per `(sector, sub_code)` cell → `data/reclustered/*.parquet`.
6. **CO2 summary** — codecarbon totals per stage.

### From your LAPTOP: pull results back (from the repo root)

```bash
mkdir -p runs/full_recluster
scp -r ubuntu@<GPU-HOST>:/home/ubuntu/13_democratiser_sobriete/policy_analysis/expert_eval/pi_recluster/data/reclustered \
       runs/full_recluster/reclustered
scp    ubuntu@<GPU-HOST>:/home/ubuntu/13_democratiser_sobriete/policy_analysis/expert_eval/pi_recluster/data/filtered.parquet \
       runs/full_recluster/filtered.parquet
scp    ubuntu@<GPU-HOST>:/home/ubuntu/13_democratiser_sobriete/policy_analysis/expert_eval/pi_recluster/data/carbon/emissions.csv \
       runs/full_recluster/emissions.csv
scp    ubuntu@<GPU-HOST>:/home/ubuntu/13_democratiser_sobriete/policy_analysis/expert_eval/pi_recluster/data/classifications_full.jsonl \
       runs/full_recluster/classifications.jsonl
```

Then run the judge locally against the new clusters (few-cents API, few minutes) and refresh `explore_reclustering.ipynb` pointing at `runs/full_recluster/`.

### Notes

- Override the classifier model with `VLLM_MODEL=<hf-repo-id>` (e.g. `Qwen/Qwen2.5-32B-Instruct`). Weights auto-download at first launch.
- Multi-GPU boxes: pass `--tensor-parallel-size N` to `classify_vllm.py` directly; `run_gpu.sh` doesn't thread it yet — either edit the call site or run `classify_vllm.py` yourself for step 3, then re-enter `run_gpu.sh` with `USE_VLLM=0` to continue from step 4.
- If you have credits and prefer the DeepSeek path: `FULL_CLASSIFY=1 bash run_gpu.sh` (requires `DEEPSEEK_API_KEY`, ~$280 spend, GPU idle during API wait).

## Getting results back

```bash
# From your LAPTOP (from the repo root):
scp -r root@<PI-HOST>:/root/13_democratiser_sobriete/policy_analysis/expert_eval/pi_recluster/data/reclustered \
       ./runs/reclustered

scp    root@<PI-HOST>:/root/13_democratiser_sobriete/policy_analysis/expert_eval/pi_recluster/data/carbon/emissions.csv \
       ./runs/reclustered/emissions.csv
```

Then locally: sample intrusion items from the new clusters, judge via the existing V4 API script, produce the 2×2 report — same code paths as the 100k run.

## Files

| File | Role |
|---|---|
| `pyproject.toml` | isolated env; `uv sync --extra gpu` adds vllm + xgrammar |
| `_carbon.py` | context manager wrapping every stage in an EmissionsTracker |
| `download_data.py` | HF download for cluster parquets + embeddings |
| `build_classify_input.py` | build the full 1.47M `to_classify.jsonl` from cluster parquets |
| `classify_vllm.py` | local vLLM classifier (Option C), guided-JSON, prefix-cached |
| `validate_gold.py` | score classifier output vs expert gold, print go/no-go verdict |
| `filter_and_stratify.py` | apply classifier filter, subset embeddings |
| `recluster.py` | stratified Leiden per `(sector, sub_code)` cell, emit per-sector parquets |
| `run.sh` | wrapper for a CPU host — default = Option A; `FULL_CLASSIFY=1` = B; `USE_VLLM=1` = C |
| `run_gpu.sh` | wrapper for a GPU host — same modes, `REPO_ROOT` anchored (default `/home/ubuntu/…`) |

## Cost estimate

| Stage | Time | Compute cost | API cost |
|---|---|---|---|
| Download data | 3–8 min | box $ | $0 |
| Filter + subset embeddings | 5–15 min | box $ | $0 |
| Stratified Leiden (Option A) | 45–90 min | box $ | $0 |
| Optional: full 1.47M classify | +1.5–2 h wall (API-bound) | box $ (idle) | ~$280 |
| **Total, Option A** | **~2 h** | **~$3–8** | **$0** |
| **Total, Option B** | **~4 h** | **~$6–15** | **~$280** |

## Sanity check before spending money

Run the pipeline once locally on the smallest sector (e.g. LOGISTICS: 9k → ~1.5k after filter) with a mocked download step, just to confirm all four stages succeed. Then boot the PI box.

```bash
# local smoke test
mkdir -p data/hf/clusters_2026-03-18
cp ~/data/wsl_sufficiency_eval/clusters_2026-03-18/LOGISTICS_* data/hf/clusters_2026-03-18/
# use the 100k classifications as input; filter_and_stratify skips missing sectors
CLASSIFICATIONS=../../../runs/api_100k/classifications.jsonl \
CLUSTERS_DIR=data/hf/clusters_2026-03-18 \
EMBEDDINGS_PATH=<path-to-a-mini-embedding-parquet-you-produce-locally> \
bash run.sh
```

If the mini run finishes cleanly and emissions.csv has three stages logged, everything is wired.

## Notes on codecarbon

`codecarbon` measures energy via CPU and RAM power estimates (based on the machine's advertised TDP + memory footprint), then multiplies by the grid carbon intensity of the box's location. If the box has no public internet at start-up, it falls back to a global average (approx. 475 g CO₂eq/kWh). In either case an `emissions.csv` line is written per stage with `project_name = pi_recluster:<stage>`.

To view after the run:

```bash
uv run python -c "
import pandas as pd
d = pd.read_csv('data/carbon/emissions.csv')
print(d.groupby('project_name')[['duration','energy_consumed','emissions']].sum().round(4))
print(f'\nTOTAL: {d.emissions.sum():.4f} kg CO2eq')
"
```
