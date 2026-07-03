# Sufficiency classification — Jean-Zay vLLM pipeline

## What this is

A per-policy classifier that tags every snippet in the corpus
(~1.5 M policies) into one of three classes — `sufficiency`,
`potential_sufficiency`, `not_sufficiency` — based on a precise definition
maintained in `prompts.py`. Designed to run as a SLURM array job on
Jean-Zay A100 nodes, one shard per GPU, vLLM in offline batch mode.

This is **not** the cluster intrusion task — that one stays on the remote
DeepSeek V4 API (see `policy_analysis/clustering/eval/`), because it's
~1k cluster-level calls where model quality matters more than throughput
and the per-call cost is negligible (~$0.08 for the whole pass).

## Two-track architecture, recap

| Task | Scale | Quality requirement | Where it runs | Cost |
|---|---:|---|---|---|
| Cluster intrusion (eval) | ~1k calls | high (frontier judgment) | DeepSeek V4 API | ~$0.08 / pass |
| Sobriety classification | ~1.5M calls | good (precise rubric) | self-hosted vLLM on Jean-Zay A100 | GPU-hours |

The two share the same `prompts.py` discipline (one source of truth per
task, version-stamped output) and the same JSONL output convention, so a
single aggregator can join them downstream.

## Model choice — what fits on one A100 80 GB

Default: **`deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`** (32B dense, BF16).
- Weights ~64 GB → comfortable on A100-80GB with KV-cache room.
- Strong instruction-following; the R1-distill series was trained on
  reasoning traces, so the model produces well-justified verdicts.
- Single GPU, `--tp 1`. Expect ~80–150 req/s with prefix-cache on a
  4k-context, short-output workload.

Swap candidates (no code change — pass `--model` and optionally
`--quantization`, `--tp`):

| Model | Fits on... | Trade-off |
|---|---|---|
| `deepseek-ai/DeepSeek-V2-Lite-Chat` | A100-40GB | 16B MoE, ~3× faster, lower quality |
| `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` (`--quantization fp8`) | 1× A100-80GB tight | strongest single-GPU option |
| `deepseek-ai/DeepSeek-V2-Chat` | 4–8× A100-80GB FP8 (`--tp 4`) | 236B MoE; nearer to V4 quality |
| `deepseek-ai/DeepSeek-V3` (671B MoE) | 2 nodes × 8× A100-80GB FP8 + Ray | last-resort; multi-node plumbing |

DeepSeek **V4** itself is not open-weight; whatever you self-host will
not give identical answers to the API run. Always re-calibrate against
the 55-item calibration set before scaling.

## Files

- `prompts.py` — system prompt, the `SUFFICIENCY_DEFINITION` placeholder
  (you fill this in with the project rubric), the JSON schema, and
  `PROMPT_VERSION`. Edit here and re-run to bump the stamp on every output row.
- `classify.py` — the vLLM offline-batch entry point. Args: `--model`,
  `--quantization`, `--tp`, `--shard`, `--num-shards`, `--data`, `--out`,
  `--limit`. Resume-safe: skips `policy_uid`s already present in the
  shard's output file. Reuses `policy_analysis.clustering.eval.loader`
  so the per-sector parquets are the single source of truth.
- `run.slurm` — SLURM array launcher. 10 shards by default; override via
  `--array=0-N` or `NUM_SHARDS=...`. Reads `MODEL`, `QUANTIZATION`, `TP`,
  `DATA_ROOT`, `OUT_DIR` from the env so you can drive everything with
  `sbatch --export`.

## How to run

### Smoke test on the API (no Jean-Zay needed)

Write a thin adapter in `test_api.py` that reuses `prompts.py` and calls
the DeepSeek V4 endpoint on, say, 200 sampled rows. Sanity-check the JSON
schema validation rate and verdict distribution. Don't scale to 1.5M
through the API — at $0.0000791/call (the rate from our intrusion task)
you'd pay ~$120 for one pass; self-hosting is far cheaper given your
Jean-Zay credits.

### Smoke test on Jean-Zay (single shard, 1k policies)

```bash
sbatch --array=0-0 --export=ALL,EXTRA_ARGS="--limit 1000",OUT_DIR=$SCRATCH/suff_smoke \
    policy_analysis/sufficiency_classification/run.slurm
```

Check `logs/suff-classify-*.out` for throughput and a few sample lines in
`$SCRATCH/suff_smoke/shard_00_of_10.jsonl`.

### Full run

```bash
sbatch policy_analysis/sufficiency_classification/run.slurm
```

10 jobs in the array → 10 GPUs in parallel → ~1.5M rows in a few hours
to a day depending on model and quant. Outputs land in
`$SCRATCH/sufficiency_runs/v0/shard_NN_of_10.jsonl`.

### Aggregate

Concatenate the shard JSONLs into a parquet and join back onto the
clustered parquets via `policy_uid`. Suggested final schema:

```
policy_uid | openalex_id | chunk_idx | sector | cluster_uid
            | category | confidence | reasoning
            | model | prompt_version
```

## Open items before running

1. **Fill in `SUFFICIENCY_DEFINITION` in `prompts.py`.** Currently a
   `TODO` placeholder; the model will refuse / hallucinate without a real
   rubric. Include 2–3 worked examples per class.
2. **Set Jean-Zay `--account` / `--partition` / `--qos` in `run.slurm`.**
   Placeholders are commented out; uncomment and fill once you have the
   project allocation in front of you.
3. **Decide on the API smoke test.** ~200 rows on V4 = trivial cost,
   gives a "ground truth" anchor to validate the self-hosted model
   against. Worth doing.
4. **Decide whether shards stratify by sector.** Current hash sharding
   is balanced by row count, not sector; a small sector imbalance per
   shard is fine for classification but matters if you want per-shard
   sector reports during the run.

## Gotchas observed elsewhere in the repo

- vLLM on Jean-Zay sometimes needs `module load cpuarch/amd` *before*
  `arch/a100` on the EPYC-host A100 nodes; the SLURM script does this.
- `trust_remote_code=True` is required for DeepSeek models because they
  ship custom modeling code in the HF repo. Already set in `classify.py`.
- Set `HF_HOME=$SCRATCH/hf_cache` (the SLURM script does) so weights
  don't blow your `$HOME` quota — the 70B model alone is ~140 GB BF16.
- Existing Jean-Zay scripts in `policy_analysis/` (e.g.
  `vllm_geo_extraction.py`, `jz_geo_extraction.sh`) use the same
  10-shard pattern; if you change `NUM_SHARDS` here, do it everywhere
  so the post-processing scripts agree.
