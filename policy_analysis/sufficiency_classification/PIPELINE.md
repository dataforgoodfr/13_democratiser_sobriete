# Sufficiency pipeline — decision log

_Last updated: 2026-06-13_

## Context

We have ~1.47M policy snippets clustered into 1,030 Leiden clusters across 11 sectors (`clusters_2026-03-18`). To deliver a sufficiency-focused policy index we need to: filter non-sufficiency policies, fix wrong-sector routing, re-cluster the survivors per sector, then verify cluster quality.

## Two-track architecture

| Track | Volume | Quality vs throughput | Where |
|---|---:|---|---|
| Classifier (sufficiency + sector) | ~1.5M calls | throughput-bound | self-hosted vLLM on Jean-Zay A100s |
| Cluster judge (intrusion task) | ~1k calls | quality-critical | DeepSeek V4 remote API |

Reasoning: classifier volume kills any per-call cost optimisation; judge volume is small enough that paying the API premium for V4 quality is worth it (~$0.10 total).

## Model choices

| Track | Model | Hardware | Trigger |
|---|---|---|---|
| Classifier — default | `DeepSeek-R1-Distill-Qwen-32B` (BF16) | 1× A100-80GB per shard, 10-shard SLURM array | first attempt |
| Classifier — escalation | `DeepSeek-V3` (671B MoE, FP8) | 2 nodes × 8× A100-80GB + Ray (TP=8, PP=2) | only if V4-vs-self-host agreement < 70% on calibration |
| Judge | DeepSeek V4 API + 4 few-shot calibration examples | — | stays remote |

CPU offloading explicitly **ruled out**: drops throughput 10–100×; vLLM doesn't even support it.

## Pipeline steps

```
1. Self-hosted classifier on 1.5M policies
   → emits category, confidence, reasoning, sector_predicted, sector_confidence
   → Jean-Zay, ~1–2h wall clock

2. Filter: drop category == not_sufficiency
   → keeps sufficiency + potential_sufficiency (~37% of 1.5M ≈ 545k policies)

3. Re-route: where sector_predicted ≠ sector_routed AND sector_confidence ≥ 4,
   move policy to predicted sector

4. Subset Qwen3-4B embeddings on HF (7.15GB) by surviving policy_uid
   → CPU pandas, < 1h

5. Re-cluster per sector via Leiden on cosine-similarity graph
   → Jean-Zay CPU big-RAM node, 1–4h

6. Judge new clusters via V4 API + few-shot calibration
   → ~$0.10 total, < 1 min

7. Produce per-cluster 2×2 report (judge accuracy × sufficiency density)
   → admit, drop, re-cluster, or discard per cluster
```

## Calibration gate before step 1 at full scale

Before committing to 1.5M API calls or 10 GPU-hours, validate the open-weight classifier against V4:

1. Take the 200 policies already classified via V4 (`runs/sufficiency_api_test/judgments.jsonl`).
2. Re-classify the same 200 with self-hosted R1-Distill-Qwen-32B.
3. Compute V4-vs-R1-Distill category agreement + confusion matrix.

**Decision rule**

| Agreement | Action |
|---|---|
| ≥ 85% | proceed with R1-Distill-Qwen-32B at full scale |
| 70–85% | try R1-Distill-Llama-70B (FP8) on single A100, recalibrate |
| < 70% | escalate to DeepSeek-V3 multi-node |

## Cost projection (decision)

| Path | API cost | GPU compute | Wall clock end-to-end |
|---|---:|---|---|
| **Chosen — self-host classifier, API judge** | ~$0.10 | Jean-Zay credits (~10–20 A100-hours) | ~3–6h |
| Fallback — full V4 API | ~$264 | none | ~9h |

## Open items

1. **`SUFFICIENCY_DEFINITION`** in `prompts.py` — currently a working placeholder using the standard academic framing (absolute demand reduction vs efficiency vs substitution). Domain expert needs to finalise.
2. **`SECTOR_TAXONOMY`** — needs writing for the `sector_predicted` field. Can be drafted from existing per-sector cluster topics for refinement.
3. **JSON schema** in `prompts.py` needs `sector_predicted` (enum of 11 sectors) and `sector_confidence` (int 1..5) added.
4. **Jean-Zay SBATCH headers** in `run.slurm` — `--account`, `--partition` (`gpu_p5`), `--qos` placeholders still to fill.
5. **Re-routing threshold** — defaulting to `sector_confidence ≥ 4`. Revisit after first run.

## Files (current state)

| File | Purpose |
|---|---|
| `policy_analysis/sufficiency_classification/prompts.py` | system prompt, `SUFFICIENCY_DEFINITION` (placeholder), JSON schema, `build_system_prompt(few_shot)` |
| `policy_analysis/sufficiency_classification/classify.py` | vLLM offline batch entry, sharded, resume-safe |
| `policy_analysis/sufficiency_classification/test_api.py` | DeepSeek V4 API client (same prompts; for calibration + smoke tests) |
| `policy_analysis/sufficiency_classification/run.slurm` | Jean-Zay 10-shard array launcher |
| `policy_analysis/sufficiency_classification/build_test_sample.py` | stratified sampler for smoke tests |
| `policy_analysis/clustering/eval/llm_judge.py` | V4 judge, now supports `--few-shot` |
| `policy_analysis/clustering/eval/prompts.py` | judge system prompt with `build_system_prompt(few_shot)` |
| `policy_analysis/clustering/eval/pipeline_report.py` | per-cluster 2×2 report generator |

## Next actions (in order)

1. Finalise `SUFFICIENCY_DEFINITION`.
2. Draft + finalise `SECTOR_TAXONOMY` block; add `sector_predicted` / `sector_confidence` to JSON schema in `prompts.py`.
3. Run V4-vs-R1-Distill calibration on 200 items once Jean-Zay access is set up.
4. Apply the decision rule above to pick model.
5. Launch full classifier run on 1.5M.
6. Filter + re-route → re-cluster per sector → judge new clusters.
7. Produce final report.

## Escalation note

If quality demands DeepSeek-V3 multi-node: budget ~1 week of engineering for Ray + multi-node vLLM plumbing on Jean-Zay, plus ~2× the GPU footprint (16 GPUs vs 10). Acceptable if V4-grade quality on classification is genuinely needed; otherwise prefer paying the $264 to the API.
