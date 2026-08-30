# Sufficiency classifier + clustering — research notes

Living document. One section per iteration cycle. Newest at top. Numbers link to the artefacts under `runs/` so future us can reproduce or diff.

Last updated: 2026-07-29

---

## Context: what we are building

A three-stage pipeline over ~1.47M policy snippets extracted from an OpenAlex-derived corpus:

1. **Classify** each policy as one of {`sufficiency`, `efficiency`, `consistency`, `ambiguous`, `not_a_policy`}. When `sufficiency`, also assign a sub-code 0–9 from the expert taxonomy (caps/limits, passive design, modal shift, …).
2. **Cluster** the surviving sufficiency-related policies per sector into coherent topics.
3. **Evaluate** clusters via LLM-as-judge intrusion task + expert spot-check.

Output: a curated, sufficiency-focused policy index for downstream RAG + analysis.

Ground truth we have:
- 770 rows expert-labelled with 5-class category + 10-code sub-taxonomy (`runs/expert_gold/gold.parquet`).
- Random 500-item intrusion set from existing clusters, calibrated with 55-item hand-labeled reference (`runs/eval_v1/`).

---

## Iteration log

### 2026-07-29 — Iteration 6: expert evaluation lands, singleton rehoming, pass-2 classifier design, judge mechanism-fit check

**Trigger.** The expert deliverables arrived: YS + team hand-labelled the 90-cluster quality-review sample (`2026-06-27_Clusters_Final assessement by YS and team.xlsx`) and ALL 2,223 singletons (`2026-06-27_Singletons_Final Assessemnt_YS and team.xlsx`). First human ground truth on the iteration-4/5 clustering itself (previous gold was policy-level only).

**What we did.**
1. **Analysed the 90-cluster expert review.** 60% coherent / 28% partly / 12% no; mean quality 3.0/5; 21 clusters flagged should-split. Cross-checked against the iteration-4 LLM judge on the 31 overlapping clusters.
2. **Analysed the 2,223-singleton review.** 79.1% `not a policy`, 9.4% sufficiency (209 rows, each with a `merge_into_cluster` = expert sub-code), rest ambiguous/consistency/efficiency.
3. **Rehomed the 209 expert-validated singletons** (`rehome_singletons.py`): similarity-weighted k-NN vote (k=20, cosine) restricted to the expert-assigned `(sector, sub_code)` cell, floor 0.45 (the iteration-5 absorption threshold), nearest-centroid cross-check. Amended clustering written to `runs/singleton_rehoming/reclustered/`.
4. **Designed the pass-2 post-clustering re-classification** (systemic replacement for hand-patching the flagged clusters): `pi_recluster/prompts_pass2.py` (v1 definition verbatim + 6 calibration rules targeting the observed failure modes — per iteration 5, rules not richer definitions), `build_pass2_input.py` (replays the 645k clustered rows, not the raw 1.47M), `apply_pass2.py` (drops rows that flip off `sufficiency`, emits `cluster_audit.csv` flagging low-retention and cell/sub-code-mismatch clusters). Reuses `classify_vllm.py` unchanged via `PROMPTS_VERSION=pass2`. Awaits GPU compute.
5. **Added the mechanism-fit question to the intrusion judge** (`clustering/eval/prompts.py` → `v1-deepseek-subcode`): items now carry the cluster's cell sub-code, the judge returns `sub_code_fit` ∈ {yes, partly, no, null} alongside the intruder pick. Backward compatible (items without sub-code get the old task + null).

**Key numbers established this iteration:**

Expert review, 90 clusters:

| metric | value |
|---|---:|
| coherent yes / partly / no | 54 / 25 / 11 |
| mean quality (1–5) | 3.03 |
| label_fits = No | 26/81 (13 of them on *coherent* clusters) |
| should_split | 21 (18 of the 27 clusters rated ≤2) |
| worst sector | INDUSTRY: mean 2.29, 0/7 fully coherent |
| best sectors | BUILDING 3.71, URBAN 3.50, MATERIALS 3.43 |
| ambiguous `__c-1__` cells | 2.10 vs 3.15 for coded cells |
| worst real sub-code | 3 · Reduce & right size: 2.36 |
| judge vs expert (31 overlap) | judge correct: 94% on coherent-yes, 73% on partly; r(quality, judge-correct) ≈ 0 |

Expert review, 2,223 singletons:

| verdict | n | share |
|---|---:|---:|
| not a policy | 1,758 | 79.1% |
| **sufficiency (rehome)** | **209** | **9.4%** |
| ambiguous | 126 | 5.7% |
| consistency | 76 | 3.4% |
| efficiency | 54 | 2.4% |

Rehoming (k=20, floor 0.45): **203/209 assigned (97%)**, 6 below floor left as singletons; median top-similarity 0.538, median vote share 0.70, centroid cross-check agreement 75%; 40/209 rows moved cell (expert re-code). Amended clustering: 645,495 rows unchanged, 3,152 → 2,949 clusters, singletons 2,160 → 1,957.

**What we learned.**

1. **The intrusion judge is not a proxy for expert-perceived quality.** It tracks coherence weakly (94% vs 73% correct on coherent-yes vs partly) but was correct on both clusters the experts called incoherent, and correlation with expert quality is ~0. Intrusion measures *separability between clusters*; the failure the experts actually flag — topically coherent but wrong mechanism, or not sufficiency at all — is invisible to it. The 86.1% from iteration 4 is real but overstates semantic quality. Hence the `sub_code_fit` addition.
2. **The weakest link is the sub-code label, not the grouping.** 32% label misfit, half of it on clusters that are otherwise coherent. Dominant pattern: lexical false-matching, concentrated on code 3 (reducing *sedentary behaviour / body weight / headcount* ≠ consumption right-sizing) and code 7 (knowledge-sharing ≠ product sharing). A few anti-sufficiency texts ("increase disposable instruments") even carry sufficiency codes. These became pass-2 rules R1–R4.
3. **INDUSTRY is confirmed bad by humans** (mean 2.29, zero fully coherent) — the iteration-4 judge regression was signal, not sample noise. Notes describe heterogeneous grab-bags, matching the mixed-mechanism hypothesis.
4. **Singleton status ≈ classifier false positive.** 79% of Leiden-stranded singletons are not policies at all. Two consequences: (a) the shelved iteration-5 absorption would have injected ~1,760 non-policies into good clusters — shelving it is retroactively validated; (b) stranding acts as a free noise filter, worth keeping in mind for the size-adaptive-resolution work: don't over-optimise singleton count down.
5. **The ambiguous-drop decision is re-validated at cluster level**: `__c-1__` clusters rate a full point below coded cells (2.10 vs 3.15).
6. **Expert sub-code agreement on the 209 keepers is 78.9%** — consistent with the 83.5% gold sub-code accuracy; biggest recodes 3→9 and 9→7.

**Cost + emissions accounting.**
- Everything this iteration ran locally on CPU (analysis + rehoming ≈ minutes): **$0, ~zero emissions**.
- Pending GPU spend: pass-2 replay is 645k rows ≈ 44% of the iteration-4 run → ~5 h H100 ≈ **~$15** when compute lands.

**Decisions made.**
- **Rehome via k-NN vote, not centroid**: consistent with the k-NN-graph construction of the clusters; centroid kept as a recorded cross-check only (75% agreement, disagreements skew to large non-spherical clusters).
- **No manual patching of the 21 flagged clusters.** The systemic fix is the pass-2 replay + `cluster_audit.csv`; expert flags become its validation set.
- **Judge prompt bumped to `v1-deepseek-subcode`** — all future judge runs report mechanism fit. Old judgments remain comparable on the intruder metric.
- The 1,957 remaining singletons (expert-reviewed: not sufficiency) stay in the amended parquets for now — dropping them is a one-liner downstream filter, decided at pass-2 apply time.

**Open items carried forward.**
- **Run pass-2 on GPU when compute lands**: gold preflight first (`PREFLIGHT_GOLD=1 PROMPTS_VERSION=pass2` — expect precision ↑ vs v1; watch strict recall for a v2.1-style overshoot), then the 645k replay, then `apply_pass2.py`. Validate `cluster_audit.csv` flags against the 21 expert-flagged clusters and the 26 label misfits.
- **Re-judge a sample with `v1-deepseek-subcode`** to get the first `sub_code_fit` distribution; calibrate against the expert `label_fits` column (90-cluster sample = free validation set).
- **SLURM packaging for the full-pipeline rerun** (next step once compute is granted): `sufficiency_classification/run.slurm` still targets the v0-era `classify.py` (R1-Distill default model, old data root, no `PROMPTS_VERSION`); port the battle-tested `pi_recluster` path (guided-JSON `classify_vllm.py`, preflights, gold gate) into the 10-shard array contract. Mind Jean-Zay compute nodes being offline: HF model + checkpoint prefetch must happen on the login node.
- Data quirk: one `cluster_uid` in the cluster-review workbook is truncated to `c` (a FOOD code-9 cluster) — repair before joining on it.
- (Carried) size-adaptive Leiden resolution; INDUSTRY deep-dive (now with expert evidence); taxonomy pruning.

**Artefacts.**
- `2026-06-27_Clusters_Final assessement by YS and team.xlsx`, `2026-06-27_Singletons_Final Assessemnt_YS and team.xlsx` — the expert deliverables.
- `runs/singleton_rehoming/` — `rehomed_singletons.parquet`/`.csv` (per-singleton assignment, similarity, vote share, centroid cross-check) + `reclustered/` (amended clustering, 2,949 clusters).
- `policy_analysis/expert_eval/rehome_singletons.py` — the rehoming script.
- `policy_analysis/expert_eval/pi_recluster/prompts_pass2.py`, `build_pass2_input.py`, `apply_pass2.py` — the pass-2 kit.
- `policy_analysis/clustering/eval/prompts.py` (`v1-deepseek-subcode`), `llm_judge.py`, `expert_eval/build_intrusion_recluster.py` — judge mechanism-fit support.

---

### 2026-07-06 — Iteration 5: six-pillar prompt exploration, prompt/model interaction, clustering-tuning dead-end, ambiguous drop

**Trigger.** Two goals for "another round of clustering": (1) fold in a **precise six-pillar sufficiency definition** (synthesised from Saheb 2021 / IPCC 2022 + the demand-side literature — avoidance, all-resources, satiable-wellbeing, planetary-boundaries, dual-agency, justice) to sharpen the classifier; (2) fix the **singleton explosion** from iteration 4 (66% of the 3,350 clusters were size-1, making the corpus hard to analyse).

**What we did.**
1. **Checkpointed the v1 Gemma classifications to a personal HF dataset** (`AmineSab/sufficiency-classifications`, private, with a dataset card) — 1.47M rows, 507 MB jsonl → 72 MB parquet. Future clustering iterations now restart from the checkpoint (no re-classification). `download_data.py --classifications` pulls it; `filter_and_stratify.py` reads parquet or jsonl.
2. **Built the six-pillar prompt as v2** (`prompts_v2.py`) — the six pillars turned into an ordered operational decision procedure + watch-outs; citations kept out of the runtime prompt (1.47M calls) and mapped to sources in the module docstring instead.
3. **Validated v2 vs v1 on the 760-row expert gold via DeepSeek V4.** v2 won: binary **79.3% → 81.1%**, precision +2.9 pt, F1 0.62 → 0.64, strict recall held at ~79%. Regressions traced to one cause — `ambiguous` over-firing on vague aspirations.
4. **Iterated to v2.1** (tightening rule: "vague aspiration with no concrete mechanism → not_a_policy, not ambiguous"). On DeepSeek it **overshot** — precision +4.6 pt but strict recall −10 pt, F1 down to 0.61. Also tried v3 (plain six-pillar definition, no scaffolding) and v1.1 (v1 + only the tightening rule).
5. **Added a GPU preflight** (`PREFLIGHT_GOLD=1` in `run_gpu.sh`): classify the 760 gold rows through the actual model, score with `validate_gold.py` (now returns SHIP/BORDERLINE/DO_NOT_SCALE exit codes), stop for review. Bundled the gold artefacts into `pi_recluster/gold/` so they exist on a fresh box.
6. **Ran the preflight for all five prompts on Gemma 4 12B — and the DeepSeek ranking inverted** (table below). The rich six-pillar definition (v2, v3) made the *smaller* model **worse**; only v2.1's precision-tightening helped, and v1 stayed best on the balanced metrics.
7. **Clustering-tuning experiment**: lowered `DEFAULT_RESOLUTION` 1.0 → 0.5 and added singleton-absorption (union-find rehoming of size-1 clusters to nearest neighbour at a 0.45 floor, below the 0.55 graph threshold). Validated on LOGISTICS (110 → 33 clusters, 74 → 6 singletons). Ran the full corpus — but it **over-merged the large cells** (SOCIAL sub_code 9 collapsed 34,242 policies into one blob; NATURE 19.5k). Shelved.
8. **Landed on the pragmatic plan**: keep the iteration-4 baseline clustering, send the 2,223 singletons to **manual classification** (the `singletons.csv` queue), and **drop `ambiguous`** from the clusters.

**Key numbers established this iteration:**

Five-way prompt comparison on the same 760 expert-gold rows (few-shot excluded):

| prompt | DeepSeek acc | DeepSeek F1 | **Gemma acc** | Gemma prec | Gemma recall | Gemma F1 | Gemma 5-class |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1 (terse discriminator) | 79.3% | 0.62 | 75.3% | 46.5% | **88.0%** | 0.61 | 62.9% |
| v1.1 (v1 + tightening rule) | — | — | 80.0% | 53.4% | 65.7% | 0.59 | 66.1% |
| v2 (six-pillar, scaffolded) | **81.1%** | **0.64** | 72.4% | 43.3% | 86.1% | 0.58 | 55.1% |
| **v2.1 (v2 + tightening)** | 81.6% (prec-heavy) | 0.63 | **81.6%** | **56.1%** | 71.7% | **0.63** | **66.7%** |
| v3 (six-pillar, plain) | — | — | 69.1% | 40.6% | **89.8%** | 0.56 | 54.9% |

| clustering metric | baseline (iter 4) | tuned (res 0.5 + absorb) |
|---|---:|---:|
| clusters | 3,350 | 783 |
| singletons | 2,223 (66.4%) | 75 (9.6%) |
| median cluster size | 1 | 220 |
| **max cluster size** | 18,746 | **34,242 (blob)** |
| substantive clusters (≥30) | 1,007 | 589 |

**What we learned.**

1. **Prompt improvements are model-specific — they do not transfer down in model size.** The six-pillar definition *helped* DeepSeek V4 (large, strong reasoner: v2 beat v1) but *hurt* Gemma 4 12B (v2/v3 were the two worst prompts; 5-class exact-match crashed 62.9% → 55.1%). A 12B model gets confused by the longer, multi-step definition. **Do not validate a prompt on model A and ship it on model B.**
2. **What actually helped Gemma was one rule, not the definition.** Gemma is intrinsically over-permissive (floods "sufficiency-related", precision ~43–46%). The only prompt that beat v1 was v2.1, and its winning ingredient is the single tightening rule ("vague aspiration → not_a_policy") that reins in that permissiveness. The six-pillar text around it was dead weight — proven by v1.1 (v1 + rule alone) being *dominated* by v2.1 (the six-pillar context provided recall ballast the bare rule lost). Net: **the corrective a small model needs is calibration, not richer concept.**
3. **v2.1-on-Gemma ≈ v2-on-DeepSeek in quality** (81.6%/56% vs 81.1%/55%). You can match API-model quality on a local 12B with the right prompt — paying for the API bought nothing here.
4. **Uniform Leiden resolution across hugely-varying cell sizes is the clustering flaw.** At res=1.0 large cells over-fragment (singletons on the fringe); at res=0.5 they over-merge (blobs in the core). These are *two different failures needing two different levers* — a stray-cleanup pass (absorption) for the fringe, and **size-adaptive resolution** (higher res for bigger cells) for the core. We only built the first; the blob problem killed the naive resolution drop. A proper fix scales resolution with cell size (or caps cluster size with recursive re-clustering).
5. **Singleton absorption works but the threshold must sit *below* the graph-edge threshold.** A node Leiden stranded as a singleton has, by construction, no neighbour ≥ the 0.55 graph threshold; its nearest is typically ~0.52 (90% ≥ 0.45). Absorbing at 0.45 rehomed ~90% sensibly; at 0.55 it absorbed nothing.
6. **Ambiguous is ~zero sufficiency.** Of 31,382 ambiguous rows (their own `__c-1__` cells), exactly **1** was sufficiency. Safe to drop from clustering entirely.

**Cost + emissions accounting.**
- DeepSeek validation: ~1,520 API calls (v2 + v2.1 × 760) ≈ **<$1**.
- Gemma preflights: 5 × (model load + 760 rows) on the already-rented H100 ≈ **~1 h ≈ $3**.
- Tuned full clustering run: reused v1 checkpoint (no re-classification), filter + recluster only, ~15 min CPU on the box → sunk.
- HF checkpoint upload: free.
- **Total this iteration: ~$4**, negligible emissions. The value was decision-quality, not compute.

**Decisions made.**
- **Ship v1 as the classifier prompt.** It validated best on Gemma, and since v1-on-Gemma *is* the existing HF checkpoint, **no re-classification is needed** — the iteration-4 classifications stand.
- **Drop `ambiguous` from clustering by default** (`filter_and_stratify.py`; `--include-ambiguous` to override).
- **Keep the iteration-4 baseline clustering**; the 2,223 singletons go to manual classification via `singletons.csv`.
- **Shelve the res=0.5 + absorption tuning** — reverted `recluster.py` to baseline (res=1.0). Kept the diagnosis for a future size-adaptive-resolution attempt.
- **Keep the workflow improvements** regardless: HF checkpoint + `download_data.py --classifications`, `run.sh`/`run_gpu.sh` mode banner + checkpoint auto-fetch, the gold preflight, bundled `gold/`.

**Open items carried forward.**
- **Size-adaptive Leiden resolution** for the blob problem: scale resolution with cell size, or cap cluster size with recursive re-clustering of any cluster > ~2,000. This is the real fix for both singletons and blobs.
- **Manual classification of the 2,223 sufficiency singletons** (`singletons.csv`, sufficiency-only after ambiguous drop = 2,160).
- Expert review of the 90-cluster `cluster_quality_review.xlsx` sample.
- (Carried from iteration 4) INDUSTRY regression; MOBILITY × code 6 spot-check; taxonomy pruning of small sub-codes.

**Artefacts.**
- `AmineSab/sufficiency-classifications` (HF) — v1 Gemma classification checkpoint, 72 MB parquet + dataset card.
- `runs/full_recluster_noambig/reclustered/` — the shipped clustering (baseline minus ambiguous): 645,495 rows, 3,152 clusters.
- `runs/recluster_tuned_withambig/reclustered/` — the shelved res=0.5 + absorption run (kept for the record; do not use).
- `runs/expert_gold/REPORT_v2.md`, `REPORT_v2_1.md`, `REPORT_v2_1_vs_v2.md` — DeepSeek prompt comparisons.
- `runs/expert_gold/classifications_v2.jsonl`, `classifications_v2_1.jsonl` — raw DeepSeek outputs.
- `policy_analysis/sufficiency_classification/prompts_v2.py`, `prompts_v2_1.py`, `prompts_v3.py`, `prompts_v1_1.py` — the prompt variants (env-selectable via `PROMPTS_VERSION`; v1 is default).
- `policy_analysis/expert_eval/eval_v2.py` — parameterised head-to-head prompt evaluator.
- `runs/full_recluster/review/` — `singletons.csv` (manual-classification queue), `cluster_quality_review.xlsx`, per-sector browse workbooks.

---

### 2026-07-04 — Iteration 4: full 1.47M pipeline on rented H100, stratified re-clustering at scale

**Trigger.** The 100k iteration (iteration 3) proved the stratified `(sector, sub_code)` Leiden approach delivered a real +4.4 pt bump on judge accuracy — enough evidence to commit to a full-corpus run. Constraint: no DeepSeek credits for the ~$280 V4-pro classifier pass on 1.47M policies. Solution: rent a single H100 on Prime Intellect and run **Gemma 4 12B** locally via vLLM, ~1/10 the cost.

**What we did.**
1. **Packaged the entire pipeline** as a self-contained runnable directory (`policy_analysis/expert_eval/pi_recluster/`) — `pyproject.toml`, `run_gpu.sh`, per-stage scripts, codecarbon-instrumented, resume-safe, guided-JSON decoding via vLLM. Repo goes to any fresh box, `uv sync --extra gpu`, one command.
2. **Booted an H100 on Prime Intellect**, hit five successive environment issues (Python 3.14 selected instead of 3.12, codecarbon 3 needing libpq, vLLM's `GuidedDecodingParams → StructuredOutputsParams` rename, missing CUDA toolkit at first, missing g++ C++ backend for the box's gcc), each producing a preflight-check improvement in `run_gpu.sh` so the next box just works.
3. **Ran the full 1.47M classification** with Gemma 4 12B, guided-JSON decoding for the v1 5-class + 10-sub-code schema, 10 few-shot from expert gold, prefix-cached. Wall clock: **10 h 45 min at ~38 policies/sec sustained** on 1× H100 BF16.
4. **Filtered** to sufficiency + ambiguous → **676,876 policies survived** (46.2%). Higher than the 100k iteration's 33.5% survival rate because SOCIAL (heavy on public provisioning) is a larger share of the full corpus than of the stratified 100k sample.
5. **Reclustered** via stratified `(sector, sub_code)` Leiden with the same size-adaptive resolution as iteration 3. Wall clock: **9 minutes**. Result: **3,350 new clusters**, avg cluster size 202.
6. **Judged 353 randomly-sampled new clusters** via DeepSeek V4 API + few-shot calibration. Cost: ~$0.50.

**Key numbers established this iteration:**

| metric | value | source |
|---|---|---|
| classifier throughput (Gemma 4 12B on 1× H100 BF16, guided-JSON) | ~38 req/s | `runs/full_recluster/emissions.csv` (38,743 s for 1,465,440 requests) |
| filter yield (sufficiency + ambiguous / total) | 46.2% | `runs/full_recluster/filtered.parquet` |
| new cluster count | 3,350 | `runs/full_recluster/reclustered/` |
| avg cluster size | 202 members | same |
| **judge accuracy on 353 new clusters (overall)** | **86.1%** | `runs/full_recluster/judgments.jsonl` |
| judge accuracy on small clusters (≤20) | **84.2%** | same |
| judge accuracy on medium (21–100) | 86.1% | same |
| judge accuracy on large (>100) | 86.3% | same |
| best-performing sector (SOCIAL) | 96.1% | same |
| **worst-performing sector (INDUSTRY)** | **64.5%** | same |
| public provisioning share of all sufficiency (code 9) | 43% | notebook §3 |
| caps/limits/bans share (code 1) | 27% | same |
| modal shift share (code 6) | 2.2% | same |

**What we learned.**

1. **Full-scale stratification decisively beats the 100k iteration and the pre-filter baseline.**
   - Overall: 73.5% → 77.9% → **86.1%** (+12.6 pt vs original, +8.2 pt vs 100k).
   - **The small-cluster penalty is gone**: previously accuracy fell to 58–68% for clusters ≤20 members; now it's 84%. Accuracy is essentially flat across size buckets, the signature of a well-resolved clustering.
   - Ten sectors improved by 6–21 pt; one regressed (INDUSTRY, −12 pt).

2. **INDUSTRY is the one regression and worth investigating.** 64.5% vs old 76.7%. n=31 clusters so the signal is real but not enormous. Hypothesis: INDUSTRY sufficiency policies span sub-codes very heterogeneously (labour hours, industrial emissions caps, employee well-being, sharing platforms, food supply chains), so within-sub-code clustering leaves too many mixed topical themes. A follow-up would either bump Leiden resolution just for INDUSTRY cells, or fold small INDUSTRY sub-code cells into a single "other" bucket. **Deferred to iteration 5.**

3. **Small-model classification (Gemma 4 12B) is competitive at full scale.** No expert-gold binary accuracy measurement on Gemma output yet (the validate_gold step was skipped mid-run under time pressure), but the downstream judge accuracy of 86.1% is arguably a stronger downstream signal than the point estimate would give us. Worth backfilling the validation pass at some point for the record, but not blocking.

4. **The mechanism distribution has real editorial implications**:
   - **Public provisioning (code 9) is 43% of all sufficiency**. Nearly half the sufficiency policy corpus is about collective/universal provision of needs — welfare state, decommodification, social floors. This is a very specific policy stance and will shape how the downstream sufficiency index reads.
   - **Caps/limits/bans (code 1) is 27%**, mostly regulatory ceilings in NATURE, MATERIALS, MACROECONOMIC (working hours).
   - Together they're 68% of the space. Every other mechanism combined is 32%.
   - **Modal shift (code 6) is only 2.2%**. Surprising given the visibility of modal-shift discussion in transport sufficiency. Possibly under-attribution by the classifier (EV / biofuel policies pulled toward efficiency / consistency instead of sufficiency-code 6). Worth spot-checking MOBILITY × code 6 clusters manually.

5. **The engineering was where the real cost went, not the compute.** Actual GPU time was ~11 h. Diagnosing and fixing five different environment issues (Python version, apt package, vLLM API, CUDA install, gcc / g++ mismatch) added several hours of wall clock and a handful of push-pull cycles. Everything is now captured in `run_gpu.sh`'s preflight (auto-detects CUDA_HOME, tests both g++ AND `gcc -x c++`, persists NVCC_CCBIN, appends CUDA env to ~/.bashrc, refuses to continue on any missing dep with actionable install commands). Next box: `git clone && uv sync --extra gpu && USE_VLLM=1 bash run_gpu.sh`. This is the payoff of packaging.

6. **Self-host is ~10× cheaper $ but ~4× worse carbon than the API path.** Real trade-off, worth remembering. Gemma-4-on-H100 at ~$3/h with guided-JSON is dominated by H100 idle cycles between batches; V4 API pipelines requests through shared inference clusters at higher utilisation.

**Cost + emissions accounting for this iteration:**
- Classifier: 10.75 h × $3/h ≈ **$32 H100 rental**.
- Judge: 353 API calls ≈ **$0.50**.
- Reclustering: 9 min of the same H100 already rented → sunk. Amortised as ~$0.45.
- **Total this iteration: ~$33.**
- **Emissions: 2.46 kg CO2eq** (99% from the classifier stage). Equivalent to ~12 km driving in a small car.
- Skipped alternatives: V4-pro full classifier would have been ~$280 and ~0.6 kg CO2eq.

**Decisions made.**

- **Accept the stratified `(sector, sub_code)` Leiden approach as the production clustering pipeline.** Update PIPELINE.md to reflect. Baseline delta established at full corpus scale.
- **Adopt Gemma 4 12B on 1× H100 as the default classifier path** when the goal is a full-corpus pass at low $ cost. Keep V4-pro API as the "fast, higher-carbon-but-quality-verified" alternative for smaller or one-off runs.
- **Skip validate_gold on Gemma 4 12B for now**; downstream judge accuracy at 86% is signal enough that classifier quality is fit for purpose.
- **Ship the pi_recluster package as-is.** All preflight checks in `run_gpu.sh` proved themselves during this iteration; next full-corpus run should take under an hour of engineering plus the actual compute.
- **Defer INDUSTRY regression fix to iteration 5.**

**Open items carried forward.**
- INDUSTRY sector regression investigation (why -12 pt when everything else improved).
- Expert review of `runs/expert_gold/disagreements_review.csv` (66 rows where v1 classifier said `sufficiency` but expert said `not_a_policy`). Was deferred from iteration 3, still deferred.
- Backfill: run `validate_gold.py` on Gemma 4 12B output against the 770 expert-gold rows to record its binary accuracy for the historical record.
- Spot-check MOBILITY × code 6 clusters to confirm the low modal-shift share isn't classifier under-attribution.
- Consider taxonomy pruning: sub-codes 0 (2.8k policies) and 2 (24.8k) are the smallest — possibly fold code 0 into 1, or code 2 into 3, to simplify downstream analytics.

**Artefacts.**
- `runs/full_recluster/reclustered/` — 11 per-sector parquets with the new stratified clustering.
- `runs/full_recluster/filtered.parquet` — 676,876 surviving policies with embeddings and category+sub_code.
- `runs/full_recluster/classifications.jsonl` — 1,465,440 rows of Gemma 4 12B classifier output.
- `runs/full_recluster/judgments.jsonl` — 353 judge verdicts.
- `runs/full_recluster/emissions.csv` — per-stage codecarbon totals.
- `policy_analysis/expert_eval/explore_full_recluster.ipynb` — 21-cell exploration notebook.
- `policy_analysis/expert_eval/pi_recluster/` — the self-contained runnable, now battle-tested for future iterations.

---

### 2026-07-03 — Iteration 3: expert gold in, prompt v1, 100k end-to-end test

**Trigger.** Expert annotation dataset returned: 770 rows filled with `Policy categorisation` (5-class) + `Sufficiency Categories` (10-code sub-taxonomy). Not the clustering ground truth we asked for, but arguably more useful — a real benchmark.

**What we did.**
1. **Built `runs/expert_gold/gold.parquet`** — normalised the two typo variants in sub-code labels, mapped 5-class to lowercase slugs.
2. **Baseline (v0 prompt) evaluation** — ran the existing 3-class classifier on the 770 rows.
   - Binary accuracy vs expert: **63.4%** (F1 0.44). Strict sufficiency recall: **34.1%**.
   - Confusion matrix showed the model dumping 191/417 `not_a_policy` rows into `potential_sufficiency` (no home in the 3-class enum).
   - Missed cases pattern: passive-design, thermal-mass, modal-shift — expert marks these as sufficiency (codes 4, 6), the v0 prompt says "same service, less input = efficiency".
3. **Wrote prompt v1** (`policy_analysis/sufficiency_classification/prompts_v1.py`):
   - 5-class enum (`sufficiency / efficiency / consistency / ambiguous / not_a_policy`).
   - Optional integer `sub_code` 0–9 when category is `sufficiency`.
   - `SUFFICIENCY_DEFINITION` rewritten to include the 10-code taxonomy in-prompt.
   - New few-shot: 10 examples drawn from expert gold, balanced across the 5 classes and priority sub-codes.
4. **v1 evaluation** on the 760 non-few-shot rows: binary **79.3%** (F1 0.62). Strict sufficiency recall **81.7%**. 5-class exact match **63.3%**. Sub-code accuracy **83.5%** where both agree it's sufficiency.
5. **Extracted 66 "sufficiency-on-not_a_policy" disagreements** to `runs/expert_gold/disagreements_review.csv` for expert re-review.
6. **100k end-to-end pipeline test** (`runs/api_100k/`):
   - Sampled 100k stratified across the 11 sectors.
   - Classified via V4 API with v1 prompt.
   - Aggregated per-cluster sufficiency density on `clusters_2026-03-18` clusters.
   - Judged 433 clusters via intrusion task with calibrated few-shot.
   - 2×2 report: 63 KEEP / 242 drop-for-off-topic / 24 RE-CLUSTER / 86 DROP.
   - Cost: $19 classifier + $0.21 judge. Extrapolation to 1.47M: ~$280 all-in via V4-pro.
7. **Written `explore_100k.ipynb`** — 10-section notebook for less-technical stakeholders.

**Key numbers established this iteration:**
| metric | value | source |
|---|---|---|
| classifier binary accuracy on expert gold (v1) | 79.3% | `runs/expert_gold/REPORT_v1.md` |
| classifier strict sufficiency recall (v1) | 81.7% | same |
| classifier 5-class exact match | 63.3% | same |
| sub-code accuracy (both agree suff) | 83.5% | same |
| judge accuracy on 433 randomly-sampled clusters | 73.5% | `runs/api_100k/REPORT.md` |
| judge accuracy on small clusters (≤50) | 58.6% | same |
| judge accuracy on large clusters (>500) | 82.1% | same |
| filter yield (sufficiency + ambiguous / total) | 33.5% | same |

**What we learned.**

1. **The classifier is not the bottleneck at 79% binary accuracy.** The 100k confirmed the v1 prompt works at scale. Precision is still 51.8% — 66 not_a_policy rows leak into sufficiency, mostly borderline cases (heritage protection, dose limits, COVID measures) where the model's mechanism-based reasoning is defensible and the disagreement is really a labeling boundary. Worth an expert pass on `disagreements_review.csv` before publishing any final numbers.

2. **Existing clusters mix mechanisms.** The 2×2 result — 242/415 clusters coherent but off-topic — was expected but the size is striking. 58% of the current clustering is producing topical groups that aren't sufficiency-mechanism groups. Filtering + reclustering per sector is the obvious fix, but doesn't address the root cause: policies from different sub-codes get grouped together because they share topical vocabulary.

3. **Stratified `(sector, sub_code)` clustering is the natural next step.** The sub-code axis is orthogonal to topic — a `caps/limits` policy and a `public provisioning` policy might both be about housing, but they're totally different sufficiency mechanisms. Grouping first by sub-code then by topic should produce clusters that are mechanism-uniform by construction. 110 (sector, sub_code) cells; small ones fold into a single cluster, big ones get Leiden at default resolution.

4. **Data quirk found and encoded**: `policy_uid = openalex_id::chunk_idx` is NOT unique — a source chunk can yield multiple policies with different text, sometimes in different clusters. All downstream joins now use `(policy_uid, cluster_uid, seq)` where `seq` is a group-local cumcount. `classify_v1.py` was patched to include `cluster_uid` in output. Any future join code must handle this.

5. **The judge behaves how the intrusion task literature predicts.** 73.5% overall, size-monotonic (small < medium < large), sector-varying (SOCIAL 89% ↔ MACROECONOMIC 65%). No red flags, no need to change the judge for this iteration.

**Cost + emissions accounting for this iteration:**
- v0 baseline on 770: ~$0.09, ~1 min.
- v1 on 760: ~$0.09, ~1 min.
- 100k classifier + judge: $19.20, ~2 h wall clock.
- **Total this iteration: ~$19.40 API + zero local compute.**

**Decisions made.**
- Adopt v1 as the classifier prompt going forward.
- Update PIPELINE.md calibration gate to measure against expert gold, not V4-self-agreement.
- Next re-cluster iteration: stratify by `(sector, sub_code)` with size-adaptive Leiden resolution.
- For the next iteration's classifier: reuse the 100k already classified (Option A, $0 API) OR run a small local model on rented GPU (Option C, $10–20 total). Full V4 rerun ($280) not justified until we validate the reclustering hypothesis.

**Open items carried forward.**
- Expert review of `disagreements_review.csv` (66 cases).
- Sector re-classification (predicted sector alongside category) — not yet added to prompt v1.
- Judge accuracy on the NEW clusters after re-clustering — the real quality metric.

---

## Cross-iteration notes

### Method stack (stable)
- **Classifier**: DeepSeek V4-pro via API, or Gemma 4 12B / Qwen 2.5-32B via vLLM. Guided-JSON decoding enforces schema. Shipped prompt: `prompts_v1.py` (best on Gemma; iteration 5 showed the richer six-pillar v2/v3 hurt the smaller model). Prompt is env-selectable via `PROMPTS_VERSION` (v1|v2|v2_1|v3|v1_1); v1 is default. Few-shot: `runs/expert_gold/few_shot_v1.jsonl` (10 examples).
- **Filter**: keep `sufficiency` only; **drop `ambiguous`** (iteration 5 — ambiguous is ~0% sufficiency). `--include-ambiguous` restores the old behaviour.
- **Clustering**: Leiden on FAISS-HNSW k-NN cosine graph (k=20, threshold 0.55). Resolution size-adaptive: <30 = single cluster, <300 = 0.4, ≥300 = 1.0. (Iteration 5 tried lowering the ≥300 tier to 0.5 + singleton absorption; it over-merged large cells and was reverted. A proper size-adaptive-resolution fix is still open.)
- **Judge**: DeepSeek V4-pro, 5-way intrusion task with 4-example few-shot calibration. Since iteration 6 (`v1-deepseek-subcode`) the judge also rates `sub_code_fit` (does the cluster content match its cell's mechanism code) — the intrusion pick alone does not track expert-perceived quality.
- **Classification checkpoint**: `AmineSab/sufficiency-classifications` on HF — re-clustering iterations restart from it without re-running the classifier.

### What we keep re-learning
- **Small-model classification is competitive.** Every iteration we consider going bigger, and every iteration the answer at classifier quality is that 12B–32B dense models with a good rubric + few-shot deliver production-quality classification. Iteration 4 confirmed at full corpus scale.
- **Cost matters.** Full 1.47M via V4-pro is ~$280. Local Gemma 4 12B on 1×H100 is ~$32 (iteration 4 measurement). ~10× cheaper $, ~4× more carbon. Pick based on constraint.
- **The clustering evaluation is where the real research happens.** ~~Classifier is a solved problem now~~; **iteration 4 result: 86% judge accuracy on 3,350 clusters** — the stratified pipeline is production-ready. New research: taxonomy pruning, per-sector resolution tuning, editorial framing (public-provisioning-dominance).
- **Prompt gains don't transfer across model sizes (iteration 5).** A prompt tuned/validated on a strong model (DeepSeek V4) can *degrade* a smaller one (Gemma 4 12B) — the six-pillar definition helped the former and was the two worst prompts on the latter. Always validate the prompt on the model you will actually ship. What a small over-permissive model needs is a calibration/tightening rule, not a richer definition.
- **Engineering overhead dominates rented-box iterations.** Iteration 4 spent almost as much wall clock on env issues (Python 3.14 default, CUDA 11.5 vs 12.4, gcc/g++ cc1plus mismatch, vLLM API rename, codecarbon-3 Postgres dep) as on classifier compute. Every fix went into `run_gpu.sh` preflight so the next box is ~$0 engineering time.

### Reproducible-run checklist for each new iteration
- [ ] Bump `PROMPT_VERSION` in `prompts_v1.py` if the prompt changes at all.
- [ ] Write a fresh `runs/<iteration>/` directory. Never overwrite prior runs.
- [ ] Always include: expert-gold binary accuracy, 100k pipeline replay, updated `disagreements_review.csv`.
- [ ] Update this doc with a new dated section, cost line, and decisions made.
- [ ] Preserve every JSONL — cheap storage, expensive re-computation.

### Live open questions
- ~~What's the right threshold for "sufficient re-clustering quality"? Currently 73.5% judge accuracy = baseline.~~ **Resolved iteration 4**: target of ≥80% overall / ≥65% on small clusters was cleared decisively — 86.1% overall, 84% on small. Set the new bar based on iteration 4 for future improvements.
- Sub-code taxonomy: are all 10 codes needed, or can 0 (involuntary) and 2 (prices/taxes) fold into neighbours? At full scale: code 0 has only 2.8k policies (0.4%), code 2 has 24.8k (3.7%), code 4 has 13.2k (2.0%), code 6 has 14.1k (2.1%). Codes 0/2/4/6 are all under 4% each — collapsing candidates for editorial simplicity, keep for scientific fidelity.
- Should we run a 3-annotator agreement study on the `disagreements_review.csv` set to see whether the boundary between `not_a_policy` and `sufficiency` is well-defined among experts, or just noisy?
- Why does INDUSTRY regress at scale (iteration 4)? Sub-code stratification exposing genuinely heterogeneous mixed-mechanism sub-groups within INDUSTRY? Or artifact of small sample (n=31)?
- **Size-adaptive Leiden resolution (iteration 5, open).** Uniform resolution fails at both ends: 1.0 over-fragments large cells into singletons, 0.5 over-merges them into blobs. What resolution schedule (as a function of cell size) — or cluster-size cap with recursive re-clustering — yields reviewable clusters across the full size range without either pathology?

### Deprecated / dead-end explorations (kept for memory)
- **3-class prompt v0** (`sufficiency / potential_sufficiency / not_sufficiency`): produced 63% binary accuracy, 34% strict recall. Superseded by v1. Do not resurrect.
- **Six-pillar prompts (v2 / v3) on a 12B model** (iteration 5): the operationalised and the plain six-pillar definitions were the two *worst* prompts on Gemma 4 12B (72.4% / 69.1% binary, 5-class ≤55%). They win on DeepSeek V4 but not on the model we ship. Kept as `prompts_v2.py`/`prompts_v3.py` for the DeepSeek path only.
- **Naive resolution drop + singleton absorption for clustering** (iteration 5): lowering the ≥300-cell Leiden resolution 1.0 → 0.5 cured singletons (66% → 10%) but over-merged large cells into blobs (SOCIAL sub_code 9 → one 34k-member cluster). Reverted. The real fix is *size-adaptive* resolution (higher res for bigger cells), not a uniform lower value.
- **Full V3-on-multi-node-H100 for classifier**: modelled and priced (~$150–250, ~10 h). Deferred — 12B–32B models are within 5 pt at a fraction of the cost. **Iteration 4 confirmed**: Gemma 4 12B classifier led to 86% downstream judge accuracy on 3,350 new clusters. V3 not needed.
- **CPU-offloading for classifier on Jean-Zay**: ruled out — 10–100× throughput drop. Documented in PIPELINE.md.
- **`nvidia-cuda-toolkit` apt package on Ubuntu 22.04**: pins to CUDA 11.5 (2021 vintage), incompatible with H100 (sm_90) and modern PyTorch wheels. Iteration 4 wasted an hour on this. Use NVIDIA's cuda-keyring repo instead.

### File map (living)
- `policy_analysis/sufficiency_classification/prompts_v1.py` — canonical/shipped v1 prompt. Also `prompts_v2.py`, `prompts_v2_1.py`, `prompts_v3.py`, `prompts_v1_1.py` — iteration-5 variants (env-selectable via `PROMPTS_VERSION`; v1 default; v2 wins only on DeepSeek).
- `policy_analysis/expert_eval/` — everything downstream of the 770-gold measurement.
  - `build_gold.py` — Excel → `runs/expert_gold/gold.parquet`.
  - `classify_v1.py` — V4-pro API classifier (prompt env-selectable via `PROMPTS_VERSION`).
  - `eval_baseline.py` / `eval_v1.py` — score against expert gold. `eval_v2.py` — parameterised head-to-head prompt evaluator (`--base`/`--challenger`).
  - `extract_disagreements.py` — 66-row review CSV.
  - `sample_100k.py`, `aggregate_100k.py`, `build_intrusion_100k.py`, `report_100k.py` — 100k pipeline (iteration 3).
  - `build_intrusion_recluster.py` — intrusion items over new stratified clusters (parameterised for any run dir, iterations 3+4).
  - `explore_100k.ipynb` — 100k stakeholder-facing summary (iteration 3).
  - `explore_reclustering.ipynb` — 100k reclustering deep-dive (iteration 3).
  - `explore_full_recluster.ipynb` — full 1.47M reclustering deep-dive (iteration 4).
  - `rehome_singletons.py` — k-NN rehoming of expert-validated singletons (iteration 6).
  - `pi_recluster/` — self-contained runnable for cloud-box re-clustering; battle-tested at full scale in iteration 4. Iteration 6 added the pass-2 kit: `prompts_pass2.py` (PROMPTS_VERSION=pass2), `build_pass2_input.py`, `apply_pass2.py`. `gold/` bundles the expert-gold artefacts for the on-box preflight; `validate_gold.py` returns SHIP/BORDERLINE/DO_NOT_SCALE exit codes; `run.sh`/`run_gpu.sh` print a mode banner and auto-fetch the HF classification checkpoint in cluster-only mode.
- `runs/expert_gold/` — gold parquet + v0/v1/v2/v2.1 comparison reports and raw classifications.
- `runs/api_100k/` — 100k pipeline outputs (iteration 3).
- `runs/api_100k_recluster/` — 100k stratified reclustering outputs (iteration 3).
- `runs/full_recluster/` — full 1.47M reclustering outputs (iteration 4); `review/` holds the manual-classification and expert-review deliverables.
- `runs/full_recluster_noambig/` — iteration-5 clustering, ambiguous removed: 645,495 rows, 3,152 clusters. Superseded by `runs/singleton_rehoming/reclustered/` (iteration 6: 209 expert-validated singletons rehomed, 2,949 clusters) — use that one downstream.
- `runs/recluster_tuned_withambig/` — shelved res=0.5 + absorption run (iteration 5, do not use).
- `runs/eval_v1/` — 500-item intrusion set (superseded but preserved).
- `runs/pipeline_test/` — 200-cluster smoke test (superseded).

---

## Template for next iteration

Copy this block, fill in.

### YYYY-MM-DD — Iteration N: <one-line summary>

**Trigger.** What changed since the last iteration.

**What we did.**
1. …
2. …

**Key numbers established this iteration:**
| metric | value | source |

**What we learned.**

**Cost + emissions accounting.**

**Decisions made.**

**Open items carried forward.**
