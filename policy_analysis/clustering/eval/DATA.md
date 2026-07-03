# Evaluation data — locked schema (v0)

Source: HuggingFace dataset `sufficiencylab/sufficiency-library`, snapshot dated 2026-03-18 (per-sector clustering outputs from Leiden).

## Local layout

```
~/data/wsl_sufficiency_eval/clusters_2026-03-18/
  BUILDING_clustered_policies_with_representatives_2026-03-18.parquet
  BUILDING_cluster_representatives_2026-03-18.csv
  ENERGY_clustered_policies_with_representatives_2026-03-18.parquet
  ENERGY_cluster_representatives_2026-03-18.csv
  ... (11 sectors total)
```

11 sectors: `BUILDING, ENERGY, FOOD, INDUSTRY, LOGISTICS, MACROECONOMIC, MATERIALS, MOBILITY, NATURE, SOCIAL, URBAN`. Total on-disk: ~92 MB.

To re-download, run `/tmp/download_hf_clusters.py` or invoke `huggingface_hub.hf_hub_download` against the 22 file names above.

## Parquet schema (per sector)

| column | dtype | notes |
|---|---|---|
| `openalex_id` | str | OpenAlex paper ID the policy text was extracted from. |
| `chunk_idx` | int64 | Chunk index inside that paper. |
| `policy_text` | str | The policy snippet — the unit being clustered. |
| `cluster_id` | int64 | Leiden cluster label, **scoped to the sector** (not globally unique). |
| `representative` | bool | Exactly one `True` row per cluster — the medoid. |

Verified across BUILDING / FOOD / LOGISTICS / URBAN: identical schema, no extra columns, no embedding column.

Observed sector sizes (row count, n_clusters): FOOD 259960/94, URBAN 120788/87, SOCIAL ~150k/?, LOGISTICS 9132/134. Cluster sizes are post-orphan-drop (min ≥ 3 everywhere).

## CSV schema (per sector)

| column | dtype | notes |
|---|---|---|
| `cluster_id` | int64 | Matches parquet's `cluster_id`. |
| `policy_text` | str | Medoid text. Same as the parquet row where `representative=True`. |
| `count` | int64 | Number of members in the cluster. |

Redundant with the `representative=True` rows of the parquet; useful as a quick standalone reference per cluster.

## Embeddings

**Not present in these files.** The original Qwen3-4B embeddings used to build the clustering live in `embeddings_policies_Qwen3-4B_2026-03-05.parquet` on HF (7.15 GB) and are not downloaded.

Consequences for v0 evaluation:
- Cluster-size distribution → computed from the parquet (no embeddings needed).
- Silhouette / medoid coherence → **deferred**, would need the 7.15 GB file or local re-embedding.
- Intrusion-task intruder selection → uses **random sampling from a different cluster within the same sector** rather than the nearest different cluster.

## Sector boundary

Sector is implicit in the file name, not a column in the data. The eval loader will inject a `sector` column when it concatenates per-sector frames. Cluster IDs are **only unique within a sector** — globally unique IDs are produced as `f"{sector}__{cluster_id}"`.

## LLM judge

DeepSeek V4 via the OpenAI-compatible endpoint at `https://api.deepseek.com`.
Env vars required (loaded automatically from `.env` via `python-dotenv`):

- `DEEPSEEK_API_KEY` — required.
- `DEEPSEEK_BASE_URL` — optional, defaults to `https://api.deepseek.com`.
- `DEEPSEEK_MODEL` — optional, defaults to `deepseek-v4-pro`; set to
  `deepseek-v4-flash` for the cheaper/faster tier.

DeepSeek's API only supports `response_format={"type": "json_object"}`. The
JSON schema lives in `prompts.py` and is enforced **locally** by
`llm_judge.validate_response` (no `jsonschema` dependency). The judge retries
once with a corrective follow-up on validation failure, then records the
error in the output line.

**Important:** V4 defaults to *thinking* mode, which consumes the token
budget on hidden reasoning and returns empty JSON content. The client passes
`extra_body={"thinking": {"type": "disabled"}}` to force non-thinking mode.
Don't remove this without testing — empty-response rate goes from ~0% to
>90% with thinking enabled.

Resume behaviour: re-running against an existing `judgments.jsonl` skips
cluster_uids with a successful prior judgment, but **does retry** errored
items. The aggregator deduplicates by keeping the latest record per uid.

## Running with uv

The repo root `pyproject.toml` lists the eval module's deps and is
configured `package = false` so `uv sync` only installs them — no build
step. From the repo root:

```
uv sync
uv run python -m policy_analysis.clustering.eval.run \
    --data ~/data/wsl_sufficiency_eval/clusters_2026-03-18 \
    --out runs/eval_v0 --n-items 50 --seed 0
```

Add `--skip-judge` for a no-API-cost dry run, or `--model deepseek-v4-flash`
to swap models.

## Open follow-ups for later iterations

- Decide whether to download `embeddings_policies_Qwen3-4B_2026-03-05.parquet` for embedding-based intrinsic metrics, or re-embed locally with the smaller Qwen3-0.6B model used elsewhere in the project (`policy_analysis/compute_embeddings.py`).
- Replace the random-intruder sampling with nearest-different-cluster sampling once embeddings are available.
- If schema-validation retries become common, consider using DeepSeek's tool-call strict mode (function calling supports strict JSON schemas) instead of the json_object + local-validate loop.
