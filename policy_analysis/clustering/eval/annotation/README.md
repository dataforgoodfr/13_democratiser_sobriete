# Expert annotation — ground-truth groupings (v0)

## Goal

We are evaluating an automatic clustering of ~1.47M policy snippets into 11 sector-specific cluster sets. To do that we need a small, expert-labelled gold set we can compare any clustering against. This folder holds that work.

## What's in `to_annotate/`

- **`{SECTOR}.csv`** — 11 files (one per sector), 70 rows each (50 stratified + 20 random).
- **`all_sectors.xlsx`** — same data, one sheet per sector, plus an `_instructions` sheet.

Columns:

| column | meaning |
|---|---|
| `policy_id` | row identifier — keep stable. |
| `sector` | the sector the row was routed to. |
| `source_cluster_id` | the current (possibly wrong) Leiden cluster — **ignore while annotating**. |
| `within_cluster_role` | `medoid` or `periphery` from the current clustering — informational. |
| `sample_kind` | `stratified` (drawn by cluster-size bucket) or `random` (uniform). Treat the same. |
| `policy_text` | the policy snippet to read. |
| `in_sector` | **empty — fill this in (task 1).** `y` = belongs to this sector, `n` = mis-routed. |
| `group_id` | **empty — fill this in (task 2).** |

## Your task

Two passes, per sector, using only `policy_text`.

### Pass 1 — sector check (`in_sector`)

For every row, decide whether it genuinely belongs to this sector or was mis-routed by the upstream classifier:
- `y` — yes, fits the sector.
- `n` — no, wrong sector (e.g. a transport policy that landed in FOOD).

Skip `n`-marked rows for pass 2.

### Pass 2 — topical grouping (`group_id`)

Among the `y` rows, group by topical similarity:
- Put rows that describe the **same topic / same kind of policy** into the **same `group_id`**.
- Use any label scheme — short tags, numbers, anything. Only **equality** between rows matters; we do not read the label content.
- Aim for groups of 2–8 rows. **Singletons are fine** for isolated policies.
- If a row genuinely does not fit any group, give it a unique label or write `unsure`.
- The rows are shuffled deliberately. Re-order freely in your spreadsheet.
- Do **not** open the existing clustering or peek at `source_cluster_id` before grouping — we want your independent judgement.

## Why these 70 / sector

- **50 stratified** by current cluster size (small ≤50, medium 51–500, large >500), capped at 2 rows per existing cluster, mixing medoids and periphery — for the grouping task. Keeps your judgement decoupled from the current clustering: changing the clustering does not invalidate your labels, because you grouped policies, not clusters.
- **20 random** uniformly drawn from the sector — to surface wrong-sector items the stratified sample might miss.

## When you're done

Save back as either updated CSVs or the same Excel workbook. We will:
- Compute the wrong-sector rate per sector from `in_sector`.
- Compute Adjusted Rand Index / Pair-F1 of any clustering against your `group_id` (restricted to `in_sector=y` rows).
