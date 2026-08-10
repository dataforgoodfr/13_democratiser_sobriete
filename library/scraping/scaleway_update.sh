#!/bin/bash
# Corpus update orchestration for the Scaleway scraping box.
#
# Jean Zay compute nodes have no internet, so everything network-bound runs
# here and moves over rsync through the JZ login node. The loop is:
#
#   1. [here]  stage-1 OpenAlex refresh (library/prescreening/stage1 scripts)
#   2. [here]  push-abstracts   → stage-1 parquets to JZ
#   3. [JZ]    jz_prescreen.slurm (Gemma prescreener) → keep-list
#   4. [here]  pull-keep-list   ← ids worth scraping
#   5. [here]  scrape           → PDFs + text extraction (Postgres queue resume)
#   6. [here]  push-corpus      → processed parquet(s) to JZ
#   7. [JZ]    extraction/classification jobs (jz_*.slurm)
#
# Key events go to $EVENTS_DIR as JSONL — same format as pi_recluster's
# _events.py, so logfire_relay.py ships them (this box HAS internet: run
#   python logfire_relay.py --events-dir "$EVENTS_DIR" --watch 300
# alongside, with LOGFIRE_TOKEN set).
#
# Required env: JZ (ssh alias for a Jean Zay login node), JZ_ROOT (absolute
# $WORK/pi_recluster path on JZ — remote $WORK can't be resolved from here).
set -euo pipefail

EVENTS_DIR=${EVENTS_DIR:-events}
STAGE1_DIR=${STAGE1_DIR:-../prescreening/stage1/output}   # stage-1 chunk parquets
KEEP_LIST=${KEEP_LIST:-keep_list.txt}
CORPUS_OUT=${CORPUS_OUT:-outputs}                          # processed_texts parquets

emit() { # emit <event> [key=value ...]
    local ev=$1 kv="" a
    shift
    for a in "$@"; do kv+=", \"${a%%=*}\": \"${a#*=}\""; done
    mkdir -p "$EVENTS_DIR"
    printf '{"ts": %s, "event": "%s", "host": "%s"%s}\n' \
        "$(date +%s)" "$ev" "$(hostname)" "$kv" >> "$EVENTS_DIR/scaleway_$$.jsonl"
}

need_jz() { : "${JZ:?set JZ to the ssh alias of a Jean Zay login node}"
            : "${JZ_ROOT:?set JZ_ROOT to the absolute \$WORK/pi_recluster path on JZ}"; }

cmd=${1:-help}
case "$cmd" in
crawl)
    # Delta crawl of OpenAlex into a fresh campaign dir (stage-1 SQLite dbs
    # are cwd-relative, so a fresh dir = a fresh crawl; theme-level resume
    # still works inside it). env:
    #   CAMPAIGN_DIR      fresh working dir, e.g. ~/crawl_2026-08
    #   OPENALEX_FROM_DATE=YYYY-MM-DD   delta window (publication date)
    #   PREV_IDS          previous ids export to diff against (optional)
    # Run inside the library venv (needs `library` importable + pyalex).
    : "${CAMPAIGN_DIR:?set CAMPAIGN_DIR to a fresh working dir for this crawl}"
    : "${OPENALEX_FROM_DATE:?set OPENALEX_FROM_DATE=YYYY-MM-DD}"
    STAGE1=$(cd "$(dirname "$0")/../prescreening/stage1" && pwd)
    mkdir -p "$CAMPAIGN_DIR"
    cd "$CAMPAIGN_DIR"
    cp -n "$STAGE1/sufficiency_keywords_regrouped_count.csv" . || true
    emit stage_start stage=crawl_ids from_date="$OPENALEX_FROM_DATE"
    python "$STAGE1/get_openalex_ids.py"
    emit stage_end stage=crawl_ids
    python - "${PREV_IDS:-}" <<'PYEOF'
import sqlite3, sys
prev = set()
if len(sys.argv) > 1 and sys.argv[1]:
    prev = {l.strip() for l in open(sys.argv[1]) if l.strip()}
ids = [r[0] for r in sqlite3.connect("openalex_ids.db").execute("SELECT id FROM works")]
delta = [i for i in ids if i not in prev]
open("openalex_ids.txt", "w").write("\n".join(delta) + "\n")
print(f"{len(ids):,} crawled, {len(prev):,} previous, {len(delta):,} delta ids")
PYEOF
    emit delta_computed ids_file=openalex_ids.txt
    emit stage_start stage=fetch_works
    python "$STAGE1/get_works_from_ids.py"
    emit stage_end stage=fetch_works
    emit stage_start stage=export_parquet
    mkdir -p outputs
    python "$STAGE1/save_to_parquet.py"
    emit stage_end stage=export_parquet
    echo "delta parquets in $CAMPAIGN_DIR/outputs — next: STAGE1_DIR=$CAMPAIGN_DIR/outputs $0 push-abstracts"
    ;;
push-abstracts)
    need_jz
    emit transfer_start what=abstracts dest="$JZ:$JZ_ROOT/prescreen/stage1"
    rsync -av --partial "$STAGE1_DIR"/chunk_*.parquet "$JZ:$JZ_ROOT/prescreen/stage1/"
    emit transfer_end what=abstracts
    echo "next (JZ login node): build_prescreen_input.py + build_queue.py, then"
    echo "  sbatch jz_prescreen.slurm    (see its header)"
    ;;
pull-keep-list)
    need_jz
    emit transfer_start what=keep_list
    rsync -av "$JZ:$JZ_ROOT/prescreen/$KEEP_LIST" .
    emit transfer_end what=keep_list ids="$(wc -l < "$KEEP_LIST" | tr -d ' ')"
    ;;
scrape)
    # Assumes the Postgres scraping queue is already populated for the new
    # ids (create_scraping_queue.py — adapt its hardcoded LIBRARY_KEY to the
    # keep-list before running). Both steps resume from queue state.
    emit stage_start stage=download_pdfs
    python download_all_pdfs.py --max-workers "${SCRAPE_WORKERS:-16}"
    emit stage_end stage=download_pdfs
    emit stage_start stage=extract_text
    python extract_text_from_pdfs.py --num-workers "${EXTRACT_WORKERS:-8}"
    emit stage_end stage=extract_text
    emit stage_start stage=extract_sections
    python extract_sections_from_raw_text.py
    emit stage_end stage=extract_sections
    ;;
push-corpus)
    need_jz
    emit transfer_start what=corpus dest="$JZ:$JZ_ROOT/corpus"
    rsync -av --partial "$CORPUS_OUT"/*.parquet "$JZ:$JZ_ROOT/corpus/"
    emit transfer_end what=corpus
    echo "next (JZ): chunk/embed, then jz_policy_extraction.sh → gold gate → classify"
    ;;
*)
    sed -n '2,20p' "$0"
    echo "usage: $0 {push-abstracts|pull-keep-list|scrape|push-corpus}"
    ;;
esac
