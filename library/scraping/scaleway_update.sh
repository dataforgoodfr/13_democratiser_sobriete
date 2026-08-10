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
