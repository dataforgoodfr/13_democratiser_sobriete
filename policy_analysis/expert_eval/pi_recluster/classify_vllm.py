"""Classify policies with a local vLLM instance instead of the DeepSeek API.

Runs offline batch inference on `to_classify.jsonl`, emits `classifications.jsonl`
in the same schema as `expert_eval/classify_v1.py` so downstream stages are
unchanged. Guided JSON decoding enforces the v1 schema at the token level, so
schema-violation retries are unnecessary.

Defaults are tuned for a single H100 (80 GB) running Gemma 4 12B in BF16 with
prefix caching. Override with CLI flags for larger models / multi-GPU setups.

Three execution modes:
  batch     --items + --out: the original single-worker flow, resumable via
            the append-mode output (a killed run loses at most one chunk).
  sharded   add --shard K --num-shards N: deterministic slice of the input
            (taken before resume filtering, so a shard's row subset is stable
            across restarts). Give each shard its own --out.
  queue     --queue-dir + --out-dir: pull fixed-size work units from an
            fsqueue (see fsqueue.py / build_queue.py) until it is drained.
            Any number of workers can run concurrently; the model is loaded
            once per worker. This is the Jean Zay mode (jz_classify.slurm).

Example:
    uv run --extra gpu python classify_vllm.py \
        --items data/to_classify.jsonl \
        --out   data/classifications_vllm.jsonl \
        --few-shot ../../../runs/expert_gold/few_shot_v1.jsonl \
        --model google/gemma-4-12B-it
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import fsqueue
from _carbon import track
from _events import emit

# The prompts_*.py files are local copies — the canonical source is
# policy_analysis/sufficiency_classification/prompts_*.py, kept in sync
# manually. Copied here so this package doesn't depend on the parent
# repo layout (which may not be fully committed when cloned on a fresh box).
# Prompt module is selectable via PROMPTS_VERSION (v1|v2|v2_1|v3); default v1 is
# the shipped prompt — it validated best on Gemma (the six-pillar v2/v3 variants
# hurt the smaller model; see RESEARCH_NOTES iteration 5).
_PROMPTS = importlib.import_module(f"prompts_{os.environ.get('PROMPTS_VERSION', 'v1')}")
CATEGORIES = _PROMPTS.CATEGORIES
PROMPT_VERSION = _PROMPTS.PROMPT_VERSION
RESPONSE_JSON_SCHEMA = _PROMPTS.RESPONSE_JSON_SCHEMA
build_system_prompt = _PROMPTS.build_system_prompt
format_user_prompt = _PROMPTS.format_user_prompt


def _load_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _load_done(path: Path) -> set[tuple[str, str | None]]:
    if not path.exists():
        return set()
    done: set[tuple[str, str | None]] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "error" in rec:
            continue
        uid = rec.get("policy_uid")
        cuid = rec.get("cluster_uid")
        if uid:
            done.add((uid, cuid))
    return done


def _build_messages(system_prompt: str, item: dict, single_turn: bool) -> list[dict]:
    """Return the messages payload for `llm.chat`.

    Some Gemma variants don't accept a `system` role; if `single_turn=True` we
    fold the system content into the user turn so the chat template stays happy.
    """
    user_content = format_user_prompt(item["sector"], item["policy_text"])
    if single_turn:
        return [{"role": "user", "content": system_prompt + "\n\n" + user_content}]
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def _record_from_output(item: dict, model_id: str, raw_text: str) -> dict:
    base = {
        "policy_uid": item["policy_uid"],
        "sector": item["sector"],
        "cluster_uid": item.get("cluster_uid"),
        "prompt_version": PROMPT_VERSION,
        "model": model_id,
    }
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return {**base, "error": f"JSONDecodeError: {e}", "raw": raw_text[:400]}

    if not isinstance(parsed, dict):
        return {**base, "error": f"not an object: {type(parsed).__name__}"}
    if parsed.get("category") not in CATEGORIES:
        return {**base, "error": f"category not in enum: {parsed.get('category')!r}"}
    return {**base, **parsed}


def _sampling_extra() -> dict:
    """Probe the structured-output API across vLLM versions.

    vLLM ≥ 0.11 renamed GuidedDecodingParams → StructuredOutputsParams and
    the SamplingParams field guided_decoding → structured_outputs. Prior
    versions kept the old names. We probe both.
    """
    try:
        from vllm.sampling_params import GuidedDecodingParams  # vLLM ≤ 0.10
        return {"guided_decoding": GuidedDecodingParams(json=RESPONSE_JSON_SCHEMA)}
    except ImportError:
        try:
            from vllm.sampling_params import StructuredOutputsParams  # vLLM ≥ 0.11
            return {"structured_outputs": StructuredOutputsParams(json=RESPONSE_JSON_SCHEMA)}
        except ImportError:
            print("[warn] no structured-output API found in this vLLM; "
                  "relying on prompt + JSON validation only. "
                  "Expect a higher failure rate on the classifier.")
            emit("structured_output_degraded")
            return {}


def _make_llm(args):
    from vllm import LLM, SamplingParams

    print(f"loading model {args.model} on TP={args.tensor_parallel_size} dtype={args.dtype}")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0, max_tokens=350, **_sampling_extra())
    return llm, sampling


def _generate_chunks(llm, sampling, system_prompt: str, items: list[dict], args):
    """Yield lists of result records, one list per --chunk-size generate() call."""
    for start in range(0, len(items), args.chunk_size):
        chunk = items[start : start + args.chunk_size]
        messages_batch = [
            _build_messages(system_prompt, it, args.single_turn) for it in chunk
        ]
        outputs = llm.chat(messages_batch, sampling, use_tqdm=True)
        yield [
            _record_from_output(it, args.model, out.outputs[0].text)
            for it, out in zip(chunk, outputs)
        ]


def _run_batch(args, exclude: set[str], system_prompt: str) -> None:
    items = _load_jsonl(Path(args.items))
    items = [it for it in items if it["policy_uid"] not in exclude]
    if args.num_shards > 1:
        # Slice before resume filtering so a shard's subset is stable across
        # restarts regardless of what is already in --out.
        items = items[args.shard :: args.num_shards]

    done = _load_done(Path(args.out))
    todo = [it for it in items if (it["policy_uid"], it.get("cluster_uid")) not in done]
    prefix = f"[shard {args.shard}/{args.num_shards}] " if args.num_shards > 1 else ""
    print(f"{prefix}{len(items):,} items after exclude, "
          f"{len(done):,} done, {len(todo):,} to classify")
    emit("run_start", mode="batch", shard=args.shard, num_shards=args.num_shards,
         items=len(items), already_done=len(done), todo=len(todo))

    if not todo:
        print("nothing to do")
        emit("run_end", mode="batch", shard=args.shard, classified=0, fails=0)
        return

    llm, sampling = _make_llm(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fails = 0
    done_so_far = 0
    t0 = time.time()
    with track("classify_vllm"), out_path.open("a") as fh:
        for records in _generate_chunks(llm, sampling, system_prompt, todo, args):
            for rec in records:
                if "error" in rec:
                    fails += 1
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done_so_far += len(records)
            dt = time.time() - t0
            rate = done_so_far / dt if dt > 0 else 0
            eta_min = (len(todo) - done_so_far) / rate / 60 if rate > 0 else None
            eta_txt = f"{eta_min:.1f}" if eta_min is not None else "?"
            print(
                f"  {prefix}[{done_so_far:,}/{len(todo):,}]  "
                f"{rate:.1f} req/s  eta {eta_txt} min  fails {fails}"
            )
            emit("chunk_done", mode="batch", shard=args.shard, done=done_so_far,
                 total=len(todo), rate_req_s=round(rate, 2),
                 eta_min=round(eta_min, 1) if eta_min is not None else None, fails=fails)

    wall_min = (time.time() - t0) / 60
    print(f"\ndone. total fails: {fails}. wall clock: {wall_min:.1f} min")
    emit("run_end", mode="batch", shard=args.shard, classified=done_so_far,
         fails=fails, wall_min=round(wall_min, 1))


def _run_queue(args, exclude: set[str], system_prompt: str) -> None:
    root = Path(args.queue_dir)
    fsqueue.init(root)
    owner = "_".join(str(x) for x in (
        os.environ.get("SLURM_JOB_ID", "local"),
        os.environ.get("SLURM_ARRAY_TASK_ID", 0),
        os.getpid(),
    ))

    # Claim before paying the model-load cost: late workers on a drained
    # queue exit in seconds.
    unit = fsqueue.claim(root, owner)
    if unit is None:
        print("queue empty — nothing to do")
        emit("run_end", mode="queue", owner=owner, units=0, fails=0)
        return

    llm, sampling = _make_llm(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    units = 0
    fails = 0
    t0 = time.time()
    emit("run_start", mode="queue", owner=owner, **fsqueue.stats(root))
    with track("classify_vllm"):
        while unit is not None:
            name = fsqueue.unit_name(unit)
            items = [it for it in _load_jsonl(unit) if it["policy_uid"] not in exclude]
            out_path = out_dir / name
            tmp = out_dir / f".{name}.tmp"
            unit_fails = 0
            with tmp.open("w") as fh:
                for records in _generate_chunks(llm, sampling, system_prompt, items, args):
                    for rec in records:
                        if "error" in rec:
                            unit_fails += 1
                        fh.write(json.dumps(rec) + "\n")
                    fsqueue.heartbeat(unit)  # keep the claim from looking stale
            os.replace(tmp, out_path)
            fsqueue.mark_done(root, unit)
            units += 1
            fails += unit_fails
            dt = time.time() - t0
            print(f"  unit {name}: {len(items):,} items, fails {unit_fails}  "
                  f"({units} unit(s), {dt / 60:.1f} min)")
            emit("unit_done", unit=name, owner=owner, items=len(items),
                 fails=unit_fails, units_done=units, **fsqueue.stats(root))
            unit = fsqueue.claim(root, owner)

    wall_min = (time.time() - t0) / 60
    print(f"\nqueue drained for this worker. units: {units}, fails: {fails}, "
          f"wall clock: {wall_min:.1f} min")
    emit("run_end", mode="queue", owner=owner, units=units, fails=fails,
         wall_min=round(wall_min, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=None, help="input JSONL (batch/sharded mode)")
    ap.add_argument("--out", default=None, help="output JSONL (batch/sharded mode)")
    ap.add_argument("--queue-dir", default=None,
                    help="fsqueue root (queue mode; see build_queue.py)")
    ap.add_argument("--out-dir", default=None,
                    help="per-unit output directory (queue mode)")
    ap.add_argument("--shard", type=int, default=0,
                    help="this worker's shard index (with --num-shards)")
    ap.add_argument("--num-shards", type=int, default=1,
                    help="deterministic input split for SLURM arrays; "
                         "give each shard its own --out")
    ap.add_argument("--few-shot", default=None)
    ap.add_argument("--exclude-ids", default=None,
                    help="text file of policy_uid to skip (for gold-set eval)")
    ap.add_argument("--model", default="google/gemma-4-12B-it")
    ap.add_argument("--dtype", default="bfloat16",
                    help="bfloat16 (H100 native) | float16 | float8_e4m3fn")
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="max context. Our prompt+output is <2k; we set 4k for headroom")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--chunk-size", type=int, default=8_000,
                    help="items per generate() call — trades throughput for flush frequency")
    ap.add_argument("--single-turn", action="store_true",
                    help="fold system role into user turn (needed for Gemma variants "
                         "that reject system)")
    args = ap.parse_args()

    if bool(args.queue_dir) == bool(args.items):
        ap.error("exactly one of --items or --queue-dir is required")
    if args.queue_dir and not args.out_dir:
        ap.error("--out-dir is required with --queue-dir")
    if args.items and not args.out:
        ap.error("--out is required with --items")
    if not 0 <= args.shard < args.num_shards:
        ap.error("--shard must be in [0, --num-shards)")

    exclude: set[str] = set()
    if args.exclude_ids:
        exclude = {
            l.strip() for l in Path(args.exclude_ids).read_text().splitlines() if l.strip()
        }

    few_shot = _load_jsonl(Path(args.few_shot)) if args.few_shot else None
    system_prompt = build_system_prompt(few_shot)
    if few_shot:
        print(f"injecting {len(few_shot)} few-shot examples into system prompt")

    if args.queue_dir:
        _run_queue(args, exclude, system_prompt)
    else:
        _run_batch(args, exclude, system_prompt)


if __name__ == "__main__":
    main()
