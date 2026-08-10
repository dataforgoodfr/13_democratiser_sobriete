"""Policy extraction with vLLM as an idempotent queue/batch worker.

Successor to `policy_analysis/policy_extraction/vllm_policy_extraction.py`
(same prompt, same Mistral model, same output fields) with the fixed-input /
duplicate-on-restart caveats removed: work comes from an fsqueue or a
sharded items file, outputs are written tmp-then-rename per unit (queue
mode) or append-with-resume (batch mode), and key events are emitted for
monitoring.

Items (see build_extract_input.py): {"policy_uid": "<openalex_id>::<chunk_idx>",
"openalex_id", "chunk_idx", "text"}. Output records keep policy_uid so
build_queue.py's done-detection works unchanged; failures carry an "error"
key and are retried on the next queue rebuild.

Example (queue mode, what jz_extract.slurm runs):
    python extract_vllm.py --queue-dir $ROOT/extract/queue \
        --out-dir $ROOT/extract/outputs \
        --prompt ../../policy_extraction/POLICIES_EXTRACTION_PROMPT.txt \
        --model $DSDIR/HuggingFace_Models/mistralai/Mistral-Small-3.2-24B-Instruct-2506
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import fsqueue
from _carbon import track
from _events import emit
from classify_vllm import _load_done, _load_jsonl

EXTRACTION_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "contains_policies": {"type": "boolean"},
        "policies": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["contains_policies", "policies"],
    "additionalProperties": False,
}


def _record_from_output(item: dict, model_id: str, raw_text: str) -> dict:
    base = {
        "policy_uid": item["policy_uid"],
        "openalex_id": item.get("openalex_id"),
        "chunk_idx": item.get("chunk_idx"),
        "model": model_id,
    }
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        return {**base, "error": f"JSONDecodeError: {e}", "raw": raw_text[:400]}
    if not isinstance(parsed, dict) or not isinstance(parsed.get("policies"), list) \
            or not isinstance(parsed.get("contains_policies"), bool):
        return {**base, "error": f"schema mismatch: {str(parsed)[:200]}"}
    return {**base,
            "contains_policies": parsed["contains_policies"],
            "policies": [p for p in parsed["policies"] if isinstance(p, str)]}


def _make_llm(args):
    from vllm import LLM, SamplingParams

    extra: dict = {}
    try:
        from vllm.sampling_params import StructuredOutputsParams  # vLLM ≥ 0.11
        extra["structured_outputs"] = StructuredOutputsParams(json=EXTRACTION_JSON_SCHEMA)
    except ImportError:
        try:
            from vllm.sampling_params import GuidedDecodingParams  # vLLM ≤ 0.10
            extra["guided_decoding"] = GuidedDecodingParams(json=EXTRACTION_JSON_SCHEMA)
        except ImportError:
            print("[warn] no structured-output API in this vLLM; expect parse failures")
            emit("structured_output_degraded")

    kwargs: dict = {}
    if "mistral" in str(args.model).lower():
        kwargs = {"tokenizer_mode": "mistral", "config_format": "mistral",
                  "load_format": "mistral"}
    print(f"loading model {args.model} on TP={args.tensor_parallel_size}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=True,
        **kwargs,
    )
    sampling = SamplingParams(temperature=0.01, max_tokens=1024, **extra)
    return llm, sampling


def _generate_chunks(llm, sampling, prompt: str, items: list[dict], args):
    for start in range(0, len(items), args.chunk_size):
        chunk = items[start : start + args.chunk_size]
        messages = [
            [{"role": "system", "content": prompt},
             {"role": "user", "content": it["text"]}]
            for it in chunk
        ]
        outputs = llm.chat(messages, sampling, use_tqdm=True)
        yield [
            _record_from_output(it, args.model, out.outputs[0].text)
            for it, out in zip(chunk, outputs)
        ]


def _run_batch(args, prompt: str) -> None:
    items = _load_jsonl(Path(args.items))
    if args.num_shards > 1:
        items = items[args.shard :: args.num_shards]
    done = _load_done(Path(args.out))
    todo = [it for it in items if (it["policy_uid"], it.get("cluster_uid")) not in done]
    print(f"{len(items):,} items, {len(done):,} done, {len(todo):,} to extract")
    emit("run_start", mode="batch", stage="extract", shard=args.shard,
         items=len(items), already_done=len(done), todo=len(todo))
    if not todo:
        print("nothing to do")
        emit("run_end", mode="batch", stage="extract", extracted=0, fails=0)
        return

    llm, sampling = _make_llm(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fails = done_so_far = 0
    t0 = time.time()
    with track("extract_vllm"), out_path.open("a") as fh:
        for records in _generate_chunks(llm, sampling, prompt, todo, args):
            for rec in records:
                if "error" in rec:
                    fails += 1
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done_so_far += len(records)
            dt = time.time() - t0
            rate = done_so_far / dt if dt > 0 else 0
            print(f"  [{done_so_far:,}/{len(todo):,}]  {rate:.1f} req/s  fails {fails}")
            emit("chunk_done", stage="extract", done=done_so_far, total=len(todo),
                 rate_req_s=round(rate, 2), fails=fails)
    emit("run_end", mode="batch", stage="extract", extracted=done_so_far, fails=fails,
         wall_min=round((time.time() - t0) / 60, 1))


def _run_queue(args, prompt: str) -> None:
    root = Path(args.queue_dir)
    fsqueue.init(root)
    owner = "_".join(str(x) for x in (
        os.environ.get("SLURM_JOB_ID", "local"),
        os.environ.get("SLURM_ARRAY_TASK_ID", 0), os.getpid()))
    unit = fsqueue.claim(root, owner)
    if unit is None:
        print("queue empty — nothing to do")
        emit("run_end", mode="queue", stage="extract", owner=owner, units=0, fails=0)
        return

    llm, sampling = _make_llm(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    units = fails = 0
    t0 = time.time()
    emit("run_start", mode="queue", stage="extract", owner=owner, **fsqueue.stats(root))
    with track("extract_vllm"):
        while unit is not None:
            name = fsqueue.unit_name(unit)
            items = _load_jsonl(unit)
            tmp = out_dir / f".{name}.tmp"
            unit_fails = 0
            with tmp.open("w") as fh:
                for records in _generate_chunks(llm, sampling, prompt, items, args):
                    for rec in records:
                        if "error" in rec:
                            unit_fails += 1
                        fh.write(json.dumps(rec) + "\n")
                    fsqueue.heartbeat(unit)
            os.replace(tmp, out_dir / name)
            fsqueue.mark_done(root, unit)
            units += 1
            fails += unit_fails
            print(f"  unit {name}: {len(items):,} items, fails {unit_fails}  "
                  f"({units} unit(s), {(time.time() - t0) / 60:.1f} min)")
            emit("unit_done", stage="extract", unit=name, owner=owner,
                 items=len(items), fails=unit_fails, **fsqueue.stats(root))
            unit = fsqueue.claim(root, owner)
    emit("run_end", mode="queue", stage="extract", owner=owner, units=units,
         fails=fails, wall_min=round((time.time() - t0) / 60, 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--queue-dir", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--prompt", required=True,
                    help="prompt file (POLICIES_EXTRACTION_PROMPT.txt)")
    ap.add_argument("--model",
                    default="mistralai/Mistral-Small-3.2-24B-Instruct-2506")
    ap.add_argument("--tensor-parallel-size", type=int,
                    default=len(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")))
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--chunk-size", type=int, default=8_000)
    args = ap.parse_args()

    if bool(args.queue_dir) == bool(args.items):
        ap.error("exactly one of --items or --queue-dir is required")
    if args.queue_dir and not args.out_dir:
        ap.error("--out-dir is required with --queue-dir")
    if args.items and not args.out:
        ap.error("--out is required with --items")
    if not 0 <= args.shard < args.num_shards:
        ap.error("--shard must be in [0, --num-shards)")

    prompt = Path(args.prompt).read_text()
    if args.queue_dir:
        _run_queue(args, prompt)
    else:
        _run_batch(args, prompt)


if __name__ == "__main__":
    main()
