"""Classify policies with a local vLLM instance instead of the DeepSeek API.

Runs offline batch inference on `to_classify.jsonl`, emits `classifications.jsonl`
in the same schema as `expert_eval/classify_v1.py` so downstream stages are
unchanged. Guided JSON decoding enforces the v1 schema at the token level, so
schema-violation retries are unnecessary.

Defaults are tuned for a single H100 (80 GB) running Gemma 4 12B in BF16 with
prefix caching. Override with CLI flags for larger models / multi-GPU setups.

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

from _carbon import track

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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
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
                    help="fold system role into user turn (needed for Gemma variants that reject system)")
    args = ap.parse_args()

    # Late import so `--help` / dry runs work on machines without CUDA
    from vllm import LLM, SamplingParams

    # vLLM ≥ 0.11 renamed GuidedDecodingParams → StructuredOutputsParams and
    # the SamplingParams field guided_decoding → structured_outputs. Prior
    # versions kept the old names. We probe both.
    _sampling_extra: dict = {}
    try:
        from vllm.sampling_params import GuidedDecodingParams  # vLLM ≤ 0.10
        _sampling_extra["guided_decoding"] = GuidedDecodingParams(json=RESPONSE_JSON_SCHEMA)
    except ImportError:
        try:
            from vllm.sampling_params import StructuredOutputsParams  # vLLM ≥ 0.11
            _sampling_extra["structured_outputs"] = StructuredOutputsParams(json=RESPONSE_JSON_SCHEMA)
        except ImportError:
            print("[warn] no structured-output API found in this vLLM; "
                  "relying on prompt + JSON validation only. "
                  "Expect a higher failure rate on the classifier.")

    items = _load_jsonl(Path(args.items))
    exclude: set[str] = set()
    if args.exclude_ids:
        exclude = {
            l.strip() for l in Path(args.exclude_ids).read_text().splitlines() if l.strip()
        }
    items = [it for it in items if it["policy_uid"] not in exclude]

    done = _load_done(Path(args.out))
    todo = [it for it in items if (it["policy_uid"], it.get("cluster_uid")) not in done]
    print(f"{len(items):,} items after exclude, {len(done):,} done, {len(todo):,} to classify")

    if not todo:
        print("nothing to do")
        return

    few_shot = _load_jsonl(Path(args.few_shot)) if args.few_shot else None
    system_prompt = build_system_prompt(few_shot)
    if few_shot:
        print(f"injecting {len(few_shot)} few-shot examples into system prompt")

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

    sampling = SamplingParams(
        temperature=0,
        max_tokens=350,
        **_sampling_extra,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fails = 0
    t0 = time.time()
    with track("classify_vllm"), out_path.open("a") as fh:
        for start in range(0, len(todo), args.chunk_size):
            chunk = todo[start : start + args.chunk_size]
            messages_batch = [
                _build_messages(system_prompt, it, args.single_turn) for it in chunk
            ]
            outputs = llm.chat(messages_batch, sampling, use_tqdm=True)
            for it, out in zip(chunk, outputs):
                raw = out.outputs[0].text
                rec = _record_from_output(it, args.model, raw)
                if "error" in rec:
                    fails += 1
                fh.write(json.dumps(rec) + "\n")
            fh.flush()
            done_so_far = start + len(chunk)
            dt = time.time() - t0
            rate = done_so_far / dt if dt > 0 else 0
            eta_min = (len(todo) - done_so_far) / rate / 60 if rate > 0 else float("inf")
            print(
                f"  [{done_so_far:,}/{len(todo):,}]  "
                f"{rate:.1f} req/s  eta {eta_min:.1f} min  fails {fails}"
            )

    print(f"\ndone. total fails: {fails}. wall clock: {(time.time() - t0)/60:.1f} min")


if __name__ == "__main__":
    main()
