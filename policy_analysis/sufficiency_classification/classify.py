"""Sufficiency classifier — vLLM offline batch mode for Jean-Zay A100 nodes.

Reads the per-sector clustered parquets, classifies every policy text into
one of three sufficiency classes, writes results to a per-shard JSONL.

Designed to run as a SLURM array job (see run.slurm). One process per GPU,
shard index passed via ``--shard``. Inside the process, vLLM batches
prompts dynamically and reuses the system-prompt prefix via its automatic
prefix-cache, so per-call cost is dominated by the (short) per-policy
suffix and the JSON output.

Defaults to ``deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`` which fits BF16 on
a single A100 80GB with ~30 GB of headroom for KV cache. Swap via ``--model``
(e.g. ``--model deepseek-ai/DeepSeek-V2-Lite-Chat`` for a 16B MoE that's
roughly 3-5x faster; or ``--model deepseek-ai/DeepSeek-R1-Distill-Llama-70B
--quantization fp8`` for stronger but tighter-fitting weights).

The same prompt module is reused by the test-on-DeepSeek-API path
(``test_api.py``, optional) so prompt + schema are version-controlled in
one place.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

from policy_analysis.clustering.eval.loader import SECTORS, load_clusters
from policy_analysis.sufficiency_classification.prompts import (
    PROMPT_VERSION,
    RESPONSE_JSON_SCHEMA,
    SYSTEM_PROMPT,
    format_user_prompt,
)

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
DEFAULT_DATA_ROOT = "~/data/wsl_sufficiency_eval/clusters_2026-03-18"
DEFAULT_OUT_DIR = "runs/sufficiency_classify"


# --------------------------------------------------------------------- input

def _policy_uid(row: pd.Series) -> str:
    """Stable identifier per policy row: openalex_id + chunk_idx is unique."""
    return f"{row['openalex_id']}::{int(row['chunk_idx'])}"


def load_inputs(data_root: str, shard: int, num_shards: int) -> pd.DataFrame:
    df = load_clusters(os.path.expanduser(data_root))
    df["policy_uid"] = df.apply(_policy_uid, axis=1)
    # Shard by a hash of policy_uid so rebalancing is deterministic and even.
    # We use Python's hash with a salt-as-numshards modulus — adequate for an
    # offline workload and reproducible because we seed via num_shards.
    df["_shard"] = df["policy_uid"].apply(lambda s: hash((num_shards, s)) % num_shards)
    return df[df["_shard"] == shard].drop(columns=["_shard"]).reset_index(drop=True)


def load_done(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    done: set[str] = set()
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in rec:
                continue
            uid = rec.get("policy_uid")
            if uid:
                done.add(uid)
    return done


# --------------------------------------------------------------------- vLLM

def build_llm(args):
    # Lazy import — vLLM is heavy and not installed on dev laptops.
    from vllm import LLM
    kwargs = dict(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,  # cache the shared system prompt
        trust_remote_code=True,      # DeepSeek models ship custom code
    )
    if args.quantization:
        kwargs["quantization"] = args.quantization
    return LLM(**kwargs)


def make_sampling_params(args):
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams
    guided = GuidedDecodingParams(json=RESPONSE_JSON_SCHEMA)
    return SamplingParams(
        temperature=0.0,
        max_tokens=args.max_tokens,
        guided_decoding=guided,
    )


def build_prompts(rows: pd.DataFrame, tokenizer) -> list[str]:
    """Render system+user messages through the model's chat template.

    Using the tokenizer's chat template guarantees we match the model's
    expected formatting (DeepSeek/Qwen/Llama all differ). vLLM's
    ``LLM.chat`` would do this too but loses some control over batching;
    we build the strings ourselves and pass them to ``LLM.generate``.
    """
    prompts: list[str] = []
    for _, r in rows.iterrows():
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": format_user_prompt(r["sector"], r["policy_text"])},
        ]
        prompts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return prompts


# --------------------------------------------------------------------- main loop

def classify_shard(args) -> Path:
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_path = out_dir / f"shard_{args.shard:02d}_of_{args.num_shards:02d}.jsonl"
    log = lambda m: print(f"[shard {args.shard}/{args.num_shards}] {m}", flush=True)

    log(f"loading inputs from {args.data} (shard {args.shard} of {args.num_shards})")
    df = load_inputs(args.data, args.shard, args.num_shards)
    done = load_done(shard_path)
    todo = df[~df["policy_uid"].isin(done)].reset_index(drop=True)
    log(f"shard size {len(df):,}; already done {len(done):,}; to classify {len(todo):,}")
    if todo.empty:
        log("nothing to do; exiting")
        return shard_path

    if args.limit:
        todo = todo.head(args.limit).reset_index(drop=True)
        log(f"--limit {args.limit} applied -> {len(todo):,} rows")

    log(f"loading model {args.model} (tp={args.tp}, quant={args.quantization or 'none'})")
    llm = build_llm(args)
    tokenizer = llm.get_tokenizer()
    sp = make_sampling_params(args)

    log("rendering prompts")
    prompts = build_prompts(todo, tokenizer)

    # vLLM batches internally; we hand it everything and let it stream
    # completions back. For a multi-million-row workload this is fine —
    # vLLM keeps the GPU saturated.
    log(f"generating {len(prompts):,} completions")
    start = time.time()
    outputs = llm.generate(prompts, sp)
    elapsed = time.time() - start
    log(f"generation done in {elapsed/60:.1f} min ({len(prompts)/max(elapsed,1):.1f} req/s)")

    log(f"writing -> {shard_path}")
    with shard_path.open("a") as f:
        for row, out in zip(todo.itertuples(index=False), outputs):
            raw = out.outputs[0].text
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as e:
                rec = {
                    "policy_uid": row.policy_uid,
                    "sector": row.sector,
                    "cluster_uid": row.cluster_uid,
                    "prompt_version": PROMPT_VERSION,
                    "model": args.model,
                    "error": f"JSONDecodeError: {e}",
                    "raw": raw[:500],
                }
            else:
                rec = {
                    "policy_uid": row.policy_uid,
                    "sector": row.sector,
                    "cluster_uid": row.cluster_uid,
                    "prompt_version": PROMPT_VERSION,
                    "model": args.model,
                    **parsed,
                }
            f.write(json.dumps(rec) + "\n")
    log("done")
    return shard_path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=DEFAULT_DATA_ROOT, help="root dir of per-sector clustered parquets")
    p.add_argument("--out", default=DEFAULT_OUT_DIR, help="output dir for shard JSONL files")
    p.add_argument("--shard", type=int, default=0, help="shard index (0..num_shards-1)")
    p.add_argument("--num-shards", type=int, default=int(os.environ.get("NUM_SHARDS", 10)),
                   help="total number of shards (default 10 to mirror jz_* convention)")
    p.add_argument("--model", default=os.environ.get("MODEL", DEFAULT_MODEL),
                   help="HuggingFace model id (default DeepSeek-R1-Distill-Qwen-32B)")
    p.add_argument("--quantization", default=os.environ.get("QUANTIZATION") or None,
                   help="vLLM quantization (fp8, awq, gptq...). Leave empty for BF16")
    p.add_argument("--dtype", default="auto", help="vLLM dtype (auto/bf16/fp16)")
    p.add_argument("--tp", type=int, default=int(os.environ.get("TP", 1)),
                   help="tensor parallel size; 1 for single-GPU, 4-8 for big MoE")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=4096,
                   help="context budget per request; policy texts are short so 4k is generous")
    p.add_argument("--max-tokens", type=int, default=200,
                   help="completion cap; the JSON answer is ~80 tokens")
    p.add_argument("--limit", type=int, default=0,
                   help="cap rows for smoke tests; 0 = no cap")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # SLURM_ARRAY_TASK_ID wins over --shard if both are set.
    env_shard = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_shard is not None:
        args.shard = int(env_shard)
    env_count = os.environ.get("SLURM_ARRAY_TASK_COUNT")
    if env_count is not None:
        args.num_shards = int(env_count)
    classify_shard(args)


if __name__ == "__main__":
    main(sys.argv[1:])
