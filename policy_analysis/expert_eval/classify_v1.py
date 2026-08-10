"""Run the v1 prompt classifier against the DeepSeek V4 API.

Standalone runner (doesn't touch v0's ``test_api.py`` so baseline stays
reproducible). Uses ``prompts_v1`` and validates against its schema.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path

from openai import APIError, AsyncOpenAI, RateLimitError

try:
    from dotenv import load_dotenv
    # repo-root .env, resolved relative to this file so it works on any box
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

import importlib

# Prompt module is selectable via PROMPTS_VERSION env (v1|v2); default v1 keeps
# every existing invocation reproducible. The chosen module must expose the same
# public symbols.
_PROMPTS = importlib.import_module(
    f"policy_analysis.sufficiency_classification.prompts_"
    f"{os.environ.get('PROMPTS_VERSION', 'v1')}"
)
CATEGORIES = _PROMPTS.CATEGORIES
PROMPT_VERSION = _PROMPTS.PROMPT_VERSION
RESPONSE_JSON_SCHEMA = _PROMPTS.RESPONSE_JSON_SCHEMA
build_system_prompt = _PROMPTS.build_system_prompt
format_user_prompt = _PROMPTS.format_user_prompt

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
CONCURRENCY = int(os.environ.get("CLASSIFY_CONCURRENCY", "48"))
MAX_RATELIMIT_RETRIES = 8
MAX_SCHEMA_RETRIES = 1
DEEPSEEK_EXTRA_BODY = {"thinking": {"type": "disabled"}}


def _validate(obj) -> str | None:
    if not isinstance(obj, dict):
        return f"top-level must be object, got {type(obj).__name__}"
    for k in RESPONSE_JSON_SCHEMA["required"]:
        if k not in obj:
            return f"missing required field: {k}"
    cat = obj.get("category")
    if cat not in CATEGORIES:
        return f"category {cat!r} not in {CATEGORIES}"
    conf = obj.get("confidence")
    if not isinstance(conf, int) or isinstance(conf, bool) or not 1 <= conf <= 5:
        return f"confidence must be int 1..5, got {conf!r}"
    reasoning = obj.get("reasoning")
    if not isinstance(reasoning, str):
        return f"reasoning must be string, got {type(reasoning).__name__}"
    if len(reasoning) > 240:
        return f"reasoning length {len(reasoning)} > 240"
    sub_code = obj.get("sub_code")
    if sub_code is not None:
        if cat != "sufficiency":
            return f"sub_code set but category={cat!r}"
        if not isinstance(sub_code, int) or isinstance(sub_code, bool) or not 0 <= sub_code <= 9:
            return f"sub_code must be int 0..9, got {sub_code!r}"
    return None


def _load_done(path: Path) -> set[tuple[str, str | None]]:
    """Return the set of already-classified (policy_uid, cluster_uid) pairs.

    We can't dedup on policy_uid alone because a chunk can be split into
    multiple policies that live in different clusters (same
    openalex_id::chunk_idx, different text).
    """
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


async def _chat(client, sem, model, messages):
    delay = 5.0
    for attempt in range(MAX_RATELIMIT_RETRIES):
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=350,
                    temperature=0,
                    response_format={"type": "json_object"},
                    extra_body=DEEPSEEK_EXTRA_BODY,
                )
                return r.choices[0].message.content or ""
            except (RateLimitError, APIError):
                if attempt == MAX_RATELIMIT_RETRIES - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
    raise RuntimeError("unreachable")


async def _classify_one(client, sem, model, system_prompt, item):
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_user_prompt(item["sector"], item["policy_text"])},
    ]
    last_err = None
    for _ in range(MAX_SCHEMA_RETRIES + 1):
        raw = await _chat(client, sem, model, msgs)
        if not raw.strip():
            last_err = "empty response"
            msgs = msgs + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Empty reply. Output the JSON object only."},
            ]
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            last_err = f"JSONDecodeError: {e}"
            msgs = msgs + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Not valid JSON ({e}). Output JSON only."},
            ]
            continue
        err = _validate(parsed)
        if err is None:
            return {
                "policy_uid": item["policy_uid"],
                "sector": item["sector"],
                "cluster_uid": item.get("cluster_uid"),
                "prompt_version": PROMPT_VERSION,
                "model": model,
                **parsed,
            }
        last_err = err
        msgs = msgs + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"Schema violation: {err}. Reply with valid JSON."},
        ]
    return {
        "policy_uid": item["policy_uid"],
        "sector": item["sector"],
        "cluster_uid": item.get("cluster_uid"),
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "error": last_err or "unknown",
    }


async def _run(items, out_path, model, system_prompt):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY not set")
    client = AsyncOpenAI(
        base_url=os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        api_key=api_key,
    )
    sem = asyncio.Semaphore(CONCURRENCY)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    coros = [_classify_one(client, sem, model, system_prompt, it) for it in items]
    done = 0
    with out_path.open("a") as f:
        for fut in asyncio.as_completed(coros):
            rec = await fut
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done += 1
            tag = "FAIL" if "error" in rec else f"{rec['category'][:4]} c{rec['confidence']}"
            if done % 50 == 0 or "error" in rec:
                print(f"  [{done}/{len(items)}] {tag} {rec['policy_uid']}", flush=True)


def _load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--few-shot", default=None)
    ap.add_argument("--exclude-ids", default=None, help="text file of policy_uid to skip")
    ap.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    args = ap.parse_args()

    items = _load_jsonl(args.items)

    exclude: set[str] = set()
    if args.exclude_ids:
        exclude = {line.strip() for line in Path(args.exclude_ids).read_text().splitlines() if line.strip()}
        print(f"excluding {len(exclude)} ids (few-shot leakage guard)")
    items = [it for it in items if it["policy_uid"] not in exclude]

    done = _load_done(Path(args.out))
    todo = [it for it in items if (it["policy_uid"], it.get("cluster_uid")) not in done]
    print(f"{len(items)} items after exclude, {len(done)} done, {len(todo)} to classify")

    few_shot = None
    if args.few_shot:
        few_shot = _load_jsonl(args.few_shot)
        print(f"injecting {len(few_shot)} calibration examples into system prompt")
    system_prompt = build_system_prompt(few_shot)

    if not todo:
        print("nothing to do")
        return

    asyncio.run(_run(todo, Path(args.out), args.model, system_prompt))


if __name__ == "__main__":
    main()
