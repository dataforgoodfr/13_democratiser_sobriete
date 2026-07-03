"""Smoke-test the sufficiency classifier against the DeepSeek V4 API.

Same prompt module as the vLLM path (``prompts.py``), so this test exercises
the rubric and the JSON schema exactly as the production run will.

Few-shot calibration: pass ``--few-shot path/to/labels.jsonl`` and the lines
will be injected into the system prompt. Lines look like the calibration
JSONL produced by ``label_calibration.py`` (or a hand-written file): one
JSON object per line with keys ``sector``, ``policy_text``, ``category``,
``reasoning``.

Idempotent: existing ``policy_uid``s in the output JSONL are skipped on
re-run; errored rows are retried.
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
    load_dotenv("/Users/aminesaboni/oss/wsl/13_democratiser_sobriete/.env")
except ImportError:
    pass

from policy_analysis.sufficiency_classification.prompts import (
    PROMPT_VERSION,
    RESPONSE_JSON_SCHEMA,
    build_system_prompt,
    format_user_prompt,
)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
CONCURRENCY = 16
MAX_RATELIMIT_RETRIES = 8
MAX_SCHEMA_RETRIES = 1
DEEPSEEK_EXTRA_BODY = {"thinking": {"type": "disabled"}}


def _validate(obj) -> str | None:
    schema = RESPONSE_JSON_SCHEMA
    if not isinstance(obj, dict):
        return f"top-level must be object, got {type(obj).__name__}"
    for k in schema["required"]:
        if k not in obj:
            return f"missing required field: {k}"
    cat = obj.get("category")
    if cat not in schema["properties"]["category"]["enum"]:
        return f"category {cat!r} not in {schema['properties']['category']['enum']}"
    conf = obj.get("confidence")
    if not isinstance(conf, int) or isinstance(conf, bool) or not 1 <= conf <= 5:
        return f"confidence must be int 1..5, got {conf!r}"
    reasoning = obj.get("reasoning")
    if not isinstance(reasoning, str):
        return f"reasoning must be string, got {type(reasoning).__name__}"
    if len(reasoning) > 200:
        return f"reasoning length {len(reasoning)} > 200"
    return None


def _load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
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
        if uid:
            done.add(uid)
    return done


async def _chat(client, sem, model, messages):
    delay = 5.0
    for attempt in range(MAX_RATELIMIT_RETRIES):
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=300,
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
    with out_path.open("a") as f:
        for fut in asyncio.as_completed(coros):
            rec = await fut
            f.write(json.dumps(rec) + "\n")
            f.flush()
            tag = "FAIL" if "error" in rec else f"{rec['category'][:4]} c{rec['confidence']}"
            print(f"  {tag:8s} {rec['policy_uid']}", flush=True)


def _load_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="JSONL of policies to classify")
    ap.add_argument("--out", required=True, help="JSONL output")
    ap.add_argument("--few-shot", default=None, help="JSONL of calibration labels to inject")
    ap.add_argument("--model", default=os.environ.get("DEEPSEEK_MODEL", DEFAULT_MODEL))
    args = ap.parse_args()

    items = _load_jsonl(args.items)
    done = _load_done(Path(args.out))
    todo = [it for it in items if it["policy_uid"] not in done]
    print(f"{len(items)} items, {len(done)} done, {len(todo)} to classify")

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
