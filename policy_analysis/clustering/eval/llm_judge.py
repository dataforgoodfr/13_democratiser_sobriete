"""LLM-as-judge intrusion task — DeepSeek V4 client.

DeepSeek's OpenAI-compatible API only supports `response_format={"type":
"json_object"}` for raw replies (no server-side JSON-schema strict mode).
The schema lives in ``prompts.py`` and is enforced locally:

  1. The model is asked for a JSON object whose required fields and ranges are
     spelled out (plus a concrete example) in the system prompt.
  2. We parse the reply, validate it against ``RESPONSE_JSON_SCHEMA`` via the
     lightweight checker in this file (no extra dependency).
  3. On validation failure we make one retry with a corrective user message
     that includes the validation error.

Env vars:
  - DEEPSEEK_API_KEY  (required)
  - DEEPSEEK_BASE_URL (optional, defaults to https://api.deepseek.com)
  - DEEPSEEK_MODEL    (optional, defaults to deepseek-v4-pro; set to
                       deepseek-v4-flash for the cheaper tier)

Idempotent: existing cluster_uids in ``out_path`` are skipped on resume.
"""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
from openai import AsyncOpenAI, RateLimitError, APIError

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .prompts import (
    PROMPT_VERSION,
    PROMPT_VERSION_FEW_SHOT,
    SYSTEM_PROMPT,
    RESPONSE_JSON_SCHEMA,
    build_system_prompt,
    format_user_prompt,
)

DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL = "deepseek-v4-pro"
CONCURRENCY = 16
MAX_RATELIMIT_RETRIES = 8
MAX_SCHEMA_RETRIES = 1

# DeepSeek V4 defaults to "thinking" mode, where hidden reasoning consumes
# the token budget and json_object replies come back empty. We explicitly
# disable it. Passed via extra_body since it's a DeepSeek-specific param
# not in the OpenAI schema. See
# https://api-docs.deepseek.com/guides/thinking_mode
DEEPSEEK_EXTRA_BODY = {"thinking": {"type": "disabled"}}


def validate_response(obj: object) -> str | None:
    """Return None if obj matches RESPONSE_JSON_SCHEMA, else an error string."""
    schema = RESPONSE_JSON_SCHEMA
    if not isinstance(obj, dict):
        return f"top-level value must be an object, got {type(obj).__name__}"
    required = schema["required"]
    missing = [k for k in required if k not in obj]
    if missing:
        return f"missing required fields: {missing}"
    extra = [k for k in obj if k not in schema["properties"]]
    if extra and not schema.get("additionalProperties", True):
        return f"unexpected fields: {extra}"
    for key, spec in schema["properties"].items():
        if key not in obj:
            continue
        v = obj[key]
        if "enum" in spec:
            if v not in spec["enum"]:
                return f"{key}={v!r} not in enum {spec['enum']}"
            continue
        if spec["type"] == "integer":
            if not isinstance(v, int) or isinstance(v, bool):
                return f"{key} must be integer, got {type(v).__name__}"
            if "minimum" in spec and v < spec["minimum"]:
                return f"{key}={v} below minimum {spec['minimum']}"
            if "maximum" in spec and v > spec["maximum"]:
                return f"{key}={v} above maximum {spec['maximum']}"
        elif spec["type"] == "string":
            if not isinstance(v, str):
                return f"{key} must be string, got {type(v).__name__}"
            if "maxLength" in spec and len(v) > spec["maxLength"]:
                return f"{key} length {len(v)} exceeds maxLength {spec['maxLength']}"
    return None


def _load_done(out_path: Path) -> set[str]:
    """Return cluster_uids that have a successful (non-error) judgment recorded.

    Errored items are intentionally NOT considered done so a re-run will retry
    them. The on-disk file may still contain stale error lines — that's fine,
    the report aggregator dedups by taking the latest valid judgment.
    """
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
            uid = rec.get("cluster_uid")
            if uid:
                done.add(uid)
    return done


async def _chat(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    messages: list[dict],
) -> str:
    delay = 5.0
    for attempt in range(MAX_RATELIMIT_RETRIES):
        async with sem:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=300,
                    temperature=0,
                    response_format={"type": "json_object"},
                    extra_body=DEEPSEEK_EXTRA_BODY,
                )
                content = response.choices[0].message.content or ""
                return content
            except RateLimitError:
                if attempt == MAX_RATELIMIT_RETRIES - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
            except APIError as e:
                # DeepSeek occasionally returns 5xx; back off and retry.
                if attempt == MAX_RATELIMIT_RETRIES - 1:
                    raise
                await asyncio.sleep(delay)
                delay *= 2
                _ = e
    raise RuntimeError("unreachable")


async def _judge_one(
    client: AsyncOpenAI,
    sem: asyncio.Semaphore,
    model: str,
    item: dict,
    system_prompt: str,
    prompt_version: str,
) -> dict:
    statements = [c["text"] for c in item["items"]]
    user_prompt = format_user_prompt(item["sector"], statements,
                                     sub_code=item.get("sub_code"))
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    last_error: str | None = None
    for schema_attempt in range(MAX_SCHEMA_RETRIES + 1):
        raw = await _chat(client, sem, model, messages)
        if not raw.strip():
            last_error = "empty response"
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": "Your previous reply was empty. Reply with the json object only."},
            ]
            continue
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as e:
            last_error = f"JSONDecodeError: {e}"
            messages = messages + [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": f"Your previous reply was not valid JSON ({e}). Reply with the json object only, no prose."},
            ]
            continue
        err = validate_response(parsed)
        if err is None:
            return {
                "cluster_uid": item["cluster_uid"],
                "sector": item["sector"],
                "size_bucket": item["size_bucket"],
                "cluster_size": item["cluster_size"],
                "gold_intruder_index": item["gold_intruder_index"],
                "sub_code": item.get("sub_code"),
                "prompt_version": prompt_version,
                "model": model,
                **parsed,
            }
        last_error = err
        messages = messages + [
            {"role": "assistant", "content": raw},
            {"role": "user", "content": f"Your previous reply failed schema validation: {err}. Reply again with a valid json object."},
        ]

    # Fall-through: never produced a valid response. Record the failure but
    # do not crash the whole run.
    return {
        "cluster_uid": item["cluster_uid"],
        "sector": item["sector"],
        "size_bucket": item["size_bucket"],
        "cluster_size": item["cluster_size"],
        "gold_intruder_index": item["gold_intruder_index"],
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "error": last_error or "unknown error",
    }


async def _run_async(items: list[dict], out_path: Path, model: str, few_shot: list[dict] | None = None) -> None:
    done = _load_done(out_path)
    todo = [it for it in items if it["cluster_uid"] not in done]
    if not todo:
        print(f"all {len(items)} items already judged in {out_path}")
        return

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    base_url = os.getenv("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL)

    client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(CONCURRENCY)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    system_prompt = build_system_prompt(few_shot)
    prompt_version = PROMPT_VERSION_FEW_SHOT if few_shot else PROMPT_VERSION

    coros = [_judge_one(client, sem, model, it, system_prompt, prompt_version) for it in todo]
    with out_path.open("a") as f:
        for fut in asyncio.as_completed(coros):
            result = await fut
            f.write(json.dumps(result) + "\n")
            f.flush()
            if "error" in result:
                print(f"  FAILED  {result['cluster_uid']}: {result['error']}")
            else:
                print(
                    f"  judged  {result['cluster_uid']} -> intruder={result['intruder_index']} "
                    f"(gold {result['gold_intruder_index']}, conf {result['confidence']})"
                )


def run_judge(
    items_path: str | Path,
    out_path: str | Path,
    model: str | None = None,
    few_shot_path: str | Path | None = None,
) -> Path:
    items_path = Path(items_path)
    out_path = Path(out_path)
    items = [json.loads(line) for line in items_path.read_text().splitlines() if line.strip()]
    few_shot = None
    if few_shot_path:
        few_shot = [json.loads(l) for l in Path(few_shot_path).read_text().splitlines() if l.strip()]
    asyncio.run(_run_async(items, out_path, model or os.getenv("DEEPSEEK_MODEL", DEFAULT_MODEL), few_shot=few_shot))
    return out_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", required=True, help="intrusion items JSONL")
    parser.add_argument("--out", required=True, help="judgments JSONL output")
    parser.add_argument("--model", default=None, help=f"override model (default {DEFAULT_MODEL})")
    parser.add_argument("--few-shot", default=None, help="path to JSONL of calibration examples to inject into the system prompt")
    args = parser.parse_args()
    path = run_judge(args.items, args.out, model=args.model, few_shot_path=args.few_shot)
    print(f"wrote {path}")
