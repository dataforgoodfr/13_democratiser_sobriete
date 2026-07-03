"""Judge prompts for the cluster-quality intrusion task.

Single source of truth for the LLM-as-judge prompt. The domain expert iterates
here; downstream code (`llm_judge.py`) re-reads it on every run. Each revision
gets a version string so reports record which prompt produced which numbers.

DeepSeek V4 only supports `response_format={"type": "json_object"}` — there is
no server-side JSON-schema enforcement for raw responses. We therefore embed
the schema *and* a concrete example directly in the system prompt, then
validate the parsed object locally (see ``validate_response`` in llm_judge.py).
"""

PROMPT_VERSION = "v0-deepseek"
# Bump this whenever a few-shot calibration file is in use so reports record it.
PROMPT_VERSION_FEW_SHOT = "v0-deepseek+fewshot"


# Authoritative schema for the judge's reply. Used both to (a) render text
# for the model and (b) validate the model's parsed JSON locally.
RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intruder_index", "topic_label", "specificity", "confidence"],
    "properties": {
        "intruder_index": {"type": "integer", "minimum": 0, "maximum": 4},
        "topic_label": {"type": "string", "maxLength": 200},
        "specificity": {"type": "integer", "minimum": 1, "maximum": 5},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
    },
}


EXAMPLE_RESPONSE = {
    "intruder_index": 2,
    "topic_label": "urban green space planning",
    "specificity": 4,
    "confidence": 4,
}


_SYSTEM_PROMPT_BASE = f"""You are a policy analyst evaluating the coherence of a topic cluster.

You will be shown five short policy statements. Four of them belong to the same
topic cluster; one is an *intruder* from a different cluster. Your job is to:
  1. Identify which statement is the intruder.
  2. Describe in a few words what the cluster (the four coherent statements) is about.
  3. Rate how *specific* the cluster's topic is, on a 1-5 scale:
     - 1 = extremely broad (could apply to many unrelated policies)
     - 5 = narrow and well-defined (a single specific policy mechanism)
  4. State your confidence in the intruder pick, on a 1-5 scale.

Reply with a single JSON object and nothing else — no prose, no markdown fences.
Required fields:
  - intruder_index: integer between 0 and 4 inclusive
  - topic_label: short string, max 10 words, describing the cluster topic
  - specificity: integer 1 to 5
  - confidence: integer 1 to 5

Example of a valid json reply:
{{"intruder_index": {EXAMPLE_RESPONSE["intruder_index"]}, "topic_label": "{EXAMPLE_RESPONSE["topic_label"]}", "specificity": {EXAMPLE_RESPONSE["specificity"]}, "confidence": {EXAMPLE_RESPONSE["confidence"]}}}
"""


def build_system_prompt(few_shot: list[dict] | None = None) -> str:
    """Return the judge system prompt, optionally with calibration examples.

    Each item in ``few_shot`` should have keys ``sector``, ``statements`` (list[5]),
    ``intruder_index``, ``topic_label``, plus optional ``specificity`` and
    ``confidence``. The block is appended to the base prompt; prompt caching
    keeps it free on repeat calls.
    """
    if not few_shot:
        return _SYSTEM_PROMPT_BASE
    import json as _json
    blocks = []
    for i, ex in enumerate(few_shot, start=1):
        ans = {
            "intruder_index": ex["intruder_index"],
            "topic_label": ex["topic_label"],
            "specificity": ex.get("specificity", 4),
            "confidence": ex.get("confidence", 5),
        }
        stmts = "\n".join(f"[{k}] {s}" for k, s in enumerate(ex["statements"]))
        blocks.append(
            f"Example {i} — sector {ex['sector']}\n{stmts}\nAnswer: {_json.dumps(ans)}"
        )
    return _SYSTEM_PROMPT_BASE + "\nCALIBRATION EXAMPLES:\n\n" + "\n\n".join(blocks) + "\n"


# Back-compat alias.
SYSTEM_PROMPT = _SYSTEM_PROMPT_BASE


USER_PROMPT_TEMPLATE = """Sector: {sector}

Statements:
[0] {s0}
[1] {s1}
[2] {s2}
[3] {s3}
[4] {s4}

Return only the json object described in the instructions.
"""


def format_user_prompt(sector: str, statements: list[str]) -> str:
    assert len(statements) == 5, "intrusion task expects exactly 5 statements"
    return USER_PROMPT_TEMPLATE.format(
        sector=sector,
        s0=statements[0],
        s1=statements[1],
        s2=statements[2],
        s3=statements[3],
        s4=statements[4],
    )
