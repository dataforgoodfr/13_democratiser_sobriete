"""Judge prompts for the cluster-quality intrusion task.

Single source of truth for the LLM-as-judge prompt. The domain expert iterates
here; downstream code (`llm_judge.py`) re-reads it on every run. Each revision
gets a version string so reports record which prompt produced which numbers.

DeepSeek V4 only supports `response_format={"type": "json_object"}` — there is
no server-side JSON-schema enforcement for raw responses. We therefore embed
the schema *and* a concrete example directly in the system prompt, then
validate the parsed object locally (see ``validate_response`` in llm_judge.py).
"""

PROMPT_VERSION = "v1-deepseek-subcode"
# Bump this whenever a few-shot calibration file is in use so reports record it.
PROMPT_VERSION_FEW_SHOT = "v1-deepseek-subcode+fewshot"


# Sufficiency sub-taxonomy, kept in sync with
# policy_analysis/sufficiency_classification/prompts_v1.py (SUB_TAXONOMY).
# Rendered into the user prompt so the judge can check mechanism fit —
# the expert review (2026-06-27) showed the intrusion task alone misses
# clusters that are topically coherent but carry the wrong mechanism label.
SUB_TAXONOMY = {
    0: ("Involuntary / non-policy demand cut",
        "coerced or crisis demand cuts that are not deliberate policy"),
    1: ("Caps, limits & bans",
        "regulatory ceilings: quotas, bans, restrictions, capacity cuts"),
    2: ("Demand-suppressing prices & taxes",
        "fiscal signals designed to cut consumption"),
    3: ("Reduce & right size",
        "lower the quantity or level of the service or product consumed"),
    4: ("Passive & climate-responsive design",
        "design out the need for active systems"),
    5: ("Proximity, compactness & land sufficiency",
        "spatially avoid the need: compact development, limit sprawl"),
    6: ("Modal & provisioning shift",
        "shift to a lighter mode or provisioning system"),
    7: ("Share, reuse, repair & prolong",
        "circularity-by-longevity for physical products; not recycling"),
    8: ("Dietary & food-system sufficiency",
        "shift food and land consumption; cut food waste"),
    9: ("Public provisioning",
        "collective or universal provision securing needs at low throughput"),
}


# Authoritative schema for the judge's reply. Used both to (a) render text
# for the model and (b) validate the model's parsed JSON locally.
RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["intruder_index", "topic_label", "specificity", "confidence",
                 "sub_code_fit"],
    "properties": {
        "intruder_index": {"type": "integer", "minimum": 0, "maximum": 4},
        "topic_label": {"type": "string", "maxLength": 200},
        "specificity": {"type": "integer", "minimum": 1, "maximum": 5},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
        "sub_code_fit": {"type": ["string", "null"],
                         "enum": ["yes", "partly", "no", None]},
    },
}


EXAMPLE_RESPONSE = {
    "intruder_index": 2,
    "topic_label": "urban green space planning",
    "specificity": 4,
    "confidence": 4,
    "sub_code_fit": "yes",
}


_SYSTEM_PROMPT_BASE = f"""You are a policy analyst evaluating the coherence of a topic cluster.

You will be shown five short policy statements. Four of them belong to the same
topic cluster; one is an *intruder* from a different cluster. The cluster may
also declare an *assigned sufficiency mechanism* (its sub-taxonomy code).
Your job is to:
  1. Identify which statement is the intruder.
  2. Describe in a few words what the cluster (the four coherent statements) is about.
  3. Rate how *specific* the cluster's topic is, on a 1-5 scale:
     - 1 = extremely broad (could apply to many unrelated policies)
     - 5 = narrow and well-defined (a single specific policy mechanism)
  4. State your confidence in the intruder pick, on a 1-5 scale.
  5. If an assigned mechanism is given, judge whether the four coherent
     statements actually use that mechanism ("yes" = clearly, "partly" =
     some do or the fit is loose, "no" = the label does not describe them —
     e.g. reducing sedentary behaviour is NOT "Reduce & right size", which
     is about consumption). If no mechanism is given, use null.

Reply with a single JSON object and nothing else — no prose, no markdown fences.
Required fields:
  - intruder_index: integer between 0 and 4 inclusive
  - topic_label: short string, max 10 words, describing the cluster topic
  - specificity: integer 1 to 5
  - confidence: integer 1 to 5
  - sub_code_fit: "yes" | "partly" | "no" | null

Example of a valid json reply:
{{"intruder_index": {EXAMPLE_RESPONSE["intruder_index"]}, "topic_label": "{EXAMPLE_RESPONSE["topic_label"]}", "specificity": {EXAMPLE_RESPONSE["specificity"]}, "confidence": {EXAMPLE_RESPONSE["confidence"]}, "sub_code_fit": "{EXAMPLE_RESPONSE["sub_code_fit"]}"}}
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
{mechanism_block}
Statements:
[0] {s0}
[1] {s1}
[2] {s2}
[3] {s3}
[4] {s4}

Return only the json object described in the instructions.
"""


def format_user_prompt(
    sector: str, statements: list[str], sub_code: int | None = None
) -> str:
    assert len(statements) == 5, "intrusion task expects exactly 5 statements"
    if sub_code is not None and sub_code in SUB_TAXONOMY:
        label, definition = SUB_TAXONOMY[sub_code]
        mechanism_block = (
            f"Assigned sufficiency mechanism: {sub_code} · {label} — {definition}.\n"
        )
    else:
        mechanism_block = ""
    return USER_PROMPT_TEMPLATE.format(
        sector=sector,
        mechanism_block=mechanism_block,
        s0=statements[0],
        s1=statements[1],
        s2=statements[2],
        s3=statements[3],
        s4=statements[4],
    )
