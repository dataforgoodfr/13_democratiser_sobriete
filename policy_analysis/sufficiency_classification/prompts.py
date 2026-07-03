"""Prompt + JSON schema for the sobriety/sufficiency classifier.

Single source of truth for the prompt. The domain expert edits
``SUFFICIENCY_DEFINITION`` to fix the rubric; the rest of the pipeline picks
up the new ``PROMPT_VERSION`` automatically and writes it into every output
row so we can diff runs.

Three-class taxonomy aligned with CLAUDE.md ("Sufficiency / Potential
Sufficiency / Not Sufficiency"). Switch to binary by collapsing the schema
enum if needed.

The judge gets *few-shot calibration examples* injected into the system
prompt at runtime — see ``build_system_prompt`` below. Because the model is
stateless, every API call carries the same examples; prompt caching makes
the repeated prefix essentially free.
"""
from __future__ import annotations

PROMPT_VERSION = "v0-sufficiency-3class-placeholder-def"

# ---------------------------------------------------------------------------
# PLACEHOLDER DEFINITION — replace with the project's authoritative wording
# before publishing any numbers. Currently uses the standard academic framing
# (sufficiency = absolute demand reduction, distinct from efficiency and
# substitution) so smoke-tests produce sensible verdicts.
# ---------------------------------------------------------------------------
SUFFICIENCY_DEFINITION = """\
Sufficiency (sobriety) policies aim to reduce the ABSOLUTE level of demand
for energy, materials, space, or services — through behavioural, structural,
organisational, or regulatory change. They are DISTINCT from:

  - efficiency policies (same service, less input per unit), and
  - substitution policies (same service, different/cleaner input).

Three classes:

  - "sufficiency": the policy explicitly aims to reduce the absolute level of
    consumption, demand, or service (caps, limits, downsizing, lifestyle
    change mandates, demand-side modulation that targets *less*, not *better*).
  - "potential_sufficiency": the policy creates conditions that could enable
    sufficiency outcomes but does not mandate them — enabling infrastructure,
    information / awareness campaigns, voluntary schemes, planning frameworks.
  - "not_sufficiency": the policy pursues efficiency, substitution, supply-side
    expansion, or goals unrelated to demand reduction.
"""

_SYSTEM_PROMPT_BASE = f"""You are a domain expert classifying public policies on energy/resource sufficiency.

DEFINITION (apply this exactly):
{SUFFICIENCY_DEFINITION}

For each policy text you receive, return STRICT JSON matching the schema below.
Do not add commentary, markdown, or anything outside the JSON object.

Schema:
{{
  "category": "sufficiency" | "potential_sufficiency" | "not_sufficiency",
  "confidence": integer 1..5  (1=guess, 5=very sure),
  "reasoning": short string, max 200 characters, explaining which clause of the
    definition led to the verdict
}}
"""


def build_system_prompt(few_shot: list[dict] | None = None) -> str:
    """Return the full system prompt, optionally with calibration examples.

    Each entry in ``few_shot`` is a dict with keys ``sector``, ``policy_text``,
    ``category``, ``reasoning``. The block is appended to the base prompt so
    the model anchors on concrete cases. Prompt caching makes the long shared
    prefix free on repeat calls.
    """
    if not few_shot:
        return _SYSTEM_PROMPT_BASE
    blocks = []
    for i, ex in enumerate(few_shot, start=1):
        ans = {
            "category": ex["category"],
            "confidence": 5,
            "reasoning": ex["reasoning"],
        }
        import json as _json
        blocks.append(
            f"Example {i} — sector {ex['sector']}\n"
            f"Policy: {ex['policy_text'].strip()}\n"
            f"Answer: {_json.dumps(ans)}"
        )
    return _SYSTEM_PROMPT_BASE + "\nCALIBRATION EXAMPLES (apply the same rubric):\n\n" + "\n\n".join(blocks) + "\n"


# Backwards-compatible alias: the vLLM script still imports SYSTEM_PROMPT.
SYSTEM_PROMPT = _SYSTEM_PROMPT_BASE


USER_PROMPT_TEMPLATE = """Sector: {sector}
Policy: {policy_text}

Classify."""

RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["sufficiency", "potential_sufficiency", "not_sufficiency"],
        },
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
        "reasoning": {"type": "string", "maxLength": 200},
    },
    "required": ["category", "confidence", "reasoning"],
    "additionalProperties": False,
}


def format_user_prompt(sector: str, policy_text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(sector=sector, policy_text=policy_text.strip())
