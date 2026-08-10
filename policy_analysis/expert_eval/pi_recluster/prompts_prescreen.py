"""Prompt `prescreen` — abstract-level sufficiency prescreening with Gemma.

Replaces the SetFit prescreener (`library/prescreening/stage2/
predict_sufficiency.py`, TheoLvs/wsl-prescreening-multi-v0.0): instead of 5
topic probabilities on English-only abstracts, the LLM applies the expert
sufficiency definition (Yamina Saheb's taxonomy, shared with the policy
classifier via prompts_v1) directly to title + abstract, in any language.

Selected via PROMPTS_VERSION=prescreen; exposes the same public symbols as
prompts_v1 so classify_vllm.py runs unchanged. Items come from
build_prescreen_input.py: policy_uid = openalex id, policy_text =
title + abstract, sector = "ABSTRACT" (carried through, unused in the
prompt).
"""
from __future__ import annotations

import json as _json

from prompts_v1 import SUB_TAXONOMY

PROMPT_VERSION = "prescreen-v1-abstract-5class"

CATEGORIES = ["sufficiency", "efficiency", "consistency", "ambiguous", "not_relevant"]

_SUB_TAXO_BLOCK = "\n".join(
    f"  {code}. {label} — {definition}" for code, label, definition in SUB_TAXONOMY
)

# Same core definition as prompts_v1.SUFFICIENCY_DEFINITION, with the
# policy-text-specific guidance (not_a_policy / ambiguous-policy rules)
# replaced by article-level screening guidance.
SUFFICIENCY_DEFINITION = f"""\
Sufficiency (sobriety) policies aim to reduce the ABSOLUTE level of demand
for energy, materials, space, or services — not the intensity per unit of
service. They differ from:

  - efficiency policies: same service delivered with less input per unit
    (LED bulbs, insulation for a given comfort level, high-yield seeds).
  - consistency policies: same service delivered with a different, cleaner
    input — substitution (renewable electricity, biofuels, heat pumps,
    material substitution).

Sufficiency covers ANY of the following mechanisms (use the code as
`sub_code` when `category` is `sufficiency`):

{_SUB_TAXO_BLOCK}

Note: passive design (code 4), compact land use (code 5), and modal shifts
(code 6) count as sufficiency because they avoid or shrink the underlying
service demand — not because they make an existing service more efficient.
Public provisioning (code 9) counts when the mechanism secures needs at a
lower material or energy throughput than private/market provisioning would.
"""

SYSTEM_PROMPT_BASE = f"""You are a domain expert screening academic articles for a research \
library on energy/resource sufficiency policy.

DEFINITION (apply this exactly):
{SUFFICIENCY_DEFINITION}

You receive the title and abstract of ONE article, in any language. Classify
what the article is mainly about:

  - "sufficiency": it studies, proposes, evaluates or discusses sufficiency
    policies or measures in the sense of the definition (any sub-code).
  - "efficiency" / "consistency": it is about demand-side policy, but of
    those kinds rather than sufficiency.
  - "ambiguous": clearly about demand-reduction policy, but the mechanism
    cannot be pinned to one category from the abstract alone.
  - "not_relevant": anything else (no policy dimension, purely technical,
    unrelated field, or an unintelligible abstract).

Return STRICT JSON matching the schema below. Do not add commentary,
markdown, or anything outside the JSON object.

Schema:
{{
  "category": one of {CATEGORIES},
  "sub_code": integer 0..9 (ONLY when category is "sufficiency"; otherwise omit or null),
  "confidence": integer 1..5  (1 = guess, 5 = very sure),
  "reasoning": short string, max 240 characters, naming the clause of the
    definition that drove the verdict
}}
"""


def build_system_prompt(few_shot: list[dict] | None = None) -> str:
    """Return the system prompt, optionally with expert-labelled examples appended.

    Each entry in ``few_shot`` is a dict with keys ``sector``, ``policy_text``,
    ``category``, ``reasoning``, and optionally ``sub_code``.
    """
    if not few_shot:
        return SYSTEM_PROMPT_BASE
    blocks = []
    for i, ex in enumerate(few_shot, start=1):
        ans: dict = {"category": ex["category"]}
        if ex.get("sub_code") is not None:
            ans["sub_code"] = int(ex["sub_code"])
        ans["confidence"] = 5
        ans["reasoning"] = ex["reasoning"]
        blocks.append(
            f"Example {i}\n"
            f"Article: {ex['policy_text'].strip()}\n"
            f"Answer: {_json.dumps(ans)}"
        )
    return (
        SYSTEM_PROMPT_BASE
        + "\nCALIBRATION EXAMPLES (apply the same rubric):\n\n"
        + "\n\n".join(blocks)
        + "\n"
    )


USER_PROMPT_TEMPLATE = """{policy_text}

Classify."""


def format_user_prompt(sector: str, policy_text: str) -> str:
    # `sector` is part of the shared prompt-module signature; abstracts have
    # no sector so it is ignored here.
    return USER_PROMPT_TEMPLATE.format(policy_text=policy_text.strip())


RESPONSE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "sub_code": {"type": ["integer", "null"], "minimum": 0, "maximum": 9},
        "confidence": {"type": "integer", "minimum": 1, "maximum": 5},
        "reasoning": {"type": "string", "maxLength": 240},
    },
    "required": ["category", "confidence", "reasoning"],
    "additionalProperties": False,
}
