"""Prompt v1 for the sufficiency classifier — aligned with expert gold.

Changes vs v0:
- 5-class enum (`sufficiency`, `efficiency`, `consistency`, `ambiguous`,
  `not_a_policy`) instead of the 3-class {suff, potential, not}. The
  `not_a_policy` slot filters junk without abusing the sufficiency label,
  and the `efficiency`/`consistency`/`ambiguous` split matches the
  domain-expert taxonomy.
- Optional integer `sub_code` in 0..9, present only when `category` is
  `sufficiency`. Coded against the taxonomy in the expert instructions sheet.
- `SUFFICIENCY_DEFINITION` rewritten to include the 10-code sub-taxonomy
  in-prompt so the model treats passive design, proximity/land use, modal
  shift, and public provisioning as sufficiency — the classes v0 was missing.
"""
from __future__ import annotations

import json as _json

PROMPT_VERSION = "v1.1-terse-plus-vague-rule-5class-10sub"

CATEGORIES = ["sufficiency", "efficiency", "consistency", "ambiguous", "not_a_policy"]

SUB_TAXONOMY = [
    (0, "Involuntary / non-policy demand cut",
     "Coerced or crisis demand cuts that are not deliberate policy (e.g. "
     "COVID lockdowns, war-time rationing outcomes)."),
    (1, "Caps, limits & bans",
     "Regulatory ceilings: quotas, bans, restrictions, extraction/abstraction "
     "limits, capacity cuts, protected areas, working-hour caps."),
    (2, "Demand-suppressing prices & taxes",
     "Fiscal signals designed to cut consumption: higher prices, "
     "environmental/carbon/consumption taxes, deterrent fees."),
    (3, "Reduce & right size",
     "Lower the quantity or level of the service or product: reduce "
     "waste/overconsumption, downscale, right-size."),
    (4, "Passive & climate-responsive design",
     "Design out the need for active systems: shading, thermal mass, natural "
     "ventilation, daylighting, orientation, cool roofs."),
    (5, "Proximity, compactness & land sufficiency",
     "Spatially avoid the need: compact mixed-use development, limit sprawl, "
     "proximity planning, urban right-sizing."),
    (6, "Modal & provisioning shift",
     "Shift to a lighter mode or provisioning system: active or public "
     "transport, rail/water/multimodal freight."),
    (7, "Share, reuse, repair & prolong",
     "Circularity-by-longevity (not recycling): reuse, repair, durability, "
     "returnable/reusable systems, sharing."),
    (8, "Dietary & food-system sufficiency",
     "Shift food and land consumption: plant-rich or extensive farming, "
     "reduce animal production, cut food waste."),
    (9, "Public provisioning",
     "Collective or universal provision securing needs at low throughput; "
     "social-floor measures (universal basic services, access, decent-work "
     "caps)."),
]

_SUB_TAXO_BLOCK = "\n".join(
    f"  {code}. {label} — {definition}" for code, label, definition in SUB_TAXONOMY
)

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

Use `not_a_policy` when the text is not actually a policy — it may be a
data description, an academic finding, an implementation detail, a
technical parameter, or unintelligible. Do NOT force such texts into a
policy category. A vague call for action, an aspiration, or a generic
"promote / transform / upgrade / manage / improve X" phrase with NO
concrete mechanism or instrument is `not_a_policy`, NOT `ambiguous`.

Use `ambiguous` only when the text clearly IS a concrete policy or measure
but its mechanism is genuinely unclear (mixed rationale, or evidence points
to more than one category with no dominant one). If you cannot point to a
concrete mechanism at all, it is `not_a_policy`.
"""

SYSTEM_PROMPT_BASE = f"""You are a domain expert classifying public policies on energy/resource sufficiency.

DEFINITION (apply this exactly):
{SUFFICIENCY_DEFINITION}

For each policy text you receive, return STRICT JSON matching the schema below.
Do not add commentary, markdown, or anything outside the JSON object.

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
            f"Example {i} — sector {ex['sector']}\n"
            f"Policy: {ex['policy_text'].strip()}\n"
            f"Answer: {_json.dumps(ans)}"
        )
    return (
        SYSTEM_PROMPT_BASE
        + "\nCALIBRATION EXAMPLES (apply the same rubric):\n\n"
        + "\n\n".join(blocks)
        + "\n"
    )


USER_PROMPT_TEMPLATE = """Sector: {sector}
Policy: {policy_text}

Classify."""


def format_user_prompt(sector: str, policy_text: str) -> str:
    return USER_PROMPT_TEMPLATE.format(sector=sector, policy_text=policy_text.strip())


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
