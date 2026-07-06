"""Prompt v3 for the sufficiency classifier — plain six-pillar definition.

Drop-in replacement for ``prompts_v1`` (identical public interface and output
schema). v3 is the middle point between v1 and v2:

  - v1: one terse discriminator (avoid vs improve vs shift). Best on Gemma so far.
  - v2: the six pillars turned into a multi-step operational procedure with
        watch-outs. Wins on DeepSeek, but the added complexity *hurts* Gemma
        (5-class exact-match fell 62.9% → 55.1%).
  - v3: the six-pillar definition stated PLAINLY as concept — the source prose,
        citations stripped, no decision procedure, no watch-outs, no operational
        elaboration. Tests whether the richer *definition* helps once the
        scaffolding that confused the smaller model is removed.

Same 5-class + 10-code taxonomy and JSON schema as v1/v2 so results and the
downstream (sector, sub_code) clustering stay directly comparable.
"""
from __future__ import annotations

import json as _json

PROMPT_VERSION = "v3-sixpillar-plain-5class-10sub"

CATEGORIES = ["sufficiency", "efficiency", "consistency", "ambiguous", "not_a_policy"]

# Identical to v1/v2 so sub_code distributions stay comparable across runs.
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
     "social-floor measures (universal basic services, access, decent-work caps)."),
]

_SUB_TAXO_BLOCK = "\n".join(
    f"  {code}. {label} — {definition}" for code, label, definition in SUB_TAXONOMY
)

# The definition, stated plainly — the source six-pillar prose with citations and
# operational enrichment removed. No "apply these tests in order", no watch-outs.
SUFFICIENCY_DEFINITION = f"""\
Sufficiency is a set of policy measures and daily practices that AVOID the demand
for energy, water, land, materials and all other natural resources while
delivering wellbeing for all within planetary boundaries.

It rests on six pillars:

1. Avoidance. The organising verb is AVOID: the absolute reduction of demand.
   Sufficiency is not efficiency ("doing more with less" — an improvement in a
   ratio) and not the substitution of cleaner supply ("shifting", e.g. to
   renewables); it is the absolute reduction of demand.
2. All natural resources. Sufficiency is defined across all natural resources —
   energy, water, land, materials and the rest — not energy or carbon alone.
3. Wellbeing for all. The purpose of avoiding demand is to deliver wellbeing,
   understood through human needs that are universal, finite and satiable: needs
   can be met, and once met, more is not better.
4. Planetary boundaries. Wellbeing must be delivered within a hard, measurable
   ecological ceiling — the biophysical safe operating space.
5. Dual agency. Sufficiency operates at two scales at once — policy measures and
   daily practices — pursued through structural policy and lived practice
   together, not by moralising individual consumers.
6. Justice. "For all" is a substantive commitment to equity, not a generic
   universalism: it asks whose needs count and whose knowledge defines need, and
   guards against a sufficiency imposed from the top down.

The other categories:
  - efficiency: same service delivered with less input per unit (LED bulbs,
    insulation for a given comfort level, high-yield seeds).
  - consistency: same service delivered with a different, cleaner input —
    substitution (renewable electricity, biofuels, heat pumps, material swap).
  - ambiguous: the text IS a policy but the mechanism is genuinely unclear.
  - not_a_policy: the text is not a policy (a data description, an academic
    finding, an implementation detail, a technical parameter, or unintelligible).

When category is `sufficiency`, set `sub_code` to the best-fitting mechanism:
{_SUB_TAXO_BLOCK}
"""

SYSTEM_PROMPT_BASE = f"""You are a domain expert classifying public policies on energy/resource sufficiency.

DEFINITION (apply this exactly):
{SUFFICIENCY_DEFINITION}

For each policy text you receive, return STRICT JSON matching the schema below.
Do not add commentary, markdown, or anything outside the JSON object.

Schema:
{{
  "category": one of {CATEGORIES},
  "sub_code": integer 0..9 (ONLY when category is "sufficiency"; otherwise null),
  "confidence": integer 1..5  (1 = guess, 5 = very sure),
  "reasoning": short string, max 240 characters, naming the pillar/clause that
    drove the verdict
}}
"""


def build_system_prompt(few_shot: list[dict] | None = None) -> str:
    """Return the system prompt, optionally with expert-labelled examples appended."""
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
