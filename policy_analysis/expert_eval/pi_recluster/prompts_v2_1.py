"""Prompt v2.1 for the sufficiency classifier — six-pillar, regression-tuned.

v2.1 vs v2 (two principled fixes from the DeepSeek gold validation):
  1. ambiguous vs not_a_policy boundary tightened. v2 over-fired `ambiguous`
     on vague calls for action ("promote regional transformation", "judicious
     management") that the expert marks `not_a_policy`. Because `ambiguous`
     counts as sufficiency-related, each was a false positive. Rule added:
     a vague call, aspiration, or generic "transformation/management/upgrade"
     phrase with NO concrete mechanism is `not_a_policy`, not `ambiguous`.
  2. Carbon price vs carbon end-of-pipe disambiguated. v2 sent a demand-
     suppressing ETS to `consistency`, contradicting sub_code 2 (which exists
     for exactly demand-suppressing carbon/consumption taxes). Rule clarified:
     a carbon/consumption PRICE, TAX or ETS meant to suppress demand is
     sufficiency (sub_code 2); carbon OFFSETS, CAPTURE or SEQUESTRATION that
     leave physical demand unchanged are NOT.

Drop-in replacement for ``prompts_v1``: identical public interface
(``PROMPT_VERSION``, ``CATEGORIES``, ``SUB_TAXONOMY``, ``RESPONSE_JSON_SCHEMA``,
``build_system_prompt``, ``format_user_prompt``) and identical output schema, so
the classifier and the downstream stratified clustering do not change. Only the
*definition* the model reasons with is rebuilt.

WHAT CHANGED vs v1
------------------
v1 leaned on a single discriminator (avoid vs improve vs shift). v2 keeps that as
the core test but adds the five other pillars as explicit gates, which is where
v1 mis-fired:

  - Pillar 2 (all natural resources) → stop rejecting non-energy demand cuts, and
    stop accepting carbon-only framings (offsets, CCS) that leave physical
    throughput unchanged.
  - Pillar 3 (satiable wellbeing) → separate demand cuts that still meet needs
    (sufficiency) from deprivation/involuntary cuts (sub_code 0 / ambiguous).
  - Pillar 4 (planetary boundaries) → require the avoided demand to have a real
    biophysical footprint; "reduce" applied to paperwork/cost/losses is not
    sufficiency.
  - Pillar 5 (dual agency) → accept measures that reshape infrastructure,
    defaults and practices, not only statutes; don't demand it be a "law".
  - Pillar 6 (justice / "for all") → surface exclusionary or vulnerable-burdening
    demand cuts as ambiguous rather than silently labelling them sufficiency.

PROVENANCE (kept out of the runtime prompt on purpose)
------------------------------------------------------
The definition is a deliberate synthesis (Saheb 2021; IPCC 2022 WGIII glossary &
ch. 5). The runtime prompt carries a one-line attribution only — 1.47M calls make
a full bibliography pure token cost with no effect on the label. The mapping from
each operational rule to its scholarly source lives here so the grounding stays
auditable:

  Pillar 1 Avoidance ............ Sachs 1993; Princen 2005; Fischer & Grießhammer
                                  2013; Brockway et al. 2021; Haberl et al. 2020
  Pillar 2 All resources ........ Georgescu-Roegen 1971; Ayres & Kneese 1969;
                                  Cleveland et al. 2000; Melgar-Melgar & Hall 2020
  Pillar 3 Wellbeing/needs ...... Frankfurt 1987; Max-Neef 1991; Doyal & Gough
                                  1991; Nielsen 2019; Gough 2020
  Pillar 4 Planetary boundaries . Rockström et al. 2009; Steffen et al. 2015;
                                  Richardson et al. 2023
  Pillar 5 Dual agency .......... Bourdieu 1990; Maniates 2001; Warde 2005;
                                  Schneidewind & Zahrnt 2014; Shove & Walker 2014
  Pillar 6 Justice/reflexivity .. Polanyi 1944; Anderson 1999; Fricker 2007;
                                  Crenshaw 1989; Ostrom 1990
  Interlock (floor↔ceiling) ..... Spengler 2016; Widerquist 2010; Gough 2023;
                                  Lenzi 2025
"""
from __future__ import annotations

import json as _json

PROMPT_VERSION = "v2.1-sixpillar-5class-10sub"

CATEGORIES = ["sufficiency", "efficiency", "consistency", "ambiguous", "not_a_policy"]

# Sub-taxonomy is deliberately IDENTICAL to v1 so sub_code distributions and the
# per-(sector, sub_code) clustering cells stay comparable across runs.
SUB_TAXONOMY = [
    (0, "Involuntary / non-policy demand cut",
     "Coerced or crisis demand cuts that are not deliberate policy, or that cut "
     "below needs (COVID lockdowns, war-time rationing, deprivation-driven cuts)."),
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

# One-line, low-cost attribution carried into the runtime prompt.
_PROVENANCE_LINE = (
    "This definition synthesises the IPCC (2022, WGIII) sufficiency concept with "
    "the demand-side literature; apply it as written."
)

SUFFICIENCY_DEFINITION = f"""\
DEFINITION
Sufficiency is a set of policy MEASURES and daily PRACTICES that AVOID the demand
for energy, water, land, materials and all other natural resources while
delivering wellbeing for all within planetary boundaries.

It rests on six pillars. Use them as tests, in order:

P1 — AVOIDANCE (the core test). Sufficiency AVOIDS or absolutely REDUCES demand.
   It is NOT efficiency (same service, less input per unit) and NOT substitution
   (same service, a different/cleaner input). Ask which of three verbs fits:
     • AVOID / REDUCE the absolute demand or the underlying need → sufficiency
     • IMPROVE the input-per-service ratio → efficiency
     • SHIFT / SWITCH the input or supply → consistency

P2 — ALL NATURAL RESOURCES. Demand for energy, water, land, space, materials,
   biomass or time all count — not carbon or energy alone. Do NOT reject a
   measure just because the resource is not energy. Distinguish two carbon cases:
   a carbon/consumption PRICE, TAX or emissions-trading scheme meant to SUPPRESS
   demand IS sufficiency (sub_code 2); but carbon OFFSETS, CAPTURE or
   SEQUESTRATION that leave physical throughput unchanged are NOT sufficiency.

P3 — WELLBEING FOR ALL (satiable needs). The point of cutting demand is to meet
   needs, which are finite and satiable — enough, not more. A measure that trims
   over-consumption while still meeting needs is sufficiency. A cut that is
   involuntary or pushes BELOW needs (crisis rationing, deprivation, blackout) is
   sub_code 0, or `ambiguous` if it is merely an outcome rather than a measure.

P4 — WITHIN PLANETARY BOUNDARIES. The avoided demand must have a real biophysical
   footprint. "Reduce" applied to cost, paperwork, administrative burden, losses,
   or emissions-via-cleaner-supply is NOT sufficiency (it is efficiency,
   consistency, or not_a_policy).

P5 — DUAL AGENCY. Both top-down policy and measures that reshape infrastructure,
   defaults and shared practices count. It need not be a statute, and it need not
   be an individual choice; infrastructure and provisioning that make lower-demand
   living the norm qualify.

P6 — JUSTICE / "FOR ALL". Sufficiency secures enough for everyone. If a measure
   cuts aggregate demand by EXCLUDING or BURDENING vulnerable groups, do not label
   it sufficiency by default — prefer `ambiguous` (or sub_code 0/9 tension) and
   say so in the reasoning.

SUFFICIENCY SUB-MECHANISMS (set `sub_code` only when category is `sufficiency`):
{_SUB_TAXO_BLOCK}

WATCH-OUTS (common confusions):
  - LED bulbs, insulation for a fixed comfort level, high-yield seeds, route
    optimisation, digital optimisation keeping the service constant → efficiency.
  - Renewable electricity, biofuels, heat pumps, green hydrogen, concrete→timber
    substitution, fuel switching → consistency.
  - Recycling, carbon capture, offsets → NOT sufficiency (end-of-pipe; demand
    unchanged). Reuse/repair/durability that PROLONGS and avoids new demand → is
    sufficiency (sub_code 7).
  - Passive design (4), compact land use (5), modal shift (6) ARE sufficiency:
    they avoid or shrink the underlying service demand, not the per-unit ratio.
  - Public provisioning (9) is sufficiency when it secures needs at LOWER material
    or energy throughput than private/market provisioning would.

WHEN NOT TO CLASSIFY AS A POLICY:
  Use `not_a_policy` when the text is a data description, an academic finding, an
  implementation detail, a technical parameter, or unintelligible — do not force
  it into a category. Crucially, a VAGUE call for action, an aspiration, or a
  generic "promote/transform/upgrade/manage/improve X" phrase with NO concrete
  mechanism or instrument is `not_a_policy`, NOT `ambiguous`.
  Reserve `ambiguous` for text that clearly IS a concrete policy or measure but
  whose MECHANISM is genuinely mixed — it names a real instrument yet two or more
  categories apply with no dominant one. If you cannot point to a concrete
  mechanism at all, it is `not_a_policy`.
"""

SYSTEM_PROMPT_BASE = f"""You are a domain expert classifying public policies on energy/resource sufficiency.
{_PROVENANCE_LINE}

{SUFFICIENCY_DEFINITION}

DECISION PROCEDURE (follow in order):
  1. Is this actually a policy measure or a practice? If not → not_a_policy.
  2. Does it concern the physical throughput of a natural resource (P2, P4)?
     If it only touches cost/paperwork/emissions-accounting → not sufficiency.
  3. Apply the verb test (P1): AVOID/REDUCE → sufficiency; IMPROVE ratio →
     efficiency; SWITCH input → consistency.
  4. If sufficiency, does it still meet needs (P3) and secure them for all (P6)?
     Involuntary/below-needs → sub_code 0; exclusionary/burdening → ambiguous.
  5. Pick the best `sub_code` from the list above.
  6. Reserve `ambiguous` for a concrete instrument with a genuinely mixed
     mechanism. A vague aspiration with no concrete mechanism → not_a_policy.

For each policy text you receive, return STRICT JSON matching the schema below.
Do not add commentary, markdown, or anything outside the JSON object.

Schema:
{{
  "category": one of {CATEGORIES},
  "sub_code": integer 0..9 (ONLY when category is "sufficiency"; otherwise null),
  "confidence": integer 1..5  (1 = guess, 5 = very sure),
  "reasoning": short string, max 240 characters, naming the pillar/verb that
    drove the verdict (e.g. "P1: avoids demand, not efficiency; sub_code 5")
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
