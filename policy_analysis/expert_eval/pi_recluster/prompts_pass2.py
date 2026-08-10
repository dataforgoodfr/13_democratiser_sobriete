"""Pass-2 prompt: v1 + expert-eval calibration rules (post-clustering replay).

Motivation (expert cluster review, 2026-06-27, 90 clusters + 2,223 singletons):
the v1-on-Gemma corpus carries systematic false positives that survive into
coherent-looking clusters. The dominant patterns:

  1. Lexical false-match on code 3 "Reduce & right size": reducing a
     *behaviour or health metric* (sedentary time, body weight, crime,
     organisational downsizing) is not consumption right-sizing. Worst-rated
     real sub-code in the expert review (mean quality 2.36/5).
  2. Lexical false-match on code 7: knowledge/experience/peer "sharing" is
     not product sharing.
  3. Anti-sufficiency direction: texts that *increase* material throughput
     (expand production, more disposables) classified as sufficiency.
  4. Awareness/education campaigns with no consumption target.
  5. Vague aspirations (the v2.1 tightening rule — the only prompt change
     that helped Gemma in iteration 5).

Per iteration 5, Gemma 4 12B responds to short calibration rules, not richer
definitions — so this module keeps the v1 definition verbatim and appends only
a DECISION RULES block. Everything else (schema, few-shot handling, user
prompt) is inherited from prompts_v1.

Select with PROMPTS_VERSION=pass2. Pair with build_pass2_input.py, which
replays the clustered corpus (not the raw extraction) through the classifier.
"""
from __future__ import annotations

from prompts_v1 import (  # noqa: F401  (re-exported for classify_vllm.py)
    CATEGORIES,
    RESPONSE_JSON_SCHEMA,
    SUB_TAXONOMY,
    SUFFICIENCY_DEFINITION,
    SYSTEM_PROMPT_BASE,
    USER_PROMPT_TEMPLATE,
    format_user_prompt,
)
from prompts_v1 import build_system_prompt as _build_system_prompt_v1

PROMPT_VERSION = "pass2-postcluster-tightened"

TIGHTENING_RULES = """
DECISION RULES (apply after the definition; they override lexical similarity):

R1. "Reduce" must target consumption of energy, materials, space, or a
    material service. Reducing a behaviour, a health metric, or an
    organisational quantity (sedentary time, body weight, crime, stress,
    corporate headcount/downsizing) is NOT sufficiency -> classify by what
    the policy actually regulates (usually not_a_policy for health-behaviour
    interventions in this corpus).
R2. "Sharing" counts (code 7) only for physical products, vehicles, spaces,
    or equipment. Knowledge-sharing, experience-sharing, peer-support and
    data-sharing are NOT sufficiency.
R3. Direction check: if the text increases or expands production,
    consumption, or disposable use, it is NEVER sufficiency, whatever
    vocabulary it uses.
R4. Awareness, education, communication or promotion campaigns are
    sufficiency ONLY if the promoted change is itself an absolute-demand
    reduction (e.g. campaign to cut meat consumption -> 8). Generic
    environmental-responsibility or healthy-lifestyle campaigns are not.
R5. A vague aspiration, call for action, or generic
    "transformation/management/upgrade" with no concrete mechanism is
    not_a_policy, not ambiguous.
R6. Efficiency traps: same service with less input per unit (insulation,
    LEDs, yield gains) is efficiency; input substitution (renewables,
    biofuels, heat pumps) is consistency. Neither is sufficiency, even
    inside an otherwise sufficiency-flavoured text.
"""

_SYSTEM_PROMPT_PASS2 = SYSTEM_PROMPT_BASE + TIGHTENING_RULES


def build_system_prompt(few_shot: list[dict] | None = None) -> str:
    """v1 prompt + rules block, few-shot appended after the rules."""
    base = _build_system_prompt_v1(few_shot)
    # splice the rules in right after the base system prompt, before few-shot
    return base.replace(SYSTEM_PROMPT_BASE, _SYSTEM_PROMPT_PASS2, 1)
