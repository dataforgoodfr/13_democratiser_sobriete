# Prompt v2_1 vs v2 — expert gold, DeepSeek V4

Same model, same 10 few-shot (excluded). Scored on **760 rows** where both versions returned a valid label.

## Headline (binary: sufficiency-related vs not)

| metric | v2 | v2_1 | Δ |
|---|---:|---:|---:|
| accuracy | 81.1% | **82.5%** | +1.4 pt |
| precision | 54.7% | **59.3%** | +4.6 pt |
| recall | 76.5% | **63.3%** | -13.3 pt |
| F1 | 0.64 | **0.61** | -0.03 |
| strict recall (expert=suff→model=suff) | 79.4% | **69.0%** | -10.3 pt |

5-class exact-match: v2 **64.1%** → v2_1 **68.0%** (+3.9 pt)

Sub-code exact-match on sufficiency: v2 **84.0%** (n=100) → v2_1 **85.1%** (n=87)

Confusion matrices (binary): v2 TP=127 FP=105 FN=39 TN=489 | v2_1 TP=105 FP=72 FN=61 TN=522

## v2_1 5-class confusion (rows = expert, cols = v2_1 model)

| category | ambiguous | consistency | efficiency | not_a_policy | sufficiency | TOTAL |
|---|---|---|---|---|---|---|
| ambiguous | 4 | 1 | 5 | 16 | 14 | 40 |
| consistency | 5 | 49 | 14 | 27 | 11 | 106 |
| efficiency | 0 | 1 | 52 | 12 | 7 | 72 |
| not_a_policy | 5 | 8 | 34 | 325 | 44 | 416 |
| sufficiency | 0 | 0 | 8 | 31 | 87 | 126 |
| TOTAL | 14 | 59 | 113 | 411 | 163 | 760 |

## Per-sector (binary accuracy, v2 vs v2_1)

| sector | n | v2_bin | v2_1_bin | Δ | v2_5cls | v2_1_5cls |
|---|---|---|---|---|---|---|
| SOCIAL | 69.0 | 0.696 | 0.623 | -0.072 | 0.609 | 0.565 |
| URBAN | 70.0 | 0.657 | 0.614 | -0.043 | 0.614 | 0.571 |
| MOBILITY | 68.0 | 0.912 | 0.882 | -0.029 | 0.75 | 0.75 |
| FOOD | 70.0 | 0.8 | 0.814 | 0.014 | 0.486 | 0.614 |
| MACROECONOMIC | 68.0 | 0.926 | 0.941 | 0.015 | 0.868 | 0.882 |
| ENERGY | 70.0 | 0.871 | 0.9 | 0.029 | 0.729 | 0.7 |
| LOGISTICS | 69.0 | 0.884 | 0.913 | 0.029 | 0.667 | 0.797 |
| MATERIALS | 70.0 | 0.8 | 0.843 | 0.043 | 0.471 | 0.614 |
| BUILDING | 67.0 | 0.806 | 0.851 | 0.045 | 0.537 | 0.582 |
| NATURE | 69.0 | 0.725 | 0.783 | 0.058 | 0.609 | 0.609 |
| INDUSTRY | 70.0 | 0.843 | 0.914 | 0.071 | 0.714 | 0.8 |

## Movement v2 → v2_1 (binary): 49 new wins, 38 regressions

### Sample new wins (v2_1 fixed what v2 got wrong)

- **BUILDING_001** expert=not_a_policy v2=ambiguous → v2_1=not_a_policy: Sustainable operation and management of buildings considering economic, environmental, and social aspects
  - v2_1: Vague aspiration with no concrete mechanism; not a policy instrument.
- **BUILDING_019** expert=not_a_policy v2=ambiguous → v2_1=not_a_policy: Adoption of innovative solutions to reduce the environmental impact of buildings
  - v2_1: Vague aspiration with no concrete mechanism; 'adoption of innovative solutions' is too generic to classify.
- **BUILDING_041** expert=efficiency v2=ambiguous → v2_1=not_a_policy: Rehabilitation of schools—hospitals—monuments
  - v2_1: Vague phrase with no concrete mechanism; merely names a domain of activity without specifying an instrument.
- **ENERGY_014** expert=not_a_policy v2=ambiguous → v2_1=not_a_policy: Promoting economic development that does not harm the environment
  - v2_1: Vague aspiration with no concrete mechanism or instrument.
- **ENERGY_035** expert=not_a_policy v2=ambiguous → v2_1=not_a_policy: Carbon neutrality by 2050
  - v2_1: Vague aspirational goal with no concrete mechanism or instrument specified.
- **ENERGY_041** expert=not_a_policy v2=ambiguous → v2_1=not_a_policy: Policy measures to implement the energy transition
  - v2_1: Vague call for action with no concrete mechanism; not a specific policy instrument.
- **ENERGY_053** expert=consistency v2=ambiguous → v2_1=not_a_policy: Measures for energy sector decarbonization
  - v2_1: Vague aspiration with no concrete mechanism; 'decarbonization' is a goal, not a specific policy instrument.
- **FOOD_001** expert=not_a_policy v2=ambiguous → v2_1=not_a_policy: Coupling of the national top-down approach of CSA-targeting with bottom-up initiatives
  - v2_1: Vague coupling of approaches without a concrete mechanism; no specific instrument to avoid/reduce demand.
- **FOOD_014** expert=not_a_policy v2=ambiguous → v2_1=not_a_policy: Policy and regulation on GM crops
  - v2_1: Vague reference to 'policy and regulation' without concrete mechanism; no specific instrument described.
- **FOOD_027** expert=not_a_policy v2=sufficiency → v2_1=not_a_policy: Teach critical thinking so people can assess the social and environmental impacts of their dietary choices.
  - v2_1: Vague aspiration with no concrete mechanism; 'teach critical thinking' is an educational goal, not a specific policy instrument.
- **FOOD_048** expert=not_a_policy v2=ambiguous → v2_1=not_a_policy: Support for agro tourism to promote sustainable development practices
  - v2_1: Vague aspiration with no concrete mechanism; 'support' and 'promote' are too generic to identify a specific policy instrument.
- **FOOD_067** expert=not_a_policy v2=ambiguous → v2_1=not_a_policy: Improved governance (environmental standards, land-use planning, social governance)
  - v2_1: Vague call for 'improved governance' with no concrete mechanism; not a specific policy instrument.

### Sample regressions (v2_1 broke what v2 got right)

- **ENERGY_009** expert=not_a_policy v2=not_a_policy → v2_1=sufficiency: Organizing strategic reserves at the national level.
  - v2_1: P1: strategic reserves cap extraction/use, avoiding demand. P2: energy resource. Sub_code 1 (caps/limits/bans).
- **ENERGY_011** expert=ambiguous v2=ambiguous → v2_1=not_a_policy: Control strategies including stepwise, long-term, region-wide and emergency measures
  - v2_1: Vague list of strategy types without concrete mechanism; no specific instrument named.
- **FOOD_006** expert=ambiguous v2=ambiguous → v2_1=not_a_policy: More sustainable agricultural practices
  - v2_1: Vague aspiration with no concrete mechanism; not a specific policy instrument.
- **FOOD_026** expert=ambiguous v2=ambiguous → v2_1=not_a_policy: Legislation forcing different farm practices, affecting the productivity of crops and pasture production
  - v2_1: Vague aspiration with no concrete mechanism; 'forcing different farm practices' is too generic to classify.
- **FOOD_037** expert=ambiguous v2=ambiguous → v2_1=not_a_policy: Adoption of sustainable soil measures
  - v2_1: Vague call for adoption without concrete mechanism; no specific instrument named.
- **FOOD_050** expert=sufficiency v2=sufficiency → v2_1=not_a_policy: Promote extensive environment-friendly farming practices that can promote biodiversity and reduce the pressure of agriculture on the ecosystem
  - v2_1: Vague aspiration with no concrete mechanism; 'promote' without specific instrument is not a policy.
- **INDUSTRY_026** expert=ambiguous v2=ambiguous → v2_1=not_a_policy: Integration of environmentally friendly practices into industrial production processes
  - v2_1: Vague aspiration with no concrete mechanism; 'integration of environmentally friendly practices' is too generic to classify.
- **LOGISTICS_033** expert=ambiguous v2=ambiguous → v2_1=not_a_policy: Establishing comprehensive regulations as well as indecisive green modes of transport
  - v2_1: Vague aspiration with no concrete mechanism; 'indecisive green modes' is unclear and not a specific instrument.
- **LOGISTICS_045** expert=sufficiency v2=sufficiency → v2_1=efficiency: Use of line standardised transport combined with multimodal transport and third-party logistics
  - v2_1: P1: improves input-per-service ratio (optimises logistics) rather than avoiding absolute demand; multimodal shift alone would be sufficiency but combined with standardisation and 3PL it is efficiency.
- **LOGISTICS_069** expert=sufficiency v2=sufficiency → v2_1=efficiency: Implementation of a tracking system to returnable packaging to identify the location and estimated time of arrival
  - v2_1: P1: tracking system improves logistics efficiency (same service with less input) rather than avoiding demand.
- **MACROECONOMIC_006** expert=not_a_policy v2=not_a_policy → v2_1=sufficiency: Assignment of at least one seat per province in the TGNA, irrespective of the size of its population, favoring rural and agricultural interests.
  - v2_1: P1: avoids demand for urban concentration by ensuring rural representation; P5: policy measure reshaping political infrastructure; sub_code 5 (proximity, compactness & land sufficiency) as it spatially balances power.
- **MACROECONOMIC_054** expert=efficiency v2=consistency → v2_1=sufficiency: Kyoto agreement with explicit emission targets
  - v2_1: P1: absolute cap on emissions (avoidance); P2: targets limit resource throughput; sub_code 1 (caps, limits & bans)
