# Prompt v2_1 vs v1 — expert gold, DeepSeek V4

Same model, same 10 few-shot (excluded). Scored on **760 rows** where both versions returned a valid label.

## Headline (binary: sufficiency-related vs not)

| metric | v1 | v2_1 | Δ |
|---|---:|---:|---:|
| accuracy | 79.3% | **82.5%** | +3.2 pt |
| precision | 51.8% | **59.3%** | +7.5 pt |
| recall | 77.7% | **63.3%** | -14.5 pt |
| F1 | 0.62 | **0.61** | -0.01 |
| strict recall (expert=suff→model=suff) | 81.7% | **69.0%** | -12.7 pt |

5-class exact-match: v1 **63.3%** → v2_1 **68.0%** (+4.7 pt)

Sub-code exact-match on sufficiency: v1 **83.5%** (n=103) → v2_1 **85.1%** (n=87)

Confusion matrices (binary): v1 TP=129 FP=120 FN=37 TN=474 | v2_1 TP=105 FP=72 FN=61 TN=522

## v2_1 5-class confusion (rows = expert, cols = v2_1 model)

| category | ambiguous | consistency | efficiency | not_a_policy | sufficiency | TOTAL |
|---|---|---|---|---|---|---|
| ambiguous | 4 | 1 | 5 | 16 | 14 | 40 |
| consistency | 5 | 49 | 14 | 27 | 11 | 106 |
| efficiency | 0 | 1 | 52 | 12 | 7 | 72 |
| not_a_policy | 5 | 8 | 34 | 325 | 44 | 416 |
| sufficiency | 0 | 0 | 8 | 31 | 87 | 126 |
| TOTAL | 14 | 59 | 113 | 411 | 163 | 760 |

## Per-sector (binary accuracy, v1 vs v2_1)

| sector | n | v1_bin | v2_1_bin | Δ | v1_5cls | v2_1_5cls |
|---|---|---|---|---|---|---|
| URBAN | 70.0 | 0.7 | 0.614 | -0.086 | 0.629 | 0.571 |
| SOCIAL | 69.0 | 0.652 | 0.623 | -0.029 | 0.594 | 0.565 |
| NATURE | 69.0 | 0.797 | 0.783 | -0.014 | 0.623 | 0.609 |
| MOBILITY | 68.0 | 0.882 | 0.882 | 0.0 | 0.721 | 0.75 |
| LOGISTICS | 69.0 | 0.884 | 0.913 | 0.029 | 0.681 | 0.797 |
| FOOD | 70.0 | 0.771 | 0.814 | 0.043 | 0.5 | 0.614 |
| INDUSTRY | 70.0 | 0.857 | 0.914 | 0.057 | 0.714 | 0.8 |
| ENERGY | 70.0 | 0.829 | 0.9 | 0.071 | 0.686 | 0.7 |
| MACROECONOMIC | 68.0 | 0.868 | 0.941 | 0.074 | 0.838 | 0.882 |
| BUILDING | 67.0 | 0.776 | 0.851 | 0.075 | 0.537 | 0.582 |
| MATERIALS | 70.0 | 0.714 | 0.843 | 0.129 | 0.443 | 0.614 |

## Movement v1 → v2_1 (binary): 58 new wins, 34 regressions

### Sample new wins (v2_1 fixed what v1 got wrong)

- **BUILDING_001** expert=not_a_policy v1=ambiguous → v2_1=not_a_policy: Sustainable operation and management of buildings considering economic, environmental, and social aspects
  - v2_1: Vague aspiration with no concrete mechanism; not a policy instrument.
- **BUILDING_019** expert=not_a_policy v1=ambiguous → v2_1=not_a_policy: Adoption of innovative solutions to reduce the environmental impact of buildings
  - v2_1: Vague aspiration with no concrete mechanism; 'adoption of innovative solutions' is too generic to classify.
- **BUILDING_028** expert=not_a_policy v1=sufficiency → v2_1=not_a_policy: Environmental education of the society in the use of alternative solutions of building thermoregulation.
  - v2_1: Vague call for education with no concrete mechanism; not a specific policy instrument.
- **BUILDING_031** expert=not_a_policy v1=sufficiency → v2_1=not_a_policy: Mandating radon activity concentration measurements in dwellings to avoid critical situations
  - v2_1: Radon measurement is a monitoring/health-safety measure, not a policy that avoids demand for natural resources (P1, P2).
- **BUILDING_043** expert=not_a_policy v1=sufficiency → v2_1=not_a_policy: Introduction of the Cold Weather Plan for England
  - v2_1: Vague plan name without concrete mechanism; no specific instrument described to classify.
- **BUILDING_048** expert=not_a_policy v1=sufficiency → v2_1=efficiency: Implement regular maintenance and inspections for educational facilities
  - v2_1: P1: maintenance improves input-per-service ratio (extends building life, reduces energy waste) rather than avoiding demand; not sufficiency.
- **ENERGY_008** expert=not_a_policy v1=ambiguous → v2_1=not_a_policy: Less integrated systems scenarios
  - v2_1: Vague phrase with no concrete mechanism; not a policy instrument.
- **ENERGY_014** expert=not_a_policy v1=ambiguous → v2_1=not_a_policy: Promoting economic development that does not harm the environment
  - v2_1: Vague aspiration with no concrete mechanism or instrument.
- **ENERGY_035** expert=not_a_policy v1=ambiguous → v2_1=not_a_policy: Carbon neutrality by 2050
  - v2_1: Vague aspirational goal with no concrete mechanism or instrument specified.
- **ENERGY_038** expert=consistency v1=ambiguous → v2_1=not_a_policy: Technological innovation in sectors of all aspects, such as reforming the current energy and production system and enhancing atmospheric CO2 capture
  - v2_1: Vague aspiration with no concrete mechanism; 'reforming' and 'enhancing' are generic calls for action, not a policy instrument.
- **ENERGY_041** expert=not_a_policy v1=ambiguous → v2_1=not_a_policy: Policy measures to implement the energy transition
  - v2_1: Vague call for action with no concrete mechanism; not a specific policy instrument.
- **ENERGY_052** expert=not_a_policy v1=ambiguous → v2_1=not_a_policy: Legislation to incentivize mitigation activity
  - v2_1: Vague aspiration with no concrete mechanism; 'incentivize mitigation activity' does not specify an instrument.

### Sample regressions (v2_1 broke what v1 got right)

- **BUILDING_026** expert=not_a_policy v1=not_a_policy → v2_1=sufficiency: ICRP recommendation of an effective dose of 50 < E ≤ 100 mSv/year
  - v2_1: P1: avoids demand by setting an absolute cap on radiation dose; sub_code 1 (caps, limits & bans)
- **ENERGY_009** expert=not_a_policy v1=not_a_policy → v2_1=sufficiency: Organizing strategic reserves at the national level.
  - v2_1: P1: strategic reserves cap extraction/use, avoiding demand. P2: energy resource. Sub_code 1 (caps/limits/bans).
- **ENERGY_068** expert=ambiguous v1=ambiguous → v2_1=consistency: Use of renewable energy and energy-efficient technologies by hotels and resorts
  - v2_1: P1: 'use of renewable energy' is substitution (cleaner input), 'energy-efficient technologies' is efficiency (same service, less input). Neither avoids absolute demand.
- **FOOD_006** expert=ambiguous v1=ambiguous → v2_1=not_a_policy: More sustainable agricultural practices
  - v2_1: Vague aspiration with no concrete mechanism; not a specific policy instrument.
- **FOOD_026** expert=ambiguous v1=ambiguous → v2_1=not_a_policy: Legislation forcing different farm practices, affecting the productivity of crops and pasture production
  - v2_1: Vague aspiration with no concrete mechanism; 'forcing different farm practices' is too generic to classify.
- **FOOD_050** expert=sufficiency v1=sufficiency → v2_1=not_a_policy: Promote extensive environment-friendly farming practices that can promote biodiversity and reduce the pressure of agriculture on the ecosystem
  - v2_1: Vague aspiration with no concrete mechanism; 'promote' without specific instrument is not a policy.
- **INDUSTRY_049** expert=sufficiency v1=sufficiency → v2_1=not_a_policy: Issue more policies that relate to the consumption stage of the textile value chain
  - v2_1: Vague call for action with no concrete mechanism or instrument.
- **LOGISTICS_033** expert=ambiguous v1=ambiguous → v2_1=not_a_policy: Establishing comprehensive regulations as well as indecisive green modes of transport
  - v2_1: Vague aspiration with no concrete mechanism; 'indecisive green modes' is unclear and not a specific instrument.
- **LOGISTICS_045** expert=sufficiency v1=sufficiency → v2_1=efficiency: Use of line standardised transport combined with multimodal transport and third-party logistics
  - v2_1: P1: improves input-per-service ratio (optimises logistics) rather than avoiding absolute demand; multimodal shift alone would be sufficiency but combined with standardisation and 3PL it is efficiency.
- **LOGISTICS_069** expert=sufficiency v1=sufficiency → v2_1=efficiency: Implementation of a tracking system to returnable packaging to identify the location and estimated time of arrival
  - v2_1: P1: tracking system improves logistics efficiency (same service with less input) rather than avoiding demand.
- **MATERIALS_011** expert=ambiguous v1=ambiguous → v2_1=not_a_policy: EU policies in the areas of water, waste and air management
  - v2_1: Vague reference to policy areas with no concrete mechanism; cannot determine if it avoids, improves, or shifts demand.
- **MOBILITY_028** expert=not_a_policy v1=not_a_policy → v2_1=sufficiency: Implementation of wildlife fences to mitigate roadkill accidents
  - v2_1: P1: avoids demand for land/resource by preventing wildlife-vehicle collisions; sub_code 1 (cap/limit/ban) as it restricts animal access to roads.
