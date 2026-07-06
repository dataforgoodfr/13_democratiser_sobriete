# Prompt v1 vs baseline — expert gold

Model: `deepseek-v4-pro`. 10 few-shot examples excluded → **760 rows** in evaluation set.
- Baseline prompt: `v0-sufficiency-3class-placeholder-def` (3-class, 6 few-shot).
- v1 prompt: `v1-expert-5class-10sub` (5-class + 10-code sub-taxonomy, 10 few-shot from expert gold).

## Headline (binary: sufficiency-related vs not)

| metric | v0 baseline | v1 | Δ |
|---|---:|---:|---:|
| accuracy | 63.4% | **79.3%** | +15.9 pt |
| precision | 33.2% | **51.8%** | +18.6 pt |
| recall | 66.9% | **77.7%** | +10.8 pt |
| F1 | 0.44 | **0.62** | +0.18 |
| strict recall (expert=suff → model=suff) | 34.1% | **81.7%** | +47.6 pt |

5-class exact-match accuracy (v1 only, v0 enum incompatible): **63.3%**

Sub-code exact-match accuracy where both label sufficiency (n=103): **83.5%**

## v1 5-class confusion (rows = expert, cols = v1 model)

| category | ambiguous | consistency | efficiency | not_a_policy | sufficiency | TOTAL |
|---|---|---|---|---|---|---|
| ambiguous | 9 | 1 | 9 | 4 | 17 | 40 |
| consistency | 4 | 51 | 20 | 17 | 14 | 106 |
| efficiency | 1 | 0 | 56 | 8 | 7 | 72 |
| not_a_policy | 28 | 8 | 52 | 262 | 66 | 416 |
| sufficiency | 0 | 0 | 9 | 14 | 103 | 126 |
| TOTAL | 42 | 60 | 146 | 305 | 207 | 760 |

## Per-sector

| sector | n | v0_bin_acc | v1_bin_acc | v1_5class_acc | n_gold_suff | v1_recall_gold_suff |
|---|---|---|---|---|---|---|
| SOCIAL | 69.0 | 0.478 | 0.652 | 0.594 | 38.0 | 0.605 |
| URBAN | 70.0 | 0.529 | 0.7 | 0.629 | 19.0 | 1.0 |
| MATERIALS | 70.0 | 0.471 | 0.714 | 0.443 | 3.0 | 1.0 |
| FOOD | 70.0 | 0.614 | 0.771 | 0.5 | 1.0 | 1.0 |
| BUILDING | 67.0 | 0.657 | 0.776 | 0.537 | 11.0 | 0.727 |
| NATURE | 69.0 | 0.594 | 0.797 | 0.623 | 23.0 | 1.0 |
| ENERGY | 70.0 | 0.857 | 0.829 | 0.686 | 6.0 | 0.833 |
| INDUSTRY | 70.0 | 0.529 | 0.857 | 0.714 | 5.0 | 1.0 |
| MACROECONOMIC | 68.0 | 0.809 | 0.868 | 0.838 | 0.0 | 0.0 |
| MOBILITY | 68.0 | 0.794 | 0.882 | 0.721 | 12.0 | 0.833 |
| LOGISTICS | 69.0 | 0.652 | 0.884 | 0.681 | 8.0 | 0.75 |

## Regressions from v0 → v1 (binary): 65

## New wins v1 vs v0 (binary): 186

### Sample regressions

- **BUILDING_010** expert=not_a_policy v0=not_sufficiency v1=sufficiency: Queensland Heritage Act 1992: Sets up a heritage council to advise the Minister on heritage matters, provides administrative procedures and authority to protect heritage places lis
  - v1 reasoning: matches sub-code 1 (cap/limit/ban): regulatory ceiling on demolition/alteration of heritage places, preserving existing building stock
- **BUILDING_019** expert=not_a_policy v0=not_sufficiency v1=ambiguous: Adoption of innovative solutions to reduce the environmental impact of buildings
  - v1 reasoning: text IS a policy but mechanism is mixed/unclear across categories
- **BUILDING_020** expert=efficiency v0=not_sufficiency v1=sufficiency: Passive house standard for residential buildings
  - v1 reasoning: matches sub-code 4 (passive/climate-responsive design avoids active system demand)
- **BUILDING_025** expert=sufficiency v0=potential_sufficiency v1=efficiency: Integration of threshold values for the materials proposed in the planning process for new buildings
  - v1 reasoning: threshold values for materials likely aim to improve performance per unit (e.g., embodied carbon limits) rather than cap absolute demand
- **BUILDING_031** expert=not_a_policy v0=not_sufficiency v1=sufficiency: Mandating radon activity concentration measurements in dwellings to avoid critical situations
  - v1 reasoning: matches sub-code 1 (cap/limit/ban): regulatory ceiling on radon exposure, an absolute limit
- **BUILDING_040** expert=not_a_policy v0=not_sufficiency v1=sufficiency: The maximized use of roof space
  - v1 reasoning: matches sub-code 5 (proximity, compactness & land sufficiency): maximizing roof space avoids sprawl and reduces land demand
- **BUILDING_045** expert=not_a_policy v0=not_sufficiency v1=sufficiency: Provision of sufficient public utilities, such as portable water and electric supply
  - v1 reasoning: matches sub-code 9 (public provisioning securing needs at low throughput)
- **BUILDING_048** expert=not_a_policy v0=not_sufficiency v1=sufficiency: Implement regular maintenance and inspections for educational facilities
  - v1 reasoning: matches sub-code 7 (share, reuse, repair & prolong): maintenance prolongs building lifespan, avoiding new construction demand
- **ENERGY_011** expert=ambiguous v0=potential_sufficiency v1=not_a_policy: Control strategies including stepwise, long-term, region-wide and emergency measures
  - v1 reasoning: describes types of control strategies, not a specific policy instrument
- **ENERGY_035** expert=not_a_policy v0=not_sufficiency v1=ambiguous: Carbon neutrality by 2050
  - v1 reasoning: Carbon neutrality can be achieved via efficiency, consistency, or sufficiency; no dominant mechanism specified.

### Sample new wins

- **BUILDING_005** expert=not_a_policy v0=potential_sufficiency v1=not_a_policy: Education and training programs
- **BUILDING_007** expert=not_a_policy v0=potential_sufficiency v1=efficiency: Excel tool for selecting less impactful solutions for building design
- **BUILDING_009** expert=not_a_policy v0=potential_sufficiency v1=efficiency: Use of POE guideline to improve the performance of government and public buildings in Malaysia.
- **BUILDING_011** expert=not_a_policy v0=potential_sufficiency v1=efficiency: Use occupancy sensors, manual counts, or devices worn by occupants to record their locations for more accurate assessments of building occupancy.
- **BUILDING_012** expert=sufficiency v0=not_sufficiency v1=sufficiency: Implement shading devices to minimize solar radiation inside buildings
- **BUILDING_026** expert=not_a_policy v0=sufficiency v1=not_a_policy: ICRP recommendation of an effective dose of 50 < E ≤ 100 mSv/year
- **BUILDING_027** expert=not_a_policy v0=potential_sufficiency v1=not_a_policy: Building capacity or maintenance management systems
- **BUILDING_029** expert=not_a_policy v0=potential_sufficiency v1=not_a_policy: National government should make it clear to stakeholders that the compulsory Energy Performance Indicators (EPIs) only reflect building-related energy use under standard user condi
- **BUILDING_036** expert=sufficiency v0=not_sufficiency v1=sufficiency: Use of thermal inertia in building constructions to reduce peak heating or cooling loads
- **BUILDING_038** expert=not_a_policy v0=potential_sufficiency v1=not_a_policy: Enhanced public education on green building practices

## Sub-code performance on sufficiency

Rows = expert sub-code, cols = v1 sub-code (only where both label `sufficiency`):

| expert_code | 0.0 | 1.0 | 2.0 | 3.0 | 4.0 | 5.0 | 6.0 | 7.0 | 8.0 | 9.0 | TOTAL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| 1 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 25 |
| 2 | 0 | 0 | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 |
| 3 | 0 | 2 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 5 |
| 4 | 0 | 0 | 0 | 0 | 7 | 0 | 0 | 0 | 0 | 0 | 7 |
| 5 | 0 | 2 | 0 | 0 | 0 | 13 | 1 | 0 | 0 | 0 | 16 |
| 6 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 2 | 0 | 0 | 10 |
| 7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 0 | 0 | 4 |
| 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 1 |
| 9 | 0 | 2 | 0 | 2 | 3 | 1 | 1 | 0 | 0 | 19 | 28 |
| TOTAL | 5 | 30 | 2 | 5 | 10 | 14 | 10 | 6 | 2 | 19 | 103 |
