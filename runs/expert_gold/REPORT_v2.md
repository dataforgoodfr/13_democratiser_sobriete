# Prompt v2 (six-pillar) vs v1 — expert gold, DeepSeek V4

Same model, same 10 few-shot (excluded). Scored on **760 rows** where both versions returned a valid label.
- v1: `v1-expert-5class-10sub` (single avoid/improve/shift discriminator)
- v2: `v2-sixpillar-5class-10sub` (six-pillar operational definition)

## Headline (binary: sufficiency-related vs not)

| metric | v1 | v2 | Δ |
|---|---:|---:|---:|
| accuracy | 79.3% | **81.1%** | +1.7 pt |
| precision | 51.8% | **54.7%** | +2.9 pt |
| recall | 77.7% | **76.5%** | -1.2 pt |
| F1 | 0.62 | **0.64** | +0.02 |
| strict recall (expert=suff→model=suff) | 81.7% | **79.4%** | -2.4 pt |

5-class exact-match: v1 **63.3%** → v2 **64.1%** (+0.8 pt)

Sub-code exact-match on sufficiency: v1 **83.5%** (n=103) → v2 **84.0%** (n=100)

Confusion matrices (binary): v1 TP=129 FP=120 FN=37 TN=474 | v2 TP=127 FP=105 FN=39 TN=489

## v2 5-class confusion (rows = expert, cols = v2 model)

| category | ambiguous | consistency | efficiency | not_a_policy | sufficiency | TOTAL |
|---|---|---|---|---|---|---|
| ambiguous | 13 | 2 | 9 | 3 | 13 | 40 |
| consistency | 7 | 55 | 19 | 11 | 14 | 106 |
| efficiency | 3 | 3 | 56 | 5 | 5 | 72 |
| not_a_policy | 26 | 10 | 67 | 263 | 50 | 416 |
| sufficiency | 1 | 0 | 10 | 15 | 100 | 126 |
| TOTAL | 50 | 70 | 161 | 297 | 182 | 760 |

## Per-sector (binary accuracy, v1 vs v2)

| sector | n | v1_bin | v2_bin | Δ | v1_5cls | v2_5cls |
|---|---|---|---|---|---|---|
| NATURE | 69.0 | 0.797 | 0.725 | -0.072 | 0.623 | 0.609 |
| URBAN | 70.0 | 0.7 | 0.657 | -0.043 | 0.629 | 0.614 |
| INDUSTRY | 70.0 | 0.857 | 0.843 | -0.014 | 0.714 | 0.714 |
| LOGISTICS | 69.0 | 0.884 | 0.884 | 0.0 | 0.681 | 0.667 |
| FOOD | 70.0 | 0.771 | 0.8 | 0.029 | 0.5 | 0.486 |
| MOBILITY | 68.0 | 0.882 | 0.912 | 0.029 | 0.721 | 0.75 |
| BUILDING | 67.0 | 0.776 | 0.806 | 0.03 | 0.537 | 0.537 |
| ENERGY | 70.0 | 0.829 | 0.871 | 0.043 | 0.686 | 0.729 |
| SOCIAL | 69.0 | 0.652 | 0.696 | 0.043 | 0.594 | 0.609 |
| MACROECONOMIC | 68.0 | 0.868 | 0.926 | 0.059 | 0.838 | 0.868 |
| MATERIALS | 70.0 | 0.714 | 0.8 | 0.086 | 0.443 | 0.471 |

## Movement v1 → v2 (binary): 34 new wins, 21 regressions

### Sample new wins (v2 fixed what v1 got wrong)

- **BUILDING_028** expert=not_a_policy v1=sufficiency → v2=not_a_policy: Environmental education of the society in the use of alternative solutions of building thermoregulation.
  - v2: Environmental education is a broad awareness measure, not a concrete policy instrument; lacks specific demand-avoidance mechanism.
- **BUILDING_031** expert=not_a_policy v1=sufficiency → v2=not_a_policy: Mandating radon activity concentration measurements in dwellings to avoid critical situations
  - v2: Measurement mandate is a monitoring/implementation detail, not a policy that directly avoids resource demand.
- **BUILDING_043** expert=not_a_policy v1=sufficiency → v2=not_a_policy: Introduction of the Cold Weather Plan for England
  - v2: The Cold Weather Plan is a public health advisory framework, not a direct resource-demand measure; it does not fit the sufficiency/efficiency/consistency classification.
- **BUILDING_048** expert=not_a_policy v1=sufficiency → v2=efficiency: Implement regular maintenance and inspections for educational facilities
  - v2: P1: maintenance improves input-per-service ratio (extends asset life, same service with less material/energy over time), not absolute demand avoidance.
- **ENERGY_008** expert=not_a_policy v1=ambiguous → v2=not_a_policy: Less integrated systems scenarios
  - v2: descriptive scenario label, not a policy measure
- **ENERGY_011** expert=ambiguous v1=not_a_policy → v2=ambiguous: Control strategies including stepwise, long-term, region-wide and emergency measures
  - v2: P1: 'control strategies' could include efficiency, consistency, or sufficiency measures; no single verb dominates.
- **ENERGY_038** expert=consistency v1=ambiguous → v2=consistency: Technological innovation in sectors of all aspects, such as reforming the current energy and production system and enhancing atmospheric CO2 capture
  - v2: P1: reforming energy system and CO2 capture are substitution/end-of-pipe, not absolute demand avoidance.
- **ENERGY_052** expert=not_a_policy v1=ambiguous → v2=not_a_policy: Legislation to incentivize mitigation activity
  - v2: descriptive statement / data / implementation detail, not a policy instrument
- **FOOD_024** expert=not_a_policy v1=ambiguous → v2=consistency: Strengthening certification schemes to encourage widespread adoption of sustainable practices
  - v2: P1: shifts to sustainable practices (cleaner input), not absolute demand reduction
- **FOOD_028** expert=not_a_policy v1=sufficiency → v2=not_a_policy: Fine of 100-300 thousand rubles for legal entities who do not comply with the requirement to label products with GMOs
  - v2: Labeling requirement is an information measure, not a direct demand-avoidance policy; it does not reduce physical throughput of resources.
- **FOOD_037** expert=ambiguous v1=efficiency → v2=ambiguous: Adoption of sustainable soil measures
  - v2: P1: 'sustainable soil measures' could involve efficiency (improve input ratio), consistency (shift inputs), or sufficiency (avoid demand). Without specifics, mechanism is mixed.
- **INDUSTRY_009** expert=not_a_policy v1=ambiguous → v2=not_a_policy: Ethical scoring matrices for procurement activities, assessing projects on moral impact, equity, and societal contribution
  - v2: Describes a tool/method (scoring matrix) rather than a concrete policy measure; no direct resource demand avoidance.

### Sample regressions (v2 broke what v1 got right)

- **BUILDING_026** expert=not_a_policy v1=not_a_policy → v2=sufficiency: ICRP recommendation of an effective dose of 50 < E ≤ 100 mSv/year
  - v2: P1: avoids demand by setting an absolute cap on radiation dose; sub_code 1 (caps, limits & bans)
- **BUILDING_041** expert=efficiency v1=not_a_policy → v2=ambiguous: Rehabilitation of schools—hospitals—monuments
  - v2: Rehabilitation could involve efficiency (insulation), consistency (material substitution), or sufficiency (passive design). Without specifics, mechanism is mixed.
- **ENERGY_068** expert=ambiguous v1=ambiguous → v2=consistency: Use of renewable energy and energy-efficient technologies by hotels and resorts
  - v2: P1: renewable energy is a switch of input (consistency); energy-efficient technologies improve input-per-service ratio (efficiency). No absolute demand avoidance.
- **FOOD_001** expert=not_a_policy v1=not_a_policy → v2=ambiguous: Coupling of the national top-down approach of CSA-targeting with bottom-up initiatives
  - v2: P1 unclear: 'coupling' could mix efficiency (CSA-targeting) with sufficiency (bottom-up initiatives); no single dominant verb.
- **INDUSTRY_035** expert=not_a_policy v1=not_a_policy → v2=ambiguous: Promoting regional industrial transformation and upgrade
  - v2: P1: 'transformation and upgrade' could involve efficiency, consistency, or sufficiency; no clear verb indicates absolute demand avoidance.
- **INDUSTRY_041** expert=efficiency v1=not_a_policy → v2=ambiguous: Judicious management of available resources
  - v2: P1: 'judicious management' could imply efficiency (improve ratio) or sufficiency (avoid demand); unclear without specifics. P2/P4: resource management touches physical throughput, but mechanism is vague.
- **INDUSTRY_049** expert=sufficiency v1=sufficiency → v2=not_a_policy: Issue more policies that relate to the consumption stage of the textile value chain
  - v2: Text is a vague call for more policies, not a concrete policy measure itself.
- **LOGISTICS_026** expert=ambiguous v1=sufficiency → v2=consistency: Maritime-only Emissions Trading System (ETS) for shipping
  - v2: P1: ETS prices carbon but does not avoid/reduce absolute demand; it shifts cost/input, not physical throughput. P2/P4: carbon-only mechanism.
- **LOGISTICS_065** expert=not_a_policy v1=not_a_policy → v2=ambiguous: Investment in port capacities and infrastructure
  - v2: P1: could be efficiency (improve throughput) or consistency (shift to water) or sufficiency (modal shift); mixed mechanism
- **MATERIALS_055** expert=not_a_policy v1=not_a_policy → v2=ambiguous: Water-sensitive programs under deep cultural features
  - v2: P1 unclear: 'water-sensitive programs' could be efficiency, consistency, or sufficiency; 'deep cultural features' is vague. No clear verb.
- **NATURE_013** expert=consistency v1=not_a_policy → v2=ambiguous: Mega-nourishments as a sustainable option for coastal management
  - v2: Mega-nourishments are a coastal management technique; unclear if they avoid demand (sufficiency), improve efficiency, or substitute inputs. Likely mixed.
- **NATURE_022** expert=efficiency v1=efficiency → v2=sufficiency: Establishment of sustainable soil management practices
  - v2: P1: avoids/reduces soil degradation demand; P2: land resource; sub_code 3 (reduce/right-size) fits sustainable soil management
