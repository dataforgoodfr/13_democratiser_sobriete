# Expert-gold baseline — current 3-class classifier

Model: `deepseek-v4-pro`, prompt `v0-sufficiency-3class-placeholder-def`, 6 few-shot examples.
Gold: 770 expert-labelled rows, 11 sectors × 70 rows.

## Headline

Binary framing (`sufficiency_related` = expert `sufficiency` ∪ `ambiguous`, model `sufficiency` ∪ `potential_sufficiency`):

- accuracy   **63.2%**
- precision  **33.4%** (113 / 338)
- recall     **66.1%** (113 / 171)
- F1         **0.44**

Strict recall on expert `sufficiency` only (model must also say `sufficiency`): **33.1%** (43 / 130)

## Expert category × model category (rows = expert, cols = model)

| category | not_sufficiency | potential_sufficiency | sufficiency | TOTAL |
|---|---|---|---|---|
| ambiguous | 23 | 15 | 3 | 41 |
| consistency | 94 | 12 | 2 | 108 |
| efficiency | 63 | 6 | 5 | 74 |
| not_a_policy | 217 | 191 | 9 | 417 |
| sufficiency | 35 | 52 | 43 | 130 |
| TOTAL | 432 | 276 | 62 | 770 |

## Per-sector

| sector | n | binary_acc | n_gold_suff | recall_gold_suff |
|---|---|---|---|---|
| MATERIALS | 70.0 | 0.471 | 3.0 | 0.333 |
| SOCIAL | 70.0 | 0.486 | 39.0 | 0.077 |
| INDUSTRY | 70.0 | 0.529 | 5.0 | 0.8 |
| URBAN | 70.0 | 0.529 | 19.0 | 0.211 |
| NATURE | 70.0 | 0.586 | 24.0 | 0.708 |
| FOOD | 70.0 | 0.614 | 1.0 | 0.0 |
| BUILDING | 70.0 | 0.643 | 12.0 | 0.333 |
| LOGISTICS | 70.0 | 0.643 | 8.0 | 0.0 |
| MACROECONOMIC | 70.0 | 0.8 | 0.0 | 0.0 |
| MOBILITY | 70.0 | 0.8 | 13.0 | 0.385 |
| ENERGY | 70.0 | 0.857 | 6.0 | 0.833 |

## Where the current schema hurts

1. **`Not a policy` (417 rows) is not modeled.** The 3-class enum cannot express 'this is junk, not a policy'; those rows are forced into `not_sufficiency` (best case) or a sufficiency bucket (false positive). Adding a 5th enum value alone would remove ~54% of the noise from downstream.
2. **`Efficiency` and `Consistency` are conflated with `not_sufficiency`.** Fine for the binary problem but destroys the mechanism-level split experts actually care about.
3. **`potential_sufficiency` has no counterpart in the expert schema.** It's how the model expresses hedging; the closest expert equivalent is `Ambiguous`. Look at the crosstab column for `potential_sufficiency` — the mass is spread across every expert class.

## Bad cases

### Missed (expert=sufficiency, model=not_sufficiency): 35

- **BUILDING_012** (BUILDING): Implement shading devices to minimize solar radiation inside buildings
  - model reasoning: shading reduces cooling load per unit of service (efficiency), not absolute demand for cooling
- **BUILDING_036** (BUILDING): Use of thermal inertia in building constructions to reduce peak heating or cooling loads
  - model reasoning: thermal inertia improves efficiency by reducing peak loads for the same comfort level, not absolute demand reduction
- **BUILDING_039** (BUILDING): Implementation of cool roofs in buildings
  - model reasoning: cool roofs reduce cooling load through material change, not absolute demand reduction; classic efficiency/substitution
- **BUILDING_049** (BUILDING): Application of phase change materials (PCMs) to the lightweight building envelope to reduce thermal loads and improve indoor temperature regulation
  - model reasoning: PCMs improve thermal performance without reducing service demand; classic efficiency measure
- **BUILDING_063** (BUILDING): Application of rainwater collection or grey water-purification systems
  - model reasoning: substitution of water sources, not absolute demand reduction
- **BUILDING_067** (BUILDING): 1966 regulations on external walls with greater thermal mass
  - model reasoning: greater thermal mass improves energy efficiency (same comfort with less heating/cooling), not absolute demand reduction
- **BUILDING_069** (BUILDING): Using 30–50% of facades as evaporative functional facades
  - model reasoning: evaporative facades improve building efficiency (cooling with less energy), not absolute demand reduction
- **LOGISTICS_015** (LOGISTICS): Repairs, modifications, changes to shipping logistics, and transportation substitutions (e.g., switching from barge to rail transport)
  - model reasoning: repairs, modifications, and modal shifts aim to improve efficiency or substitute transport modes, not reduce absolute demand for logistics services
- **LOGISTICS_018** (LOGISTICS): MARPOL Annex V: Prevention of Pollution by Garbage from Ships, dealing with different types of garbage and specifying the distances from land and the manner in which they may be di
  - model reasoning: Regulates disposal methods, not absolute reduction in material demand or consumption.
- **LOGISTICS_045** (LOGISTICS): Use of line standardised transport combined with multimodal transport and third-party logistics
  - model reasoning: optimises logistics through standardisation and multimodality, aiming for efficiency, not absolute demand reduction

### Hallucinated sufficiency on `Not a policy`: 9

- **BUILDING_026** (BUILDING): ICRP recommendation of an effective dose of 50 < E ≤ 100 mSv/year
  - model reasoning: dose limit caps absolute radiation exposure, directly reducing demand for medical imaging services
- **ENERGY_018** (ENERGY): Heat allowances for energy vulnerable consumers
  - model reasoning: Heat allowances cap energy use for vulnerable consumers, directly limiting absolute demand.
- **MATERIALS_036** (MATERIALS): Adequate use of chlorination for laundry
  - model reasoning: Adequate use implies reducing excess chlorine consumption, an absolute reduction in material demand through behavioral change.
- **MOBILITY_008** (MOBILITY): Closure of transportation infrastructures connecting different parts of Sondrio during floods
  - model reasoning: closure reduces absolute transport demand during floods, not efficiency or substitution
- **SOCIAL_012** (SOCIAL): Behavioral intervention aimed to advance bedtime earlier into the evening (the Transdiagnostic Sleep and Circadian Intervention; TranS-C)
  - model reasoning: behavioural change that reduces absolute energy demand by shifting sleep schedule, not improving efficiency of lighting/heating
- **SOCIAL_037** (SOCIAL): Borrowing livestock from friends or relatives
  - model reasoning: sharing/borrowing reduces absolute demand for new livestock ownership, a structural/behavioural change for less consumption
- **URBAN_003** (URBAN): Government measures in response to the COVID-19 pandemic that reduced anthropogenic activities (industrial, transportation) during the DLP-2020.
  - model reasoning: Measures directly reduced absolute demand for transportation and industrial activities, not efficiency or substitution.
- **URBAN_009** (URBAN): Restricting the use of green spaces during pandemics to mitigate disease transmission
  - model reasoning: restricting use of green spaces reduces absolute demand for that service during pandemics
- **URBAN_038** (URBAN): Implementation of restrictive measures in residential areas to reduce noise pollution
  - model reasoning: restrictive measures directly reduce absolute noise levels, a form of demand reduction
