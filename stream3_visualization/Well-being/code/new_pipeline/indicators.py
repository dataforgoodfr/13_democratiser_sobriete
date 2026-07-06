"""
Indicator Registry
==================
Single source of truth for ALL EWBI indicators.

Each indicator is a dict with:
  code        – unique identifier (e.g. "HQ-SILC-1")
  name        – full English description
  source      – data source: "EU-SILC" | "HBS" | "LFS" | "EHIS"
  variable    – raw source variable(s) used
  condition   – human-readable flag condition
  level       – "household" (uses household weight) or "personal" (uses person weight)
  dimension   – thematic group in EWBI hierarchy
  in_ewbi     – True if already in the published EWBI retained indicator set

To add an indicator: add an entry here and implement the flag logic in the
corresponding extractor (extract_eusilc.py / extract_hbs.py / etc.).
To remove one: set in_ewbi=False and delete its CSV from output/indicators/.
"""

# ─── EWBI retained set (original published list) ─────────────────────────────
EWBI_RETAINED = {
    # Energy & Housing
    "HQ-SILC-1", "HQ-SILC-2", "HQ-SILC-3", "HQ-SILC-4",
    "HQ-SILC-5", "HQ-SILC-6", "HQ-SILC-7", "HQ-SILC-8",
    "HE-SILC-2",
    # Equality
    "ES-SILC-1", "ES-SILC-2",
    "EC-SILC-2", "EC-SILC-3", "EC-SILC-4",
    # Health
    "AH-SILC-2", "AH-SILC-3", "AH-SILC-4",
    "AC-SILC-3", "AC-SILC-4",
    # Education
    "IS-SILC-3", "IS-SILC-4", "IS-SILC-5",
    # Quality of Jobs
    "RT-SILC-1", "RT-SILC-2",
    "RT-LFS-1", "RT-LFS-2", "RT-LFS-3", "RT-LFS-4",
    "RT-LFS-5", "RT-LFS-6", "RT-LFS-7", "RT-LFS-8",
    "RU-LFS-1",
}

# ─── Full indicator list ──────────────────────────────────────────────────────
_RAW = [

    # ══ EU-SILC – HOUSEHOLD ══════════════════════════════════════════════════
    # Weight: DB090  |  Income decile: from HY020

    ("HQ-SILC-1", "Overcrowded dwelling",
     "EU-SILC", "HH030 (computed)", "rooms < required_rooms",
     "household", "Energy & Housing"),

    ("HQ-SILC-2", "Cannot replace worn-out furniture",
     "EU-SILC", "HD080", "in [2,3]",
     "household", "Energy & Housing"),

    ("HQ-SILC-3", "Cannot keep dwelling comfortably cool",
     "EU-SILC", "HC070", "== 2",
     "household", "Energy & Housing"),

    ("HQ-SILC-4", "Dwelling too dark",
     "EU-SILC", "HS160", "== 1",
     "household", "Energy & Housing"),

    ("HQ-SILC-5", "Noise from street or neighbours",
     "EU-SILC", "HS170", "== 1",
     "household", "Energy & Housing"),

    ("HQ-SILC-6", "Leaking roof, damp or rot",
     "EU-SILC", "HH040", "== 1",
     "household", "Energy & Housing"),

    ("HQ-SILC-7", "Pollution or crime in the area",
     "EU-SILC", "HS180", "== 1",
     "household", "Energy & Housing"),

    ("HQ-SILC-8", "No renovation measures",
     "EU-SILC", "HC003", "in [4, 99]",
     "household", "Energy & Housing"),

    ("HE-SILC-1", "Cannot keep dwelling comfortably warm",
     "EU-SILC", "HC060", "== 2",
     "household", "Energy & Housing"),

    ("HE-SILC-2", "Arrears on utility bills",
     "EU-SILC", "HS021", "in [1, 2]",
     "household", "Equality"),

    ("HH-SILC-1", "Arrears on mortgage or rent",
     "EU-SILC", "HS011", "in [1, 2]",
     "household", "Equality"),

    ("AN-SILC-1", "Cannot afford meat/fish/veg every 2nd day",
     "EU-SILC", "HS050", "== 2",
     "household", "Equality"),

    ("ES-SILC-1", "Cannot face unexpected financial expenses",
     "EU-SILC", "HS060", "== 2",
     "household", "Equality"),

    ("ES-SILC-2", "Hard to make ends meet",
     "EU-SILC", "HS120", "in [1, 2]",
     "household", "Equality"),

    ("TS-SILC-1", "Cannot afford 1-week holiday away from home",
     "EU-SILC", "HS040", "== 2",
     "household", "Equality"),

    ("EC-SILC-4", "Persons living alone",
     "EU-SILC", "HH030 (computed)", "household_size == 1",
     "household", "Equality"),

    # ══ EU-SILC – PERSONAL ═══════════════════════════════════════════════════
    # Weight: RB050  |  Income decile: from household HY020 merged via household ID

    ("EL-SILC-1", "Not satisfied with life",
     "EU-SILC", "PW010", "< 3",
     "personal", "Equality"),

    ("AH-SILC-1", "Bad self-perceived health",
     "EU-SILC", "PH010", "in [4, 5]",
     "personal", "Health"),

    ("AH-SILC-2", "Living with a chronic illness",
     "EU-SILC", "PH020", "== 1",
     "personal", "Health"),

    ("AH-SILC-3", "Limited by health problems",
     "EU-SILC", "PH030", "in [1, 2]",
     "personal", "Health"),

    ("AH-SILC-4", "Unable to work due to long-term illness",
     "EU-SILC", "PL086", "> 0",
     "personal", "Health"),

    ("AC-SILC-1", "Could not afford medical care",
     "EU-SILC", "PH050", "== 1",
     "personal", "Health"),

    ("AC-SILC-3", "Unmet need for medical examination",
     "EU-SILC", "PH060", "== 1",
     "personal", "Health"),

    ("AC-SILC-4", "Unmet need for dental examination",
     "EU-SILC", "PH040", "== 1",
     "personal", "Health"),

    ("IC-SILC-1", "Cannot regularly participate in leisure activity",
     "EU-SILC", "PD060", "in [2, 3]",
     "personal", "Equality"),

    ("IC-SILC-2", "Cannot spend small amount on self",
     "EU-SILC", "PD070", "in [2, 3]",
     "personal", "Equality"),

    ("EC-SILC-1", "Cannot meet friends/family monthly",
     "EU-SILC", "PD050", "in [2, 3]",
     "personal", "Equality"),

    ("EC-SILC-2", "Not trusting others",
     "EU-SILC", "PW191", "< 3",
     "personal", "Equality"),

    ("EC-SILC-3", "Cannot get together with friends/family",
     "EU-SILC", "PD050", "in [2, 3]",
     "personal", "Equality"),

    ("IS-SILC-3", "No formal education (age > 15)",
     "EU-SILC", "PE041", "== 0 or NaN (age > 15)",
     "personal", "Education"),

    ("IS-SILC-4", "Not participating in formal training",
     "EU-SILC", "PE010", "== 2",
     "personal", "Education"),

    ("IS-SILC-5", "No secondary education",
     "EU-SILC", "PE041", "in [0, 100]",
     "personal", "Education"),

    ("RT-SILC-1", "Adults on fixed-term contracts",
     "EU-SILC", "PL141", "== 2 (pre-2021) or in [11,12] (2021+), age > 17",
     "personal", "Quality of Jobs"),

    ("RT-SILC-2", "Adults working part-time",
     "EU-SILC", "PL145", "== 2 (age > 17)",
     "personal", "Quality of Jobs"),

    ("RU-SILC-1", "Unemployed for 6+ months",
     "EU-SILC", "PL080", "> 5",
     "personal", "Quality of Jobs"),

    # ══ EU-SILC – SUPPLEMENTARY (ad-hoc waves) ═══════════════════════════════

    ("SP-SILC-1", "Not participating in voluntary activity",
     "EU-SILC", "PS101 (2015) / PS110 (2022)", "== 6",
     "personal", "Social Participation"),

    ("SP-SILC-2", "No active citizenship",
     "EU-SILC", "PS102", "in [2, 3, 4]",
     "personal", "Social Participation"),

    ("GE-SILC-1", "Gender employment gap",
     "EU-SILC", "PL031 / RB211, RB090", "rate(men employed) - rate(women employed), age 20-64",
     "personal", "Equality"),

    ("GE-SILC-2", "Gender pay gap",
     "EU-SILC", "PY010G, RB090", "(1 - median_wage_women_in_decile/median_wage_men_overall)*100, employed age 20-64",
     "personal", "Equality"),

    # ══ HBS – HOUSEHOLD BUDGET SURVEY ════════════════════════════════════════
    # Weight: HA10  |  Income: EUR_HH095  |  Years: 2010, 2015, 2020
    # Indicator = share of households whose equivalized expense (col / HB062)
    # exceeds 2× (or falls below 0.5×) the national decile-balanced weighted
    # median equivalized expense.  Each income decile contributes 1/10 to the
    # national median (2-level weighting: decile balance + HA10 sub-weight).
    # Exception: IE-HBS uses raw HA10 restricted to positive spenders only.

    ("HH-HBS-1", "Rent equiv. expense > 2× national median",
     "HBS", "EUR_HE041", "equiv_expense > 2 × national_median_equiv_expense",
     "household", "Energy & Housing"),

    ("HH-HBS-2", "Rent equiv. expense < 0.5× national median",
     "HBS", "EUR_HE041", "equiv_expense < 0.5 × national_median_equiv_expense",
     "household", "Energy & Housing"),

    ("HH-HBS-3", "Housing & utilities equiv. expense > 2× national median",
     "HBS", "EUR_HE04", "equiv_expense > 2 × national_median_equiv_expense",
     "household", "Energy & Housing"),

    ("HH-HBS-4", "Housing & utilities equiv. expense < 0.5× national median",
     "HBS", "EUR_HE04", "equiv_expense < 0.5 × national_median_equiv_expense",
     "household", "Energy & Housing"),

    ("AC-HBS-1", "Health equiv. expense > 2× national median",
     "HBS", "EUR_HE06", "equiv_expense > 2 × national_median_equiv_expense",
     "household", "Health"),

    ("AC-HBS-2", "Health equiv. expense < 0.5× national median",
     "HBS", "EUR_HE06", "equiv_expense < 0.5 × national_median_equiv_expense",
     "household", "Health"),

    ("AE-HBS-1", "Food equiv. expense > 2× national median",
     "HBS", "EUR_HE01", "equiv_expense > 2 × national_median_equiv_expense",
     "household", "Equality"),

    ("AE-HBS-2", "Food equiv. expense < 0.5× national median",
     "HBS", "EUR_HE01", "equiv_expense < 0.5 × national_median_equiv_expense",
     "household", "Equality"),

    ("EC-HBS-1", "Communications equiv. expense > 1.5× national median (non-zero spenders)",
     "HBS", "EUR_HJ08", "equiv_expense > 1.5 × national_median_equiv_expense (median & flag: non-zero spenders only)",
     "household", "Equality"),

    ("EC-HBS-2", "Communications equiv. expense < 0.2× national median (non-zero spenders)",
     "HBS", "EUR_HJ08", "equiv_expense < 0.2 × national_median_equiv_expense (median & flag: non-zero spenders only)",
     "household", "Equality"),

    ("IE-HBS-1", "Education equiv. expense > 2× national median (non-zero spenders)",
     "HBS", "EUR_HE10", "equiv_expense > 2 × national_median_equiv_expense (median & flag: non-zero spenders only)",
     "household", "Education"),

    ("IE-HBS-2", "Education equiv. expense < 0.5× national median (non-zero spenders)",
     "HBS", "EUR_HE10", "equiv_expense < 0.5 × national_median_equiv_expense (median & flag: non-zero spenders only)",
     "household", "Education"),

    ("TT-HBS-1", "Transport equiv. expense > 2× national median",
     "HBS", "EUR_HE07", "equiv_expense > 2 × national_median_equiv_expense",
     "household", "Equality"),

    ("TT-HBS-2", "Transport equiv. expense < 0.5× national median",
     "HBS", "EUR_HE07", "equiv_expense < 0.5 × national_median_equiv_expense",
     "household", "Equality"),

    ("TS-HBS-1", "Travel & accommodation equiv. expense > 1.5× national median (non-zero spenders)",
     "HBS", "EUR_HJ90", "equiv_expense > 1.5 × national_median_equiv_expense (median & flag: non-zero spenders only)",
     "household", "Equality"),

    ("TS-HBS-2", "Travel & accommodation equiv. expense < 0.2× national median (non-zero spenders)",
     "HBS", "EUR_HJ90", "equiv_expense < 0.2 × national_median_equiv_expense (median & flag: non-zero spenders only)",
     "household", "Equality"),

    ("IC-HBS-1", "Recreation & culture equiv. expense > 2× national median",
     "HBS", "EUR_HE09", "equiv_expense > 2 × national_median_equiv_expense",
     "household", "Equality"),

    ("IC-HBS-2", "Recreation & culture equiv. expense < 0.5× national median",
     "HBS", "EUR_HE09", "equiv_expense < 0.5 × national_median_equiv_expense",
     "household", "Equality"),

    ("HH-HBS-5", "Housing & Energy overburden (shelter+energy > 40% of net income)",
     "HBS", "EUR_HE041+EUR_HE042+EUR_HE045", "(shelter+energy) / EUR_HH099 > 0.40",
     "household", "Energy & Housing"),

    # ══ LFS – LABOUR FORCE SURVEY ════════════════════════════════════════════
    # Weight: COEFFY  |  Income decile: from EU-SILC (merged by country/year/decile)
    # Note: LFS has no own income variable → decile is approximated via STAPRO/ISCO

    ("RT-LFS-1", "Working multiple jobs",
     "LFS", "NUMJOB", "in [2, 3, 4]",
     "personal", "Quality of Jobs"),

    ("RT-LFS-2", "Wishing to work more hours",
     "LFS", "WISHMORE", "== 2",
     "personal", "Quality of Jobs"),

    ("RT-LFS-3", "Overtime or extra hours",
     "LFS", "EXTRAHRS", "> 0",
     "personal", "Quality of Jobs"),

    ("RT-LFS-4", "No flexibility in working time choice",
     "LFS", "VARITIME", "in [3, 4]",
     "personal", "Quality of Jobs"),

    ("RT-LFS-5", "Shift work in main job",
     "LFS", "SHIFTWK", "== 1",
     "personal", "Quality of Jobs"),

    ("RT-LFS-6", "Night work in main job",
     "LFS", "NIGHTWK", "in [1, 2]",
     "personal", "Quality of Jobs"),

    ("RT-LFS-7", "Saturday work in main job",
     "LFS", "SATWK", "in [1, 2]",
     "personal", "Quality of Jobs"),

    ("RT-LFS-8", "Sunday work in main job",
     "LFS", "SUNWK", "in [1, 2]",
     "personal", "Quality of Jobs"),

    ("RU-LFS-1", "Unemployed persons (ILO definition)",
     "LFS", "ILOSTAT", "== 2",
     "personal", "Quality of Jobs"),

    ("EL-LFS-2", "No adequate childcare services available",
     "LFS", "NEEDCARE", "in [1, 2]",
     "personal", "Education"),

    # ══ EHIS – EUROPEAN HEALTH INTERVIEW SURVEY ═══════════════════════════════
    # Weight: WGT (waves 2-3) / PWGT (wave 1)  |  Income: quintile-based
    # Waves: 2006-2009 (wave 1), 2013-2015 (wave 2), 2018-2020 (wave 3)

    ("AN-EHIS-1", "Poor long-term health or chronic condition",
     "EHIS", "HA1A", "in [2, 3, 4]",
     "personal", "Health"),

    ("AE-EHIS-1", "Low fruit consumption (less than once a day)",
     "EHIS", "FV1", "== 4",
     "personal", "Health"),

    ("AE-EHIS-2", "Sleeping problems",
     "EHIS", "DH3", "in [4, 5]",
     "personal", "Health"),

    ("AN-EHIS-2", "Obesity (high BMI category)",
     "EHIS", "BMI", "== 4",
     "personal", "Health"),

    ("EC-EHIS-1", "No social support network",
     "EHIS", "SS1", "== 1",
     "personal", "Equality"),

    ("ED-EHIS-1", "Poor mental health",
     "EHIS", "HA1B", "in [2, 3, 4]",
     "personal", "Health"),

    ("AH-EHIS-2", "No physical activity outside work",
     "EHIS", "PE6", "== 0",
     "personal", "Health"),

    ("AC-EHIS-1", "Unmet dental care needs",
     "EHIS", "UN2C", "== 1",
     "personal", "Health"),

    ("AB-EHIS-1", "Current smoker",
     "EHIS", "SK1", "== 1",
     "personal", "Health"),

    ("AB-EHIS-2", "Hazardous alcohol consumption",
     "EHIS", "AL1", "== 1",
     "personal", "Health"),

    ("AB-EHIS-3", "Difficulty affording healthcare",
     "EHIS", "AC1A", "== 1",
     "personal", "Health"),
]

# ─── Build structured indicator list ─────────────────────────────────────────
INDICATORS: list[dict] = []
for _row in _RAW:
    code, name, source, variable, condition, level, dimension = _row
    INDICATORS.append(dict(
        code=code,
        name=name,
        source=source,
        variable=variable,
        condition=condition,
        level=level,
        dimension=dimension,
        in_ewbi=(code in EWBI_RETAINED),
    ))

# ─── Convenience lookups ──────────────────────────────────────────────────────
BY_CODE:   dict[str, dict] = {ind["code"]: ind for ind in INDICATORS}
BY_SOURCE: dict[str, list[dict]] = {}
for ind in INDICATORS:
    BY_SOURCE.setdefault(ind["source"], []).append(ind)

CODES_EUSILC = [i["code"] for i in BY_SOURCE.get("EU-SILC", [])]
CODES_HBS    = [i["code"] for i in BY_SOURCE.get("HBS", [])]
CODES_LFS    = [i["code"] for i in BY_SOURCE.get("LFS", [])]
CODES_EHIS   = [i["code"] for i in BY_SOURCE.get("EHIS", [])]

# ─── Quick summary ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Total indicators : {len(INDICATORS)}")
    for src, lst in sorted(BY_SOURCE.items()):
        ewbi = sum(1 for i in lst if i["in_ewbi"])
        print(f"  {src:8s}  {len(lst):3d} indicators  ({ewbi} in current EWBI)")
    print()
    print(f"{'CODE':<14} {'IN EWBI':<9} {'SOURCE':<9} {'LEVEL':<11} {'DIMENSION':<22} {'VARIABLE':<22} NAME")
    print("-" * 130)
    for ind in INDICATORS:
        ewbi_flag = "✓ YES" if ind["in_ewbi"] else "  no"
        print(f"{ind['code']:<14} {ewbi_flag:<9} {ind['source']:<9} {ind['level']:<11} "
              f"{ind['dimension']:<22} {ind['variable']:<22} {ind['name']}")
