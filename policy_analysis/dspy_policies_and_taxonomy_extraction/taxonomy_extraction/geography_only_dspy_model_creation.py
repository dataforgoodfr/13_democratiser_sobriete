import dspy
import os
import json
import random
from dotenv import load_dotenv
from datetime import datetime
from dspy.teleprompt import MIPROv2

# ---------------------------------------------------------------------
# IMPORTS: TAXONOMY
# ---------------------------------------------------------------------
# Ensure these imports match your file structure. 
# For your Impact/Planetary model, you would import those Enums here.
from dspy_policies_and_taxonomy_extraction.taxonomy_definition.geographical_taxonomy import (
    Geographical_scope,
    Studied_country,
    Regional_group,
)

from utils import canonicalize, normalize_to_set

# ---------------------------------------------------------------------
# 1. ENV + DATA LOADING (WITH NORMALIZATION FIX)
# ---------------------------------------------------------------------

load_dotenv()

# --- HELPER: ROBUST NORMALIZATION ---
def create_enum_mapper(enum_class):
    """
    Creates a dictionary mapping normalized strings to valid Enum values.
    Example: 'climate change' -> 'climate_change', 'Climate-Change' -> 'climate_change'
    """
    mapper = {}
    for e in enum_class:
        # Normalize the valid enum value itself (e.g. 'climate_change')
        key = e.value.lower().strip().replace(" ", "_").replace("-", "_")
        mapper[key] = e.value
    return mapper

def normalize_and_map(raw_list, enum_mapper):
    """
    Takes a list of raw strings from JSON (e.g. ["Climate Change"])
    and maps them to valid Enum values (e.g. ["climate_change"]).
    """
    if not raw_list:
        return []
    
    # Handle single string case if data is not a list
    if isinstance(raw_list, str):
        raw_list = [raw_list]

    clean_results = []
    for item in raw_list:
        if not item: 
            continue
        # Normalize the input string to match the Enum format
        # 1. Lowercase, 2. Strip spaces, 3. Replace space/hyphen with underscore
        norm_item = item.lower().strip().replace(" ", "_").replace("-", "_")
        
        if norm_item in enum_mapper:
            clean_results.append(enum_mapper[norm_item])
        else:
            # Optional: Print warning if a label in JSON doesn't exist in Taxonomy
            # print(f"Warning: Label '{item}' not found in taxonomy.")
            pass
            
    return clean_results

# --- PREPARE MAPPERS ---
# Create mappers for your specific Enums
# (For your Impact model, create mappers for Planetary_boundaries, Human_needs, etc.)
geo_scope_mapper = create_enum_mapper(Geographical_scope)
country_mapper = create_enum_mapper(Studied_country)
regional_mapper = create_enum_mapper(Regional_group)


golden_dataset = []
path = "dspy_policies_and_taxonomy_extraction/model_training_data/gold_taxonomy.jsonl"

if not os.path.exists(path):
    raise FileNotFoundError(f"Data file '{path}' not found.")

with open(path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        data = json.loads(line)
        
        # --- APPLIED FIX: Normalize data before creating Example ---
        # This handles cases where JSON has "Global" but Enum is "global",
        # or "Climate Change" vs "climate_change".
        
        norm_scopes = normalize_and_map(data.get("geographical_scopes", []), geo_scope_mapper)
        norm_countries = normalize_and_map(data.get("main_country_focus", []), country_mapper)
        
        # Regional group is often a single string, handle it carefully
        raw_region = data.get("regional_group")
        norm_region_list = normalize_and_map([raw_region] if raw_region else [], regional_mapper)
        norm_region = norm_region_list[0] if norm_region_list else ""

        # --- NOTE FOR YOUR IMPACT MODEL ---
        # For planetary boundaries, you would do:
        # pb_mapper = create_enum_mapper(Planetary_boundaries)
        # norm_pb = normalize_and_map(data.get("planetary_boundaries", []), pb_mapper)

        example = dspy.Example(
            question=data["question"],
            regional_group=norm_region,
            geographical_scopes=norm_scopes,
            main_country_focus=norm_countries,
            # Add your impact fields here if this was the impact script:
            # planetary_boundaries=norm_pb
        )

        golden_dataset.append(example.with_inputs("question"))

random.seed(42)
random.shuffle(golden_dataset)

trainset = golden_dataset[40:]
devset = golden_dataset[:40]

# ---------------------------------------------------------------------
# 2. GEOGRAPHY SIGNATURE
# ---------------------------------------------------------------------

class GeographySignature(dspy.Signature):
    """
    Step 1: Identify all explicit location mentions verbatim.
    Step 2: Map mentions to countries.
    Step 3: Map countries to regional groups and scopes.
    Do not output intermediate steps.
    """

    question = dspy.InputField(
        desc=(
            "A single paragraph from the conclusion section of a scientific study. "
            "The paragraph may or may not mention any geographical information."
        )
    )

    regional_group = dspy.OutputField(
        desc=(
            f"ONE value from {[e.value for e in Regional_group]}. "
            "Return empty if NO regional group is explicitly mentioned."
        )
    )

    geographical_scopes = dspy.OutputField(
        desc=(
            f"A LIST of values from {[e.value for e in Geographical_scope]}. "
            "Return an empty list if none are explicitly mentioned."
        )
    )

    main_country_focus = dspy.OutputField(
        desc=(
            f"A LIST of values from {[e.value for e in Studied_country]}. "
            "Return an empty list if no countries are explicitly mentioned."
        )
    )


# ---------------------------------------------------------------------
# 3. MODEL CONFIG
# ---------------------------------------------------------------------

lm = dspy.LM(
    model="mistral/mistral-small-3.2-24b-instruct-2506",
    api_key=os.getenv("SCALEWAY_API_KEY"),
    api_base="https://api.scaleway.ai/a2dc0d31-c47f-47f1-b0b9-9877dd4eb2b5/v1"
)

dspy.configure(lm=lm)


# ---------------------------------------------------------------------
# 4. GEOGRAPHY-ONLY MODULE
# ---------------------------------------------------------------------

class GeographyExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predictor = dspy.Predict(GeographySignature)

    def forward(self, question: str):
        raw = self.predictor(question=question)

        return dspy.Prediction(
            regional_group=canonicalize(
                raw.regional_group, Regional_group, allow_list=False
            ),
            geographical_scopes=canonicalize(
                raw.geographical_scopes, Geographical_scope, allow_list=True
            ),
            main_country_focus=canonicalize(
                raw.main_country_focus, Studied_country, allow_list=True
            ),
        )


# ---------------------------------------------------------------------
# 5. SIMPLE GEOGRAPHY F1 METRIC (MIPRO-FRIENDLY)
# ---------------------------------------------------------------------
def geography_f1_metric(example, pred, trace=None):
    tp = fp = fn = 0

    fields = [
        "regional_group",
        "geographical_scopes",
        "main_country_focus",
    ]

    for field in fields:
        gold = normalize_to_set(getattr(example, field))
        predicted = normalize_to_set(getattr(pred, field))

        # Correct abstention → weak true positive
        if not gold and not predicted:
            tp += 0.3
            continue

        tp += len(gold & predicted)
        fp += len(predicted - gold)
        fn += len(gold - predicted)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return round(2 * precision * recall / (precision + recall), 4)


# ---------------------------------------------------------------------
# 6. OPTIMIZATION
# ---------------------------------------------------------------------

print("Starting geography-only optimization...")

# Note: 'auto' is experimental, use 'heavy' or 'light' if needed, 
# or explicit minibatch size.
optimizer = MIPROv2(metric=geography_f1_metric
                    # ,auto="heavy"
                    )

compiled_program = optimizer.compile(
    GeographyExtractor(),
    trainset=trainset
)


# ---------------------------------------------------------------------
# 7. EVALUATION
# ---------------------------------------------------------------------

print("Evaluating on dev set...")

evaluator = dspy.Evaluate(
    devset=devset,
    metric=geography_f1_metric,
    display_table=True,
    display_progress=False
)

score = evaluator(compiled_program)
print("Final Geography F1:", score)

# Save results
optimized_score = evaluator(compiled_program, save_as_json=f"results.json")

# ---------------------------------------------------------------------
# 8. SAVE MODEL
# ---------------------------------------------------------------------

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path = f"saved_dspy_model/geography_model_{timestamp}"

compiled_program.save(model_path, save_program=True)
print(f"Optimized geography model saved to {model_path}")