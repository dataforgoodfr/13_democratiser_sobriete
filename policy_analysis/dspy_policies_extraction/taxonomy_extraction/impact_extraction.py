import dspy
import csv
from enum import Enum
from typing import List, Dict, Any, Union
import os
from dotenv import load_dotenv

import dspy
from dspy.teleprompt import MIPROv2

import json
import os
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
from geographical_taxonomy import (
    Geographical_scope,
    Studied_country,
    Regional_group,
)
from Impact_taxonomy import (
    Human_needs,    
    Natural_ressource,
    Wellbeing,
    Justice_consideration,
    Planetary_boundaries
)
import re

import ast
from taxonomy_utils import canonicalize, NULL_STRINGS, clean_token


load_dotenv()

golden_dataset = []
path = "taxonomy_extraction/transformed_data_all.jsonl"

if os.path.exists(path):
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # skip empty lines
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {idx}: {e}")
                continue

            example = dspy.Example(
                question=data["question"],
                # Geography
                regional_group=data.get("regional_group"),
                geographical_scopes=data.get("geographical_scopes", []),
                main_country_focus=data.get("main_country_focus", []),
                # Impact / Taxonomy
                human_needs=data.get("human_needs", []),
                natural_ressource=data.get("natural_ressource", []),
                wellbeing=data.get("wellbeing", []),
                justice_consideration=data.get("justice_consideration", []),
                planetary_boundaries=data.get("planetary_boundaries", [])
            )

            golden_dataset.append(
                example.with_inputs("question")
            )
else:
    raise FileNotFoundError(f"Data file '{path}' not found.")


trainset = golden_dataset
devset = golden_dataset


class GeographySignature(dspy.Signature):
    """
    Analyze the provided CONCLUSION paragraph to infer the study's geographical focus.
    """
    
    question = dspy.InputField(
        desc="A single paragraph from the conclusion section of a scientific study."
    )
 
    regional_group = dspy.OutputField(
        desc=f"ONE value from {[e.value for e in Regional_group]} or empty if unknown. "
    )

    geographical_scopes = dspy.OutputField(
        desc=f"A LIST of values from {[e.value for e in Geographical_scope]}. "
    )

    main_country_focus = dspy.OutputField(
        desc=f"A LIST of values from {[e.value for e in Studied_country]}. "
                )

class ImpactSignature(dspy.Signature):
    """
    Analyze the provided CONCLUSION paragraph to extract the specific impacts that were the core subject of the study's findings.
    """

    question = dspy.InputField(
        desc="A single paragraph from the conclusion section of a scientific study."
    )

    human_needs = dspy.OutputField(
        desc=f"A LIST of values from {[e.value for e in Human_needs]}. "
    )

    natural_ressource = dspy.OutputField(
        desc=f"A LIST of values from {[e.value for e in Natural_ressource]}." )
    

    wellbeing = dspy.OutputField(
        desc=f"A LIST of values from {[e.value for e in Wellbeing]}. "
    )

    justice_consideration = dspy.OutputField(
        desc=f"A LIST of values from {[e.value for e in Justice_consideration]}. "
    )

    planetary_boundaries = dspy.OutputField(
        desc=f"A LIST of values from {[e.value for e in Planetary_boundaries]}. "
    )


# Configure the Language Model, passing the schema directly to force structured output
# This often bypasses serialization issues in the adapter layer.
"""lm = dspy.LM(
    model='openai/gpt-4o-mini', 
    api_key=os.getenv('OPENAI_API_KEY')
)"""

lm = dspy.LM(
        model="openai/mistral-small-3.2-24b-instruct-2506",
        api_key=os.getenv('SCALEWAY_API_KEY'),
        api_base="https://api.scaleway.ai/a2dc0d31-c47f-47f1-b0b9-9877dd4eb2b5/v1"
        )


dspy.configure(lm=lm)

class GeographyExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        # Module 1: Focus only on Geography
        self.geo_predictor = dspy.ChainOfThought(GeographySignature)
        # Module 2: Focus only on Impact
        self.impact_predictor = dspy.ChainOfThought(ImpactSignature)

    def forward(self, question: str):
        # Run them in parallel or sequence
        geo_raw = self.geo_predictor(question=question)
        impact_raw = self.impact_predictor(question=question)
        
             # UPDATED: Return all fields, canonicalized
        return dspy.Prediction(
            # Geography
            regional_group=canonicalize(
                geo_raw.regional_group, Regional_group, allow_list=False
            ),
            geographical_scopes=canonicalize(
                geo_raw.geographical_scopes, Geographical_scope, allow_list=True
            ),
            main_country_focus=canonicalize(
                geo_raw.main_country_focus, Studied_country, allow_list=True
            ),
            # Impact
            human_needs=canonicalize(
                impact_raw.human_needs, Human_needs, allow_list=True
            ),
            natural_ressource=canonicalize(
                impact_raw.natural_ressource, Natural_ressource, allow_list=True
            ),
            wellbeing=canonicalize(
                impact_raw.wellbeing, Wellbeing, allow_list=True
            ),
            justice_consideration=canonicalize(
                impact_raw.justice_consideration, Justice_consideration, allow_list=True
            ),
            planetary_boundaries=canonicalize(
                impact_raw.planetary_boundaries, Planetary_boundaries, allow_list=True
            )
        )


def calculate_f1(gold, pred):
    gold_set = normalize_to_set(gold)
    pred_set = normalize_to_set(pred)

    # Perfect empty match
    if not gold_set and not pred_set:
        return 1.0

    # One empty, one not
    if not gold_set or not pred_set:
        return 0.0

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def get_match_counts(gold, pred, null_match_weight=0.5):
    """
    Returns (TP, FP, FN) for a single field.
    If both gold and pred are empty, returns TP = null_match_weight (default 0.5).
    """
    gold_set = normalize_to_set(gold)
    pred_set = normalize_to_set(pred)
    
    # 1. Handle Null vs Null case
    # If both are empty, we reward the model, but less than a full match (1.0)
    if not gold_set and not pred_set:
        return null_match_weight, 0, 0

    # 2. Standard Calculation for non-empty fields
    # Intersection is True Positives
    tp = len(gold_set & pred_set)
    # Predicted but not in Gold is False Positives
    fp = len(pred_set - gold_set)
    # In Gold but not Predicted is False Negatives
    fn = len(gold_set - pred_set)
    
    return tp, fp, fn

def geography_f1_metric(example, pred, trace=None):
    # Initialize global counters
    total_tp, total_fp, total_fn = 0, 0, 0
    
    # 1. Define Field Weights
    # Default is 1.0. Lower value = less importance. Higher value = more importance.
    field_weights = {
        'geographical_scopes': 0.3,
        'main_country_focus':2.0,
        'regional_group': 1.0,
        'human_needs': 0.7,
        'natural_ressource': 0.7,
        'wellbeing': 0.7,
        'justice_consideration': 0.7,
        'planetary_boundaries': 0.7,  
    }

    # List of fields to evaluate
    fields = [
        ('regional_group', pred.regional_group),
        ('geographical_scopes', pred.geographical_scopes),
        ('main_country_focus', pred.main_country_focus),
        ('human_needs', pred.human_needs),
        ('natural_ressource', pred.natural_ressource),
        ('wellbeing', pred.wellbeing),
        ('justice_consideration', pred.justice_consideration),
        ('planetary_boundaries', pred.planetary_boundaries),
    ]
    
    for field_name, pred_value in fields:
        gold_value = getattr(example, field_name)
        
        # Calculate raw counts (including your null logic from before)
        tp, fp, fn = get_match_counts(gold_value, pred_value, null_match_weight=0.5)
        
        # 2. Apply Weighting
        # Get weight for this field (default to 1.0 if not in dictionary)
        weight = field_weights.get(field_name, 1.0)
        
        # Multiply counts by the weight
        total_tp += (tp * weight)
        total_fp += (fp * weight)
        total_fn += (fn * weight)

    # Calculate Global F1
    denominator_precision = total_tp + total_fp
    denominator_recall = total_tp + total_fn

    # Avoid division by zero
    if denominator_precision == 0 or denominator_recall == 0:
        # If everything is empty/zero (and no TPs occurred), score is 0
        if total_tp == 0:
            return 0.0
        
    precision = total_tp / denominator_precision if denominator_precision > 0 else 0.0
    recall = total_tp / denominator_recall if denominator_recall > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
        
    return round(2 * precision * recall / (precision + recall), 2)


print("Starting optimization...")

optimizer = MIPROv2(metric=geography_f1_metric
                    #,auto="heavy"
                    )

compiled_program = optimizer.compile(GeographyExtractor(), trainset=trainset)

print("Evaluating optimized model on Dev Set...")
optimized_evaluator = dspy.Evaluate(
    devset=devset,
    metric=geography_f1_metric,
    display_progress=False,
    display_table=True
)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


optimized_score = optimized_evaluator(compiled_program,save_as_json=f"{timestamp}.json")
print(optimized_score)

score_str = f"{round(optimized_score.score,2)}".replace(".", "_") 


optimized_evaluator(
    compiled_program
)

print(f"Final Score on Validation Set (optimized): {optimized_score}%")

# --- Saving the optimized model ---
model_path = f"{score_str}"
compiled_program.save(model_path,save_program=True)
print(f"Optimized model saved to {model_path}")
