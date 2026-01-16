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
import re

import ast
from utils import canonicalize, NULL_STRINGS, clean_token,normalize_to_set


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
 
            )

            golden_dataset.append(
                example.with_inputs("question")
            )
else:
    raise FileNotFoundError(f"Data file '{path}' not found.")

synthetic_dataset = []
path = "synthetic_data-all.jsonl"

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
 
            )

            synthetic_dataset.append(
                example.with_inputs("question")
            )
else:
    raise FileNotFoundError(f"Data file '{path}' not found.")


trainset = synthetic_dataset[:20]
devset = golden_dataset


class RegionalGroupSignature(dspy.Signature):
    """
    Analyze the provided CONCLUSION paragraph to infer the study's geographical focus.
    """
    
    question = dspy.InputField(
        desc="A single paragraph from the conclusion section of a scientific study."
    )
 
    regional_group = dspy.OutputField(
        desc=f"ONE value from {[e.value for e in Regional_group]} or empty if unknown. The regional group studied in the paper if any"
    )

class GeographicalScopeSignature(dspy.Signature):
    """
    Analyze the provided CONCLUSION paragraph to infer the study's geographical focus.
    """
    
    question = dspy.InputField(
        desc="A single paragraph from the conclusion section of a scientific study."
    )
 
    geographical_scopes = dspy.OutputField(
        desc=f"A LIST of values from {[e.value for e in Geographical_scope]}. The geographical scope(s) studied in the paper if any"
    )

class MainCountryFocusSignature(dspy.Signature):
    """
    Analyze the provided CONCLUSION paragraph to infer the study's geographical focus.
    """
    
    question = dspy.InputField(
        desc="A single paragraph from the conclusion section of a scientific study."
    )
 
    main_country_focus = dspy.OutputField(
        desc=f"A LIST of values from {[e.value for e in Studied_country]}. The main country focus studied in the paper if any."
        "Do not imply country from regional scope."
                )

lm = dspy.LM(
    model='openai/gpt-4o-mini', 
    api_key=os.getenv('OPENAI_API_KEY')
)

lm = dspy.LM(
        model="openai/mistral-small-3.2-24b-instruct-2506",
        api_key=os.getenv('SCALEWAY_API_KEY'),
        api_base="https://api.scaleway.ai/a2dc0d31-c47f-47f1-b0b9-9877dd4eb2b5/v1"
        )


dspy.configure(lm=lm)

class RegionalGroupExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.regional_predictor = dspy.Predict(RegionalGroupSignature)

    def forward(self, question: str):
        # Run them in parallel or sequence
        regional_predictor = self.regional_predictor(question=question)
        return dspy.Prediction(
            # Geography
            regional_group=canonicalize(
                regional_predictor.regional_group, Regional_group, allow_list=False
            )
            )

class GeographicalScopeExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        
        self.geographical_scope_predictor = dspy.Predict(GeographicalScopeSignature)
    def forward(self, question: str):
        # Run them in parallel or sequence
        geographical_scope_predictor = self.geographical_scope_predictor(question=question)
        return dspy.Prediction(
            # Geography
            geographical_scopes=canonicalize(
                geographical_scope_predictor.geographical_scopes, Geographical_scope, allow_list=True
            )
        )


class MainCountryFocusExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.main_country_focus_predictor = dspy.Predict(MainCountryFocusSignature)

    def forward(self, question: str):
        main_country_focus_predictor = self.main_country_focus_predictor(question=question)
        return dspy.Prediction(
            main_country_focus=canonicalize(
                main_country_focus_predictor.main_country_focus, Studied_country, allow_list=True
            )
            )

def regional_group_accuracy(example, pred, trace=None):
    gold = example.regional_group
    pred = pred.regional_group
    return 1.0 if gold == pred else 0.0

def scopes_micro_f1(example, pred, trace=None):
    gold = normalize_to_set(example.geographical_scopes)
    pred = normalize_to_set(pred.geographical_scopes)

    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0

    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)

def country_micro_f1(example, pred, trace=None):
    gold = normalize_to_set(example.main_country_focus)
    pred = normalize_to_set(pred.main_country_focus)

    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0

    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)

    if tp == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


print("Starting optimization...")

rg_optimizer = MIPROv2(metric=regional_group_accuracy,auto="heavy")
rg_model = rg_optimizer.compile(
    RegionalGroupExtractor(),
    trainset=trainset
)

scope_optimizer = MIPROv2(metric=scopes_micro_f1,auto="heavy")
scope_model = scope_optimizer.compile(
    GeographicalScopeExtractor(),
    trainset=trainset
)

country_optimizer = MIPROv2(metric=country_micro_f1,auto="heavy")
country_model = country_optimizer.compile(
    MainCountryFocusExtractor(),
    trainset=trainset
)


rg_evaluator = dspy.Evaluate(
    devset=devset,
    metric=regional_group_accuracy,
    display_progress=True,
    display_table=True
)

rg_score = rg_evaluator(rg_model, save_as_json=f"rg_score.json")
print("Regional Group Accuracy:")

scope_evaluator = dspy.Evaluate(
    devset=devset,
    metric=scopes_micro_f1,
    display_progress=True,
    display_table=True
)

scope_score = scope_evaluator(scope_model,save_as_json=f"scope_score.json")
print("Geographical Scope micro-F1:", scope_score.score)

country_evaluator = dspy.Evaluate(
    devset=devset,
    metric=country_micro_f1,
    display_progress=True,
    display_table=True
)

country_score = country_evaluator(country_model,save_as_json=f"country_score.json")
print("Main Country Focus micro-F1:", country_score.score)

results = {
    "Regional Group": {
        "metric": "Accuracy",
        "score": rg_score.score
    },
    "Geographical Scope": {
        "metric": "Micro-F1",
        "score": scope_score.score
    },
    "Main Country Focus": {
        "metric": "Micro-F1",
        "score": country_score.score
    }
}

print("\nFinal Results: {}".format(results))