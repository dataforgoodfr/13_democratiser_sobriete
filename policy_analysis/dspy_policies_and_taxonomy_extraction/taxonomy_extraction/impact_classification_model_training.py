import sys
import os
import json
import random
import re
from datetime import datetime
from dotenv import load_dotenv

import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import joblib

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
load_dotenv()

# ---------------------------------------------------------------------
# 0. IMPORT TAXONOMY ENUMS
# ---------------------------------------------------------------------
from dspy_policies_and_taxonomy_extraction.taxonomy_definition.impact_taxonomy import (
    Human_needs,
    Natural_ressource,
    Wellbeing,
    Justice_consideration,
    Planetary_boundaries,
)

# ---------------------------------------------------------------------
# ENUM NORMALIZATION
# ---------------------------------------------------------------------
# These sets contain snake_case strings (e.g., 'climate_change', 'freshwater_use')
HUMAN_NEEDS_ENUM = {e.name.lower() for e in Human_needs}
NATURAL_RESOURCE_ENUM = {e.name.lower() for e in Natural_ressource}
WELLBEING_ENUM = {e.name.lower() for e in Wellbeing}
JUSTICE_ENUM = {e.name.lower() for e in Justice_consideration}
PLANETARY_ENUM = {e.name.lower() for e in Planetary_boundaries}

# ---------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------
DATA_PATH_GOLD = "dspy_policies_and_taxonomy_extraction/model_training_data/gold_taxonomy.jsonl"
DATA_PATH_SYNTHETIC = "dspy_policies_and_taxonomy_extraction/model_training_data/synthetic_taxonomy.jsonl"

MODEL_ROOT = "saved_impact_models"

# ---------------------------------------------------------------------
# 1. NORMALIZATION HELPERS
# ---------------------------------------------------------------------
def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_labels(labels, allowed_enum, allow_unknown=False):
    """
    Normalizes input labels to snake_case and validates them against the provided Enum set.
    """
    if not labels:
        return ["unknown"] if allow_unknown else []

    if isinstance(labels, str):
        labels = [labels]

    cleaned = []
    for l in labels:
        if not l:
            continue
            
        # --- FIX: FORCE SNAKE_CASE ---
        # 1. Lowercase and strip
        l_norm = l.lower().strip()
        # 2. Replace spaces and hyphens with underscores 
        #    (e.g., "Climate Change" -> "climate_change", "Land-System Change" -> "land_system_change")
        l_norm = re.sub(r"[\s\-]+", "_", l_norm)
        # 3. Remove any other special characters just in case
        l_norm = re.sub(r"[^\w_]", "", l_norm)

        # --- VALIDATE ---
        if l_norm in allowed_enum:
            cleaned.append(l_norm)
        # Optional: Print warning if label is rejected (good for debugging)
        # else: 
        #    print(f"Warning: Label '{l}' normalized to '{l_norm}' not found in Enum.")

    if not cleaned and allow_unknown:
        return ["unknown"]

    return cleaned

# ---------------------------------------------------------------------
# 2. LOAD DATASETS
# ---------------------------------------------------------------------
def load_taxonomy_jsonl(path, source_name):
    if not os.path.exists(path):
        print(f"Warning: Data file '{path}' not found.")
        return []

    dataset = []

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[{source_name}] Skipping invalid JSON on line {idx}: {e}")
                continue

            # Apply normalization that converts "Climate Change" -> "climate_change"
            example = {
                "text": normalize_text(data.get("question", "")),
                "human_needs": normalize_labels(data.get("human_needs", []), HUMAN_NEEDS_ENUM),
                "natural_ressource": normalize_labels(data.get("natural_ressource", []), NATURAL_RESOURCE_ENUM),
                "wellbeing": normalize_labels(data.get("wellbeing", []), WELLBEING_ENUM),
                "justice_consideration": normalize_labels(data.get("justice_consideration", []), JUSTICE_ENUM),
                "planetary_boundaries": normalize_labels(data.get("planetary_boundaries", []), PLANETARY_ENUM),
                "source": source_name,
            }

            # Only add if there is text
            if example["text"]:
                dataset.append(example)

    print(f"Loaded {len(dataset)} examples from {source_name}")
    return dataset


gold_dataset = load_taxonomy_jsonl(DATA_PATH_GOLD, "gold")
synthetic_dataset = load_taxonomy_jsonl(DATA_PATH_SYNTHETIC, "synthetic")

# Combine datasets (Uncomment synthetic if needed)
golden_dataset = gold_dataset # + synthetic_dataset 
print(f"Total combined dataset size: {len(golden_dataset)}")

# Shuffle after combining
random.seed(32)
random.shuffle(golden_dataset)

# ---------------------------------------------------------------------
# 3. FIELD DEFINITIONS
# ---------------------------------------------------------------------
impact_fields = [
    "human_needs",
    "natural_ressource",
    "wellbeing",
    "justice_consideration",
    "planetary_boundaries",
]

ENUM_MAP = {
    "human_needs": HUMAN_NEEDS_ENUM,
    "natural_ressource": NATURAL_RESOURCE_ENUM,
    "wellbeing": WELLBEING_ENUM,
    "justice_consideration": JUSTICE_ENUM,
    "planetary_boundaries": PLANETARY_ENUM,
}

texts = [ex["text"] for ex in golden_dataset]

# ---------------------------------------------------------------------
# 4. LABEL BINARIZATION (ENUM-LOCKED)
# ---------------------------------------------------------------------
labels_dict = {field: [ex[field] for ex in golden_dataset] for field in impact_fields}
mlb_dict = {}
Y_dict = {}

print("\n--- Binarizing Labels ---")
for field, lists in labels_dict.items():
    # Force the classes to be exactly what's in the Enum
    mlb = MultiLabelBinarizer(classes=sorted(ENUM_MAP[field]))
    Y = mlb.fit_transform(lists)

    mlb_dict[field] = mlb
    Y_dict[field] = Y

    print(f"{field}: {Y.shape}")
    # print(f"  classes: {mlb.classes_}")
    print(f"  positives per class: {Y.sum(axis=0)}")
    
    # Sanity check
    if Y.sum() == 0:
        print(f"  WARNING: 0 positives found for {field}. Check normalization.")

# ---------------------------------------------------------------------
# 5. EMBEDDINGS
# ---------------------------------------------------------------------
print("\nComputing Sentence-BERT embeddings...")
sbert_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
X = sbert_model.encode(texts, show_progress_bar=True)

# ---------------------------------------------------------------------
# 6. TRAIN / DEV SPLIT
# ---------------------------------------------------------------------
if len(texts) > 5:
    indices = list(range(len(texts)))
    train_idx, dev_idx = train_test_split(indices, test_size=0.1, random_state=32)

    X_train = X[train_idx]
    X_dev = X[dev_idx]
    Y_train_dict = {f: Y_dict[f][train_idx] for f in Y_dict}
    Y_dev_dict = {f: Y_dict[f][dev_idx] for f in Y_dict}
else:
    print("\nDataset too small for split. Training on all data.")
    X_train, X_dev = X, X
    Y_train_dict = Y_dict
    Y_dev_dict = Y_dict

# ---------------------------------------------------------------------
# 7. TRAIN CLASSIFIERS
# ---------------------------------------------------------------------
classifiers = {}
print("\nTraining Classifiers...")

for field in impact_fields:
    y_train = Y_train_dict[field]
    
    # Check if there are any labels to train on
    if y_train.sum() == 0:
        print(f"Skipping {field}: no positive labels in training set")
        continue

    clf = OneVsRestClassifier(
        LogisticRegression(max_iter=500, class_weight="balanced")
    )
    clf.fit(X_train, y_train)
    classifiers[field] = clf

    print(f"Trained classifier for {field}")

# ---------------------------------------------------------------------
# 8. EVALUATION
# ---------------------------------------------------------------------
print("\nEvaluating...")
results = pd.DataFrame()
f1_scores = {}

for field, clf in classifiers.items():
    y_true = Y_dev_dict[field]
    y_pred = clf.predict(X_dev)

    y_true_labels = mlb_dict[field].inverse_transform(y_true)
    y_pred_labels = mlb_dict[field].inverse_transform(y_pred)

    results[f"{field}_true"] = [list(l) for l in y_true_labels]
    results[f"{field}_pred"] = [list(l) for l in y_pred_labels]

    f1 = f1_score(y_true, y_pred, average="micro")
    f1_scores[field] = round(f1, 4)

    print(f"{field}: F1-micro = {f1_scores[field]}")

# ---------------------------------------------------------------------
# 9. SAVE OUTPUTS
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = os.path.join(MODEL_ROOT, f"impact_model_only_{timestamp}")
os.makedirs(model_dir, exist_ok=True)

output_csv = os.path.join(model_dir, "dev_predictions.csv")
results.to_csv(output_csv, index=False)

joblib.dump(sbert_model, os.path.join(model_dir, "sentence_bert_model.pkl"))
for field in classifiers:
    joblib.dump(classifiers[field], os.path.join(model_dir, f"{field}_clf.pkl"))
    joblib.dump(mlb_dict[field], os.path.join(model_dir, f"{field}_mlb.pkl"))

print(f"\nModels and predictions saved to {model_dir}")