import sys
import os
import random
import re
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import pandas as pd
import sklearn
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold, GridSearchCV, cross_val_predict
from sklearn.metrics import f1_score, fbeta_score, precision_score, recall_score, make_scorer
import joblib

sklearn.set_config(enable_metadata_routing=True)  # allows passing sample_with through CV

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


# ---------------------------------------------------------------------
# 0. IMPORT TAXONOMY ENUMS
# ---------------------------------------------------------------------
from dspy_policies_and_taxonomy_extraction.taxonomy_definition.impact_taxonomy import (
    Need,
    Resource,
    Wellbeing,
    Justice,
    PlanetaryBoundary,
)

# ---------------------------------------------------------------------
# PATH SETUP
# ---------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------
# ENUM NORMALIZATION
# ---------------------------------------------------------------------
# These sets contain snake_case strings (e.g., 'climate_change', 'freshwater_use')
Need_ENUM = {e.name.lower() for e in Need}
NATURAL_RESOURCE_ENUM = {e.name.lower() for e in Resource}
WELLBEING_ENUM = {e.name.lower() for e in Wellbeing}
JUSTICE_ENUM = {e.name.lower() for e in Justice}
PLANETARY_ENUM = {e.name.lower() for e in PlanetaryBoundary}

# ---------------------------------------------------------------------
# PATHS & CONFIG
# ---------------------------------------------------------------------
DATA_PATH_GOLD = "../model_training_data/gold_impact_taxonomy_concat_with_embeddings_2026-03-11.parquet"
DATA_PATH_SYNTH = "../model_training_data/sample_2000_impact_taxonomy_gemini3_flash.parquet"
EMB_COL = "embedding"
MODEL_ROOT = "saved_impact_models"

# Sample weight multiplier for gold data points relative to synthetic ones.
# 1.0 → equal weight (no effect). Higher values make the model trust gold labels more
# during both training and CV evaluation. E.g. 5.0 → each gold example counts as 5
# synthetic examples when fitting and when computing metrics.
GOLD_SAMPLE_WEIGHT = 5.0

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
    if labels is None or len(labels) == 0:
        return ["unknown"] if allow_unknown else []

    if isinstance(labels, str):
        labels = [labels]

    cleaned = []
    for label in labels:
        if not label:
            continue
            
        # --- FIX: FORCE SNAKE_CASE ---
        # 1. Lowercase and strip
        l_norm = label.lower().strip()
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
        #    print(f"Warning: Label '{label}' normalized to '{l_norm}' not found in Enum.")

    if not cleaned and allow_unknown:
        return ["unknown"]

    return cleaned

# ---------------------------------------------------------------------
# 2. LOAD DATASET
# ---------------------------------------------------------------------
def load_taxonomy_parquet(path, source_name):
    if not os.path.exists(path):
        print(f"Warning: Data file '{path}' not found.")
        return []

    df = pd.read_parquet(path)
    dataset = []

    for idx, row in df.iterrows():
        text = normalize_text(row.get('text', row.get('question', '')))
        if not text:
            continue

        embedding = row.get(EMB_COL)
        if embedding is None:
            print(f"Warning: row {idx} has no embedding, skipping.")
            continue

        example = {
            "text": text,
            "embedding": embedding.astype(np.float32),
            "Need": normalize_labels(row.get("human_needs", []), Need_ENUM),
            "Resource": normalize_labels(row.get("natural_resources", []), NATURAL_RESOURCE_ENUM),
            "wellbeing": normalize_labels(row.get("wellbeing", []), WELLBEING_ENUM),
            "Justice": normalize_labels(row.get("justice_considerations", []), JUSTICE_ENUM),
            "PlanetaryBoundary": normalize_labels(row.get("planetary_boundaries", []), PLANETARY_ENUM),
            "source": source_name,
        }
        dataset.append(example)

    print(f"Loaded {len(dataset)} examples from {source_name}")
    return dataset


gold_dataset = load_taxonomy_parquet(DATA_PATH_GOLD, "gold")
synth_dataset = load_taxonomy_parquet(DATA_PATH_SYNTH, "synthetic")

random.seed(32)
train_dataset = gold_dataset + synth_dataset
random.shuffle(train_dataset)
print(f"Train size: {len(train_dataset)}  (gold={len(gold_dataset)}, synth={len(synth_dataset)})")
print(f"Gold sample weight: {GOLD_SAMPLE_WEIGHT}")

# Per-sample weights: gold examples get GOLD_SAMPLE_WEIGHT, synthetic get 1.0.
sample_weights = np.array(
    [GOLD_SAMPLE_WEIGHT if ex["source"] == "gold" else 1.0 for ex in train_dataset],
    dtype=np.float32,
)

golden_dataset = train_dataset

# ---------------------------------------------------------------------
# 3. FIELD DEFINITIONS
# ---------------------------------------------------------------------
impact_fields = [
    "Need",
    "Resource",
    "wellbeing",
    "Justice",
    "PlanetaryBoundary",
]

ENUM_MAP = {
    "Need": Need_ENUM,
    "Resource": NATURAL_RESOURCE_ENUM,
    "wellbeing": WELLBEING_ENUM,
    "Justice": JUSTICE_ENUM,
    "PlanetaryBoundary": PLANETARY_ENUM,
}

texts = [ex["text"] for ex in train_dataset]

# ---------------------------------------------------------------------
# 4. LABEL BINARIZATION (ENUM-LOCKED, training data only)
# ---------------------------------------------------------------------
labels_dict = {field: [ex[field] for ex in train_dataset] for field in impact_fields}
mlb_dict = {}
Y_dict = {}

print("\n--- Binarizing Labels (train set) ---")
for field, lists in labels_dict.items():
    # Force the classes to be exactly what's in the Enum
    mlb = MultiLabelBinarizer(classes=sorted(ENUM_MAP[field]))
    Y = mlb.fit_transform(lists)

    mlb_dict[field] = mlb
    Y_dict[field] = Y

    print(f"{field}: {Y.shape[1]} classes")
    print(f"  positives per class: {Y.sum(axis=0)}")

    # Sanity check
    if Y.sum() == 0:
        print(f"  WARNING: 0 positives found for {field}. Check normalization.")

# ---------------------------------------------------------------------
# 5. EMBEDDINGS
# ---------------------------------------------------------------------
print("\nLoading precomputed embeddings from parquet...")
X = np.stack([ex["embedding"] for ex in train_dataset])
print(f"Embedding matrix shape: {X.shape}")

# ---------------------------------------------------------------------
# 6. K-FOLD CROSS-VALIDATION & HYPERPARAMETER GRID
# ---------------------------------------------------------------------
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=32)

param_grid = {
    "estimator__C": [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
}

print(f"\nCross-validation: {N_SPLITS}-fold KFold")
print(f"Param grid: {param_grid}")

# ---------------------------------------------------------------------
# 7. GRID SEARCH + FIT FINAL CLASSIFIERS
# ---------------------------------------------------------------------
classifiers = {}
best_params_dict = {}
best_cv_scores = {}

print("\nRunning GridSearchCV for each field...")

for field in impact_fields:
    Y = Y_dict[field]
    if Y.sum() == 0:
        print(f"  Skipping {field}: no positive labels in dataset")
        continue

    print(f"\n[{field}] GridSearchCV...")

    base_clf = OneVsRestClassifier(
        LogisticRegression(max_iter=500, class_weight="balanced").set_fit_request(sample_weight=True)
    )  # Enable passing sample_weight through CV
    # fbeta with beta=2 gives more importance to recall over precision
    gs = GridSearchCV(
        base_clf,
        param_grid,
        cv=kf,
        scoring=make_scorer(fbeta_score, beta=2, average="micro", zero_division=0),
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    gs.fit(X, Y, sample_weight=sample_weights)

    classifiers[field] = gs.best_estimator_
    best_params_dict[field] = gs.best_params_
    best_cv_scores[field] = round(gs.best_score_, 4)

    print(f"  Best params : {gs.best_params_}")
    print(f"  Best CV F1  : {gs.best_score_:.4f}")

# ---------------------------------------------------------------------
# 8. EVALUATION — Out-of-fold CV with sample-weighted metrics
#    Fitting uses sample_weight so gold examples are trusted more.
#    Reported metrics are also weighted by the same weights, so gold
#    examples contribute more to the reported P/R/F1 scores.
# ---------------------------------------------------------------------
print("\n" + "=" * 70)
print("EVALUATION RESULTS (out-of-fold CV, sample-weighted metrics)")
print("=" * 70)
results = pd.DataFrame()
f1_scores = {}

for field, clf in classifiers.items():
    Y = Y_dict[field]

    print(f"\n[{field}]  best params: {best_params_dict[field]}")

    y_pred_oof = cross_val_predict(
        clf, X, Y, cv=kf, method="predict",
        params={"sample_weight": sample_weights}, n_jobs=-1,
    )
    classes = mlb_dict[field].classes_
    support = Y.sum(axis=0)

    # Weighted micro metrics
    p_val  = precision_score(Y, y_pred_oof, average="micro", zero_division=0, sample_weight=sample_weights)
    r_val  = recall_score(Y, y_pred_oof, average="micro", zero_division=0, sample_weight=sample_weights)
    f1_val = f1_score(Y, y_pred_oof, average="micro", zero_division=0, sample_weight=sample_weights)
    f1_scores[field] = round(f1_val, 4)

    print(f"  micro (weighted CV): P={p_val:.4f}  R={r_val:.4f}  F1={f1_val:.4f}")

    y_true_labels = mlb_dict[field].inverse_transform(Y)
    y_pred_labels = mlb_dict[field].inverse_transform(y_pred_oof)
    results[f"{field}_true"] = [list(lbl) for lbl in y_true_labels]
    results[f"{field}_pred"] = [list(lbl) for lbl in y_pred_labels]

    # Weighted per-label metrics
    per_label_p  = precision_score(Y, y_pred_oof, average=None, zero_division=0, sample_weight=sample_weights)
    per_label_r  = recall_score(Y, y_pred_oof, average=None, zero_division=0, sample_weight=sample_weights)
    per_label_f1 = f1_score(Y, y_pred_oof, average=None, zero_division=0, sample_weight=sample_weights)

    print(f"  {'Label':<35} {'Support':>7}  {'P':>6}  {'R':>6}  {'F1':>6}")
    print(f"  {'-'*35}  {'-'*7}  {'-'*6}  {'-'*6}  {'-'*6}")
    for cls, p, r, f, s in zip(classes, per_label_p, per_label_r, per_label_f1, support, strict=False):
        print(f"  {cls:<35} {int(s):>7}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}")

# ---------------------------------------------------------------------
# 9. SAVE OUTPUTS
# ---------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = os.path.join(MODEL_ROOT, f"impact_model_only_{timestamp}")
os.makedirs(model_dir, exist_ok=True)

output_csv = os.path.join(model_dir, "dev_predictions.csv")
results.to_csv(output_csv, index=False)

#joblib.dump(sbert_model, os.path.join(model_dir, "sentence_bert_model.pkl"))
for field in classifiers:
    joblib.dump(classifiers[field], os.path.join(model_dir, f"{field}_clf.pkl"))
    joblib.dump(mlb_dict[field], os.path.join(model_dir, f"{field}_mlb.pkl"))

print(f"\nModels and predictions saved to {model_dir}")