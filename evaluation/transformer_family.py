# family_eval.py
# Evaluate trained TransformerFamilyModel on an eval dataset (e.g., BLDB / divergent) using BL-only rows.
# The only “mode” difference here is whether you INCLUDE or EXCLUDE rows labeled "other beta-lactamase".
#
# Outputs:
#   1) global metrics CSV (macro precision/recall/F1 + accuracy + loss)
#   2) per-class metrics CSV (precision/recall/F1/support)
#   3) confusion matrix PNG (optional; commented)
#
# Notes:
# - Family evaluation is done on BL sequences only (is_bla==1).
# - Uses LabelEncoder saved during training to map class indices consistently.
# - Forces full label space when computing per-class + macro metrics (avoids length mismatch).

import os
import joblib
import numpy as np
import pandas as pd

# If you're running on a login/CPU node, uncomment:
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

import matplotlib.pyplot as plt

from tokenization import MAX_LEN, encode_and_pad
from model import TransformerFamilyModel, PositionalAdder, AttnMaskExpand, Encoder


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ============================
# CONFIG
# ============================
DATASET = "discovery"            # "gold" or "discovery"
EVAL_SET = "divergent3"           # "bldb" or "divergent" (add others if needed)
SPLIT = "cluster"                 # "random" or "cluster" (matches your training dir convention)

INCLUDE_OTHER_BETA_LACTAMASE = True   # True = keep "other beta-lactamase"; False = drop it
OTHER_LABEL = "other beta-lactamase"

# ============================
# Paths
# ============================
BASE_DIR = "/home/wlodarsm/projects/def-mcarthur/wlodarsm/eccb"

# --- evaluation data ---
if EVAL_SET == "bldb":
    CSV_PATH = f"{BASE_DIR}/bldb/bldb_training_data.csv"
elif EVAL_SET == "divergent":
    CSV_PATH = f"{BASE_DIR}/divergent/dataset4_pos_annotated.csv"
elif EVAL_SET =="divergent2":
    CSV_PATH = f"{BASE_DIR}/divergent/dataset4_bins_40_50.csv"
elif EVAL_SET == "divergent3":
    CSV_PATH = f"{BASE_DIR}/divergent/dataset4_bins_50_60.csv"
else:
    raise ValueError("EVAL_SET must be 'bldb' or 'divergent'")

# --- model + encoder paths (trained on Gold/Discovery) ---
# Directory naming convention for family:
#   - if include "other beta-lactamase": family/
#   - if exclude it: family_wo_other/
FAMILY_DIR = "family_w_other" if INCLUDE_OTHER_BETA_LACTAMASE else "family_wo_other"

MODEL_PATH = f"{BASE_DIR}/models/{SPLIT}_split/{FAMILY_DIR}/{DATASET}/best_family.keras"
ENCODER_PKL = f"{BASE_DIR}/models/{SPLIT}_split/{FAMILY_DIR}/{DATASET}/family.pkl"

# --- outputs ---
OUT_DIR = f"{BASE_DIR}/models/eval/{SPLIT}_split/{FAMILY_DIR}/{EVAL_SET}/{DATASET}"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_GLOBAL_CSV = os.path.join(
    OUT_DIR,
    f"{EVAL_SET}_{DATASET}_{SPLIT}_eval_family_global_metrics.csv"
)
OUT_PER_CLASS_CSV = os.path.join(
    OUT_DIR,
    f"{EVAL_SET}_{DATASET}_{SPLIT}_eval_family_per_class_metrics.csv"
)
OUT_CM_PNG = os.path.join(
    OUT_DIR,
    f"{EVAL_SET}_{DATASET}_{SPLIT}_eval_family_confusion_matrix.png"
)

print("=== Config ===")
print("DATASET:", DATASET)
print("EVAL_SET:", EVAL_SET)
print("SPLIT:", SPLIT)
print("INCLUDE_OTHER_BETA_LACTAMASE:", INCLUDE_OTHER_BETA_LACTAMASE)
print("CSV_PATH:", CSV_PATH)
print("MODEL_PATH:", MODEL_PATH)
print("ENCODER_PKL:", ENCODER_PKL)
print("OUT_DIR:", OUT_DIR)

# ============================
# Load data
# ============================
df = pd.read_csv(CSV_PATH)

# Family label column varies across your CSVs; prefer "Label" if present.
family_col_candidates = ["Label", "Family", "AMR Gene Family", "Family Label", "Beta-lactamase Family"]
FAMILY_COL = next((c for c in family_col_candidates if c in df.columns), None)
if FAMILY_COL is None:
    raise ValueError(f"Could not find a family label column. Tried: {family_col_candidates}. Found: {list(df.columns)}")

required = ["Sequence", "is_bla", FAMILY_COL]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"CSV must contain columns: {required}. Missing: {missing}")

df = df.dropna(subset=required).copy()
df["Sequence"] = df["Sequence"].astype(str)
df["is_bla"] = df["is_bla"].astype(int)
df[FAMILY_COL] = df[FAMILY_COL].astype(str).str.strip()

# BL-only filter
df = df[df["is_bla"] == 1].copy()

# Optional exclusion of "other beta-lactamase"
if not INCLUDE_OTHER_BETA_LACTAMASE:
    df = df[df[FAMILY_COL] != OTHER_LABEL].copy()

print(f"\nFamily evaluation samples (is_bla==1, after filters): {len(df)}")
print("Family label distribution (top 25):")
print(df[FAMILY_COL].value_counts().head(25))

if len(df) == 0:
    raise ValueError("No samples remain after BL-only + filtering. Check your eval set and INCLUDE_OTHER_BETA_LACTAMASE.")

# ============================
# Load encoder (class mapping)
# ============================
le_family = joblib.load(ENCODER_PKL)
class_names = list(map(str, le_family.classes_))
n_classes = len(class_names)

# Keep only rows whose labels exist in encoder classes (safety)
df = df[df[FAMILY_COL].isin(class_names)].copy()
if len(df) == 0:
    raise ValueError(
        "After filtering to encoder classes, no samples remain. "
        "This likely means your eval labels don't match training label space."
    )

y_true = le_family.transform(df[FAMILY_COL].values).astype(int)

print(f"\nUsing encoder with {n_classes} classes.")
print(f"Filtered eval samples: {len(df)}")

# ============================
# Encode sequences
# ============================
X = encode_and_pad(df["Sequence"].values, max_len=MAX_LEN)

y_dict = {"family_output": y_true}

BATCH_SIZE = 256
eval_ds = (
    tf.data.Dataset.from_tensor_slices((X, y_dict))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ============================
# Load model (with custom objects)
# ============================
print("\nLoading trained Family model...")

custom_objects = {
    "TransformerFamilyModel": TransformerFamilyModel,
    "PositionalAdder": PositionalAdder,
    "AttnMaskExpand": AttnMaskExpand,
    "Encoder": Encoder,
}

family_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects=custom_objects,
    compile=False
)

family_model.compile(
    loss={"family_output": tf.keras.losses.SparseCategoricalCrossentropy()},
    metrics={"family_output": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]}
)

# ============================
# Keras loss + acc
# ============================
keras_metrics = family_model.evaluate(eval_ds, return_dict=True, verbose=1)

# ============================
# Manual predictions for macro + per-class + confusion matrix
# ============================
y_pred = []
for xb, _ in eval_ds:
    probs = family_model(xb, training=False)["family_output"]  # (B, C)
    y_pred.append(tf.argmax(probs, axis=1).numpy().astype(int))
y_pred = np.concatenate(y_pred)

acc = float(accuracy_score(y_true, y_pred))

# Force fixed label space so per-class arrays are always length n_classes
labels = list(range(n_classes))

prec, rec, f1, supp = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=labels,
    average=None,
    zero_division=0
)

prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=labels,
    average="macro",
    zero_division=0
)

cm = confusion_matrix(y_true, y_pred, labels=labels)

# ============================
# Save global metrics
# ============================
global_row = {
    "n": int(len(y_true)),
    "accuracy": acc,
    "macro_precision": float(prec_macro),
    "macro_recall": float(rec_macro),
    "macro_f1": float(f1_macro),
    "cross_entropy": float(keras_metrics.get("loss", np.nan)),
    "keras_acc": float(keras_metrics.get("family_output_acc", np.nan)),
    "model_path": MODEL_PATH,
    "encoder_path": ENCODER_PKL,
    "data_path": CSV_PATH,
    "dataset": DATASET,
    "split": SPLIT,
    "eval_set": EVAL_SET,
    "include_other_beta_lactamase": bool(INCLUDE_OTHER_BETA_LACTAMASE),
    "family_label_col": FAMILY_COL,
}

pd.DataFrame([global_row]).to_csv(OUT_GLOBAL_CSV, index=False)

# ============================
# Save per-class metrics
# ============================
per_class_df = pd.DataFrame({
    "class": class_names,
    "precision": prec,
    "recall": rec,
    "f1": f1,
    "support": supp.astype(int),
})
per_class_df.to_csv(OUT_PER_CLASS_CSV, index=False)

"""
# ============================
# Save confusion matrix PNG
# ============================
fig = plt.figure(figsize=(10, 9))
ax = fig.add_subplot(111)
im = ax.imshow(cm, interpolation="nearest")
fig.colorbar(im, ax=ax)

ax.set_title("Family Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("True")

ax.set_xticks(np.arange(n_classes))
ax.set_yticks(np.arange(n_classes))
ax.set_xticklabels(class_names, rotation=90)
ax.set_yticklabels(class_names)

fig.tight_layout()
fig.savefig(OUT_CM_PNG, dpi=300)
plt.close(fig)
"""

# ============================
# Print summary + paths
# ============================
print("\nFamily evaluation (BL-only) summary:")
print(f"n={len(y_true)}")
print(f"accuracy:        {acc:.4f}")
print(f"macro_precision: {prec_macro:.4f}")
print(f"macro_recall:    {rec_macro:.4f}")
print(f"macro_f1:        {f1_macro:.4f}")
print(f"cross_entropy:   {global_row['cross_entropy']:.4f}")

print("\nSaved outputs:")
print("-", OUT_GLOBAL_CSV)
print("-", OUT_PER_CLASS_CSV)
# print("-", OUT_CM_PNG)
