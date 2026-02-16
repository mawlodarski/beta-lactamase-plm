# ambler_eval.py
# Evaluate TransformerAmblerModel and write ONE unified CSV + console summary

import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

from tokenization import MAX_LEN, encode_and_pad
from model import TransformerAmblerModel, PositionalAdder, AttnMaskExpand, Encoder

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# User config
# ----------------------------
DATASET     = "dapt_discovery"
EVAL_SET    = "divergent4"
SPLIT       = "cluster"
AMBLER_MODE = "full"

BASE_DIR = "/home/wlodarsm/projects/def-mcarthur/wlodarsm/eccb"

# ----------------------------
# Resolve evaluation CSV
# ----------------------------
if EVAL_SET == "bldb":
    CSV_PATH = f"{BASE_DIR}/bldb/bldb_training_data.csv"
elif EVAL_SET == "divergent":
    CSV_PATH = f"{BASE_DIR}/divergent/dataset4_pos_annotated.csv"
elif EVAL_SET == "divergent2":
    CSV_PATH = f"{BASE_DIR}/divergent/dataset4_bins_40_50.csv"
elif EVAL_SET == "divergent3":
    CSV_PATH = f"{BASE_DIR}/divergent/dataset4_bins_50_60.csv"
elif EVAL_SET == "divergent4":
    CSV_PATH = f"{BASE_DIR}/divergent/dataset4_bins_gt60.csv"
elif EVAL_SET == "bacterial":
    CSV_PATH = f"{BASE_DIR}/dapt/bacterial_proteins_50k.csv"
elif EVAL_SET == "hydrolases2":
    CSV_PATH = f"{BASE_DIR}/dapt/hydrolases_50k.csv"
else:
    raise ValueError(f"Unknown EVAL_SET: {EVAL_SET}")

# ----------------------------
# Model + encoder paths
# ----------------------------
if AMBLER_MODE == "full":
    MODEL_PATH = f"{BASE_DIR}/models/{SPLIT}_split/ambler/{DATASET}/best_ambler.keras"
    ENCODER_PKL = f"{BASE_DIR}/models/{SPLIT}_split/ambler/{DATASET}/ambler.pkl"
else:
    MODEL_PATH = f"{BASE_DIR}/models/{SPLIT}_split/ambler_collapsed/{DATASET}/best_ambler.keras"
    ENCODER_PKL = f"{BASE_DIR}/models/{SPLIT}_split/ambler_collapsed/{DATASET}/ambler.pkl"

OUT_DIR = f"{BASE_DIR}/models/eval/{SPLIT}_split/ambler_{AMBLER_MODE}/{EVAL_SET}/{DATASET}/outputs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_ONE_CSV = os.path.join(
    OUT_DIR,
    f"{EVAL_SET}_{DATASET}_{SPLIT}_ambler_eval.csv"
)

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(CSV_PATH)

required = ["Sequence", "is_bla", "Ambler Class"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required).copy()
df["is_bla"] = df["is_bla"].astype(int)
df["Ambler Class"] = df["Ambler Class"].astype(str).str.strip()

if AMBLER_MODE == "collapsed":
    df["Ambler Class"] = df["Ambler Class"].replace({"B1": "B", "B2": "B", "B3": "B"})

# Ambler eval is BL-only
df = df[df["is_bla"] == 1].copy()
if len(df) == 0:
    raise ValueError("No BL sequences found (is_bla == 1).")

# ----------------------------
# Load encoder
# ----------------------------
le = joblib.load(ENCODER_PKL)
classes = list(map(str, le.classes_))
labels = list(range(len(classes)))

df = df[df["Ambler Class"].isin(classes)].copy()
if len(df) == 0:
    raise ValueError("After filtering to encoder classes, no samples remain.")

y_true = le.transform(df["Ambler Class"].values).astype(int)

# ----------------------------
# Encode + dataset
# ----------------------------
X = encode_and_pad(df["Sequence"].values, max_len=MAX_LEN)
eval_ds = tf.data.Dataset.from_tensor_slices((X, y_true)).batch(256)

# ----------------------------
# Load model
# ----------------------------
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={
        "TransformerAmblerModel": TransformerAmblerModel,
        "PositionalAdder": PositionalAdder,
        "AttnMaskExpand": AttnMaskExpand,
        "Encoder": Encoder,
    },
    compile=False
)

# ----------------------------
# Predict
# ----------------------------
y_pred = []
for xb, _ in eval_ds:
    probs = model(xb, training=False)["ambler_output"]
    y_pred.append(tf.argmax(probs, axis=1).numpy().astype(int))
y_pred = np.concatenate(y_pred)

# ----------------------------
# Metrics
# ----------------------------
acc = float(accuracy_score(y_true, y_pred))

prec_m, rec_m, f1_m, _ = precision_recall_fscore_support(
    y_true, y_pred, labels=labels, average="macro", zero_division=0
)

cm = confusion_matrix(y_true, y_pred, labels=labels)

# Per-class one-vs-rest counts (these are fine per class)
tp = np.diag(cm).astype(int)
fn = (cm.sum(axis=1) - tp).astype(int)
fp = (cm.sum(axis=0) - tp).astype(int)
tn = (cm.sum() - (tp + fp + fn)).astype(int)

# IMPORTANT: total samples in multiclass is cm.sum(), NOT tp+tn+fp+fn
n_total = int(cm.sum())
tp_total = int(np.trace(cm))              # correct predictions
err_total = int(n_total - tp_total)       # incorrect predictions
fp_total = err_total                      # micro-style
fn_total = err_total                      # micro-style
tn_total = np.nan                         # not meaningful as a single scalar in multiclass

# Safety checks
assert n_total == len(y_true) == len(y_pred)
assert tp_total + err_total == n_total

# ----------------------------
# Assemble ONE row
# ----------------------------
row = {
    "model": f"{DATASET}_{AMBLER_MODE}",
    "bin": EVAL_SET,
    "accuracy": acc,
    "macro_precision": float(prec_m),
    "macro_recall": float(rec_m),
    "macro_f1": float(f1_m),

    # Multiclass-safe totals
    "tp": tp_total,
    "tn": tn_total,
    "fp": fp_total,
    "fn": fn_total,
    "total": n_total,
}

for cls in ["A", "B", "C", "D"]:
    if cls in classes:
        i = classes.index(cls)
        row.update({
            f"{cls}_tp": int(tp[i]),
            f"{cls}_tn": int(tn[i]),
            f"{cls}_fp": int(fp[i]),
            f"{cls}_fn": int(fn[i]),
            f"{cls}_support": int(cm[i].sum()),
        })
    else:
        row.update({
            f"{cls}_tp": 0,
            f"{cls}_tn": 0,
            f"{cls}_fp": 0,
            f"{cls}_fn": 0,
            f"{cls}_support": 0,
        })

# ----------------------------
# Write CSV
# ----------------------------
pd.DataFrame([row]).to_csv(OUT_ONE_CSV, index=False)

# ----------------------------
# Print summaries
# ----------------------------
print("\n================ Ambler Evaluation Summary ================")
print(f"CSV_PATH:   {CSV_PATH}")
print(f"MODEL_PATH: {MODEL_PATH}")
print(f"ENCODER:    {ENCODER_PKL}")
print(f"n_total:    {n_total}")
print(f"classes:    {classes}")
print("-----------------------------------------------------------")
for k in row:
    print(f"{k}: {row[k]}")
print("===========================================================")

print("\n================ TOTAL =================")
print(f"TOTAL SAMPLES (cm.sum): {n_total}")
print(f"CORRECT (trace):        {tp_total}")
print(f"INCORRECT:              {err_total}")
print("=======================================")

print(f"\nCSV written to: {OUT_ONE_CSV}")
