# ambler_eval_esm2_lr.py
# Evaluate an ESM-2 (embedding + LR) Ambler classifier and write ONE unified CSV + console summary.

import os
import joblib
import ast
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ============================
# CONFIG (edit these)
# ============================
SET = "divergent_50_60"
CSV_PATH   = f"{SET}_with_esm2.csv"
MODEL_PATH = "discovery_cluster_esm2_softmax_full.joblib"

MODEL_STR = "ESM-2 discovery (Embedding + softmax)"  # CSV "model"
BIN_STR   = "50-60"                               # CSV "bin"

OUT_DIR = "ambler_full_eval_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_ONE_CSV = os.path.join(OUT_DIR, f"{SET}_ambler_eval.csv")

EMB_COL = "ESM-2_embedding"
AMBLER_COL = "Ambler Class"
BL_FILTER_COL = "is_bla"
FILTER_TO_BL = True

# "full" => A,B1,B2,B3,C,D
# "collapsed" => A,B,C,D (B1/B2/B3 -> B)
AMBLER_MODE = "full"  # "full" or "collapsed"

# If True, in AMBLER_MODE="full" we ALSO write a collapsed B_* aggregate (B1+B2+B3 summed)
ADD_COLLAPSED_B_AGGREGATE_IN_FULL_MODE = True

# ============================
# Ambler label systems
# ============================
FULL_CLASSES = ["A", "B1", "B2", "B3", "C", "D"]
COLLAPSED_CLASSES = ["A", "B", "C", "D"]

def normalize_ambler(s: str) -> str:
    return str(s).strip().upper()

def apply_collapse(label: str) -> str:
    if label in {"B1", "B2", "B3"}:
        return "B"
    return label

def parse_embedding(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return np.asarray(x, dtype=np.float32)
    return np.asarray(ast.literal_eval(x), dtype=np.float32)

# ============================
# Load model (bundle-safe)
# ============================
bundle = joblib.load(MODEL_PATH)

if isinstance(bundle, dict) and "model" in bundle:
    clf = bundle["model"]
    model_label_names = bundle.get("label_names", None)   # int -> label string (training)
    model_mode = bundle.get("ambler_mode", None)
else:
    clf = bundle
    model_label_names = None
    model_mode = None

# ============================
# Determine evaluation class space
# ============================
if AMBLER_MODE not in {"full", "collapsed"}:
    raise ValueError('AMBLER_MODE must be "full" or "collapsed"')

if AMBLER_MODE == "full":
    class_space = FULL_CLASSES
else:
    class_space = COLLAPSED_CLASSES

label_to_idx = {c: i for i, c in enumerate(class_space)}
idx_to_label = {i: c for c, i in label_to_idx.items()}

# ============================
# Validate model label space (prevents silent misalignment)
# ============================
if model_label_names is not None:
    trained_space = [model_label_names[i] for i in sorted(model_label_names.keys())]
    if trained_space != class_space:
        raise ValueError(
            "Model label space does not match evaluation AMBLER_MODE.\n"
            f"  MODEL_PATH: {MODEL_PATH}\n"
            f"  Model trained labels: {trained_space}\n"
            f"  Eval AMBLER_MODE={AMBLER_MODE} labels: {class_space}\n"
            "Fix: point MODEL_PATH to the correct model, or switch AMBLER_MODE."
        )

# ============================
# Load data
# ============================
df = pd.read_csv(CSV_PATH)

required = [EMB_COL, AMBLER_COL]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

if FILTER_TO_BL:
    if BL_FILTER_COL not in df.columns:
        raise ValueError(f"FILTER_TO_BL=True but missing column: {BL_FILTER_COL}")
    df = df[df[BL_FILTER_COL].astype(int) == 1].copy()

# Normalize labels
df[AMBLER_COL] = df[AMBLER_COL].apply(normalize_ambler)

# Collapse if requested
if AMBLER_MODE == "collapsed":
    df[AMBLER_COL] = df[AMBLER_COL].apply(apply_collapse)

df = df[df[AMBLER_COL].isin(class_space)].copy()
if len(df) == 0:
    raise ValueError("No rows remain after filtering to Ambler label space.")

y_true = df[AMBLER_COL].map(label_to_idx).astype(int).values
n_samples = int(len(y_true))
n_classes = int(len(class_space))

X = np.vstack(df[EMB_COL].apply(parse_embedding).values)

# ============================
# Predict
# ============================
y_pred = np.asarray(clf.predict(X)).astype(int)

# ============================
# Metrics (global)
# ============================
acc = float(accuracy_score(y_true, y_pred))
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    labels=list(range(n_classes)),
    average="macro",
    zero_division=0
)

# Sample-level totals (total == n_samples)
tp_global = int(np.sum(y_true == y_pred))
fp_global = int(np.sum(y_true != y_pred))
fn_global = fp_global
tn_global = 0
total = int(n_samples)

# ============================
# Confusion matrix + per-class OVR components
# ============================
labels = list(range(n_classes))
cm = confusion_matrix(y_true, y_pred, labels=labels).astype(int)

tp = np.diag(cm).astype(int)
fn = (cm.sum(axis=1) - tp).astype(int)
fp = (cm.sum(axis=0) - tp).astype(int)
tn = (cm.sum() - (tp + fp + fn)).astype(int)

support = cm.sum(axis=1).astype(int)

# ============================
# Assemble ONE row
# ============================
row = {
    "model": str(MODEL_STR),
    "bin": str(BIN_STR),
    "ambler_mode": str(AMBLER_MODE),
    "accuracy": float(acc),
    "macro_precision": float(macro_p),
    "macro_recall": float(macro_r),
    "macro_f1": float(macro_f1),
    "tp": tp_global,
    "tn": tn_global,
    "fp": fp_global,
    "fn": fn_global,
    "total": total,
}

# Per-class blocks
def add_class_block(label: str):
    if label not in label_to_idx:
        row[f"{label}_tp"] = 0
        row[f"{label}_tn"] = 0
        row[f"{label}_fp"] = 0
        row[f"{label}_fn"] = 0
        row[f"{label}_support"] = 0
        return

    i = label_to_idx[label]
    row[f"{label}_tp"] = int(tp[i])
    row[f"{label}_tn"] = int(tn[i])
    row[f"{label}_fp"] = int(fp[i])
    row[f"{label}_fn"] = int(fn[i])
    row[f"{label}_support"] = int(support[i])

if AMBLER_MODE == "collapsed":
    # A/B/C/D blocks
    for lab in ["A", "B", "C", "D"]:
        add_class_block(lab)

    columns = [
        "model", "bin", "ambler_mode",
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "tp", "tn", "fp", "fn", "total",
        "A_tp", "A_tn", "A_fp", "A_fn", "A_support",
        "B_tp", "B_tn", "B_fp", "B_fn", "B_support",
        "C_tp", "C_tn", "C_fp", "C_fn", "C_support",
        "D_tp", "D_tn", "D_fp", "D_fn", "D_support",
    ]

else:
    # Full mode: A/B1/B2/B3/C/D blocks
    for lab in ["A", "B1", "B2", "B3", "C", "D"]:
        add_class_block(lab)

    # Optional: collapsed B aggregate for convenience
    if ADD_COLLAPSED_B_AGGREGATE_IN_FULL_MODE:
        b_idxs = [label_to_idx[b] for b in ["B1", "B2", "B3"] if b in label_to_idx]
        row["B_tp"] = int(tp[b_idxs].sum()) if b_idxs else 0
        row["B_tn"] = int(tn[b_idxs].sum()) if b_idxs else 0
        row["B_fp"] = int(fp[b_idxs].sum()) if b_idxs else 0
        row["B_fn"] = int(fn[b_idxs].sum()) if b_idxs else 0
        row["B_support"] = int(support[b_idxs].sum()) if b_idxs else 0

    columns = [
        "model", "bin", "ambler_mode",
        "accuracy", "macro_precision", "macro_recall", "macro_f1",
        "tp", "tn", "fp", "fn", "total",
        "A_tp", "A_tn", "A_fp", "A_fn", "A_support",
        "B1_tp", "B1_tn", "B1_fp", "B1_fn", "B1_support",
        "B2_tp", "B2_tn", "B2_fp", "B2_fn", "B2_support",
        "B3_tp", "B3_tn", "B3_fp", "B3_fn", "B3_support",
        "C_tp", "C_tn", "C_fp", "C_fn", "C_support",
        "D_tp", "D_tn", "D_fp", "D_fn", "D_support",
    ]

    if ADD_COLLAPSED_B_AGGREGATE_IN_FULL_MODE:
        columns += ["B_tp", "B_tn", "B_fp", "B_fn", "B_support"]

out_df = pd.DataFrame([row], columns=columns)
out_df.to_csv(OUT_ONE_CSV, index=False)

# ============================
# Print summary
# ============================
print("\n================ Ambler Evaluation Summary (ESM-2 LR) ================")
print(f"MODEL_PATH:  {MODEL_PATH}")
print(f"CSV_PATH:    {CSV_PATH}")
print(f"model:       {row['model']}")
print(f"bin:         {row['bin']}")
print(f"AMBLER_MODE: {AMBLER_MODE}")
print("---------------------------------------------------------------------")
print(f"n_samples:        {n_samples}")
print(f"n_classes:        {n_classes}")
print(f"accuracy:         {row['accuracy']:.6f}")
print(f"macro_precision:  {row['macro_precision']:.6f}")
print(f"macro_recall:     {row['macro_recall']:.6f}")
print(f"macro_f1:         {row['macro_f1']:.6f}")
print("---------------------------------------------------------------------")
print("Sample-level totals:")
print(f"tp:    {row['tp']}")
print(f"tn:    {row['tn']}")
print(f"fp:    {row['fp']}")
print(f"fn:    {row['fn']}")
print(f"total: {row['total']}  (== n_samples)")
print("---------------------------------------------------------------------")

if AMBLER_MODE == "collapsed":
    for lab in ["A", "B", "C", "D"]:
        print(
            f"{lab}: tp={row[f'{lab}_tp']} tn={row[f'{lab}_tn']} "
            f"fp={row[f'{lab}_fp']} fn={row[f'{lab}_fn']} support={row[f'{lab}_support']}"
        )
else:
    for lab in ["A", "B1", "B2", "B3", "C", "D"]:
        print(
            f"{lab}: tp={row[f'{lab}_tp']} tn={row[f'{lab}_tn']} "
            f"fp={row[f'{lab}_fp']} fn={row[f'{lab}_fn']} support={row[f'{lab}_support']}"
        )
    if ADD_COLLAPSED_B_AGGREGATE_IN_FULL_MODE:
        print(
            f"B (B1+B2+B3): tp={row['B_tp']} tn={row['B_tn']} "
            f"fp={row['B_fp']} fn={row['B_fn']} support={row['B_support']}"
        )

print("=====================================================================")
print(f"CSV written to: {OUT_ONE_CSV}\n")
