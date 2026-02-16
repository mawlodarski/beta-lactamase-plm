import pandas as pd 
import numpy as np 
import joblib
import os
import ast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
#load model
clf = joblib.load("best_bl_lr_gold.joblib")

# evaluate seqs
# CONFIG
CSV_PATH = "clean_hydrolases_with_esm2.csv"   # change as needed
MODEL_PATH = "best_bl_lr_discovery.joblib"
OUT_CSV = "clean_hydrolases_metrics.csv"
THRESH = 0.5
# ============================
# Load model + data
# ============================
clf = joblib.load(MODEL_PATH)

df = pd.read_csv(CSV_PATH)
df["is_bla"] = 0

y = df["is_bla"].values

# Parse ESM2 embeddings from single column
X = np.vstack(
    df["ESM-2_embedding"].apply(lambda s: np.array(ast.literal_eval(s), dtype=np.float32)).values
)

# ============================
# Predict
# ============================
p = clf.predict_proba(X)[:, 1]
y_hat = (p >= THRESH).astype(int)

# ============================
# Confusion counts
# ============================
y_true = y.astype(int)
y_pred = y_hat.astype(int)

tp = int(np.sum((y_true == 1) & (y_pred == 1)))
tn = int(np.sum((y_true == 0) & (y_pred == 0)))
fp = int(np.sum((y_true == 0) & (y_pred == 1)))
fn = int(np.sum((y_true == 1) & (y_pred == 0)))

# ============================
# Metrics
# ============================
metrics = {
    "n": len(y),
    "prevalence": float(np.mean(y)),
    "threshold": THRESH,

    "tp": tp,
    "tn": tn,
    "fp": fp,
    "fn": fn,

    "accuracy": accuracy_score(y_true, y_pred),
    "precision": precision_score(y_true, y_pred, zero_division=0),
    "recall": recall_score(y_true, y_pred),
    "f1": f1_score(y_true, y_pred, zero_division=0),
    "auroc": roc_auc_score(y_true, p),
    "auprc": average_precision_score(y_true, p),
}

# Best-F1 threshold
thresholds = np.linspace(0.01, 0.99, 99)
f1s = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in thresholds]
metrics["best_f1"] = max(f1s)
metrics["best_f1_threshold"] = thresholds[np.argmax(f1s)]

# ============================
# Output
# ============================
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

pd.DataFrame([metrics]).to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CS