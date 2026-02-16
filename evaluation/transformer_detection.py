# bl_detector_eval.py
# Evaluate trained TransformerBLDetector on BLDB/hydrolases

# ============================
# CONFIG
# ============================
DATASET = "discovery"        # "gold" or "discovery"
SPLIT   = "random_split"  # "random_split" or "cluster_split"

EVAL_SET = "hydrolases"  # "bldb" or hydrolases or divergent

import os
import numpy as np
import pandas as pd

# If you're running on a login/CPU node, uncomment this to suppress CUDA init:
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

import tensorflow as tf

from tokenization import MAX_LEN, encode_and_pad
from model import TransformerBLDetector, PositionalAdder, AttnMaskExpand, Encoder

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---- paths ----
# ============================
# Paths (derived from CONFIG)
# ============================

BASE_DIR = "/home/wlodarsm/projects/def-mcarthur/wlodarsm/eccb"

# --- evaluation data ---
if EVAL_SET == "hydrolases":
    CSV_PATH = f"{BASE_DIR}/raw_data/negs/hydrolases.csv"
elif EVAL_SET == "bldb":
    CSV_PATH = f"{BASE_DIR}/bldb/bldb_training_data.csv"
elif EVAL_SET == "divergent":
    CSV_PATH = f"{BASE_DIR}/divergent/dataset4_pos_annotated.csv"
elif EVAL_SET =="divergent2":
    CSV_PATH = f"{BASE_DIR}/divergent/dataset4_bins_40_50.csv"
elif EVAL_SET == "divergent3":
    CSV_PATH = f"{BASE_DIR}/divergent/dataset4_bins_50_60.csv"
elif EVAL_SET == "divergent4":
    CSV_PATH = f"{BASE_DIR}/divergent/dataset4_bins_gt60.csv"
elif EVAL_SET == "bacterial":
    CSV_PATH = f"{BASE_DIR}/raw_data/negs/bacterial.csv"
else:
    raise ValueError("EVAL_SET must be 'hydrolases' or 'bldb' or 'divergent'")

MODEL_PATH = (
f"{BASE_DIR}/models/{SPLIT}/bl_detector/{DATASET}/" f"best_bl_detector.keras"
)
OUT_DIR = (
f"{BASE_DIR}/models/eval/{SPLIT}/bl_detector/{EVAL_SET}/{DATASET}"
)

os.makedirs(OUT_DIR, exist_ok=True)

OUT_CSV = os.path.join(
    OUT_DIR,
    f"{EVAL_SET}_eval_"
    f"{DATASET}_{SPLIT}.csv"
)

# ---- load data ----
df = pd.read_csv(CSV_PATH)

if "Sequence" not in df.columns or "is_bla" not in df.columns:
    raise ValueError("CSV must contain Sequence and is_bla columns")

df = df.dropna(subset=["Sequence", "is_bla"]).copy()
df["is_bla"] = df["is_bla"].astype(int)

print(f"evaluation samples: {len(df)}")
print("BL prevalence:", df["is_bla"].mean())

# ---- encode ----
X = encode_and_pad(df["Sequence"].astype(str).values, max_len=MAX_LEN)
y = df["is_bla"].astype(np.float32).values
y_dict = {"bla_output": y}

# ---- dataset ----
BATCH_SIZE = 256
eval_ds = (
    tf.data.Dataset.from_tensor_slices((X, y_dict))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ---- load model (with custom objects) ----
print("Loading trained BL detector...")

custom_objects = {
    "TransformerBLDetector": TransformerBLDetector,
    #"TransformerHierarchicalBLModel": TransformerHierarchicalBLModel,
    "PositionalAdder": PositionalAdder,
    "AttnMaskExpand": AttnMaskExpand,
    "Encoder": Encoder,
}

bl_model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects=custom_objects,
    compile=False
)

# ---- compile for evaluation ----
bl_model.compile(
    loss={"bla_output": tf.keras.losses.BinaryCrossentropy()},
    metrics={"bla_output": [
        tf.keras.metrics.BinaryAccuracy(name="acc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auroc", curve="ROC"),
        tf.keras.metrics.AUC(name="auprc", curve="PR"),
    ]}
)


    # ---- evaluate ----
metrics = bl_model.evaluate(eval_ds, return_dict=True)

print("\nevaluation metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# ---- compute TP/TN/FP/FN counts ----
# Predict in the same order as eval_ds yields labels
y_prob = bl_model.predict(eval_ds, verbose=0)

# Handle possible dict output or array output
if isinstance(y_prob, dict):
    y_prob = y_prob["bla_output"]

y_prob = np.asarray(y_prob).reshape(-1)
y_true = y.reshape(-1).astype(int)

THRESH = 0.5
y_pred = (y_prob >= THRESH).astype(int)

tp = int(np.sum((y_true == 1) & (y_pred == 1)))
tn = int(np.sum((y_true == 0) & (y_pred == 0)))
fp = int(np.sum((y_true == 0) & (y_pred == 1)))
fn = int(np.sum((y_true == 1) & (y_pred == 0)))
total = int(len(y_true))

# ---- write false positives to FASTA ----
fp_mask = (y_true == 0) & (y_pred == 1)
fp_df = df.loc[fp_mask].copy()

FP_FASTA = os.path.join(
    OUT_DIR,
    f"{EVAL_SET}_false_positives_{DATASET}_{SPLIT}.fasta"
)

with open(FP_FASTA, "w") as fh:
    for i, row in fp_df.iterrows():
        seq = str(row["Sequence"])
        header = f">fp_{i}"
        fh.write(header + "\n")
        fh.write(seq + "\n")

print(f"Saved {len(fp_df)} false positives to:\n{FP_FASTA}")

print("\nconfusion counts (threshold=0.5):")
print(f"TP: {tp}  TN: {tn}  FP: {fp}  FN: {fn}  Total: {total}")

# Add counts to metrics dict (so they go into the same CSV row)
metrics.update({
    "threshold": THRESH,
    "tp": tp,
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "total": total,
})

# ---- save metrics (CSV only) ----
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv(OUT_CSV, index=False)

print(f"\nSaved evaluation metrics to:\n{OUT_CSV}")
