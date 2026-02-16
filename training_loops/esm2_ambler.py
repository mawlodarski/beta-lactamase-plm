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
    precision_recall_fscore_support,
    f1_score,
    confusion_matrix
)

#TURN AMBLER INTO CATEGORICAL

col = "Ambler Class"

ambler_map = {
    "A": 0,
    "B1": 1,
    "B2": 2,
    "B3": 3,
    "C": 4,
    "D": 5
}

# ----------------------------
# User config
# ----------------------------
DATASET = "discovery"
SPLIT = "cluster"         # "random" or "cluster"
SEED = 42
TEST_SIZE = 0.1

# For cluster split: what to do with singleton clusters (cluster size == 1)
SINGLETON_POLICY = "train"   # "train" or "random"

# NEW: label mode
AMBLER_MODE = "full"    # "collapsed" or "full"

df = pd.read_csv(f"{DATASET}_with_esm2.csv")

# ----------------------------
# Config
# ----------------------------
EMB_COL = "ESM-2_embedding"
BLA_COL = "is_bla"
AMBLER_COL = "Ambler Class"   # values: "A","B1","B2","B3","C","D"

BIN_NAME = "all"

# ----------------------------
# Label spaces
# ----------------------------
# Collapsed output: A, B, C, D
collapsed_names = {0: "A", 1: "B", 2: "C", 3: "D"}
collapse_map = {"A": 0, "B1": 1, "B2": 1, "B3": 1, "C": 2, "D": 3}

# Full output: A, B1, B2, B3, C, D
full_order = ["A", "B1", "B2", "B3", "C", "D"]
full_names = {i: lab for i, lab in enumerate(full_order)}
full_map = {lab: i for i, lab in enumerate(full_order)}

if AMBLER_MODE not in {"collapsed", "full"}:
    raise ValueError('AMBLER_MODE must be "collapsed" or "full"')

if AMBLER_MODE == "collapsed":
    MODEL_NAME = "ESM2_softmax_ambler_collapsed"
    y_col = "Ambler_y"
    label_names = collapsed_names
    label_map = collapse_map
    labels = [0, 1, 2, 3]
else:
    MODEL_NAME = "ESM2_softmax_ambler_full"
    y_col = "Ambler_y"
    label_names = full_names
    label_map = full_map
    labels = list(range(len(full_order)))  # [0..5]

MODEL_OUT = f"{DATASET}_{SPLIT}_esm2_softmax_{AMBLER_MODE}.joblib"
OUT_CSV = f"{DATASET}_{SPLIT}_esm2_softmax_{AMBLER_MODE}_metrics.csv"

# ----------------------------
# Filter to BL + build labels
# ----------------------------
bla = df[df[BLA_COL].astype(int) == 1].copy()

bla[AMBLER_COL] = bla[AMBLER_COL].astype(str).str.strip()

if AMBLER_MODE == "collapsed":
    bla[y_col] = bla[AMBLER_COL].map(collapse_map)
else:
    bla[y_col] = bla[AMBLER_COL].map(full_map)

bla = bla[bla[y_col].notna()].copy()
bla[y_col] = bla[y_col].astype(int)

print("Filtered rows (is_bla==1):", len(bla))
print(f"{AMBLER_MODE} y distribution:")
print(
    bla[y_col]
    .value_counts()
    .sort_index()
    .rename(index=label_names)
)

# ----------------------------
# Split logic
# ----------------------------
if SPLIT == "random":
    X = np.vstack(
        bla[EMB_COL].apply(
            lambda s: np.asarray(ast.literal_eval(s), dtype=np.float32)
        ).values
    )
    y = bla[y_col].values

    print("X shape:", X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y
    )

elif SPLIT == "cluster":
    if "cluster_id" not in bla.columns:
        raise ValueError("Missing required column for cluster split: cluster_id")

    tmp = bla.copy()
    tmp["cluster_id"] = tmp["cluster_id"].astype(str).str.strip()

    cluster_counts = tmp["cluster_id"].value_counts()
    valid_clusters = cluster_counts[cluster_counts >= 2].index
    singleton_clusters = cluster_counts[cluster_counts == 1].index

    df_valid = tmp[tmp["cluster_id"].isin(valid_clusters)].copy()
    df_single = tmp[tmp["cluster_id"].isin(singleton_clusters)].copy()

    train_valid, test_valid = train_test_split(
        df_valid,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=df_valid["cluster_id"],
    )

    if len(df_single) == 0:
        train_df = train_valid
        test_df = test_valid
    else:
        if SINGLETON_POLICY == "train":
            train_single = df_single
            test_single = df_single.iloc[0:0].copy()
        elif SINGLETON_POLICY == "random":
            if len(df_single) == 1:
                train_single = df_single
                test_single = df_single.iloc[0:0].copy()
            else:
                train_single, test_single = train_test_split(
                    df_single,
                    test_size=TEST_SIZE,
                    random_state=SEED,
                    shuffle=True,
                )
        else:
            raise ValueError('SINGLETON_POLICY must be "train" or "random"')

        train_df = pd.concat([train_valid, train_single], ignore_index=True)
        test_df = pd.concat([test_valid, test_single], ignore_index=True)

    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    X_train = np.vstack(
        train_df[EMB_COL].apply(
            lambda s: np.asarray(ast.literal_eval(s), dtype=np.float32)
        ).values
    )
    y_train = train_df[y_col].values

    X_test = np.vstack(
        test_df[EMB_COL].apply(
            lambda s: np.asarray(ast.literal_eval(s), dtype=np.float32)
        ).values
    )
    y_test = test_df[y_col].values

    print(f"\nCluster split summary ({AMBLER_MODE} Ambler):")
    print("Total rows:", len(tmp))
    print("Train rows:", len(train_df))
    print("Test rows: ", len(test_df))
    print("Total clusters:", tmp["cluster_id"].nunique())
    print("Singleton clusters (size=1):", int((cluster_counts == 1).sum()))
    train_clusters = set(train_df["cluster_id"].astype(str))
    test_clusters = set(test_df["cluster_id"].astype(str))
    print("Clusters in train:", len(train_clusters))
    print("Clusters in test: ", len(test_clusters))
    print("Overlapping clusters (expected):", len(train_clusters & test_clusters))

else:
    raise ValueError('SPLIT must be "random" or "cluster"')

# ----------------------------
# Softmax classifier
# ----------------------------
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=5000,
        class_weight="balanced"
    ))
])

clf.fit(X_train, y_train)

# ----------------------------
# SAVE MODEL
# ----------------------------
joblib.dump(
    {
        "model": clf,
        "label_names": label_names,          # int -> label string
        "label_map_used": label_map,         # original string -> int (collapse or full)
        "ambler_mode": AMBLER_MODE,
        "embedding_col": EMB_COL,
        "dataset": DATASET,
        "split": SPLIT,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "singleton_policy": SINGLETON_POLICY if SPLIT == "cluster" else None
    },
    MODEL_OUT
)
print(f"\nModel saved to: {MODEL_OUT}")

# ----------------------------
# Predictions
# ----------------------------
y_pred = clf.predict(X_test)

# ----------------------------
# Overall metrics
# ----------------------------
acc = accuracy_score(y_test, y_pred)
macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average="macro", zero_division=0
)

total = len(y_test)
tp = int(np.sum(y_test == y_pred))
fp = int(np.sum(y_test != y_pred))
fn = fp
tn = 0

# ----------------------------
# Per-class confusion components
# ----------------------------
cm = confusion_matrix(y_test, y_pred, labels=labels)

per_class = {}
for i, cls in enumerate(labels):
    cls_tp = int(cm[i, i])
    cls_fn = int(cm[i, :].sum() - cls_tp)
    cls_fp = int(cm[:, i].sum() - cls_tp)
    cls_tn = int(cm.sum() - (cls_tp + cls_fn + cls_fp))
    cls_support = int(cm[i, :].sum())

    per_class[cls] = {
        "tp": cls_tp,
        "tn": cls_tn,
        "fp": cls_fp,
        "fn": cls_fn,
        "support": cls_support
    }

# ----------------------------
# Build output row (schema)
# ----------------------------
row = {
    "model": MODEL_NAME,
    "bin": BIN_NAME,
    "accuracy": float(acc),
    "macro_precision": float(macro_p),
    "macro_recall": float(macro_r),
    "macro_f1": float(macro_f1),
    "tp": tp,
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "total": int(total),
}

# class-level columns
for cls, name in label_names.items():
    stats = per_class.get(cls, {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "support": 0})
    row[f"{name}_tp"] = stats["tp"]
    row[f"{name}_tn"] = stats["tn"]
    row[f"{name}_fp"] = stats["fp"]
    row[f"{name}_fn"] = stats["fn"]
    row[f"{name}_support"] = stats["support"]

# fixed leading columns
columns = [
    "model", "bin", "accuracy", "macro_precision", "macro_recall", "macro_f1",
    "tp", "tn", "fp", "fn", "total",
]

# append class columns in label order
for cls in labels:
    name = label_names[cls]
    columns.extend([f"{name}_tp", f"{name}_tn", f"{name}_fp", f"{name}_fn", f"{name}_support"])

out_df = pd.DataFrame([row], columns=columns)

print("\n=== Final Evaluation Row ===")
print(out_df.to_string(index=False))

# ----------------------------
# Save final CSV
# ----------------------------
out_df.to_csv(OUT_CSV, index=False)
print(f"\nMetrics CSV saved to: {OUT_CSV}")

# ----------------------------
# Optional: softmax probabilities
# ----------------------------
proba = clf.predict_proba(X_test)
print("\nSoftmax probs shape:", proba.shape)
