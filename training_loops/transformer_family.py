# family_train.py
# Train TransformerFamilyModel on Discovery (random split)
# Filters OUT non-family labels:
#   - other ARG
#   - other beta-lactamase
#   - bacterial protein
#
# Logs per-epoch TRAIN+VAL (single CSV):
#   - loss
#   - accuracy
#   - macro precision/recall/F1
# Tracks best epoch by val_macro_f1, and ONLY for that best epoch:
#   - writes train per-class CSV
#   - writes val per-class CSV
#   - writes val confusion matrix PNG
# Saves best model as best_family.keras and final model snapshot as family_final.keras.

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)

from tokenization import MAX_LEN, vocab_size, encode_and_pad
from model import TransformerFamilyModel


SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# Paths
# ----------------------------
CSV_PATH = "/home/wlodarsm/projects/def-mcarthur/wlodarsm/eccb/gold/gold_training_data.csv"
OUT_DIR = "/home/wlodarsm/projects/def-mcarthur/wlodarsm/eccb/models/cluster_split/family_wo_other/family_gold"
os.makedirs(OUT_DIR, exist_ok=True)

HISTORY_CSV = os.path.join(OUT_DIR, "family_epoch_metrics.csv")
ENCODER_PKL = os.path.join(OUT_DIR, "family.pkl")
ENCODER_JSON = os.path.join(OUT_DIR, "family.json")
BEST_MODEL_PATH = os.path.join(OUT_DIR, "best_family.keras")
FINAL_MODEL_PATH = os.path.join(OUT_DIR, "family_final.keras")
CM_PNG = os.path.join(OUT_DIR, "family_confusion_matrix.png")

BEST_TRAIN_PER_CLASS_CSV = os.path.join(OUT_DIR, "family_train_per_class_best.csv")
BEST_VAL_PER_CLASS_CSV = os.path.join(OUT_DIR, "family_val_per_class_best.csv")

# ----------------------------
# Load + filter data
# ----------------------------
df = pd.read_csv(CSV_PATH)

required = ["Sequence", "is_bla", "Label"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required).copy()
df["is_bla"] = df["is_bla"].astype(int)
df["Label"] = df["Label"].astype(str).str.strip()

# Family classification trained/evaluated on BL sequences only
df = df[df["is_bla"] == 1].copy()

# Filter OUT non-family categories explicitly
EXCLUDE_LABELS = {
    "other ARG",
    "other beta-lactamase",
    "bacterial protein",
}
df = df[~df["Label"].isin(EXCLUDE_LABELS)].copy()

print(f"Family samples (is_bla==1 and Label not in exclude): {len(df)}")
print("\nLabel distribution (top 30):")
print(df["Label"].value_counts().head(30))

if len(df) == 0:
    raise ValueError("No family samples remain after filtering. Check Label values / exclude list.")

# ----------------------------
# LabelEncoder
# ----------------------------
le_family = LabelEncoder()
le_family.fit(df["Label"].values)

class_names = list(map(str, le_family.classes_))
n_classes = len(class_names)

y_all = le_family.transform(df["Label"].values).astype(np.int32)

joblib.dump(le_family, ENCODER_PKL)
with open(ENCODER_JSON, "w") as f:
    json.dump(class_names, f, indent=2)

print("\nSaved Family LabelEncoder:")
print("-", ENCODER_PKL)
print("-", ENCODER_JSON)
print(f"n_family_classes: {n_classes}")

'''
# ----------------------------
# Train/val split (stratified on family label)
# ----------------------------
train_df, val_df, y_train, y_val = train_test_split(
    df,
    y_all,
    test_size=0.10,
    random_state=SEED,
    stratify=y_all
)

print(f"\nTrain n={len(train_df)} | Val n={len(val_df)}")
print("Train class counts (top 20):\n", pd.Series(y_train).value_counts().head(20))
print("Val class counts (top 20):\n", pd.Series(y_val).value_counts().head(20))
'''

# --- CLUSTER SPLIT (stratify by cluster_id, handle singletons safely) ---
if "cluster_id" not in df.columns:
    raise ValueError("Missing required column: cluster_id")

# ensure stable dtype for stratification
df = df.copy()
df["cluster_id"] = df["cluster_id"].astype(str).str.strip()

# Count cluster sizes
cluster_counts = df["cluster_id"].value_counts(dropna=False)

valid_clusters = cluster_counts[cluster_counts >= 2].index
singleton_clusters = cluster_counts[cluster_counts == 1].index

df_valid = df[df["cluster_id"].isin(valid_clusters)].copy()
df_single = df[df["cluster_id"].isin(singleton_clusters)].copy()

# 1) Stratified split on clusters with >=2 members
train_valid, val_valid = train_test_split(
    df_valid,
    test_size=0.10,
    random_state=SEED,
    shuffle=True,
    stratify=df_valid["cluster_id"],
)

# 2) Singletons: cannot stratify
# - "train": keep all singleton clusters in train (most conservative / avoids "lucky" val)
# - "random": randomly assign ~10% of singletons to val (better size matching, noisier)
SINGLETON_POLICY = "train"  # "train" or "random"

if len(df_single) == 0:
    train_df = train_valid
    val_df = val_valid
else:
    if SINGLETON_POLICY == "train":
        train_single = df_single
        val_single = df_single.iloc[0:0].copy()
    elif SINGLETON_POLICY == "random":
        if len(df_single) == 1:
            train_single = df_single
            val_single = df_single.iloc[0:0].copy()
        else:
            train_single, val_single = train_test_split(
                df_single,
                test_size=0.10,
                random_state=SEED,
                shuffle=True,
            )
    else:
        raise ValueError('SINGLETON_POLICY must be "train" or "random"')

    train_df = pd.concat([train_valid, train_single], ignore_index=True)
    val_df = pd.concat([val_valid, val_single], ignore_index=True)

# Final shuffle for cleanliness
train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

# Construct y_train/y_val from the split dataframes (aligned to the split order)
# Assumes df has a string family column used to build y_all earlier (e.g., df["Label"]).
# Replace "Label" with your family column name if different.
FAMILY_COL = "Label"
if FAMILY_COL not in train_df.columns or FAMILY_COL not in val_df.columns:
    raise ValueError(f"Missing required family column: {FAMILY_COL}")

y_train = le_family.transform(train_df[FAMILY_COL].astype(str).values).astype(np.int32)
y_val = le_family.transform(val_df[FAMILY_COL].astype(str).values).astype(np.int32)

print("=== Cluster split summary (family model) ===")
print("Total rows:", len(df))
print("Train rows:", len(train_df))
print("Val rows:  ", len(val_df))
print("Total clusters:", df["cluster_id"].nunique())
print("Singleton clusters (size=1):", int((cluster_counts == 1).sum()))

train_clusters = set(train_df["cluster_id"])
val_clusters = set(val_df["cluster_id"])
print("Clusters in train:", len(train_clusters))
print("Clusters in val:  ", len(val_clusters))
print("Overlapping clusters (expected with your goal):", len(train_clusters & val_clusters))

print(f"\nTrain n={len(train_df)} | Val n={len(val_df)}")
print("Train class counts (top 20):\n", pd.Series(y_train).value_counts().head(20))
print("Val class counts (top 20):\n", pd.Series(y_val).value_counts().head(20))

# ----------------------------
# Encode sequences
# ----------------------------
X_train = encode_and_pad(train_df["Sequence"].astype(str).values, max_len=MAX_LEN)
X_val = encode_and_pad(val_df["Sequence"].astype(str).values, max_len=MAX_LEN)

y_train_dict = {"family_output": y_train}
y_val_dict = {"family_output": y_val}

# ----------------------------
# tf.data
# ----------------------------
BATCH_SIZE = 256

train_ds = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train_dict))
    .shuffle(min(len(X_train), 20000), seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val_dict))
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ----------------------------
# Model
# ----------------------------
family_model = TransformerFamilyModel(
    vocab_size=vocab_size,
    n_family_classes=n_classes,
    emb_dim=128,
    num_heads=4,
    ff_dim=512,
    max_len=MAX_LEN,
    num_layers=4,
    dropout=0.1
)

_ = family_model(tf.zeros((1, MAX_LEN), dtype=tf.int32), training=False)
family_model.summary()

family_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={"family_output": tf.keras.losses.SparseCategoricalCrossentropy()},
    metrics={"family_output": [
        tf.keras.metrics.SparseCategoricalAccuracy(name="acc"),
    ]}
)

# ----------------------------
# Best-epoch logger (writes per-class only once, for best epoch)
# ----------------------------
class FamilyBestOnlyLogger(tf.keras.callbacks.Callback):
    def __init__(self, train_ds, val_ds, class_names, out_dir):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.class_names = list(class_names)
        self.out_dir = out_dir

        self.rows = []
        self.best_epoch = None
        self.best_val_macro_f1 = None

        self.best_train_pc = None
        self.best_val_pc = None
        self.best_cm = None

    def _collect_true_pred(self, ds):
        y_true_all = []
        y_pred_all = []
        for x, y in ds:
            probs = self.model(x, training=False)["family_output"]  # (B, C)
            y_pred = tf.argmax(probs, axis=1).numpy().astype(int)
            y_true = y["family_output"].numpy().astype(int)
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
        return np.concatenate(y_true_all), np.concatenate(y_pred_all)

    def _macro_and_perclass(self, y_true, y_pred):
        acc = float(accuracy_score(y_true, y_pred))

        labels = np.arange(len(self.class_names), dtype=int)

        # macro over the fixed label space
        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            y_true, y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        )

        # per-class over the fixed label space (length == n_classes always)
        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred,
            labels=labels,
            average=None,
            zero_division=0,
        )

        per_class_df = pd.DataFrame({
            "class": self.class_names,
            "precision": p.astype(float),
            "recall": r.astype(float),
            "f1": f.astype(float),
            "support": s.astype(int),
        })

        return acc, float(p_macro), float(r_macro), float(f_macro), per_class_df

    def _write_confusion_png(self, cm, epoch):
        cm = cm.astype(int)
        n = len(self.class_names)

        fig_w = max(10, 0.30 * n)
        fig_h = max(8, 0.30 * n)

        fig = plt.figure(figsize=(fig_w, fig_h))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation="nearest")
        fig.colorbar(im, ax=ax)

        ax.set_title(f"Family Confusion Matrix (VAL) - Best Epoch {epoch}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(self.class_names, rotation=90, ha="center", fontsize=6)
        ax.set_yticklabels(self.class_names, fontsize=6)

        if n <= 25:
            for i in range(n):
                for j in range(n):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        fig.tight_layout()
        fig.savefig(CM_PNG, dpi=300)
        plt.close(fig)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ep = int(epoch + 1)

        # Collect preds for TRAIN + VAL
        y_true_tr, y_pred_tr = self._collect_true_pred(self.train_ds)
        tr_acc, tr_p, tr_r, tr_f, tr_pc = self._macro_and_perclass(y_true_tr, y_pred_tr)

        y_true_va, y_pred_va = self._collect_true_pred(self.val_ds)
        va_acc, va_p, va_r, va_f, va_pc = self._macro_and_perclass(y_true_va, y_pred_va)

        row = {
            "epoch": ep,
            "train_loss": float(logs.get("loss", np.nan)),
            "val_loss": float(logs.get("val_loss", np.nan)),
            "train_acc": tr_acc,
            "val_acc": va_acc,
            "train_macro_precision": tr_p,
            "train_macro_recall": tr_r,
            "train_macro_f1": tr_f,
            "val_macro_precision": va_p,
            "val_macro_recall": va_r,
            "val_macro_f1": va_f,
        }
        self.rows.append(row)

        # Best epoch by VAL macro-F1
        if (self.best_val_macro_f1 is None) or (va_f > self.best_val_macro_f1):
            self.best_val_macro_f1 = va_f
            self.best_epoch = ep

            # stash best artifacts in memory
            self.best_train_pc = tr_pc.copy()
            self.best_val_pc = va_pc.copy()
            self.best_cm = confusion_matrix(
                y_true_va, y_pred_va, labels=list(range(len(self.class_names)))
            )

            # save best model snapshot immediately
            self.model.save(BEST_MODEL_PATH)

        # Write epoch CSV with is_best
        hist_df = pd.DataFrame(self.rows)
        hist_df["is_best"] = (hist_df["epoch"] == self.best_epoch).astype(int)
        hist_df.to_csv(HISTORY_CSV, index=False)

    def on_train_end(self, logs=None):
        # Write ONLY best-epoch per-class CSVs
        if self.best_epoch is None:
            return

        train_pc = self.best_train_pc.copy()
        train_pc["epoch"] = int(self.best_epoch)
        train_pc["split"] = "train"
        train_pc.to_csv(BEST_TRAIN_PER_CLASS_CSV, index=False)

        val_pc = self.best_val_pc.copy()
        val_pc["epoch"] = int(self.best_epoch)
        val_pc["split"] = "val"
        val_pc.to_csv(BEST_VAL_PER_CLASS_CSV, index=False)

        # Confusion matrix PNG for best epoch
        self._write_confusion_png(self.best_cm, self.best_epoch)


# ----------------------------
# Callbacks
# ----------------------------
metrics_logger = FamilyBestOnlyLogger(
    train_ds=train_ds,
    val_ds=val_ds,
    class_names=class_names,
    out_dir=OUT_DIR
)

callbacks = [
    metrics_logger,
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=5,
        restore_best_weights=True
    ),
]

# ----------------------------
# Train
# ----------------------------
EPOCHS = 50
family_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# Save final snapshot (end-of-training weights)
family_model.save(FINAL_MODEL_PATH)

print("\nSaved:")
print("-", HISTORY_CSV)
print("-", BEST_MODEL_PATH)
print("-", FINAL_MODEL_PATH)
print("-", CM_PNG)
print("-", BEST_TRAIN_PER_CLASS_CSV)
print("-", BEST_VAL_PER_CLASS_CSV)
