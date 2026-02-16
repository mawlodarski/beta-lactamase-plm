# ambler_train.py
# Train TransformerAmblerModel on BL-only sequences (is_bla==1)
# Supports:
#   - AMBLER_MODE: "full" (A,B1,B2,B3,C,D) or "collapsed" (A,B,C,D; B1/B2/B3 -> B)
#   - SPLIT: "random", "cluster", or "disjoint"
#   - DAPT: optionally load MLM weights (skip_mismatch=True) before training
#
# Logs per-epoch TRAIN + VAL:
#   - loss
#   - accuracy
#   - macro precision/recall/F1
# Writes:
#   - ambler_epoch_metrics.csv (with is_best)
#   - per-class CSV per epoch (train + val)
#   - best_ambler.keras (best by val_macro_f1)
#   - ambler_final.keras (end-of-training snapshot)
#   - ambler_confusion_matrix.png (VAL confusion matrix for best epoch)
#   - train_df.csv / val_df.csv (the split used)

import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    accuracy_score,
)

from tokenization import MAX_LEN, vocab_size, encode_and_pad
from model import TransformerAmblerModel


# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# User config
# ----------------------------
DATASET = "gold"            # "gold" or "discovery"
OUT_NAME = "gold"      # output folder name, e.g. "gold", "dapt_gold", "discovery", "dapt_discovery"
SPLIT = "cluster"            # "random", "cluster", "disjoint"
AMBLER_MODE = "collapsed"   # "full" or "collapsed"
DAPT = False

BASE_DIR = "/home/wlodarsm/projects/def-mcarthur/wlodarsm/eccb"
CSV_PATH = f"{BASE_DIR}/{DATASET}/{DATASET}_training_data.csv"

MLM_WEIGHTS = f"{BASE_DIR}/models/dapt_mlm/best_mlm.weights.h5"

# Split settings
TEST_SIZE = 0.10
BATCH_SIZE = 256
EPOCHS = 30
LR = 1e-4

# For cluster split (non-disjoint) singleton behavior
SINGLETON_POLICY = "train"  # "train" (recommended) or "random"

# ----------------------------
# Output paths
# ----------------------------
if AMBLER_MODE == "full":
    OUT_DIR = f"{BASE_DIR}/models/{SPLIT}_split/ambler/{OUT_NAME}"
else:
    OUT_DIR = f"{BASE_DIR}/models/{SPLIT}_split/ambler_collapsed/{OUT_NAME}"

os.makedirs(OUT_DIR, exist_ok=True)

HISTORY_CSV = os.path.join(OUT_DIR, "ambler_epoch_metrics.csv")
ENCODER_PKL = os.path.join(OUT_DIR, "ambler.pkl")
ENCODER_JSON = os.path.join(OUT_DIR, "ambler.json")
BEST_MODEL_PATH = os.path.join(OUT_DIR, "best_ambler.keras")
FINAL_MODEL_PATH = os.path.join(OUT_DIR, "ambler_final.keras")
CM_PNG = os.path.join(OUT_DIR, "ambler_confusion_matrix.png")
TRAIN_DF_CSV = os.path.join(OUT_DIR, "train_df.csv")
VAL_DF_CSV = os.path.join(OUT_DIR, "val_df.csv")


# ----------------------------
# Load + filter data
# ----------------------------
df = pd.read_csv(CSV_PATH)

required = ["Sequence", "is_bla", "Ambler Class"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required).copy()
df["Sequence"] = df["Sequence"].astype(str)
df["is_bla"] = df["is_bla"].astype(int)
df["Ambler Class"] = df["Ambler Class"].astype(str).str.strip().str.upper()

# Train Ambler on BL-only
df = df[df["is_bla"] == 1].copy()
if len(df) == 0:
    raise ValueError("No BL sequences found (is_bla == 1). Ambler training requires BL-only rows.")

# Apply collapse mapping before label filtering if requested
if AMBLER_MODE == "collapsed":
    df["Ambler Class"] = df["Ambler Class"].replace({"B1": "B", "B2": "B", "B3": "B"})

print("Ambler training samples (is_bla==1):", len(df))
print("Ambler class distribution (raw):")
print(df["Ambler Class"].value_counts())

# ----------------------------
# Fixed label space (stable indices)
# ----------------------------
if AMBLER_MODE == "full":
    AMBLER_CLASSES = ["A", "B1", "B2", "B3", "C", "D"]
else:
    AMBLER_CLASSES = ["A", "B", "C", "D"]

# Keep only expected labels
df = df[df["Ambler Class"].isin(AMBLER_CLASSES)].copy()
if len(df) == 0:
    raise ValueError("After filtering to expected Ambler classes, no samples remain.")

print("\nAfter filtering to expected Ambler classes:")
print(df["Ambler Class"].value_counts())

le_ambler = LabelEncoder()
le_ambler.classes_ = np.array(AMBLER_CLASSES, dtype=object)

# Save encoder for consistent eval mapping
joblib.dump(le_ambler, ENCODER_PKL)
with open(ENCODER_JSON, "w") as f:
    json.dump(list(map(str, le_ambler.classes_)), f, indent=2)

class_names = list(map(str, le_ambler.classes_))
n_classes = len(class_names)

print("\nSaved Ambler LabelEncoder:")
print("-", ENCODER_PKL)
print("-", ENCODER_JSON)
print("Class space:", class_names)

y_all = le_ambler.transform(df["Ambler Class"].values).astype(np.int32)


# ----------------------------
# Train/val split
# ----------------------------
if SPLIT == "random":
    train_df, val_df, y_train, y_val = train_test_split(
        df,
        y_all,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y_all,
        shuffle=True,
    )

elif SPLIT == "cluster":
    if "cluster_id" not in df.columns:
        raise ValueError("Missing required column for cluster split: cluster_id")

    tmp = df.copy()
    tmp["cluster_id"] = tmp["cluster_id"].astype(str).str.strip()

    cluster_counts = tmp["cluster_id"].value_counts()

    valid_clusters = cluster_counts[cluster_counts >= 2].index
    singleton_clusters = cluster_counts[cluster_counts == 1].index

    df_valid = tmp[tmp["cluster_id"].isin(valid_clusters)].copy()
    df_single = tmp[tmp["cluster_id"].isin(singleton_clusters)].copy()

    train_valid, val_valid = train_test_split(
        df_valid,
        test_size=TEST_SIZE,
        random_state=SEED,
        shuffle=True,
        stratify=df_valid["cluster_id"],
    )

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
                    test_size=TEST_SIZE,
                    random_state=SEED,
                    shuffle=True,
                )
        else:
            raise ValueError('SINGLETON_POLICY must be "train" or "random"')

        train_df = pd.concat([train_valid, train_single], ignore_index=True)
        val_df = pd.concat([val_valid, val_single], ignore_index=True)

    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    y_train = le_ambler.transform(train_df["Ambler Class"].values).astype(np.int32)
    y_val = le_ambler.transform(val_df["Ambler Class"].values).astype(np.int32)

    print("\nCluster split summary (Ambler):")
    print("Total rows:", len(tmp))
    print("Train rows:", len(train_df))
    print("Val rows:  ", len(val_df))
    print("Total clusters:", tmp["cluster_id"].nunique())
    print("Singleton clusters (size=1):", int((cluster_counts == 1).sum()))

    train_clusters = set(train_df["cluster_id"].astype(str))
    val_clusters = set(val_df["cluster_id"].astype(str))
    print("Clusters in train:", len(train_clusters))
    print("Clusters in val:  ", len(val_clusters))
    print("Overlapping clusters (expected in this mode):", len(train_clusters & val_clusters))

elif SPLIT == "disjoint":
    if "cluster_id" not in df.columns:
        raise ValueError("Missing required column for disjoint split: cluster_id")

    tmp = df.copy()
    tmp["cluster_id"] = tmp["cluster_id"].astype(str).str.strip()

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    train_idx, val_idx = next(gss.split(tmp, groups=tmp["cluster_id"]))

    train_df = tmp.iloc[train_idx].copy().reset_index(drop=True)
    val_df = tmp.iloc[val_idx].copy().reset_index(drop=True)

    y_train = le_ambler.transform(train_df["Ambler Class"].values).astype(np.int32)
    y_val = le_ambler.transform(val_df["Ambler Class"].values).astype(np.int32)

    train_clusters = set(train_df["cluster_id"].astype(str))
    val_clusters = set(val_df["cluster_id"].astype(str))
    overlap = train_clusters & val_clusters

    print("\nDisjoint split summary (Ambler):")
    print("Train rows:", len(train_df))
    print("Val rows:  ", len(val_df))
    print("Train clusters:", len(train_clusters))
    print("Val clusters:  ", len(val_clusters))
    print("Overlapping clusters (should be 0):", len(overlap))

else:
    raise ValueError('SPLIT must be "random", "cluster", or "disjoint"')

print("\nTrain n:", len(train_df), "Val n:", len(val_df))
print("Train class counts:\n", pd.Series(y_train).value_counts().sort_index())
print("Val class counts:\n", pd.Series(y_val).value_counts().sort_index())

# Save split tables
train_df.to_csv(TRAIN_DF_CSV, index=False)
val_df.to_csv(VAL_DF_CSV, index=False)

# ----------------------------
# Encode sequences
# ----------------------------
X_train = encode_and_pad(train_df["Sequence"].values, max_len=MAX_LEN)
X_val = encode_and_pad(val_df["Sequence"].values, max_len=MAX_LEN)

y_train_dict = {"ambler_output": y_train}
y_val_dict = {"ambler_output": y_val}

# ----------------------------
# tf.data
# ----------------------------
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
ambler_model = TransformerAmblerModel(
    vocab_size=vocab_size,
    n_ambler_classes=n_classes,
    emb_dim=128,
    num_heads=4,
    ff_dim=512,
    max_len=MAX_LEN,
    num_layers=4,
    dropout=0.1
)

_ = ambler_model(tf.zeros((1, MAX_LEN), dtype=tf.int32), training=False)
ambler_model.summary()

if DAPT:
    if not os.path.exists(MLM_WEIGHTS):
        raise FileNotFoundError(f"DAPT weights not found: {MLM_WEIGHTS}")
    print("\nLoading DAPT weights into Ambler model (partial; skip mismatches)...")
    ambler_model.load_weights(MLM_WEIGHTS, skip_mismatch=True)
    print("DAPT weight load complete.")

# Compile
ambler_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss={"ambler_output": tf.keras.losses.SparseCategoricalCrossentropy()},
    metrics={"ambler_output": [tf.keras.metrics.SparseCategoricalAccuracy(name="acc")]}
)

# ----------------------------
# Macro + per-class + confusion-matrix logger
# ----------------------------
class AmblerMacroLogger(tf.keras.callbacks.Callback):
    def __init__(self, train_ds, val_ds, class_names, out_dir, best_model_path, cm_png):
        super().__init__()
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.class_names = list(class_names)
        self.out_dir = out_dir
        self.best_model_path = best_model_path
        self.cm_png = cm_png

        self.rows = []
        self.best_epoch = None
        self.best_val_macro_f1 = None
        self.best_cm = None

        self.epoch_csv = os.path.join(out_dir, "ambler_epoch_metrics.csv")

    def _collect_true_pred(self, ds):
        y_true_all = []
        y_pred_all = []
        for x, y in ds:
            probs = self.model(x, training=False)["ambler_output"]
            y_pred = tf.argmax(probs, axis=1).numpy().astype(int)
            y_true = y["ambler_output"].numpy().astype(int)
            y_true_all.append(y_true)
            y_pred_all.append(y_pred)
        return np.concatenate(y_true_all), np.concatenate(y_pred_all)

    def _macro_and_perclass(self, y_true, y_pred):
        acc = float(accuracy_score(y_true, y_pred))

        p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        p, r, f, s = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        per_class_df = pd.DataFrame({
            "class": self.class_names,
            "precision": p,
            "recall": r,
            "f1": f,
            "support": s.astype(int),
        })

        return acc, float(p_macro), float(r_macro), float(f_macro), per_class_df

    def _save_perclass(self, df_pc, epoch, split_name):
        out = df_pc.copy()
        out["epoch"] = int(epoch)
        out["split"] = split_name
        out_path = os.path.join(self.out_dir, f"ambler_{split_name}_per_class_epoch_{epoch}.csv")
        out.to_csv(out_path, index=False)

    def _write_confusion_png(self):
        if self.best_cm is None:
            return

        cm = self.best_cm.astype(int)
        n = len(self.class_names)

        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, interpolation="nearest")
        fig.colorbar(im, ax=ax)

        ax.set_title(f"Ambler Confusion Matrix (VAL) - Best Epoch {self.best_epoch}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(self.class_names, rotation=45, ha="right")
        ax.set_yticklabels(self.class_names)

        for i in range(n):
            for j in range(n):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        fig.tight_layout()
        fig.savefig(self.cm_png, dpi=300)
        plt.close(fig)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        ep = int(epoch + 1)

        y_true_tr, y_pred_tr = self._collect_true_pred(self.train_ds)
        tr_acc, tr_p, tr_r, tr_f, tr_pc = self._macro_and_perclass(y_true_tr, y_pred_tr)
        self._save_perclass(tr_pc, ep, "train")

        y_true_va, y_pred_va = self._collect_true_pred(self.val_ds)
        va_acc, va_p, va_r, va_f, va_pc = self._macro_and_perclass(y_true_va, y_pred_va)
        self._save_perclass(va_pc, ep, "val")

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

        if (self.best_val_macro_f1 is None) or (va_f > self.best_val_macro_f1):
            self.best_val_macro_f1 = va_f
            self.best_epoch = ep
            self.best_cm = confusion_matrix(
                y_true_va, y_pred_va, labels=list(range(len(self.class_names)))
            )
            self.model.save(self.best_model_path)

        hist_df = pd.DataFrame(self.rows)
        hist_df["is_best"] = (hist_df["epoch"] == self.best_epoch).astype(int)
        hist_df.to_csv(self.epoch_csv, index=False)

    def on_train_end(self, logs=None):
        self._write_confusion_png()


metrics_logger = AmblerMacroLogger(
    train_ds=train_ds,
    val_ds=val_ds,
    class_names=class_names,
    out_dir=OUT_DIR,
    best_model_path=BEST_MODEL_PATH,
    cm_png=CM_PNG
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
ambler_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Save final snapshot
ambler_model.save(FINAL_MODEL_PATH)

# ----------------------------
# Print outputs line-by-line
# ----------------------------
print("\nSaved outputs:")
print("train_df:", TRAIN_DF_CSV)
print("val_df:", VAL_DF_CSV)
print("epoch_metrics:", HISTORY_CSV)
print("encoder_pkl:", ENCODER_PKL)
print("encoder_json:", ENCODER_JSON)
print("best_model:", BEST_MODEL_PATH)
print("final_model:", FINAL_MODEL_PATH)
print("confusion_matrix_png:", CM_PNG)
print("per_class_csvs: ambler_train_per_class_epoch_*.csv, ambler_val_per_class_epoch_*.csv")
