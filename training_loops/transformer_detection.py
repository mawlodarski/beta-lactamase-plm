# bl_detector.py
# train TransformerBLDetector on gold_training_data.csv

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

from tokenization import MAX_LEN, vocab_size, encode_and_pad
from model import TransformerBLDetector, PositionalAdder, AttnMaskExpand, Encoder

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

DATASET = "gold" # gold, discovery
OUT_NAME = "dapt_gold" # gold, discovery, dapt_discovery
SPLIT = "random"
DAPT = True

CSV_PATH = f"/home/wlodarsm/projects/def-mcarthur/wlodarsm/eccb/{DATASET}/{DATASET}_training_data.csv"
OUT_DIR = f"/home/wlodarsm/projects/def-mcarthur/wlodarsm/eccb/models/{SPLIT}_split/bl_detector/{OUT_NAME}"
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Required columns
if "Sequence" not in df.columns:
    raise ValueError("Missing required column: Sequence")
if "is_bla" not in df.columns:
    raise ValueError("Missing required column: is_bla")

# Basic cleanup
df = df.dropna(subset=["Sequence", "is_bla"]).copy()
df["is_bla"] = df["is_bla"].astype(int)

MLM_WEIGHTS = "/home/wlodarsm/projects/def-mcarthur/wlodarsm/eccb/models/dapt_mlm/best_mlm.weights.h5"

### RANDOM SPLIT
# Stratified split on is_bla (keeps class balance stable)
if SPLIT == "random":
    train_df, val_df = train_test_split(
        df,
        test_size=0.10,
        random_state=SEED,
        stratify=df["is_bla"]
    )
### CLUSTER SPLIT (stratify by cluster_id, but handle singleton clusters safely)
elif SPLIT == "cluster":
    if "cluster_id" not in df.columns:
        raise ValueError("Missing required column: cluster_id")

    # ensure stable dtype for stratification
    df["cluster_id"] = df["cluster_id"].astype(str)

    # Count cluster sizes
    cluster_counts = df["cluster_id"].value_counts()

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

    # 2) Singletons: cannot stratify; split randomly (or keep all in train)
    # Choose ONE behavior:
    SINGLETON_POLICY = "random"   # "random" or "train" (train means no singletons in validation)

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
            raise ValueError('SINGLETON_POLICY must be "random" or "train"')

        train_df = pd.concat([train_valid, train_single], ignore_index=True)
        val_df = pd.concat([val_valid, val_single], ignore_index=True)

    # Final shuffle for cleanliness
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Sanity prints
    print("=== Cluster split summary ===")
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
# --- TRUE DISJOINT CLUSTER SPLIT (clusters are groups) ---
#random 90/10 for negatives
# --- HYBRID SPLIT ---
# Goal:
#   - is_bla==1 (beta-lactamases): DISJOINT by cluster_id (no cluster overlap train vs val)
#   - is_bla==0 (negatives: bacterial protein + other ARG + anything else): RANDOM split
elif SPLIT ==  "disjoint":
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    if "cluster_id" not in df.columns:
        raise ValueError("Missing required column: cluster_id")

    df = df.copy()
    df["cluster_id"] = df["cluster_id"].astype(str).str.strip()
    df["is_bla"] = df["is_bla"].astype(int)

    # Split positives (BL) by group-disjoint clusters
    df_pos = df[df["is_bla"] == 1].copy()
    df_neg = df[df["is_bla"] == 0].copy()

    if len(df_pos) == 0:
        raise ValueError("No is_bla==1 samples found for disjoint cluster split.")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=SEED)
    pos_train_idx, pos_val_idx = next(gss.split(df_pos, groups=df_pos["cluster_id"]))

    train_pos = df_pos.iloc[pos_train_idx].copy()
    val_pos   = df_pos.iloc[pos_val_idx].copy()

    # Split negatives randomly (no cluster constraint)
    # If you want exact 90/10 negative split:
    if len(df_neg) == 0:
        train_neg = df_neg
        val_neg = df_neg
    else:
        train_neg, val_neg = train_test_split(
            df_neg,
            test_size=0.10,
            random_state=SEED,
            shuffle=True,
            # optional: stratify=df_neg["Label"]  # only if you want label-balance inside negatives
        )

    # Combine
    train_df = pd.concat([train_pos, train_neg], ignore_index=True)
    val_df   = pd.concat([val_pos,   val_neg], ignore_index=True)

    # Shuffle for cleanliness
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    val_df   = val_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # --- Sanity checks ---
    print("=== Hybrid split summary ===")
    print("Total rows:", len(df))
    print("Train rows:", len(train_df))
    print("Val rows:  ", len(val_df))

    print("\nBL prevalence:")
    print("Train is_bla mean:", train_df["is_bla"].mean())
    print("Val   is_bla mean:", val_df["is_bla"].mean())

    # Confirm DISJOINT clusters for positives only
    train_pos_clusters = set(train_df.loc[train_df["is_bla"] == 1, "cluster_id"])
    val_pos_clusters   = set(val_df.loc[val_df["is_bla"] == 1, "cluster_id"])
    overlap = train_pos_clusters & val_pos_clusters
    print("\nPositive (is_bla==1) clusters:")
    print("Train pos clusters:", len(train_pos_clusters))
    print("Val   pos clusters:", len(val_pos_clusters))
    print("Overlapping pos clusters (should be 0):", len(overlap))
    if len(overlap) > 0:
        print("Example overlapping clusters:", list(sorted(overlap))[:10])

#save dfs
train_df.to_csv(f"{OUT_DIR}/train_df.csv", index=False)
val_df.to_csv(f"{OUT_DIR}/val_df.csv", index=False)

###
# Encode
X_train = encode_and_pad(train_df["Sequence"].astype(str).values, max_len=MAX_LEN)
X_val = encode_and_pad(val_df["Sequence"].astype(str).values, max_len=MAX_LEN)

y_train = train_df["is_bla"].astype(np.float32).values
y_val = val_df["is_bla"].astype(np.float32).values

y_train_dict = {"bla_output": y_train}
y_val_dict = {"bla_output": y_val}

# tf.data
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

# Model
bl_model = TransformerBLDetector(
    vocab_size=vocab_size,
    emb_dim=128,
    num_heads=4,
    ff_dim=512,
    max_len=MAX_LEN,
    num_layers=4,
    dropout=0.1
)

# Force build for correct summary
_ = bl_model(tf.zeros((1, MAX_LEN), dtype=tf.int32), training=False)
bl_model.summary()

'''
if DAPT:
    print("Loading DAPT weights into BL detector (partial, skip mismatches)...")
    bl_model.load_weights(MLM_WEIGHTS, skip_mismatch=True)
    print("DAPT weight load complete.")

# Compile
bl_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss={"bla_output": tf.keras.losses.BinaryCrossentropy()},
    metrics={"bla_output": [
        tf.keras.metrics.BinaryAccuracy(name="acc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auroc", curve="ROC"),
        tf.keras.metrics.AUC(name="auprc", curve="PR"),
    ]}
)

# ---- epoch-by-epoch logging callback (robust is_best) ----
class MetricsLogger(tf.keras.callbacks.Callback):
    def __init__(self, out_csv: str, out_json: str | None = None, monitor: str = "val_bla_output_auprc"):
        super().__init__()
        self.out_csv = out_csv
        self.out_json = out_json
        self.monitor = monitor
        self.rows = []
        self.best_epoch = None
        self.best_value = None
        self._resolved_monitor = None

    def _resolve_monitor(self, logs: dict):
        # Prefer explicit monitor if present
        if self.monitor in logs:
            return self.monitor

        # Otherwise auto-detect a validation AUPRC key
        candidates = [k for k in logs.keys() if k.startswith("val_") and "auprc" in k]
        if candidates:
            # deterministic choice
            return sorted(candidates)[0]

        return None

    def _write(self):
        hist_df = pd.DataFrame(self.rows)
        hist_df["is_best"] = 0
        if self.best_epoch is not None and 1 <= self.best_epoch <= len(hist_df):
            hist_df.loc[hist_df["epoch"] == self.best_epoch, "is_best"] = 1

        hist_df.to_csv(self.out_csv, index=False)
        if self.out_json is not None:
            hist_df.to_json(self.out_json, orient="records", indent=2)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # Resolve monitor key on first epoch where logs are available
        if self._resolved_monitor is None:
            self._resolved_monitor = self._resolve_monitor(logs)
            print("MetricsLogger resolved monitor key:", self._resolved_monitor)
            print("Available log keys:", sorted(list(logs.keys())))

        row = {"epoch": int(epoch + 1)}
        for k, v in logs.items():
            try:
                row[k] = float(v)
            except Exception:
                row[k] = v
        self.rows.append(row)

        # Update best tracking
        if self._resolved_monitor is not None and self._resolved_monitor in logs:
            current = float(logs[self._resolved_monitor])
            if (self.best_value is None) or (current > self.best_value):
                self.best_value = current
                self.best_epoch = int(epoch + 1)

        self._write()

history_csv = os.path.join(OUT_DIR, "bl_detector_epoch_metrics.csv")

metrics_logger = MetricsLogger(out_csv=history_csv)

# callbacks
callbacks = [
    metrics_logger,
    tf.keras.callbacks.EarlyStopping(
        monitor="val_bla_output_auprc",  # AUPRC is better for imbalance
        mode="max",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(OUT_DIR, "best_bl_detector.keras"),
        monitor="val_bla_output_auprc",
        mode="max",
        save_best_only=True
    )
]

# Train
EPOCHS = 30
bl_model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=callbacks)

# Final eval (best weights if EarlyStopping ran)
metrics = bl_model.evaluate(val_ds, return_dict=True)
print("\nValidation metrics:")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

print(f"\nSaved per-epoch metrics to:\n- {history_csv}")

# Save trained model
bl_model.save(f"{OUT_DIR}_bl_detector_128d_512ff_4encs_4heads_256batch.keras")
'''