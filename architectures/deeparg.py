import numpy as np
from scipy.sparse import load_npz
import tensorflow as tf

# -------------------------
# Inputs
# -------------------------
PREFIX = "gold_deeparg"
X_TRAIN_PATH = f"{PREFIX}.X_train.npz"
X_VAL_PATH   = f"{PREFIX}.X_val.npz"
Y_TRAIN_PATH = f"{PREFIX}.y_train.npy"
Y_VAL_PATH   = f"{PREFIX}.y_val.npy"

# -------------------------
# Load data
# -------------------------
X_train = load_npz(X_TRAIN_PATH).tocsr()
X_val   = load_npz(X_VAL_PATH).tocsr()
y_train = np.load(Y_TRAIN_PATH).astype(np.float32)
y_val   = np.load(Y_VAL_PATH).astype(np.float32)

n_features = X_train.shape[1]
print("X_train:", X_train.shape, "nnz:", X_train.nnz)
print("X_val  :", X_val.shape,   "nnz:", X_val.nnz)
print("n_features:", n_features)

# -------------------------
# CSR -> tf.sparse.SparseTensor
# -------------------------
def csr_to_sparse_tensor(X_csr):
    X_csr = X_csr.tocsr()
    X_csr.sort_indices()
    coo = X_csr.tocoo()
    indices = np.vstack([coo.row, coo.col]).T.astype(np.int64)
    values  = coo.data.astype(np.float32)
    shape   = np.array(coo.shape, dtype=np.int64)
    return tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=shape)

X_train_sp = csr_to_sparse_tensor(X_train)
X_val_sp   = csr_to_sparse_tensor(X_val)

# -------------------------
# Model (DeepARG-like MLP)
# -------------------------
inputs = tf.keras.Input(shape=(n_features,), sparse=True)
x = tf.keras.layers.Dense(2000, activation="relu")(inputs)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1000, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(500, activation="relu")(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(100, activation="relu")(x)
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.AUC(curve="ROC", name="auroc"),
        tf.keras.metrics.AUC(curve="PR",  name="auprc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

# -------------------------
# Train
# -------------------------
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_auprc", mode="max", patience=5, restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "gold_deeparg_mlp.keras", monitor="val_auprc", mode="max", save_best_only=True
    ),
]

history = model.fit(
    X_train_sp, y_train,
    validation_data=(X_val_sp, y_val),
    epochs=50,
    batch_size=256,
    callbacks=callbacks,
    verbose=2,
)

# -------------------------
# Save raw validation probabilities for thresholding
# -------------------------
val_prob = model.predict(X_val_sp, batch_size=256).reshape(-1).astype(np.float32)
np.save("gold_deeparg_val_prob.npy", val_prob)
print("Saved: gold_deeparg_mlp.keras and gold_deeparg_val_prob.npy")
