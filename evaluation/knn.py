from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas as pd
import ast
import tensorflow as tf

from tokenization import encode_and_pad, MAX_LEN
from model import TransformerAmblerModel

# ============================================================
# INPUTS 
# ============================================================
CSV_BINS = "bins.csv" # evaluation data         
MODEL_KERAS = discovery_model #input model  

KNN_K = 10
KNN_METRIC = "cosine"


# ============================================================
# Load data
# ============================================================
df = pd.read_csv(CSV_BINS)

df["Ambler Class"] = (
    df["Ambler Class"]
    .astype(str)
    .str.replace(r"^B[123]$", "B", regex=True)
)

# ------------------------------------------------------------
# ESM-2 embeddings
# ------------------------------------------------------------
df_esm = df.dropna(subset=["ESM-2_embedding", "Ambler Class"]).copy()
labels_esm = df_esm["Ambler Class"].values

X_esm = np.vstack(
    df_esm["ESM-2_embedding"].apply(
        lambda s: np.array(ast.literal_eval(s), dtype=np.float32)
    ).values
)

# L2 normalize
norms = np.linalg.norm(X_esm, axis=1, keepdims=True)
norms[norms == 0] = 1.0
X_esm = X_esm / norms


# ------------------------------------------------------------
# Discovery CLS embeddings
# ------------------------------------------------------------
df_disc = df.dropna(subset=["Sequence", "Ambler Class"]).copy()
labels_disc = df_disc["Ambler Class"].values

tokens = encode_and_pad(
    df_disc["Sequence"].astype(str).tolist(),
    max_len=MAX_LEN
).astype(np.int32)

model = tf.keras.models.load_model(
    MODEL_KERAS,
    compile=False,
    custom_objects={"TransformerAmblerModel": TransformerAmblerModel},
)

emb = model.get_sequence_embedding(tokens, batch_size=256).astype(np.float32)

# L2 normalize
norms = np.linalg.norm(emb, axis=1, keepdims=True)
norms[norms == 0] = 1.0
emb = emb / norms


# ============================================================
# kNN functions
# ============================================================
def _knn_majority_vote(neigh_labels_2d):
    preds = []
    for row in neigh_labels_2d:
        vals, counts = np.unique(row, return_counts=True)
        m = counts.max()
        tied = vals[counts == m]
        preds.append(np.sort(tied)[0])
    return np.asarray(preds, dtype=str)


def knn_eval_embeddings(X, y, k=10, metric="cosine"):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y).astype(str)

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric)
    nn.fit(X)

    idx = nn.kneighbors(X, return_distance=False)[:, 1:]  # drop self
    neigh_labels = y[idx]
    y_pred = _knn_majority_vote(neigh_labels)

    acc = float(accuracy_score(y, y_pred))
    macro_f1 = float(f1_score(y, y_pred, average="macro"))

    return acc, macro_f1


# ============================================================
# Run kNN
# ============================================================
print("\n===== kNN neighborhood quantification (original embedding space) =====")
print(f"Settings: k={KNN_K}, metric={KNN_METRIC}\n")

disc_acc, disc_f1 = knn_eval_embeddings(emb, labels_disc, k=KNN_K, metric=KNN_METRIC)
print(f"Discovery CLS: acc={disc_acc:.4f}  macro-F1={disc_f1:.4f}")

esm_acc, esm_f1 = knn_eval_embeddings(X_esm, labels_esm, k=KNN_K, metric=KNN_METRIC)
print(f"ESM-2       : acc={esm_acc:.4f}  macro-F1={esm_f1:.4f}")

print("\nInterpretation:")
print("  Higher macro-F1 = cleaner class-consistent neighborhoods.")
