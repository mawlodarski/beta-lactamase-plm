import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, save_npz
#train loop is in model architecture script

REF_IDS = "ref_ids.txt"
SPLIT_CSV = "gold_split.csv"

TRAIN_M8 = "train_gold_vs_ref.m8"
VAL_M8   = "val_gold_vs_ref.m8"

OUT_PREFIX = "gold_deeparg"

# --------------------------
# Load reference ID -> column
# --------------------------
ref = pd.read_csv(REF_IDS, header=None, names=["sseqid"], dtype=str)
ref["col"] = np.arange(len(ref), dtype=np.int32)
ref_map = dict(zip(ref["sseqid"], ref["col"]))
M = len(ref)

# --------------------------
# Load labels + row order
# --------------------------
df = pd.read_csv(SPLIT_CSV, usecols=["seq_id", "is_bla", "split"], dtype={"seq_id": str})
df["is_bla"] = df["is_bla"].astype(int)

train_ids = df.loc[df["split"]=="train", "seq_id"].tolist()
val_ids   = df.loc[df["split"]=="val",   "seq_id"].tolist()

y_train = df.loc[df["split"]=="train", "is_bla"].to_numpy(dtype=np.int8)
y_val   = df.loc[df["split"]=="val",   "is_bla"].to_numpy(dtype=np.int8)

train_row = {sid: i for i, sid in enumerate(train_ids)}
val_row   = {sid: i for i, sid in enumerate(val_ids)}

# --------------------------
# Helper: build X from m8
# --------------------------
def build_X(m8_path, row_map, n_rows, M):
    # m8 columns: qseqid sseqid bitscore evalue pident length
    m8 = pd.read_csv(
        m8_path, sep="\t", header=None,
        names=["qseqid","sseqid","bitscore","evalue","pident","length"],
        usecols=[0,1,2],
        dtype={"qseqid": str, "sseqid": str, "bitscore": np.float32}
    )

    # Keep only queries that are in this split (safety)
    m8 = m8[m8["qseqid"].isin(row_map)]

    # Map to row/col
    m8["row"] = m8["qseqid"].map(row_map)
    m8["col"] = m8["sseqid"].map(ref_map)

    # Drop anything that didn't map (safety)
    m8 = m8.dropna(subset=["row","col"])
    m8["row"] = m8["row"].astype(np.int32)
    m8["col"] = m8["col"].astype(np.int32)

    # Max bitscore per (row, col)
    grp = m8.groupby(["row","col"], sort=False)["bitscore"].max().reset_index()

    rows = grp["row"].to_numpy(np.int32)
    cols = grp["col"].to_numpy(np.int32)
    data = grp["bitscore"].to_numpy(np.float32)

    X = coo_matrix((data, (rows, cols)), shape=(n_rows, M)).tocsr()

    # Normalize each row by its max (avoid densifying)
    row_max = X.max(axis=1).toarray().ravel().astype(np.float32)
    nz = row_max > 0
    inv = np.zeros_like(row_max)
    inv[nz] = 1.0 / row_max[nz]

    # left-multiply by diagonal(inv)
    X = X.multiply(inv[:, None]).tocsr()
    return X

print("Building X_train...")
X_train = build_X(TRAIN_M8, train_row, len(train_ids), M)
print("Building X_val...")
X_val = build_X(VAL_M8, val_row, len(val_ids), M)

save_npz(f"{OUT_PREFIX}.X_train.npz", X_train)
save_npz(f"{OUT_PREFIX}.X_val.npz", X_val)
np.save(f"{OUT_PREFIX}.y_train.npy", y_train)
np.save(f"{OUT_PREFIX}.y_val.npy", y_val)

print("Done.")
print("X_train:", X_train.shape, "nnz:", X_train.nnz)
print("X_val  :", X_val.shape,   "nnz:", X_val.nnz)
