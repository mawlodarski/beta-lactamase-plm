import os
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"
os.environ["OPENBLAS_NUM_THREADS"] = "32"
os.environ["NUMEXPR_NUM_THREADS"] = "32"


import csv
import json
import numpy as np
import pandas as pd
from fastembed_bio import ProteinEmbedding
from tqdm import tqdm

IN_CSV = "clean_bacterial.csv"
OUT_CSV = "clean_bacterial_with_esm2.csv"
BATCH_SIZE = 128

df = pd.read_csv(IN_CSV, low_memory=False)
seqs = df["Sequence"].astype(str).tolist()

model = ProteinEmbedding("facebook/esm2_t12_35M_UR50D")

with open(OUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)

    # header: all original cols + embedding col
    out_cols = list(df.columns) + ["ESM-2_embedding"]
    writer.writerow(out_cols)

    for start in tqdm(range(0, len(seqs), BATCH_SIZE), desc="Embedding+write"):
        batch = seqs[start:start + BATCH_SIZE]
        rows = df.iloc[start:start + len(batch)]

        emb_iter = model.embed(batch)  # yields 1 embedding per sequence in same order

        for row_vals, emb in zip(rows.itertuples(index=False, name=None), emb_iter):
            emb = np.asarray(emb, dtype=np.float32)

            # store full embedding as a JSON list in ONE cell
            emb_json = json.dumps(emb.tolist())

            writer.writerow(list(row_vals) + [emb_json])

print("Done:", OUT_CSV)
