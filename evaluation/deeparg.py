#!/usr/bin/env python3
"""

Load a trained DeepARG-like BL detector (Keras), evaluate on an input CSV that
contains protein sequences and ground-truth labels, and output a single-row CSV
with these columns:

  model, bin, recall, tp, tn, fp, fn, total

Required input CSV columns:
  - Sequence
  - is_bla

Required files/resources:
  - Keras model (.keras)
  - DIAMOND DB basename (without .dmnd), e.g. gold_ref_panel
  - ref_ids.txt (one ref ID per line, defines feature column order)
"""

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
import tensorflow as tf


def write_fasta(ids, seqs, out_faa: Path):
    with open(out_faa, "w") as f:
        for sid, seq in zip(ids, seqs):
            sid = str(sid)
            seq = str(seq).replace(" ", "").replace("\n", "").upper().strip()
            if not seq:
                continue
            f.write(f">{sid}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")


def run_diamond_blastp(ref_db: str, query_faa: Path, out_m8: Path, threads: int, evalue: float, max_target_seqs: int):
    cmd = [
        "diamond", "blastp",
        "--db", ref_db,
        "--query", str(query_faa),
        "--out", str(out_m8),
        "--outfmt", "6", "qseqid", "sseqid", "bitscore", "evalue", "pident", "length",
        "--max-target-seqs", str(max_target_seqs),
        "--evalue", str(evalue),
        "--threads", str(threads),
    ]
    subprocess.run(cmd, check=True)


def load_ref_map(ref_ids_path: Path):
    ref = pd.read_csv(ref_ids_path, header=None, names=["sseqid"], dtype=str)
    ref["col"] = np.arange(len(ref), dtype=np.int32)
    ref_map = dict(zip(ref["sseqid"], ref["col"]))
    return ref, ref_map


def build_feature_matrix(m8_path: Path, query_ids, ref_map, n_rows: int, n_cols: int):
    row_map = {str(sid): i for i, sid in enumerate(query_ids)}

    m8 = pd.read_csv(
        m8_path, sep="\t", header=None,
        names=["qseqid", "sseqid", "bitscore", "evalue", "pident", "length"],
        usecols=[0, 1, 2],
        dtype={"qseqid": str, "sseqid": str, "bitscore": np.float32},
    )

    m8["row"] = m8["qseqid"].map(row_map)
    m8["col"] = m8["sseqid"].map(ref_map)
    m8 = m8.dropna(subset=["row", "col"])

    if len(m8) == 0:
        X = csr_matrix((n_rows, n_cols), dtype=np.float32)
        return X

    m8["row"] = m8["row"].astype(np.int32)
    m8["col"] = m8["col"].astype(np.int32)

    grp = m8.groupby(["row", "col"], sort=False)["bitscore"].max().reset_index()

    rows = grp["row"].to_numpy(np.int32)
    cols = grp["col"].to_numpy(np.int32)
    data = grp["bitscore"].to_numpy(np.float32)

    X = coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols)).tocsr()

    row_max = X.max(axis=1).toarray().ravel().astype(np.float32)
    nz = row_max > 0
    inv = np.zeros_like(row_max, dtype=np.float32)
    inv[nz] = 1.0 / row_max[nz]
    X = X.multiply(inv[:, None]).tocsr()

    return X


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.astype(np.int8)
    y_pred = y_pred.astype(np.int8)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    total = tp + tn + fp + fn
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    return tp, tn, fp, fn, total, recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="Path to Keras model (.keras)")
    ap.add_argument("--ref_db", required=True, help="DIAMOND DB basename (no .dmnd), e.g., gold_ref_panel")
    ap.add_argument("--ref_ids", required=True, help="ref_ids.txt (one ref ID per line, defines feature columns)")
    ap.add_argument("--input_csv", required=True, help="Input CSV with Sequence and is_bla")
    ap.add_argument("--out_csv", required=True, help="Output metrics CSV (single row)")
    ap.add_argument("--model_name", default="DeepARG-like (custom)", help="Value for 'model' column")
    ap.add_argument("--threshold", type=float, default=0.5, help="Probability threshold (default 0.5)")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--evalue", type=float, default=1e-3)
    ap.add_argument("--max_target_seqs", type=int, default=10000)
    ap.add_argument("--tmp_dir", default="tmp_eval", help="Temp dir for FASTA + m8")
    ap.add_argument("--pred_out", default=None, help="Optional per-sequence predictions CSV")

    args = ap.parse_args()

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if "Sequence" not in df.columns:
        raise ValueError("Input CSV must contain column 'Sequence'.")
    if "is_bla" not in df.columns:
        raise ValueError("Input CSV must contain column 'is_bla' (0/1).")

    if "seq_id" in df.columns:
        q_ids = df["seq_id"].astype(str).tolist()
    else:
        q_ids = [f"q{i}" for i in range(len(df))]
        df["seq_id"] = q_ids

    seqs = df["Sequence"].astype(str).tolist()
    y_true = df["is_bla"].astype(int).to_numpy(np.int8)

    query_faa = tmp_dir / "queries.faa"
    m8_path = tmp_dir / "queries_vs_ref.m8"

    write_fasta(q_ids, seqs, query_faa)

    run_diamond_blastp(
        ref_db=args.ref_db,
        query_faa=query_faa,
        out_m8=m8_path,
        threads=args.threads,
        evalue=args.evalue,
        max_target_seqs=args.max_target_seqs,
    )

    ref_df, ref_map = load_ref_map(Path(args.ref_ids))
    n_cols = len(ref_df)
    n_rows = len(q_ids)

    X = build_feature_matrix(
        m8_path=m8_path,
        query_ids=q_ids,
        ref_map=ref_map,
        n_rows=n_rows,
        n_cols=n_cols,
    )

    # Predict (dense to avoid Keras SparseTensor issues)
    X_dense = X.toarray().astype(np.float32, copy=False)

    model = tf.keras.models.load_model(args.model_path, compile=False)
    prob = model.predict(X_dense, batch_size=256, verbose=0).reshape(-1).astype(np.float32)
    y_pred = (prob >= float(args.threshold)).astype(np.int8)

    # Optional per-sequence output
    if args.pred_out is not None:
        out_pred = df[["seq_id", "is_bla"]].copy()
        out_pred["prob"] = prob
        out_pred["pred"] = y_pred
        out_pred.to_csv(args.pred_out, index=False)

    tp, tn, fp, fn, total, recall = confusion_counts(y_true, y_pred)

    out = pd.DataFrame([{
        "model": args.model_name,
        "bin": "all",
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }])

    out.to_csv(args.out_csv, index=False)
    print(f"Wrote metrics: {args.out_csv}")
    if args.pred_out is not None:
        print(f"Wrote per-sequence predictions: {args.pred_out}")


if __name__ == "__main__":
    main()
