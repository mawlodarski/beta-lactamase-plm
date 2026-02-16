#!/usr/bin/env python3
import argparse
import pandas as pd


def fasta_ids(path):
    ids = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith(">"):
                ids.append(line[1:].strip().split()[0])
    return ids


def tblout_hit_ids(path):
    hits = set()
    with open(path, "r") as f:
        for line in f:
            if not line or line.startswith("#"):
                continue
            hits.add(line.split()[0])  # target name = query id
    return hits


def main():
    ap = argparse.ArgumentParser(description="Evaluate HMMER on NEGATIVES-ONLY FASTA.")
    ap.add_argument("--neg_faa", required=True, help="FASTA containing only negatives")
    ap.add_argument("--tblout", required=True, help="hmmsearch --tblout output for this FASTA")
    ap.add_argument("--out_csv", required=True, help="Output metrics CSV (single row)")
    ap.add_argument("--model_name", default="Gold HMMER BL")
    ap.add_argument("--bin_name", default="all")
    args = ap.parse_args()

    neg_ids = set(fasta_ids(args.neg_faa))
    hit_ids = tblout_hit_ids(args.tblout)

    fp = len(neg_ids & hit_ids)
    tn = len(neg_ids - hit_ids)

    # Not defined for negatives-only; set to 0 by design
    tp = 0
    fn = 0

    total = tn + fp
    recall = ""  # recall undefined for negatives-only

    out = pd.DataFrame([{
        "model": args.model_name,
        "bin": args.bin_name,
        "recall": recall,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
    }])

    out.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
