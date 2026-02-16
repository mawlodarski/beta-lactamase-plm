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
    ap = argparse.ArgumentParser(description="Evaluate HMMER on POSITIVES-ONLY FASTA.")
    ap.add_argument("--pos_faa", required=True, help="FASTA containing only positives")
    ap.add_argument("--tblout", required=True, help="hmmsearch --tblout output for this FASTA")
    ap.add_argument("--out_csv", required=True, help="Output metrics CSV (single row)")
    ap.add_argument("--model_name", default="Gold HMMER BL")
    ap.add_argument("--bin_name", default="all")
    args = ap.parse_args()

    pos_ids = set(fasta_ids(args.pos_faa))
    hit_ids = tblout_hit_ids(args.tblout)

    tp = len(pos_ids & hit_ids)
    fn = len(pos_ids - hit_ids)

    # Not defined for positives-only; set to 0 by design
    tn = 0
    fp = 0

    total = tp + fn
    recall = tp / (tp + fn) if (tp + fn) else 0.0

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
