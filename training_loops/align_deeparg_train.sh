diamond blastp \
  --db gold_ref_panel \
  --query train_gold.faa \
  --out train_gold_vs_ref.m8 \
  --outfmt 6 qseqid sseqid bitscore evalue pident length \
  --max-target-seqs 10000 \
  --evalue 1e-3 \
  --threads 32 \
  > train_blastp.log 2>&1 &
