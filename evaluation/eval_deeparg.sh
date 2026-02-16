python deeparg.py \
  --model_path discovery_deeparg/discovery_deeparg_mlp.keras \
  --ref_db discovery_deeparg/discovery_ref_panel \
  --ref_ids discovery_deeparg/ref_ids.txt \
  --input_csv bacterial.csv \
  --out_csv discovery_deeparg/bacterial_metrics.csv \
  --model_name "DeepARG-like discovery (alignment-dissimilarity neural network)" \
  --threshold 0.5 \
  --threads 32 > discovery.log 2>&1 &
