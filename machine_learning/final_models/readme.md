# For walk_forward TCN

python tcn_walk_forward.py \
  --results-sqlite-path results.sqlite \
  --preds-table PREDS_TABLE \
  --metrics-output-dir metrics_json \
  --data-start 2015-01-01 \
  --data-end 2025-12-31 \
  --lookback 60 \
  --seed 0

# For final training TCN

python tcn_train_final.py \
  --data-start 2015-01-01 \
  --data-end 2025-12-31 \
  --lookback 60 \
  --seed 0 \
  --artifacts-root runs
