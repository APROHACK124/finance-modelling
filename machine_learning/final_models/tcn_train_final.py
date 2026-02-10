
"""
tcn_train_final.py

Train the FINAL (deployment) TCN model using the "final inference fold" produced by the annual
walk-forward schedule.

Logic:
- Build annual fold schedule on historical data
- Create deploy fold AFTER the last validated year (train_fit_end_date = last val_end_date)
- Train using data available "as-of" train_fit_end_date:
    * use samples with timestamp <= train_last_labeled_date
    * and enforce target_timestamp <= train_fit_end_date  (anti-leakage)
- Save artifact (model + scaler + metadata) to disk

This script does NOT write predictions to SQLite. It only trains + saves the final model.

All models (CNN/XGB/etc) should use the same final fold builder logic (from walk_forward_common.py).
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict

import numpy as np
import pandas as pd

import torch

from tcn_walk_forward import (
    Config,
    build_final_inference_fold,
    build_feature_cols,
    build_schedule,
    build_tcn_samples,
    default_tcn_cfg,
    load_raw_dataframe,
    make_run_meta,
    save_tcn_artifact,
    set_global_seed,
    stable_hash,
    train_tcn_one_fold,
)

from machine_learning.final_models.walk_forward_common import validate_annual_fold_schedule


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-start", type=str, default="2015-01-01")
    p.add_argument("--data-end", type=str, default="2025-12-31")
    p.add_argument("--lookback", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--symbols", type=str, default="", help="Comma-separated symbols. If empty, uses TARGET_STOCKS.")
    p.add_argument("--artifacts-root", type=str, default="runs")
    return p.parse_args()


def main():
    args = parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()] if args.symbols else None

    cfg = Config(
        seed=int(args.seed),
        lookback=int(args.lookback),
        data_start=args.data_start,
        data_end=args.data_end,
        symbols=symbols,
        tcn_cfg=default_tcn_cfg(),
        artifacts_root=args.artifacts_root,
        fold_schedule_path=f"annual_fold_schedule_h22.csv",
    )

    set_global_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Use same versioning logic as walk-forward, but different run_id for the final artifact
    wf_run_meta = make_run_meta(cfg)
    final_run_id = f"{wf_run_meta['run_id']}_FINAL"
    final_model_version = wf_run_meta["model_version"]  # keep model_version identical for the config

    artifacts_dir = os.path.join(cfg.artifacts_root, final_run_id, "final_model")
    os.makedirs(artifacts_dir, exist_ok=True)

    print("DEVICE:", device)
    print("FINAL_RUN_ID:", final_run_id)
    print("MODEL_VERSION:", final_model_version)
    print("ARTIFACT_DIR:", artifacts_dir)

    df_raw = load_raw_dataframe(cfg)
    feature_cols = build_feature_cols(df_raw)
    trading_dates = pd.DatetimeIndex(sorted(df_raw["timestamp"].unique()))

    schedule = build_schedule(cfg, trading_dates)
    schedule = validate_annual_fold_schedule(schedule)
    if schedule.empty:
        raise ValueError("Fold schedule is empty; cannot build final fold.")

    final_fold = build_final_inference_fold(schedule, trading_dates, horizon_td=cfg.horizon_td)
    print("Final fold:")
    print(final_fold.to_dict())

    # Build training dataset for final fold (anti-leakage)
    train_last = pd.to_datetime(final_fold["train_last_labeled_date"])
    train_fit_end = pd.to_datetime(final_fold["train_fit_end_date"])

    data = build_tcn_samples(
        df_raw,
        feature_cols=feature_cols,
        lookback=cfg.lookback,
        horizon_td=cfg.horizon_td,
        start_date=None,
        end_date=train_last,
        require_y=True,
    )

    meta = data["meta"].copy()
    ok = pd.to_datetime(meta["target_timestamp"], errors="coerce") <= train_fit_end
    ok &= pd.to_datetime(meta["timestamp"], errors="coerce") <= train_last

    X = data["X"][ok.to_numpy()]
    y = data["y"][ok.to_numpy()]
    meta = meta.loc[ok].reset_index(drop=True)

    if X.shape[0] == 0:
        raise ValueError("No training samples for final fold after leakage filters.")

    bundle = train_tcn_one_fold(
        X,
        y,
        meta,
        cfg=cfg.tcn_cfg or default_tcn_cfg(),
        device=device,
    )

    final_meta: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": final_run_id,
        "model_version": final_model_version,
        "base_walk_forward_run_id": wf_run_meta["run_id"],
        "fold": final_fold.to_dict(),
        "train_info": {
            "train_size": bundle["train_size"],
            "val_size": bundle["val_size"],
            "best_val_mean_dailyIC": bundle.get("best_val_mean_dailyIC"),
        },
        "horizon_td": cfg.horizon_td,
        "lookback": cfg.lookback,
        "feature_cols": feature_cols,
        "tcn_cfg": cfg.tcn_cfg or default_tcn_cfg(),
    }

    save_tcn_artifact(bundle, artifacts_dir, final_meta)

    # Also save final_fold as json (human-readable)
    with open(os.path.join(artifacts_dir, "final_fold.json"), "w", encoding="utf-8") as f:
        json.dump(final_fold.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    print("Saved final model artifact to:", artifacts_dir)


if __name__ == "__main__":
    main()
