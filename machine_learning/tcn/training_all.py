import sys
import os
PROJECT_ROOT = os.path.abspath('../..')
sys.path.append(PROJECT_ROOT)

# AJUSTA ESTOS IMPORTS a tu proyecto real:
from machine_learning.data_collectors import build_ml_dataframe, build_supervised_dataset
from machine_learning.evaluators import eval_regression_extended

from machine_learning.tcn.train_tcn import train_eval_tcn, TrainTCNConfig
from machine_learning.artifacts import load_model_artifact_auto

from machine_learning.data_collectors import (
    build_ml_dataframe,
    build_supervised_dataset,
    time_split_masks,
    purged_ts_cv_splits,
    TARGET_HORIZONS,
    TARGET_LOOKBACKS,
    parse_feat_lag
)
from database_tier1 import TARGET_STOCKS
from python_scripts.LLM_analysis.preprocess_store_database import get_connection

import pandas as pd

from train_walk_forward_tcn import run_walk_forward_tcn, ExperimentConfig, TrainHP
from walk_forward import WalkForwardConfig

# 1) Cargar datos (long: symbol, timestamp, OHLCV, indicadores, etc.)
conn = get_connection()

timeframe = "1Day"
symbols = TARGET_STOCKS

start = None
end = None

include_indicators = False
indicator_names = []
# indicator_names = ['RSI_14', 'BBB_20_2.0', 'BBP_20_2.0', 'ATRr_14']

include_economic_indicators = False
econ_indicator_names = []
# econ_indicator_names = ['CPI', 'UNEMPLOYMENT']

include_fmp = False
fmp_feature_names = []
keep_fmp_asof_date = False
fmp_prefix = 'fmp'

# -----------------------
# ELIGE LOOKBACK AQUÍ
# -----------------------
lookback = TARGET_LOOKBACKS[3]  # <-- cámbialo

# 3 horizontes baseline (puedes editar)
#horizons = [5, 20, 60]
horizon = TARGET_HORIZONS[2]

base_feature_cols = ['open', 'high', 'low', 'close', 'volume', 'trade_count']

lags_by_feature = None
default_lags = lookback


feature_cols = base_feature_cols + indicator_names + econ_indicator_names + fmp_feature_names


print(f"lb={lookback}, h={horizon}")





df = build_ml_dataframe(
    conn,
    symbols=symbols,
    timeframe="1Day",
    start="2015-01-01",
    end="2025-12-31",
    include_indicators=True,
    include_econ=True,
    include_fmp=False,
)

# 2) feature_cols (excluir no-features)
non_feature_cols = {"symbol", "timestamp", "timeframe"}
feature_cols = [c for c in df.columns if c not in non_feature_cols]

# 3) Sanity check explícito: build_supervised_dataset clásico (horizon=5, lags_by_feature=None)
X_wide_5, y_5, meta_5 = build_supervised_dataset(
    df,
    feature_cols=feature_cols,
    lookback=60,
    horizon=5,
    price_col="close",
    group_col="symbol",
    timestamp_col="timestamp",
    lags_by_feature=None,   # explícito, como pediste
)
print("Sanity check horizon=5:", X_wide_5.shape, y_5.shape, meta_5.columns)
assert "target_timestamp" in meta_5.columns, "meta debe incluir target_timestamp"


# 4) Walk-forward + TCN
cfg = ExperimentConfig(
    lookback=TARGET_LOOKBACKS[3],
    horizons=(5, 20, 60),
    wf=WalkForwardConfig(
        target_col="target_timestamp",
        train_span=252*5,   # rolling 5 años (en target timestamps)
        val_span=126,       # 6m
        test_span=126,      # 6m
        step_span=126,      # reentreno semestral
        embargo_span=0,
        min_train_span=252*3,
    ),
    seed=0,
    run_base_dir="../runs",
    run_name=None,
    device="cuda",
    train_hp=TrainHP(num_workers=8, pin_memory=True)
)

# (Opcional) selección simple de hiperparámetros (ejemplo: probar ic_lambda)
hp_candidates = [
    {"loss_hp": {"ic_lambda": 0.0}},   # solo SmoothL1
    {"loss_hp": {"ic_lambda": 0.2}},   # SmoothL1 + IC loss
]

out = run_walk_forward_tcn(
    df,
    feature_cols=feature_cols,
    build_supervised_dataset_fn=build_supervised_dataset,
    eval_fn=eval_regression_extended,
    cfg=cfg,
    hp_candidates=hp_candidates,
)

print("run_dir:", out["run_dir"])
print("final verify diffs:", out["final"]["verify"])
print("tabla folds:\n", out["agg_table"])