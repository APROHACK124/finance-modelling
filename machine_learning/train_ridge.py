import json
from pathlib import Path
from xml.etree.ElementInclude import include

import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from machine_learning.data_collectors import (
    build_ml_dataframe,
    build_supervised_dataset,
    time_split_masks,
)
from machine_learning.evaluators import eval_regression

from python_scripts.LLM_analysis.preprocess_store_database import get_connection
from database_tier1 import TARGET_STOCKS




    

def main():
    

    


    # 5 Save the artifact

    run_dir = f"runs/ridge_{timeframe}_lb{lookback}_h{horizon}"

    config = {
        "model": "ridge",
        "timeframe": timeframe,
        "symbols": list(symbols),
        "lookback": lookback,
        "horizon": horizon,
        "base_feature_cols": base_feature_cols,
        "X_shape": [int(X.shape[0]), int(X.shape[1])],
        "trainval_rows": int(X_tv.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "meta_time_min": str(pd.to_datetime(meta["timestamp"]).min()),
        "meta_time_max": str(pd.to_datetime(meta["timestamp"]).max()),
        "best_alpha": best_alpha,
        "cv_best_score_neg_mse": float(gs.best_score_),
    }

    metrics = {
        "test": metrics_test,
        "best_alpha": best_alpha
    }

    save_ridge_artifact(
        run_dir=run_dir,
        pipeline=best_pipe,
        config=config,
        metrics=metrics, 
        feature_names=list(X.columns)
    )

    print(f"Saved to {run_dir}")



    



