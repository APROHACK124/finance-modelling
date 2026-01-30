from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, Iterable, Optional, Union, List, Tuple, Sequence

import joblib
import numpy as np
import pandas as pd

from machine_learning.evaluators import eval_regression_extended

from machine_learning.data_collectors import *
from machine_learning.evaluators import eval_regression
from machine_learning.models import CNN1DRegressor



def model_baseline_naive(X_raw_df: pd.DataFrame, y_test: pd.DataFrame, test_sl,
                         horizon: int):
    
    close_now = X_raw_df['close_lag0'].to_numpy(dtype=np.float64)
    close_past = X_raw_df[f'close_lag{horizon}'].to_numpy(dtype=np.float64)

    y_pred_baseline_all = np.log(close_now / close_past).astype(np.float32)

    y_pred_baseline_test = y_pred_baseline_all[test_sl]

    print("Baseline (Momentum) - TEST:", eval_regression(y_test.to_numpy(dtype=np.float64), y_pred_baseline_test))

# Model 2 - Ridge Regression (sklearn)

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit

def model_ridge_regression(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.DataFrame, y_test: pd.DataFrame,
                         horizon: int, print_important_features=False):
    alphas = np.logspace(-4, 4, 25)
    tscv = TimeSeriesSplit(n_splits=5)
    ridge_cv = RidgeCV(alphas=alphas, cv=tscv, scoring="neg_mean_squared_error")
    ridge_cv.fit(X_train, y_train)

    print("Best alpha:", ridge_cv.alpha_)

    y_pred_ridge_test = ridge_cv.predict(X_test).astype(np.float64)

    print("Ridge - Test:", eval_regression(y_test.to_numpy(np.float64), y_pred_ridge_test))

    if print_important_features:
        coef = ridge_cv.coef_
        top_idx = np.argsort(np.abs(coef))[::-1][:15]
        top_features = X_train.columns[top_idx]

        for f, c in zip(top_features, coef[top_idx]):
            print(f"{f:40s} {c:+.6f}")


import re
from typing import Iterable, List

_LAGCOL_RE = re.compile(r"^(?P<base>.+)_lag(?P<k>\d+)$")

def _unique_preserve_order(xs: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _infer_base_features_from_lagged_cols(cols: Iterable[str]) -> List[str]:
    bases = []
    for c in cols:
        m = _LAGCOL_RE.match(str(c))
        if m:
            bases.append(m.group("base"))
    return _unique_preserve_order(bases)

def _looks_lagged_feature_list(features: Iterable[str]) -> bool:
    for f in features:
        if _LAGCOL_RE.match(str(f)):
            return True
    return False



import json, os
from pathlib import Path
import joblib

def save_sklearn_artifact(run_dir: str, model, x_scaler, config: dict, metrics: dict):
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, p / "model.joblib")
    joblib.dump(x_scaler, p / "x_scaler.joblib")

    with open(p / "config.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    with open(p / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=float)



def load_ridge_artifact(run_dir: str) -> Dict[str, Any]:
    """
    Loads an artifact stored by save_ridge_artifact:
    - pipeline.joblib (StandardScaler + Ridge)
    - feature_names.json
    - config.json (optional)
    - metrics.json (optional)
    """

    p = Path(run_dir)

    pipeline_path = p / "pipeline.joblib"
    feat_path = p / "feature_names.json"

    if not pipeline_path.exists():
        raise FileNotFoundError(f"Doesn't exists {pipeline_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Doesn't exists {feat_path}")
    
    pipeline = joblib.load(pipeline_path)

    with open(feat_path, "r") as f:
        feature_names = json.load(f)

    config = {}
    metrics = {}

    config_path = p / "config.json"
    metrics_path = p / "metrics.json"

    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    return {
        "run_dir": str(p),
        "pipeline": pipeline,
        "feature_names": feature_names,
        "config": config,
        "metrics": metrics,
    }


def predict_with_ridge_artifact(
    artifact: Dict[str, Any],
    X: pd.DataFrame,
    *,
    strict: bool = True,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Predicts with a saved Ridge artifact, alligning columns.

    strict = True:
        - If columns are missing -> error

    strict = False:
        - Missing columns are filled with fill_value
    
    - Extra columns are ignored
    """

    feat: List[str] = list(artifact["feature_names"])

    missing = [c for c in feat if c not in X.columns]

    
    extra = [c for c in X.columns if c not in feat]

    if missing and strict:
        raise ValueError(
            "X is not compatible with the saved model.\n"
            f"Missing {len(missing)} columns, ej: {missing[:10]}\n"
        )
    
    X_aligned = X.reindex(columns=feat, fill_value=fill_value)

    pipe = artifact["pipeline"]
    y_pred = pipe.predict(X_aligned)

    return np.asarray(y_pred, dtype = np.float64)


# Function to discover stored ridges

def discover_saved_ridges(runs_root: str = "runs") -> List[str]:
    """
    Returns a list of run_dirs that appear to be stored ridges
    Criteria: existing pipeline.joblib and feature_names.json.
    """

    root = Path(runs_root)
    if not root.exists():
        return []
    
    run_dirs = []

    for p in root.iterdir():
        if not p.is_dir():
            continue
        if (p / "pipeline.joblib").exists() and (p / "feature_names.json").exists():
            run_dirs.append(str(p))

    return sorted(run_dirs)

def _predict_generic(
    model_or_fn: Union[Any, Callable[[pd.DataFrame], np.ndarray]],
    X: pd.DataFrame
) -> np.ndarray:
    # Callable
    if callable(model_or_fn) and not hasattr(model_or_fn, "predict"):
        y = model_or_fn(X)
        return np.asarray(y, dtype=np.float64)
    
    # sklearn-like
    if hasattr(model_or_fn, "predict"):
        y = model_or_fn.predict(X)
        return np.asarray(y, dtype=np.float64)
    
    raise TypeError("candidate should be a model with .predict(X) or a function F(X) -> pred")


def compare_candidate_to_saved_ridges(
    candidate: Union[Any, Callable[[pd.DataFrame], np.ndarray]],
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    *,
    saved_run_dirs: Optional[Iterable[str]] = None,
    runs_root: str = "runs",
    candidate_name: str = "candidate",
    eval_fn: Optional[Callable[[np.ndarray, np.ndarray], Any]] = None,
    strict_feature_match: bool = True,
    meta_test: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compares a new model (candidate) against stored ridges
    Returns a dataframe of metrics per model

    - eval_fn: by default tries to use machine_learning.evaluators.eval_regression if its passed
    otherwise it will use some commoon metrics
    - meta_test: if passed (with 'symbol'), adds metrics of hit-rate per symbol
    """

    y_true = np.asarray(y_test, dtype=np.float64)

    rows = []

    def add_row(name: str, y_pred: np.ndarray, extra: Dict[str, Any]):
        row: Dict[str, Any] = {"model": name, **extra}

        if eval_fn is not None:
            m = eval_fn(y_true, y_pred)
            if isinstance(m, dict):
                row.update(m)    
            else:
                row["metric"] = m

        else:
            # Fallback
            err = y_pred - y_true
            row["mae"] = float(np.mean(np.abs(err)))
            row["rmse"] = float(np.sqrt(np.mean(err**2)))

        
    y_pred_cand = _predict_generic(candidate, X_test)
    add_row(candidate_name, y_pred_cand, {"source": "candidate"})

    if saved_run_dirs is None:
        saved_run_dirs = discover_saved_ridges(runs_root=runs_root)

    for rd in saved_run_dirs:
        art = load_ridge_artifact(rd)

        cfg = art.get("config", {}) or {}
        name = cfg.get("run_name") or Path(rd).name

        y_pred = predict_with_ridge_artifact(art, X_test, strict=strict_feature_match)

        add_row(name, y_pred, {"source": rd, "best_alpha": cfg.get("best_alpha", -12345)})

    out = pd.DataFrame(rows)

    if "rmse" in out.columns:
        out = out.sort_values("rmse", ascending=True).reset_index(drop=True)
    elif "mae" in out.columns:
        out = out.sort_values("mae", ascending=True).reset_index(drop=True)

    return out


def _parse_lb_h_from_dirname(run_dir: str) -> Tuple[Optional[int], Optional[int]]:
    """
    Tries to parse lookback/horizon from directory name.
    Example: runs/ridge_1Day_lb30_h1_indicators0_econ0_fmp0
    """

    name = Path(run_dir).name
    m = re.search(r"_lb(\d+)_h(\d+)_indicators(\d+)_econ(\d+)_fmp(\d+)", name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def _parse_timeframe_from_dirname(run_dir: str) -> Optional[str]:
    name = Path(run_dir).name
    m = re.search(r"ridge_(.+?)_lb\d+_h\d+_indicators\d+_econ\d+_fmp\d+", name)
    if not m:
        return None
    return str(m.group(1))

def _parse_flags_from_dirname(run_dir: str) -> Dict[str, Optional[int]]:
    """
    Optional: parse _indicators{0/1}_econ{0/1}_fmp{0/1} if present.
    """
    name = Path(run_dir).name
    m = re.search(r"_indicators(\d+)_econ(\d+)_fmp(\d+)", name)
    if not m:
        return {"indicators": None, "econ": None, "fmp": None}
    return {"indicators": int(m.group(1)), "econ": int(m.group(2)), "fmp": int(m.group(3))}


def evaluate_saved_ridges(
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    *,
    current_lookback: int,
    horizon: int,
    evaluator: Callable[[np.ndarray, np.ndarray], Dict[str, Any]],
    runs_root: str = "runs",
    timeframe: Optional[str] = "1Day",
    strict: bool = True,
    fill_value: float = 0.0,
    require_ridge_step: bool = True,
    verbose: bool = True,
    return_report: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Discover Ridge artifacts under `runs_root`, filter those with lookback < current_lookback
    and horizon == `horizon`, then evaluate on the provided (X_test, y_test).

    Filtering:
    - ridge_lookback < current_lookback
    - ridge_horizon == horizon
    - if timeframe is not None -> require matching timeframe (from config or folder name)

    Feature compatibility:
    - strict=True: if the ridge expects columns not in X_test -> skip (recommended for fair comparison)
    - strict=False: missing columns filled with `fill_value`

    Returns:
    - df_ok: only successfully evaluated ridges
    - optionally df_report: includes skipped entries with reasons (return_report=True)
    """
     
    y_true = np.asarray(y_test, dtype=np.float64).reshape(-1)

    run_dirs = discover_saved_ridges(runs_root=runs_root)
    report_rows: List[Dict[str, Any]] = []
    ok_rows: List[Dict[str, Any]] = []

    for rd in run_dirs:
        row: Dict[str, Any] = {
            "run_dir": rd,
            "status": "skip",
            "skip_reason": None,
            "ridge_lookback": None,
            "ridge_horizon": None,
            "ridge_timeframe": None,
        }

        # Load artifact
        try:
            art = load_ridge_artifact(rd)
        except Exception as e:
            row["skip_reason"] = f"load_error: {e}"
            report_rows.append(row)
            if verbose:
                print(f"[skip] {rd} -> load_error: {e}")
            continue

        if require_ridge_step:
            pipe = art.get("pipeline", None)
            has_ridge = bool(getattr(pipe, "named_steps", None)) and ("ridge" in pipe.named_steps) # type: ignore

            if not has_ridge:
                row["skip_reason"] = "not_a_ridge_pipeline"
                report_rows.append(row)
                continue

        cfg = art.get("config") or {}

        lb = cfg.get("lookback", None)
        h = cfg.get("horizon", None)
        tf = cfg.get("timeframe", None)

        # Validate parsed params
        if lb is None or h is None:
            row["skip_reason"] = "cannot_parse_lookback_or_horizon"
            report_rows.append(row)
            if verbose:
                print(f"[skip] {rd} -> cannot_parse_lookback_or_horizon")
            continue

        lb = int(lb)
        h = int(h)

        row["ridge_lookback"] = lb
        row["ridge_horizon"] = h
        row["ridge_timeframe"] = tf

        # filtering

        if h != int(horizon):
            row["skip_reason"] = f"horizon_mismatch (got{h})"
            report_rows.append(row)
            continue

        if lb > int(current_lookback):
            row["skip_reason"] = f"lookback_not_smaller (got{lb})"
            report_rows.append(row)
            continue

        if timeframe is not None and tf is not None and str(tf) != str(timeframe):
            row["skip_reason"] = f"timeframe_mismatch (got{tf})"
            report_rows.append(row)
            continue

        try:
            y_pred = predict_with_ridge_artifact(
                art,
                X_test,
                strict=strict,
                fill_value=fill_value,
            )
        except Exception as e:
            row["skip_reason"] = f"predict_error: {e}"
            report_rows.append(row)
            if verbose:
                print(f"[skip] {rd} -> predict_error: {e}")
            continue

        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        if y_pred.shape[0] != y_true.shape[0]:
            row["skip_reason"] = f"pred_len_mismatch (pred{y_pred.shape[0]} vs true {y_true.shape[0]})"
            report_rows.append(row)
            continue

        # evaluate
        metrics = evaluator(y_true, y_pred)

        ok_row = dict(row)
        ok_row["status"] = "ok"
    
        ok_row["best_alpha"] = cfg.get("best_alpha", None)
        ok_row["cv_best_score_neg_mse"] = cfg.get("cv_best_score_neg_mse", None)

        # flatten metrics onto columns
        for k, v in metrics.items():
            ok_row[f"test_{k}"] = v

        ok_rows.append(ok_row)
        report_rows.append(ok_row)

    df_ok = pd.DataFrame(ok_rows)
    df_report = pd.DataFrame(report_rows)

    if return_report:
        return df_ok, df_report
    return df_ok

def predict_ridge_by_run_dir(
    X: pd.DataFrame,
    run_dir: str,
    horizon: int,
    strict: bool,
    fill_value
) -> np.ndarray:
    rd = run_dir
    art = load_ridge_artifact(rd)
    cfg = art.get("config") or {}

    # --- Extract params (config first, then dirname fallback) ---
    lb = cfg.get("lookback", None)
    h = cfg.get("horizon", None)
    tf = cfg.get("timeframe", None)

    if lb is None or h is None:
        lb2, h2 = _parse_lb_h_from_dirname(rd)
        if lb is None:
            lb = lb2
        if h is None:
            h = h2

    if tf is None:
        tf = _parse_timeframe_from_dirname(rd)

    
    lb = int(lb) # type: ignore
    h = int(h) # type: ignore

    if h != int(horizon):
        print('Error: horizon different to the expected')
        return np.ndarray([])

    prediction = predict_with_ridge_artifact(
        art,
        X,
        strict=strict,
        fill_value=fill_value
    )
    return prediction.reshape(-1)



def evaluate_saved_ridges_extended(
    *,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[Union[pd.Series, np.ndarray]],
    meta_val: Optional[pd.DataFrame],
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    meta_test: Optional[pd.DataFrame],
    current_lookback: int,
    horizon: int,
    runs_root: str = "runs",
    timeframe: Optional[str] = None,
    strict: bool = True,
    fill_value: float = 0.0,
    deadzone: float = 0.0,
    quantile: float = 0.1,
    min_group_size: int = 20,
    periods_per_year: int = 252,
    conformal_alphas: Tuple[float, ...] = (0.1, 0.05),
    allow_equal_lookback: bool = False,
    return_report: bool = False,
    verbose: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Evaluate saved Ridge artifacts using eval_regression_extended, with optional
    validation set for calibration (conformal) and val metrics.

    Filters:
      - ridge_horizon == horizon
      - ridge_lookback < current_lookback   (or <= if allow_equal_lookback=True)
      - if timeframe provided -> must match

    Returns:
      df_ok: evaluated models (metrics flattened with val_/test_ prefixes)
      optionally df_report: includes skipped with reasons
    """
    y_test_np = np.asarray(y_test, dtype=np.float64).reshape(-1)

    has_val = (X_val is not None) and (y_val is not None)
    if has_val:
        y_val_np = np.asarray(y_val, dtype=np.float64).reshape(-1)
        if len(X_val) != len(y_val_np): # type: ignore
            raise ValueError("X_val and y_val must have same length.")
        if meta_val is not None and len(meta_val) != len(y_val_np):
            raise ValueError("meta_val must align with X_val/y_val.")
    else:
        y_val_np = None  # type: ignore

    if len(X_test) != len(y_test_np):
        raise ValueError("X_test and y_test must have same length.")
    if meta_test is not None and len(meta_test) != len(y_test_np):
        raise ValueError("meta_test must align with X_test/y_test.")

    run_dirs = discover_saved_ridges(runs_root=runs_root)

    ok_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    for rd in run_dirs:
        row_base: Dict[str, Any] = {
            "model_type": "ridge",
            "model_id": rd,
            "run_dir": rd,
            "status": "skip",
            "skip_reason": None,
            "ridge_lookback": None,
            "ridge_horizon": None,
            "ridge_timeframe": None,
            "ridge_best_alpha": None,
            "include_indicators": None,
            "include_econ": None,
            "include_fmp": None,
        }

        # --- Load artifact ---
        try:
            art = load_ridge_artifact(rd)
        except Exception as e:
            row_base["skip_reason"] = f"load_error: {e}"
            report_rows.append(row_base)
            if verbose:
                print(f"[skip] {rd} | load_error: {e}")
            continue

        cfg = art.get("config") or {}

        # --- Extract params (config first, then dirname fallback) ---
        lb = cfg.get("lookback", None)
        h = cfg.get("horizon", None)
        tf = cfg.get("timeframe", None)

        if lb is None or h is None:
            lb2, h2 = _parse_lb_h_from_dirname(rd)
            if lb is None:
                lb = lb2
            if h is None:
                h = h2

        if tf is None:
            tf = _parse_timeframe_from_dirname(rd)

        if lb is None or h is None:
            row_base["skip_reason"] = "cannot_parse_lookback_or_horizon"
            report_rows.append(row_base)
            continue

        lb = int(lb)
        h = int(h)

        row_base["ridge_lookback"] = lb
        row_base["ridge_horizon"] = h
        row_base["ridge_timeframe"] = tf

        # flags (config preferred)
        row_base["include_indicators"] = cfg.get("include_indicators", None)
        row_base["include_econ"] = cfg.get("include_econ", None)
        row_base["include_fmp"] = cfg.get("include_fmp", None)

        if row_base["include_indicators"] is None or row_base["include_econ"] is None or row_base["include_fmp"] is None:
            flags = _parse_flags_from_dirname(rd)
            if row_base["include_indicators"] is None:
                row_base["include_indicators"] = flags["indicators"]
            if row_base["include_econ"] is None:
                row_base["include_econ"] = flags["econ"]
            if row_base["include_fmp"] is None:
                row_base["include_fmp"] = flags["fmp"]

        row_base["ridge_best_alpha"] = cfg.get("best_alpha", None)

        # --- Filters ---
        if h != int(horizon):
            row_base["skip_reason"] = f"horizon_mismatch (got {h})"
            report_rows.append(row_base)
            continue

        if allow_equal_lookback:
            ok_lb = lb <= int(current_lookback)
        else:
            ok_lb = lb < int(current_lookback)

        if not ok_lb:
            row_base["skip_reason"] = f"lookback_not_allowed (got {lb})"
            report_rows.append(row_base)
            continue

        if timeframe is not None and tf is not None and str(tf) != str(timeframe):
            row_base["skip_reason"] = f"timeframe_mismatch (got {tf})"
            report_rows.append(row_base)
            continue

        # --- Predict val (optional) ---
        if has_val:
            try:
                y_val_pred = predict_with_ridge_artifact(
                    art,
                    X_val,  # type: ignore[arg-type]
                    strict=strict,
                    fill_value=fill_value,
                ).reshape(-1)
            except Exception as e:
                row_base["skip_reason"] = f"predict_val_error: {e}"
                report_rows.append(row_base)
                if verbose:
                    print(f"[skip] {rd} | predict_val_error: {e}")
                continue

            metrics_val = eval_regression_extended(
                y_val_np,  # type: ignore[arg-type]
                y_val_pred,
                deadzone=deadzone,
                meta=meta_val,
                time_col="timestamp",
                group_col="symbol",
                periods_per_year=periods_per_year,
                quantile=quantile,
                min_group_size=min_group_size,
                conformal_calib=None,  # no conformal on val if val is used as calib for test
                conformal_alphas=conformal_alphas,
            )
        else:
            y_val_pred = None
            metrics_val = None

        # --- Predict test ---
        try:
            y_test_pred = predict_with_ridge_artifact(
                art,
                X_test,
                strict=strict,
                fill_value=fill_value,
            ).reshape(-1)
        except Exception as e:
            row_base["skip_reason"] = f"predict_test_error: {e}"
            report_rows.append(row_base)
            if verbose:
                print(f"[skip] {rd} | predict_test_error: {e}")
            continue

        # --- Metrics test (with conformal if we have val) ---
        metrics_test = eval_regression_extended(
            y_test_np,
            y_test_pred,
            deadzone=deadzone,
            meta=meta_test,
            time_col="timestamp",
            group_col="symbol",
            periods_per_year=periods_per_year,
            quantile=quantile,
            min_group_size=min_group_size,
            conformal_calib=(y_val_np, y_val_pred) if has_val else None,  # type: ignore[arg-type]
            conformal_alphas=conformal_alphas,
        )

        # --- Flatten to row ---
        row = dict(row_base)
        row["status"] = "ok"
        row["skip_reason"] = None

        if metrics_val is not None:
            for k, v in metrics_val.items():
                row[f"val_{k}"] = v
        for k, v in metrics_test.items():
            row[f"test_{k}"] = v

        ok_rows.append(row)
        report_rows.append(row)

    df_ok = pd.DataFrame(ok_rows)
    df_report = pd.DataFrame(report_rows)

    # Nice sorting default if available
    if not df_ok.empty:
        sort_key = None
        for candidate in ["val_DailyRankIC_mean", "val_PearsonCorr(IC)", "test_DailyRankIC_mean", "test_PearsonCorr(IC)"]:
            if candidate in df_ok.columns:
                sort_key = candidate
                break
        if sort_key is not None:
            df_ok = df_ok.sort_values(sort_key, ascending=False).reset_index(drop=True)

    if return_report:
        return df_ok, df_report
    return df_ok


DEFAULT_FILENAMES = {
    "config": ["config.json", "config.pkl", "artifact_config.json"],
    "feature_names": ["feature_names.json", "features.json"],
    "scaler": ["scaler.pkl", "scaler.joblib"],
    "model": ["model.pt", "model.pth", "model_state.pt", "state_dict.pt"],
    "artifact": ["artifact.pkl", "artifact.joblib"],  # si guardas todo en un solo archivo
    "metrics": ["metrics.json"],
}

import torch

def _first_existing(run_dir: Union[str, Path], candidates: Sequence[str]) -> Optional[Path]:
    rd = Path(run_dir)
    for name in candidates:
        p = rd / name
        if p.exists():
            return p
    return None

from dataclasses import asdict, is_dataclass

def _jsonify(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj

def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def _load_pickle(path: Path) -> Any:
    # joblib suele ser mejor para scalers/sklearn
    return joblib.load(path)

def _safe_torch_load(path: Path) -> Any:
    # map_location cpu para que sea portable
    return torch.load(path, map_location="cpu")

TORCH_MODEL_BUILDERS: Dict[str, Callable[[Dict[str, Any], int], torch.nn.Module]] = {}

def register_torch_model_builder(model_name: str):
    """Decorator para registrar un builder: (config, input_dim) -> nn.Module"""
    def deco(fn: Callable[[Dict[str, Any], int], torch.nn.Module]):
        TORCH_MODEL_BUILDERS[model_name] = fn
        return fn
    return deco

@register_torch_model_builder("mlp_regressor")
def _build_mlp_1_from_config(config: Dict[str, Any], input_dim: int) -> torch.nn.Module:
    from machine_learning.models import MLPRegressor
    mlp_cfg = config.get("mlp_config") or {}
    hidden_sizes = mlp_cfg.get("hidden_sizes", (256, 128))
    dropout = float(mlp_cfg.get("dropout", 0.10))
    batch_norm = bool(mlp_cfg.get("batch_norm", True))

    # hidden_sizes puede venir como list desde JSON -> OK
    return MLPRegressor(
        input_dim=int(input_dim),
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        batch_norm=batch_norm,
    )

# def load_torch_artifact(run_dir: str) -> Dict[str, Any]:
#     """
#     Devuelve un dict con al menos:
#       - config: dict
#       - feature_names: list[str]
#       - scaler: objeto con .transform o None
#       - model_obj: nn.Module o None
#       - model_state: state_dict o None
#     """
#     rd = Path(run_dir)
#     if not rd.exists():
#         raise FileNotFoundError(f"run_dir no existe: {run_dir}")

#     # 1) Si existe un artifact.pkl que ya trae todo, úsalo
#     p_art = _first_existing(rd, DEFAULT_FILENAMES["artifact"])
#     if p_art is not None:
#         art = _load_pickle(p_art)
#         # Intenta normalizar claves
#         config = art.get("config") or art.get("cfg") or {}
#         feature_names = art.get("feature_names") or config.get("feature_names")
#         scaler = art.get("scaler", None)
#         model_state = art.get("model_state") or art.get("state_dict")
#         model_obj = art.get("model", None)
#         return {
#             "run_dir": str(rd),
#             "config": config,
#             "feature_names": list(feature_names) if feature_names is not None else None,
#             "scaler": scaler,
#             "model_state": model_state,
#             "model_obj": model_obj,
#             "raw": art,
#         }

#     # 2) Layout multi-archivos (config, scaler, state_dict, etc.)
#     p_cfg = _first_existing(rd, DEFAULT_FILENAMES["config"])
#     if p_cfg is None:
#         raise FileNotFoundError(f"No encuentro config en {run_dir}. Busqué: {DEFAULT_FILENAMES['config']}")

#     if p_cfg.suffix.lower() == ".json":
#         config = _load_json(p_cfg)
#     else:
#         config = _load_pickle(p_cfg)


#     # 2) metrics
#     p_metrics = _first_existing(rd, DEFAULT_FILENAMES["metrics"])
#     if p_metrics is None:
#         raise FileNotFoundError(f"No encuentro metrics en {run_dir}. Busqué: {DEFAULT_FILENAMES['metrics']}")

#     if p_metrics.suffix.lower() == ".json":
#         metrics = _load_json(p_metrics)
#     else:
#         metrics = _load_pickle(p_metrics)

#     # feature_names
#     p_feats = _first_existing(rd, DEFAULT_FILENAMES["feature_names"])
#     feature_names = None
#     if p_feats is not None:
#         feature_names = _load_json(p_feats) if p_feats.suffix.lower() == ".json" else _load_pickle(p_feats)
#     else:
#         # fallback: algunos guardan en config
#         feature_names = config.get("feature_names", None)

#     if feature_names is None:
#         raise FileNotFoundError(
#             f"No encuentro feature_names (archivo ni en config) en {run_dir}. "
#             f"Busqué: {DEFAULT_FILENAMES['feature_names']} o config['feature_names']"
#         )

#     # scaler
#     p_scaler = _first_existing(rd, DEFAULT_FILENAMES["scaler"])
#     scaler = _load_pickle(p_scaler) if p_scaler is not None else None

#     # model
#     p_model = _first_existing(rd, DEFAULT_FILENAMES["model"])
#     if p_model is None:
#         raise FileNotFoundError(f"No encuentro model/state_dict en {run_dir}. Busqué: {DEFAULT_FILENAMES['model']}")

#     obj = _safe_torch_load(p_model)
#     model_obj = obj if isinstance(obj, torch.nn.Module) else None

#     # state_dict puede venir directo o empacado
#     model_state = None
#     if model_obj is None:
#         if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
#             # puede ser state_dict puro o un dict con state_dict adentro
#             if any(k in obj for k in ["state_dict", "model_state", "model"]):
#                 model_state = obj.get("state_dict") or obj.get("model_state") or obj.get("model")
#             else:
#                 model_state = obj
#         else:
#             raise ValueError(f"Archivo de modelo en {p_model} no es nn.Module ni dict state_dict interpretable.")

#     return {
#         "run_dir": str(rd),
#         "config": config,
#         "feature_names": list(feature_names),
#         "scaler": scaler,
#         "model_state": model_state,
#         "model_obj": model_obj,
#         "paths": {
#             "config": str(p_cfg),
#             "feature_names": str(p_feats) if p_feats else None,
#             "scaler": str(p_scaler) if p_scaler else None,
#             "model": str(p_model),
#         },
#         "metrics": metrics,
#     }


def _read_feature_names(path: Path) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Acepta list o cualquier iterable razonable
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "feature_names" in data and isinstance(data["feature_names"], list):
        return data["feature_names"]
    return list(data)

def load_torch_artifact(
    run_dir: Path,
    *,
    map_location: str = "cpu",
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Loader genérico para artefactos PyTorch tipo 'model.pt' + 'feature_names.json'
    (CNN incluidos), compatible con tu loader viejo.
    Requiere que exista build_torch_model_from_payload(model_name, payload, config).
    """
    config = config or {}
    metrics = metrics or {}

    model_pt = run_dir / "model.pt"
    feat_path = run_dir / "feature_names.json"

    if not model_pt.exists():
        raise FileNotFoundError(f"Missing {model_pt}")

    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path} (required for torch artifacts)")

    feature_names = _read_feature_names(feat_path)

    payload = torch.load(model_pt, map_location=map_location)

    # Soporta configs viejas y nuevas
    model_name = (
        config.get("model_name")
        or config.get("model")          # viejo
        or config.get("model_type")     # nuevo
        or payload.get("model_name")
        or payload.get("model")
        or "torch_model"
    )
    if model_name == "torch_model":
        print('wtf', run_dir, config)

    # Tu registry/factory (de tu versión vieja)
    model = build_torch_model_from_payload(str(model_name), payload, config)
    model.eval()

    # scaler opcional (acepta joblib o pkl)
    scaler = None
    for scaler_fname in ("scaler.joblib", "scaler.pkl"):
        sp = run_dir / scaler_fname
        if sp.exists():
            scaler = joblib.load(sp)
            break

    return {
        "run_dir": str(run_dir),
        "family": "torch",
        "framework": "pytorch",
        "model_name": str(model_name),
        "model": model,
        "scaler": scaler,
        "feature_names": list(feature_names),
        "config": config,
        "metrics": metrics,
        "payload": payload,  # útil para debug
    }

import numpy as np
import pandas as pd
from typing import Any, Optional, Sequence

def _align_and_transform_features(
    X: pd.DataFrame,
    feature_names: Sequence[str],
    scaler: Optional[Any],
    *,
    strict: bool = True,
    fill_value: float = 0.0,
    dtype=np.float32,
) -> np.ndarray:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X debe ser pd.DataFrame")

    feat = list(feature_names)

    # 1) Alinear columnas (y orden)
    if strict:
        missing = [c for c in feat if c not in X.columns]
        if missing:
            raise ValueError(
                f"Faltan features requeridas: {missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        X_aligned = X.loc[:, feat]
    else:
        X_aligned = X.reindex(columns=feat)

    # 2) Limpiar NaN/inf
    X_aligned = (
        X_aligned
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill_value)
    )

    # 3) ESCALAR usando DataFrame (para conservar feature names y evitar warning)
    if scaler is not None:
        # Si el scaler fue fitteado con feature_names, sklearn valida nombres y orden.
        # Como X_aligned ya está alineado, no debería avisar.
        X_scaled = scaler.transform(X_aligned)  # <- DataFrame IN, numpy OUT
        X_np = np.asarray(X_scaled, dtype=dtype)
    else:
        X_np = X_aligned.to_numpy(dtype=dtype, copy=False)

    # 4) Seguridad extra
    X_np = np.nan_to_num(X_np, nan=fill_value, posinf=fill_value, neginf=fill_value)
    return X_np


def _batched_torch_predict(
    model: torch.nn.Module,
    X_np: np.ndarray,
    *,
    device: Union[str, torch.device] = "gpu",
    batch_size: int = 8192,
) -> np.ndarray:
    dev = torch.device(device)
    model = model.to(dev)
    model.eval()

    n = X_np.shape[0]
    out = np.empty(n, dtype=np.float64)

    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X_np[i:i+batch_size]).to(dev)
            yb = model(xb)
            yb = yb.detach().cpu().numpy().reshape(-1).astype(np.float64)
            out[i:i+len(yb)] = yb

    return out


    model_name = config.get("model", None)
    if model_name is None:
        raise ValueError(f"config['model'] no existe en {run_dir}. Ej: 'mlp_regressor'")

    feature_names: List[str] = art["feature_names"]
    input_dim = len(feature_names)

    # construir modelo
    if art.get("model_obj") is not None:
        model = art["model_obj"]
    else:
        if model_name not in TORCH_MODEL_BUILDERS:
            raise KeyError(
                f"No hay builder registrado para model='{model_name}'. "
                f"Registra uno con @register_torch_model_builder('{model_name}')"
            )
        model = TORCH_MODEL_BUILDERS[model_name](config, input_dim=input_dim)
        state = art["model_state"]
        model.load_state_dict(state)

    # features
    X_np = _align_and_transform_features(
        X,
        feature_names=feature_names,
        scaler=art.get("scaler", None),
        strict=strict,
        fill_value=fill_value,
        dtype=np.float32,
    )

    # pred
    y_pred = _batched_torch_predict(model, X_np, device=device, batch_size=batch_size)
    return y_pred.reshape(-1)

def predict_torch_by_run_dir(
    X: pd.DataFrame,
    run_dir: str,
    *,
    device: Union[str, torch.device] = "gpu",
    batch_size: int = 8192,
    strict: bool = True,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Predice usando un artefacto guardado (MLP/CNN/Transformer) en run_dir.
    No hay que reescribir esto por arquitectura: solo registras un builder.
    """
    art = load_torch_artifact(run_dir)
    config = art["config"]
    model_name = config.get("model", None)
    if model_name is None:
        raise ValueError(f"config['model'] no existe en {run_dir}. Ej: 'mlp_regressor'")

    feature_names: List[str] = art["feature_names"]
    input_dim = len(feature_names)

    # construir modelo
    if art.get("model_obj") is not None:
        model = art["model_obj"]
    else:
        if model_name not in TORCH_MODEL_BUILDERS:
            raise KeyError(
                f"No hay builder registrado para model='{model_name}'. "
                f"Registra uno con @register_torch_model_builder('{model_name}')"
            )
        model = TORCH_MODEL_BUILDERS[model_name](config, input_dim=input_dim)
        state = art["model_state"]
        model.load_state_dict(state)

    # features
    X_np = _align_and_transform_features(
        X,
        feature_names=feature_names,
        scaler=art.get("scaler", None),
        strict=strict,
        fill_value=fill_value,
        dtype=np.float32,
    )

    # pred
    y_pred = _batched_torch_predict(model, X_np, device=device, batch_size=batch_size)
    return y_pred.reshape(-1)


def discover_saved_runs(
    runs_root: str = "runs",
    *,
    must_contain_any: Sequence[str] = ("config.json", "model.pt", "artifact.pkl"),
) -> List[str]:
    root = Path(runs_root)
    if not root.exists():
        return []

    out: List[str] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        # heurística: si contiene alguno de esos archivos, es un run_dir
        if any((p / fname).exists() for fname in must_contain_any):
            out.append(str(p))
    return sorted(out)

def evaluate_saved_torch_models_extended(
    *,
    X_val: Optional[pd.DataFrame],
    y_val: Optional[Union[pd.Series, np.ndarray]],
    meta_val: Optional[pd.DataFrame],
    X_test: pd.DataFrame,
    y_test: Union[pd.Series, np.ndarray],
    meta_test: Optional[pd.DataFrame],
    current_lookback: int,
    horizon: int,
    runs_root: str = "runs",
    timeframe: Optional[str] = None,
    model_names: Optional[Tuple[str, ...]] = None,  # ej ("mlp_regressor","cnn_regressor")
    strict: bool = True,
    fill_value: float = 0.0,
    deadzone: float = 0.0,
    quantile: float = 0.1,
    min_group_size: int = 20,
    periods_per_year: int = 252,
    conformal_alphas: Tuple[float, ...] = (0.1, 0.05),
    allow_equal_lookback: bool = False,
    return_report: bool = False,
    verbose: bool = False,
    device: Union[str, torch.device] = "gpu",
    batch_size: int = 8192,
    time_col: str = "timestamp",
    group_col: str = "symbol",
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Evalúa artefactos Torch guardados (MLP/CNN/Transformer) con eval_regression_extended.
    Similar a evaluate_saved_ridges_extended pero genérico para Torch.

    Requisitos mínimos del artefacto:
      - config con keys: model, lookback, horizon (o parseable del dirname)
      - feature_names
      - model state_dict
      - scaler (opcional)
    """
    y_test_np = np.asarray(y_test, dtype=np.float64).reshape(-1)

    has_val = (X_val is not None) and (y_val is not None)
    if has_val:
        y_val_np = np.asarray(y_val, dtype=np.float64).reshape(-1)
        if len(X_val) != len(y_val_np):  # type: ignore
            raise ValueError("X_val and y_val must have same length.")
        if meta_val is not None and len(meta_val) != len(y_val_np):
            raise ValueError("meta_val must align with X_val/y_val.")
    else:
        y_val_np = None  # type: ignore

    if len(X_test) != len(y_test_np):
        raise ValueError("X_test and y_test must have same length.")
    if meta_test is not None and len(meta_test) != len(y_test_np):
        raise ValueError("meta_test must align with X_test/y_test.")

    run_dirs = discover_saved_runs(runs_root=runs_root)

    ok_rows: List[Dict[str, Any]] = []
    report_rows: List[Dict[str, Any]] = []

    for rd in run_dirs:
        row_base: Dict[str, Any] = {
            "model_type": None,
            "model_id": Path(rd).name,
            "run_dir": rd,
            "status": "skip",
            "skip_reason": None,
            "lookback": None,
            "horizon": None,
            "timeframe": None,
            "include_indicators": None,
            "include_econ": None,
            "include_fmp": None,
        }

        # --- Load artifact ---
        try:
            art = load_torch_artifact(rd)
        except Exception as e:
            row_base["skip_reason"] = f"load_error: {e}"
            report_rows.append(row_base)
            if verbose:
                print(f"[skip] {rd} | load_error: {e}")
            continue

        cfg = art.get("config") or {}
        model_type = cfg.get("model", None)
        row_base["model_type"] = model_type

        # filtrar por model_names si aplica
        if model_names is not None and model_type is not None and model_type not in model_names:
            row_base["skip_reason"] = f"model_not_in_filter (got {model_type})"
            report_rows.append(row_base)
            continue

        # params (config primero)
        lb = cfg.get("lookback", None)
        h = cfg.get("horizon", None)
        tf = cfg.get("timeframe", None)

        if lb is None or h is None:
            lb2, h2 = _parse_lb_h_from_dirname(rd)
            if lb is None:
                lb = lb2
            if h is None:
                h = h2

        if tf is None:
            tf = _parse_timeframe_from_dirname(rd)

        if lb is None or h is None:
            row_base["skip_reason"] = "cannot_parse_lookback_or_horizon"
            report_rows.append(row_base)
            continue

        lb = int(lb)
        h = int(h)
        row_base["lookback"] = lb
        row_base["horizon"] = h
        row_base["timeframe"] = tf

        # flags: si no están en config, parse del dirname
        row_base["include_indicators"] = cfg.get("include_indicators", None)
        row_base["include_econ"] = cfg.get("include_econ", None)
        row_base["include_fmp"] = cfg.get("include_fmp", None)
        if row_base["include_indicators"] is None or row_base["include_econ"] is None or row_base["include_fmp"] is None:
            flags = _parse_flags_from_dirname(rd)
            if row_base["include_indicators"] is None:
                row_base["include_indicators"] = flags["indicators"]
            if row_base["include_econ"] is None:
                row_base["include_econ"] = flags["econ"]
            if row_base["include_fmp"] is None:
                row_base["include_fmp"] = flags["fmp"]

        # --- Filters ---
        if h != int(horizon):
            row_base["skip_reason"] = f"horizon_mismatch (got {h})"
            report_rows.append(row_base)
            continue

        ok_lb = (lb <= int(current_lookback)) if allow_equal_lookback else (lb < int(current_lookback))
        if not ok_lb:
            row_base["skip_reason"] = f"lookback_not_allowed (got {lb})"
            report_rows.append(row_base)
            continue

        if timeframe is not None and tf is not None and str(tf) != str(timeframe):
            row_base["skip_reason"] = f"timeframe_mismatch (got {tf})"
            report_rows.append(row_base)
            continue

        # --- Predict val (optional) ---
        if has_val:
            try:
                y_val_pred = predict_torch_by_run_dir(
                    X_val, rd,  # type: ignore[arg-type]
                    device=device,
                    batch_size=batch_size,
                    strict=strict,
                    fill_value=fill_value,
                ).reshape(-1)
            except Exception as e:
                row_base["skip_reason"] = f"predict_val_error: {e}"
                report_rows.append(row_base)
                if verbose:
                    print(f"[skip] {rd} | predict_val_error: {e}")
                continue

            metrics_val = eval_regression_extended(
                y_val_np,  # type: ignore[arg-type]
                y_val_pred,
                deadzone=deadzone,
                meta=meta_val,
                time_col=time_col,
                group_col=group_col,
                periods_per_year=periods_per_year,
                quantile=quantile,
                min_group_size=min_group_size,
                conformal_calib=None,
                conformal_alphas=conformal_alphas,
            )
        else:
            y_val_pred = None
            metrics_val = None

        # --- Predict test ---
        try:
            y_test_pred = predict_torch_by_run_dir(
                X_test, rd,
                device=device,
                batch_size=batch_size,
                strict=strict,
                fill_value=fill_value,
            ).reshape(-1)
        except Exception as e:
            row_base["skip_reason"] = f"predict_test_error: {e}"
            report_rows.append(row_base)
            if verbose:
                print(f"[skip] {rd} | predict_test_error: {e}")
            continue

        # --- Metrics test (con conformal si hay val) ---
        metrics_test = eval_regression_extended(
            y_test_np,
            y_test_pred,
            deadzone=deadzone,
            meta=meta_test,
            time_col=time_col,
            group_col=group_col,
            periods_per_year=periods_per_year,
            quantile=quantile,
            min_group_size=min_group_size,
            conformal_calib=(y_val_np, y_val_pred) if has_val else None,  # type: ignore[arg-type]
            conformal_alphas=conformal_alphas,
        )

        # --- Flatten ---
        row = dict(row_base)
        row["status"] = "ok"
        row["skip_reason"] = None

        # útil: best_epoch/best_val_loss si existen
        row["best_epoch"] = cfg.get("best_epoch", None)
        row["best_val_loss"] = cfg.get("best_val_loss", None)

        if metrics_val is not None:
            for k, v in metrics_val.items():
                row[f"val_{k}"] = v
        for k, v in metrics_test.items():
            row[f"test_{k}"] = v

        ok_rows.append(row)
        report_rows.append(row)

    df_ok = pd.DataFrame(ok_rows)
    df_report = pd.DataFrame(report_rows)

    if not df_ok.empty:
        sort_key = None
        for candidate in ["val_DailyRankIC_mean", "val_SpearmanCorr(RankIC)", "test_DailyRankIC_mean", "test_SpearmanCorr(RankIC)"]:
            if candidate in df_ok.columns:
                sort_key = candidate
                break
        if sort_key is not None:
            df_ok = df_ok.sort_values(sort_key, ascending=False).reset_index(drop=True)

    return (df_ok, df_report) if return_report else df_ok

def discover_artifacts_by_horizon(horizon: int):
    run_dirs = discover_saved_runs()
    artifacts = []

    for dir in run_dirs:
        
        name = Path(dir).name
        if name[0:5] == 'ridge':
            continue
        if name[0:3] == 'xgb':
            continue
        
        dir = Path(dir)

        config = _read_json_if_exists(dir / "config.json")
        metrics = _read_json_if_exists(dir / "metrics.json")

        art = load_torch_artifact(dir, config=config, metrics=metrics)
        if art['config']['horizon'] != horizon:
            continue

        artifacts.append(art)

    return artifacts



import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import torch

# Ajusta este import según dónde tengas tu clase
# from machine_learning.models import MLPRegressor
# o si está en el notebook, asegúrate que exista en scope


def load_mlp_artifact(run_dir: str, *, map_location: str = "cpu") -> Dict[str, Any]:
    from machine_learning.models import MLPRegressor
    """
    Loads an MLP artifact saved by save_mlp_artifact.
    Expects:
      - model.pt (state_dict + input_dim + config)
      - scaler.joblib
      - feature_names.json
      - config.json (optional)
      - metrics.json (optional)
    Returns a dict with:
      - model (torch nn.Module, eval mode)
      - scaler (StandardScaler)
      - feature_names (list[str])
      - config (dict)
      - metrics (dict)
      - run_dir (str)
    """
    p = Path(run_dir)
    model_path = p / "model.pt"
    scaler_path = p / "scaler.joblib"
    feat_path = p / "feature_names.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing {scaler_path}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing {feat_path}")

    payload = torch.load(model_path, map_location=map_location)
    scaler = joblib.load(scaler_path)

    with open(feat_path, "r") as f:
        feature_names = json.load(f)

    config = {}
    metrics = {}

    config_path = p / "config.json"
    metrics_path = p / "metrics.json"

    if config_path.exists():
        with open(config_path, "r") as f:
            config = json.load(f)

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

    # Reconstruct model
    mlp_cfg = payload.get("config", {}) or {}
    input_dim = int(payload["input_dim"])

    model = MLPRegressor(
        input_dim=input_dim,
        hidden_sizes=mlp_cfg.get("hidden_sizes", (64, 32)),
        dropout=float(mlp_cfg.get("dropout", 0.0)),
        batch_norm=bool(mlp_cfg.get("batch_norm", False)),
    )

    model.load_state_dict(payload["state_dict"])
    model.eval()

    return {
        "run_dir": str(p),
        "model": model,
        "scaler": scaler,
        "feature_names": list(feature_names),
        "config": config,
        "metrics": metrics,
    }

from torch.utils.data import DataLoader, TensorDataset


def predict_with_mlp_artifact(
    artifact: Dict[str, Any],
    X: pd.DataFrame,
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 4096,
    strict: bool = True,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Predict with saved MLP artifact.
    Steps:
      1) Align X columns to artifact["feature_names"]
      2) Apply artifact["scaler"].transform
      3) Torch forward pass in eval/no_grad
    """
    feat: List[str] = list(artifact["feature_names"])
    scaler = artifact["scaler"]
    model = artifact["model"]

    # Align columns
    missing = [c for c in feat if c not in X.columns]
    if missing and strict:
        raise ValueError(
            "X no es compatible con el modelo guardado.\n"
            f"Faltan {len(missing)} columnas. Ej: {missing[:10]}"
        )

    X_aligned = X.reindex(columns=feat, fill_value=fill_value)

    # IMPORTANT: StandardScaler expects same column order; reindex en ese orden.
    # Scaler output is float64; convert to float32 (como en training)
    X_scaled = scaler.transform(X_aligned).astype(np.float32)

    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    ds = TensorDataset(torch.from_numpy(X_scaled))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            yb = model(xb)
            # model puede devolver (B,1) o (B,)
            yb = yb.reshape(-1).detach().cpu().numpy()
            preds.append(yb)

    y_pred = np.concatenate(preds, axis=0).astype(np.float64)
    return y_pred


def predict_all_features_artifact_to_compare(
    artifact: Dict[str, Any],
    timeframe: str,
    symbols: List[str],
    *,
    price_col: str = "close",
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    device=None,
    batch_size: int = 4096,
    strict: bool = True,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Builds fresh data using the artifact config, builds supervised dataset, applies the saved scaler,
    runs the saved MLP model, and returns pred_df with y_true/y_pred.

    Returns:
      pred_df columns: symbol, timestamp, target_timestamp, y_true, y_pred, error
    """
    cfg = artifact.get("config", {}) or {}

    include_indicators = bool(cfg.get("include_indicators", False))

    # OJO: tu config usa include_economic_indicators, pero build_ml_dataframe usa include_econ
    include_econ = bool(cfg.get("include_econ", cfg.get("include_economic_indicators", False)))
    econ_indicator_names = cfg.get("econ_indicator_names", None)

    include_fmp = bool(cfg.get("include_fmp", False))
    fmp_feature_names = cfg.get("fmp_feature_names", None)
    fmp_prefix = cfg.get("fmp_prefix", "fmp_")
    keep_fmp_asof_date = bool(cfg.get("keep_fmp_asof_date", False))

    # Construye DF con las mismas “familias” de features que el modelo esperaba
    conn = get_connection()
    df = build_ml_dataframe(
        conn,
        symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        include_indicators=include_indicators,

        include_econ=include_econ,
        econ_indicator_names=econ_indicator_names if include_econ else None,

        include_fmp=include_fmp,
        fmp_feature_names=fmp_feature_names,
        fmp_prefix=fmp_prefix,
        keep_fmp_asof_date=keep_fmp_asof_date,
    )

    if df.empty:
        raise ValueError("build_ml_dataframe devolvió df vacío (no hay datos con esos parámetros).")

    # Dataset supervisado (mismo lookback/horizon/features usados al entrenar)
    base_feature_cols = cfg["base_feature_cols"]
    lookback = int(cfg["lookback"])
    horizon = int(cfg["horizon"])

    X, y, meta = build_supervised_dataset(
        df=df,
        feature_cols=base_feature_cols,
        lookback=lookback,
        horizon=horizon,
        price_col=price_col,
        group_col=group_col,
        timestamp_col=timestamp_col,
        lags_by_feature=cfg.get("lags_by_feature", None),
        default_lags=cfg.get("default_lags", None),
    )

    # Predicción con artefacto (alinea columnas + scaler + torch)
    y_pred = predict_with_mlp_artifact(
        artifact,
        X,
        device=device,
        batch_size=batch_size,
        strict=strict,
        fill_value=fill_value,
    )

    # Arma pred_df comparable (igual a tu pred_df_test)
    pred_df = meta.copy()
    pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
    pred_df["target_timestamp"] = pd.to_datetime(pred_df["target_timestamp"])
    pred_df["y_true"] = np.asarray(y, dtype=np.float64).reshape(-1)
    pred_df["y_pred"] = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    pred_df["error"] = pred_df["y_pred"] - pred_df["y_true"]

    return pred_df


# even more generic (full chatgpt)

def _read_json_if_exists(p: Path) -> dict:
    if p.exists():
        with open(p, "r") as f:
            return json.load(f)
    return {}

def _read_json(path: Union[str, Path]) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))

def build_torch_model_from_payload(model_name: str, payload: dict, config: dict) -> torch.nn.Module:
    """
    Central registry for torch model reconstruction.

    Add new model types here (cnn1d_regressor, transformer_regressor, etc.)
    without changing the inference pipeline.
    """
    from machine_learning.models import MLPRegressor
    model_name = str(model_name)

    if model_name in TORCH_MODEL_BUILDERS:
        input_dim = int(payload["input_dim"])
        model = TORCH_MODEL_BUILDERS[model_name](config, input_dim=input_dim) # type: ignore
        model.load_state_dict(payload["state_dict"])
        return model

    if model_name == "mlp_regressor":
        mlp_cfg = payload.get("config", {}) or config.get("mlp_config", {}) or {}
        input_dim = int(payload["input_dim"])

        # hidden_sizes puede venir como list -> conviértelo a tuple
        hs = mlp_cfg.get("hidden_sizes", (64, 32))
        if isinstance(hs, list):
            hs = tuple(hs)

        model = MLPRegressor(
            input_dim=input_dim,
            hidden_sizes=hs,
            dropout=float(mlp_cfg.get("dropout", 0.0)),
            batch_norm=bool(mlp_cfg.get("batch_norm", False)),
        )
        model.load_state_dict(payload["state_dict"])
        return model
    elif model_name == "cnn1d_regressor":
        cnn_cfg = payload.get("config", {}) or config.get("cnn_config", {}) or {}
        input_dim = int(payload["input_dim"])
        n_channels= cnn_cfg.get("n_channels")
        seq_len = cnn_cfg.get("seq_len")
        perm = cnn_cfg.get("perm")
        conv_channels = cnn_cfg.get("conv_channels")
        kernel_size = cnn_cfg.get("kernel_size")
        dilations = cnn_cfg.get("dilations")
        dropout = cnn_cfg.get("dropout")
        use_bn = cnn_cfg.get("use_bn")
        head_hidden = cnn_cfg.get("head_hidden")
        out_dim = cnn_cfg.get("out_dim")
        time_order = cnn_cfg.get("time_order")


        model = CNN1DRegressor(
            input_dim=input_dim,
            n_channels=n_channels, # type: ignore
            seq_len=seq_len, # type: ignore
            perm=perm, # type: ignore
            conv_channels=conv_channels, # type: ignore
            kernel_size=kernel_size, # type: ignore
            dilations=dilations, # type: ignore
            dropout=dropout, # type: ignore
            use_bn=use_bn, # type: ignore
            head_hidden=head_hidden, # type: ignore
            out_dim=out_dim # type: ignore
        )

        model.load_state_dict(payload['state_dict'])
        return model

    # ---- FUTURO: CNN/Transformer ----
    # elif model_name == "cnn1d_regressor":
    #     ...
    # elif model_name == "transformer_regressor":
    #     ...

    raise ValueError(
        f"Unknown torch model_name='{model_name}'.\n"
        f"Add it to build_torch_model_from_payload registry."
    )



# def load_model_artifact_auto(run_dir: str, *, map_location: str = "cpu") -> Dict[str, Any]:
#     """
#     Auto-detects and loads a saved artifact (sklearn pipeline vs torch model).

#     Returns a unified dict with keys:
#       - run_dir, family ("sklearn"|"torch"), model_name, config, metrics, feature_names
#       - plus loaded objects:
#          sklearn: pipeline
#          torch: model, scaler (optional)
#     """
#     p = Path(run_dir)
#     if not p.exists():
#         raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

#     # Prefer config.json if present (helps decide model_name)
#     config = _read_json_if_exists(p / "config.json")
#     metrics = _read_json_if_exists(p / "metrics.json")

#     # --- SKLEARN style: pipeline.joblib exists ---
#     if (p / "pipeline.joblib").exists() and (p / "feature_names.json").exists():
#         art = load_ridge_artifact(str(p))
#         cfg = art.get("config", {}) or {}
#         if "model" not in cfg:
#             cfg["model"] = "ridge"
#         art["config"] = cfg
#         art["metrics"] = art.get("metrics", {}) or metrics
#         art["family"] = "sklearn"
#         art["model_name"] = cfg.get("model", "ridge")
#         return art

#     # --- TORCH style: model.pt exists ---
#     if (p / "model.pt").exists():
#         feat_path = p / "feature_names.json"
#         if not feat_path.exists():
#             raise FileNotFoundError(f"Missing {feat_path}")

#         with open(feat_path, "r") as f:
#             feature_names = json.load(f)

#         payload = torch.load(p / "model.pt", map_location=map_location)

#         # Determine model_name
#         model_name = config.get("model", None)
#         if model_name is None:
#             # fallback: if payload stores something later
#             model_name = payload.get("model_name", "torch_model")

#         # Load scaler if present
#         scaler = None
#         scaler_path = p / "scaler.joblib"
#         if scaler_path.exists():
#             scaler = joblib.load(scaler_path)

#         # Build torch model via registry
#         model = build_torch_model_from_payload(model_name, payload, config)

#         model.eval()

#         return {
#             "run_dir": str(p),
#             "family": "torch",
#             "model_name": model_name,
#             "model": model,
#             "scaler": scaler,
#             "feature_names": list(feature_names),
#             "config": config,
#             "metrics": metrics,
#             "payload": payload,  # opcional (útil para debug)
#         }

#     raise ValueError(
#         f"Could not detect artifact type in: {run_dir}\n"
#         f"Expected either (pipeline.joblib + feature_names.json) OR model.pt."
#     )

def _cfg_get(cfg: dict, key: str, default=None, aliases: Optional[List[str]] = None):
    if key in cfg:
        return cfg[key]
    if aliases:
        for a in aliases:
            if a in cfg:
                return cfg[a]
    return default

from numbers import Integral

def _as_lag_list(v, *, fallback_lookback: int):
    """
    Normaliza especificación de lags:
      - int => range(int)
      - iterable => list(int)
      - str => json o csv o número
      - None => None
    """
    if v is None:
        return None

    # int / numpy int
    if isinstance(v, Integral):
        n = int(v)
        return list(range(n))

    # strings: "60", "[0,1,2]", "0,1,2"
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return None
        # "60"
        if s.lstrip("+-").isdigit():
            return list(range(int(s)))
        # "[0,1,2]" o "60" via json
        try:
            return _as_lag_list(json.loads(s), fallback_lookback=fallback_lookback)
        except Exception:
            # "0,1,2"
            parts = [p.strip() for p in s.split(",") if p.strip()]
            if len(parts) == 1 and parts[0].lstrip("+-").isdigit():
                return list(range(int(parts[0])))
            return [int(p) for p in parts]

    # iterables
    try:
        return [int(x) for x in v]
    except TypeError:
        # valor escalar raro
        return [int(v)]


def build_Xy_meta_from_artifact_config(
    artifact: Dict[str, Any],
    *,
    timeframe: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    price_col: str = "close",
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    conn=None,
    enforce_timeframe_match: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Builds X, y, meta using the artifact's config (indicators/econ/fmp, lookback/horizon, lags spec).
    """
    cfg = artifact.get("config", {}) or {}

    cfg_timeframe = _cfg_get(cfg, "timeframe", None)
    if timeframe is None:
        timeframe = cfg_timeframe
    if timeframe is None:
        raise ValueError("timeframe must be provided (or exist in artifact config).")

    if enforce_timeframe_match and cfg_timeframe is not None and str(cfg_timeframe) != str(timeframe):
        raise ValueError(f"timeframe mismatch: artifact={cfg_timeframe} vs requested={timeframe}")

    if symbols is None:
        symbols = _cfg_get(cfg, "symbols", None)
    if not symbols:
        raise ValueError("symbols must be provided (or exist in artifact config).")

    include_indicators = bool(_cfg_get(cfg, "include_indicators", False))
    indicator_names = _cfg_get(cfg, 'indicators_used', ['indicator_names'])
    include_econ = bool(_cfg_get(cfg, "include_econ", False, aliases=["include_economic_indicators"]))
    econ_names = _cfg_get(cfg, "econ_indicator_names", None)
    include_fmp = bool(_cfg_get(cfg, "include_fmp", False))
    fmp_feature_names = _cfg_get(cfg, "fmp_feature_names", None)
    fmp_prefix = _cfg_get(cfg, "fmp_prefix", "fmp_")
    keep_fmp_asof_date = bool(_cfg_get(cfg, "keep_fmp_asof_date", False))

    base_feature_cols = _cfg_get(cfg, "base_feature_cols", None)
    if not base_feature_cols:
        raise ValueError("artifact config missing base_feature_cols")

    lookback = int(_cfg_get(cfg, "lookback"))
    horizon = int(_cfg_get(cfg, "horizon"))

    lags_by_feature = _cfg_get(cfg, "lags_by_feature", None)
    default_lags = _cfg_get(cfg, "default_lags", None)

    if conn is None:
        conn = get_connection()

    df = build_ml_dataframe(
        conn,
        symbols=symbols,
        timeframe=timeframe,
        start=start,
        end=end,
        include_indicators=include_indicators,
        include_econ=include_econ,
        econ_indicator_names=econ_names if include_econ else None,
        indicator_names=indicator_names,
        include_fmp=include_fmp,
        fmp_feature_names=fmp_feature_names,
        fmp_prefix=fmp_prefix,
        keep_fmp_asof_date=keep_fmp_asof_date,
    )


    if df.empty:
        raise ValueError("build_ml_dataframe returned empty df (no data).")

    X, y, meta = build_supervised_dataset(
        df=df,
        feature_cols=base_feature_cols,
        lookback=lookback,
        horizon=horizon,
        price_col=price_col,
        group_col=group_col,
        timestamp_col=timestamp_col,
        lags_by_feature=lags_by_feature,
        default_lags=default_lags,
    )

    return X, y, meta


from torch.utils.data import DataLoader, TensorDataset

import numpy as np
import pandas as pd
import torch
from typing import Any, Dict, Optional, Sequence, Union
from torch.utils.data import DataLoader, TensorDataset

_SEQ_MODELS = {"tcn_regressor", "cnn1d_regressor", "transformer_regressor"}


def _wide_lagged_to_3d(
    X_wide: pd.DataFrame,
    base_features: Sequence[str],
    lookback: int,
    *,
    time_order: str = "oldest_to_newest",
    fill_value: float = 0.0,
    strict: bool = True,
) -> np.ndarray:
    """
    X_wide columns expected: {feat}_lag0..{feat}_lag{lookback-1}.
    Returns: (N, F, L) float32
    time_order:
      - oldest_to_newest: L axis = lag=lookback-1 ... lag=0
      - newest_to_oldest: L axis = lag=0 ... lag=lookback-1
    """
    base = list(base_features)
    L = int(lookback)
    if time_order not in {"oldest_to_newest", "newest_to_oldest"}:
        time_order = "oldest_to_newest"

    lags = list(range(L - 1, -1, -1)) if time_order == "oldest_to_newest" else list(range(0, L))
    required = [f"{f}_lag{k}" for k in lags for f in base]  # note: time-major then features

    missing = [c for c in required if c not in X_wide.columns]
    if missing and strict:
        raise ValueError(f"Faltan lag cols para secuencia. Ej: {missing[:10]}")
    if missing and not strict:
        X_wide = X_wide.copy()
        for c in missing:
            X_wide[c] = float(fill_value)

    X_block = (
        X_wide.loc[:, required]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(fill_value)
        .to_numpy(dtype=np.float32, copy=False)
    )  # (N, L*F)

    N = X_block.shape[0]
    F = len(base)
    X_block = X_block.reshape(N, L, F)          # (N, L, F)
    X3 = np.transpose(X_block, (0, 2, 1))       # (N, F, L)
    return X3


def _scale_sequence_3d(scaler: Any, X3: np.ndarray, *_, **kwargs) -> np.ndarray:
    """
    Compatible con:
      - scalers que aceptan 3D directo
      - scalers entrenados en (N*L, F) -> esperan F features
      - scalers entrenados en (N, F*L) -> esperan F*L features (tu CNN viejo)
    kwargs opcionales para alinear por nombres:
      - base_features: list[str]
      - lookback: int
      - time_order: "oldest_to_newest" | "newest_to_oldest"
      - wide_feature_names: list[str] (orden usado en entrenamiento, si no hay feature_names_in_)
    """
    X3 = np.asarray(X3, dtype=np.float32)
    if scaler is None:
        return X3

    N, F, L = X3.shape

    # 0) custom scalers que soportan 3D
    try:
        out = scaler.transform(X3)
        out = np.asarray(out, dtype=np.float32)
        if out.shape == X3.shape:
            return out
    except Exception:
        pass

    # 1) cuántas features espera el scaler
    n_in = getattr(scaler, "n_features_in_", None)
    if n_in is None and hasattr(scaler, "mean_"):
        try:
            n_in = int(len(scaler.mean_))
        except Exception:
            n_in = None

    base_features = kwargs.get("base_features")
    time_order = str(kwargs.get("time_order", "oldest_to_newest"))
    wide_feature_names = kwargs.get("wide_feature_names")

    # Si sklearn guardó nombres, eso es lo más confiable
    fn_in = getattr(scaler, "feature_names_in_", None)
    if fn_in is not None:
        wide_feature_names = list(map(str, fn_in))

    # A) scaler entrenado en wide (espera F*L)
    if n_in == F * L:
        # nuestro flatten natural (feature-major sobre X3): (N, F*L)
        Xw = np.ascontiguousarray(X3).reshape(N, F * L)

        # Si tenemos nombres, reordenamos por nombre para que el scaler reciba exactamente su orden
        if (base_features is not None) and (wide_feature_names is not None) and (len(wide_feature_names) == F * L):
            # Nombres en el orden "nuestro" (el que corresponde a Xw)
            # OJO: L axis en X3 depende de time_order usado al armar X3.
            lag_nums = list(range(L - 1, -1, -1)) if time_order == "oldest_to_newest" else list(range(0, L))
            ours_names = [f"{f}_lag{k}" for f in list(base_features) for k in lag_nums]

            pos = {name: i for i, name in enumerate(ours_names)}
            idx_map = [pos.get(name, -1) for name in list(map(str, wide_feature_names))]

            if all(i >= 0 for i in idx_map):
                # Pasar a orden del scaler
                Xw_in = Xw[:, idx_map]
                Xw_scaled = np.asarray(scaler.transform(Xw_in), dtype=np.float32)

                # Volver a nuestro orden
                inv = np.empty(len(idx_map), dtype=int)
                for scaler_i, ours_i in enumerate(idx_map):
                    inv[ours_i] = scaler_i
                Xw_scaled = Xw_scaled[:, inv]

                return Xw_scaled.reshape(N, F, L)

        # fallback: asumir que el orden coincide
        Xw_scaled = np.asarray(scaler.transform(Xw), dtype=np.float32)
        return Xw_scaled.reshape(N, F, L)

    # B) scaler per-feature (espera F)
    if n_in == F or n_in is None:
        X2 = np.transpose(X3, (0, 2, 1)).reshape(-1, F)  # (N*L, F)
        X2t = scaler.transform(X2)
        Xt = np.asarray(X2t, dtype=np.float32).reshape(N, L, F).transpose(0, 2, 1)
        return Xt

    # C) último recurso
    try:
        Xw = np.ascontiguousarray(X3).reshape(N, F * L)
        Xw_scaled = scaler.transform(Xw)
        return np.asarray(Xw_scaled, dtype=np.float32).reshape(N, F, L)
    except Exception:
        X2 = np.transpose(X3, (0, 2, 1)).reshape(-1, F)
        X2t = scaler.transform(X2)
        return np.asarray(X2t, dtype=np.float32).reshape(N, L, F).transpose(0, 2, 1)


def _is_cnn1d_model(model, cfg: dict, model_name: str) -> bool:
    cls = model.__class__.__name__.lower()
    if "cnn1d" in cls:
        return True
    mn = str(model_name or "").lower()
    if "cnn" in mn:
        return True
    if isinstance(cfg, dict) and ("cnn_config" in cfg):
        return True
    return False


def _expected_wide_lag_cols_for_seq(
    *,
    artifact: dict,
    scaler: object,
    base_features: list[str],
    lookback: int,
) -> list[str]:
    # 1) Si el scaler fue fit con DataFrame, sklearn guarda feature_names_in_
    if scaler is not None and hasattr(scaler, "feature_names_in_"):
        cols = [str(c) for c in list(scaler.feature_names_in_)]
        if len(cols) == len(base_features) * int(lookback):
            return cols

    # 2) Si el artifact guardó feature_names (normal en CNN viejos), usar eso
    feat = artifact.get("feature_names")
    if isinstance(feat, (list, tuple)):
        feat = [str(c) for c in feat]
        if len(feat) == len(base_features) * int(lookback):
            return feat

    # 3) Fallback: feature-major, lag ascendente
    L = int(lookback)
    return [f"{f}_lag{k}" for f in base_features for k in range(L)]


def predict_with_loaded_artifact(
    artifact: Dict[str, any],
    X: pd.DataFrame,
    *,
    device: Optional[torch.device] = None,
    batch_size: int = 8192,
    strict: bool = True,
    fill_value: float = 0.0,
) -> np.ndarray:
    """
    Predicts y_hat from X using loaded artifact, regardless of family.
    - sklearn: uses predict_with_ridge_artifact(...) if you have it, else _predict_generic
    - torch:
        * MLP-like: align -> scaler -> forward on 2D
        * TCN-like: reshape wide-lags -> 3D -> scaler -> forward (N, out_dim)
    """
    family = artifact.get("family", None)
    if family is None:
        raise ValueError("artifact missing 'family' key (use load_model_artifact_auto).")

    if family == "sklearn":
        # keep your ridge pipeline behavior if you have it
        if "pipeline" in artifact:
            yhat = artifact["pipeline"].predict(X)
            return np.asarray(yhat, dtype=np.float64)
        # fallback
        yhat = _predict_generic(artifact.get("model"), X)
        return np.asarray(yhat, dtype=np.float64)

    if family != "torch":
        raise ValueError(f"Unknown artifact family: {family}")


    cfg = artifact.get("config", {}) or {}
    model_name = artifact.get("model_name", cfg.get("model", "torch_model"))
    model = artifact["model"]
    scaler = artifact.get("scaler", None)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    # ---- SEQUENCE MODELS (TCN/CNN/etc.) ----
    if str(model_name) in _SEQ_MODELS:
        lookback = int(_cfg_get(cfg, "lookback", None) or (_parse_lb_h_from_dirname(artifact["run_dir"])[0] or 0))
        if lookback <= 0:
            raise ValueError("No pude inferir lookback (config['lookback'] faltante).")

        time_order = _cfg_get(cfg, "seq_order", "oldest_to_newest", aliases=["time_order", "sequence_layout"])

        # base_features: SIEMPRE desde config para modelos seq
        base_features = cfg.get("base_feature_cols") or cfg.get("base_features")
        if not base_features:
            # último recurso: inferir de columnas laggeadas del X
            base_features = _infer_base_features_from_lagged_cols(list(X.columns))
        base_features = list(base_features)

        # --- CNN1DRegressor: input 2D (B, C*T) ---
        if _is_cnn1d_model(model, cfg, model_name):
            wide_cols = _expected_wide_lag_cols_for_seq(
                artifact=artifact,
                scaler=scaler,
                base_features=base_features,
                lookback=lookback,
            )

            Xw = (
                X.reindex(columns=wide_cols, fill_value=fill_value)
                .replace([np.inf, -np.inf], np.nan)
                .fillna(fill_value)
            )

            if scaler is not None:
                X_np = np.asarray(scaler.transform(Xw), dtype=np.float32)
            else:
                X_np = Xw.to_numpy(dtype=np.float32, copy=False)

            ds = TensorDataset(torch.from_numpy(X_np))
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

            preds = []
            with torch.no_grad():
                for (xb,) in loader:
                    xb = xb.to(device)
                    yb = model(xb)  # <-- 2D OK
                    preds.append(yb.detach().cpu().numpy())

            y = np.concatenate(preds, axis=0)
            return np.asarray(y, dtype=np.float64)

        # --- TCN (u otros que sí esperan 3D): input (B, C, T) ---
        X3 = _wide_lagged_to_3d(
            X,
            base_features=base_features,
            lookback=lookback,
            time_order=time_order,
            strict=strict,
            fill_value=fill_value,
        )
        X3 = _scale_sequence_3d(
            scaler,
            X3,
            base_features=base_features,
            time_order=time_order,
            wide_feature_names=artifact.get("feature_names"),
        )

        ds = TensorDataset(torch.from_numpy(X3))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

        preds = []
        with torch.no_grad():
            for (xb,) in loader:
                xb = xb.to(device)
                yb = model(xb)  # <-- 3D OK para TCN
                preds.append(yb.detach().cpu().numpy())

        y = np.concatenate(preds, axis=0)
        return np.asarray(y, dtype=np.float64)



# def predict_artifact_to_compare(
#     run_dir: str,
#     *,
#     timeframe: Optional[str] = None,
#     symbols: Optional[List[str]] = None,
#     start: Optional[Union[str, pd.Timestamp]] = None,
#     end: Optional[Union[str, pd.Timestamp]] = None,
#     price_col: str = "close",
#     group_col: str = "symbol",
#     timestamp_col: str = "timestamp",
#     device: Optional[torch.device] = None,
#     batch_size: int = 8192,
#     strict: bool = True,
#     fill_value: float = 0.0,
#     conn=None,
#     enforce_timeframe_match: bool = True,
# ):
#     """
#     One-stop function:
#       - auto-load artifact
#       - rebuild df + supervised dataset according to config
#       - predict
#       - return pred_df (meta + y_true + y_pred + error)
#     """
#     artifact = load_model_artifact_auto(run_dir)

#     X, y, meta = build_Xy_meta_from_artifact_config(
#         artifact,
#         timeframe=timeframe,
#         symbols=symbols,
#         start=start,
#         end=end,
#         price_col=price_col,
#         group_col=group_col,
#         timestamp_col=timestamp_col,
#         conn=conn,
#         enforce_timeframe_match=enforce_timeframe_match,
#     )

#     y_pred = predict_with_loaded_artifact(
#         artifact,
#         X,
#         device=device,
#         batch_size=batch_size,
#         strict=strict,
#         fill_value=fill_value,
#     )

#     pred_df = meta.copy()
#     pred_df["timestamp"] = pd.to_datetime(pred_df["timestamp"])
#     pred_df["target_timestamp"] = pd.to_datetime(pred_df["target_timestamp"])
#     pred_df["y_true"] = np.asarray(y, dtype=np.float64).reshape(-1)
#     pred_df["y_pred"] = np.asarray(y_pred, dtype=np.float64).reshape(-1)
#     pred_df["run_dir"] = run_dir
#     pred_df["model_name"] = artifact.get("model_name", "unknown")
#     pred_df["family"] = artifact.get("family", "unknown")

#     return pred_df, meta


# from typing import Any, Dict, List, Optional, Union
# import numpy as np
# import pandas as pd

# def predict_artifact_live(
#     run_dir: str,
#     *,
#     timeframe: Optional[str] = None,
#     symbols: Optional[List[str]] = None,
#     start: Optional[Union[str, pd.Timestamp]] = None,
#     end: Optional[Union[str, pd.Timestamp]] = None,
#     price_col: str = "close",
#     group_col: str = "symbol",
#     timestamp_col: str = "timestamp",
#     select: str = "latest",          # "latest" | "tail" | "all"
#     tail_n: int = 1,
#     add_prices: bool = True,
#     add_ranks: bool = True,
#     q: float = 0.1,
#     device=None,
#     batch_size: int = 8192,
#     strict: bool = True,
#     fill_value: float = 0.0,
# ) -> pd.DataFrame:
#     """
#     Live/paper inference:
#       - returns predictions even for the most recent rows where y_true does not exist yet.

#     Output columns:
#       symbol, timestamp, target_timestamp_est, y_pred, (optional) close, pred_price, rank, bucket
#     """
#     artifact = load_model_artifact_auto(run_dir)
#     cfg = artifact.get("config", {}) or {}

#     # timeframe/symbols fallback to artifact config
#     cfg_timeframe = cfg.get("timeframe", None)
#     if timeframe is None:
#         timeframe = cfg_timeframe
#     if timeframe is None:
#         raise ValueError("timeframe must be provided or exist in artifact config.")

#     if symbols is None:
#         symbols = cfg.get("symbols", None)
#     if not symbols:
#         raise ValueError("symbols must be provided or exist in artifact config.")

#     include_indicators = bool(_cfg_get(cfg, "include_indicators", False))
#     indicator_names = _cfg_get(cfg, 'indicators_used', ['indicator_names'])
#     include_econ = bool(_cfg_get(cfg, "include_econ", False, aliases=["include_economic_indicators"]))
#     econ_names = _cfg_get(cfg, "econ_indicator_names", None)
#     include_fmp = bool(_cfg_get(cfg, "include_fmp", False))
#     fmp_feature_names = _cfg_get(cfg, "fmp_feature_names", None)
#     fmp_prefix = _cfg_get(cfg, "fmp_prefix", "fmp_")
#     keep_fmp_asof_date = bool(_cfg_get(cfg, "keep_fmp_asof_date", False))
    
#     base_feature_cols = cfg.get("base_feature_cols", None)
#     if not base_feature_cols:
#         raise ValueError("artifact config missing base_feature_cols")

#     lookback = int(cfg.get("lookback"))
#     horizon = int(cfg.get("horizon"))

#     # 1) Build df with same feature families as training
#     conn = get_connection()
#     df = build_ml_dataframe(
#         conn,
#         symbols=symbols,
#         timeframe=timeframe,
#         start=start,
#         end=end,
#         include_indicators=include_indicators,
#         include_econ=include_econ,
#         econ_indicator_names=econ_names if include_econ else None,
#         indicator_names=indicator_names,
#         include_fmp=include_fmp,
#         fmp_feature_names=fmp_feature_names,
#         fmp_prefix=fmp_prefix,
#         keep_fmp_asof_date=keep_fmp_asof_date,
#     )
#     if df.empty:
#         raise ValueError("build_ml_dataframe returned empty df")

#     # 2) Build X/meta for inference (no y)
#     X_live, meta_live = build_inference_dataset(
#         df=df,
#         feature_cols=base_feature_cols,
#         lookback=lookback,
#         group_col=group_col,
#         timestamp_col=timestamp_col,
#         lags_by_feature=cfg.get("lags_by_feature", None),
#         default_lags=cfg.get("default_lags", None),
#         select=select,
#         tail_n=tail_n,
#     )

#     # 3) Predict
#     y_pred = predict_with_loaded_artifact(
#         artifact,
#         X_live,
#         device=device,
#         batch_size=batch_size,
#         strict=strict,
#         fill_value=fill_value,
#     )

#     pred_df = meta_live.copy()
#     pred_df[timestamp_col] = pd.to_datetime(pred_df[timestamp_col])
#     pred_df["y_pred"] = np.asarray(y_pred, dtype=np.float64).reshape(-1)

#     # target timestamp estimate (for reporting)
#     pred_df["target_timestamp_est"] = estimate_target_timestamp(pred_df[timestamp_col], timeframe, horizon)
#     pred_df["horizon"] = horizon
#     pred_df["timeframe"] = timeframe
#     pred_df["run_dir"] = run_dir
#     pred_df["model_name"] = artifact.get("model_name", "unknown")

#     # 4) Add current price + implied future price (since y_pred is log-return)
#     if add_prices and price_col in df.columns:
#         px = df[[group_col, timestamp_col, price_col]].copy()
#         px[timestamp_col] = pd.to_datetime(px[timestamp_col])
#         pred_df = pred_df.merge(px, on=[group_col, timestamp_col], how="left")
#         pred_df["pred_price"] = pred_df[price_col] * np.exp(pred_df["y_pred"])

#     # 5) Add ranks/buckets to trade (optional)
#     if add_ranks:
#         # rank within each timestamp (for select="all" or "tail")
#         pred_df["rank"] = pred_df.groupby(timestamp_col)["y_pred"].rank(ascending=False, method="first")
#         pred_df["rank_pct"] = pred_df.groupby(timestamp_col)["y_pred"].rank(pct=True, ascending=False)

#         # simple bucket: long/top q, short/bottom q, else neutral
#         if q is not None and 0.0 < q < 0.5:
#             pred_df["bucket"] = "neutral"
#             pred_df.loc[pred_df["rank_pct"] <= q, "bucket"] = "long"
#             pred_df.loc[pred_df["rank_pct"] >= (1.0 - q), "bucket"] = "short"

#     return pred_df

# TCN

def make_run_dir(base_dir: Union[str, Path] = "runs", run_name: Optional[str] = None) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    name = run_name or f"tcn_{ts}"
    run_dir = base / name
    i = 0
    final = run_dir
    while final.exists():
        i += 1
        final = base / f"{name}_{i}"
    final.mkdir(parents=True, exist_ok=False)
    return final

import time, platform

def save_tcn_artifact(
    run_dir: Union[str, Path],
    *,
    config: Dict[str, Any],
    model: torch.nn.Module,
    scaler: Any,
    feature_names: Sequence[str],
    horizons: Sequence[int],
    metrics: Dict[str, Any],
    training_history: Optional[Sequence[Dict[str, Any]]] = None,
) -> Path:
    """
    Files created:
      - config.json
      - metrics.json
      - feature_names.json
      - model.pt
      - scaler.pkl
      - training_history.json (optional)
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = dict(config)
    config.setdefault("artifact_type", "torch")
    config.setdefault("framework", "pytorch")
    config.setdefault("model_type", "tcn_regressor")
    config.setdefault("created_at", time.strftime("%Y-%m-%d %H:%M:%S"))
    config.setdefault("python_version", platform.python_version())
    config.setdefault("platform", platform.platform())
    config.setdefault("torch_version", torch.__version__)
    config.setdefault("device", str(next(model.parameters()).device) if any(True for _ in model.parameters()) else "unknown")
    config.setdefault("horizons", list(map(int, horizons)))

    model_path = run_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    scaler_path = run_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)

    feat_path = run_dir / "feature_names.json"
    with feat_path.open("w", encoding="utf-8") as f:
        json.dump({"feature_names": list(feature_names)}, f, ensure_ascii=False, indent=2)

    cfg_path = run_dir / "config.json"
    with cfg_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2, default=_jsonify)

    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=_jsonify)

    if training_history is not None:
        hist_path = run_dir / "training_history.json"
        with hist_path.open("w", encoding="utf-8") as f:
            json.dump(list(training_history), f, ensure_ascii=False, indent=2, default=_jsonify)

    return run_dir

from tcn.models.tcn import TCNConfig, TCNRegressor
import importlib
import sys
import types
def _alias_module_for_pickle(old_module: str, candidates: list[str]) -> bool:
    """
    Hace que `old_module` sea importable aliasándolo al primer candidate existente.
    Útil para cargar pickles viejos cuando cambió el layout del proyecto.
    """
    for cand in candidates:
        try:
            mod = importlib.import_module(cand)
        except Exception:
            continue

        parts = old_module.split(".")
        if len(parts) > 1:
            pkg_name = parts[0]
            if pkg_name not in sys.modules:
                pkg = types.ModuleType(pkg_name)
                pkg.__path__ = []  # marcar como package
                sys.modules[pkg_name] = pkg
            else:
                pkg = sys.modules[pkg_name]
                if not hasattr(pkg, "__path__"):
                    pkg.__path__ = []  # type: ignore[attr-defined]

        sys.modules[old_module] = mod
        if len(parts) > 1:
            setattr(sys.modules[parts[0]], parts[1], mod)

        return True

    return False


def load_tcn_artifact(run_dir: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
    
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.json"
    model_path = run_dir / "model.pt"
    scaler_path = run_dir / "scaler.pkl"
    feat_path = run_dir / "feature_names.json"

    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config.json in {run_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.pt in {run_dir}")
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler.pkl in {run_dir}")
    if not feat_path.exists():
        raise FileNotFoundError(f"Missing feature_names.json in {run_dir}")

    config = json.loads(cfg_path.read_text(encoding="utf-8"))
    feat_obj = json.loads(feat_path.read_text(encoding="utf-8"))
    feature_names = feat_obj.get("feature_names", [])

    model_cfg = TCNConfig(
        num_features=int(config["num_features"]),
        channels=tuple(config["channels"]),
        kernel_size=int(config["kernel_size"]),
        dropout=float(config.get("dropout", 0.0)),
        output_dim=int(config["output_dim"]),
    )
    model = TCNRegressor(model_cfg)
    state = torch.load(model_path, map_location=map_location)
    model.load_state_dict(state)
    model.eval()

    scaler = None
    if scaler_path.exists():
        try:
            scaler = joblib.load(scaler_path)
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", "")
            if missing in {"data.seq_dataset", "data"}:
                ok = _alias_module_for_pickle(
                    "data.seq_dataset",
                    candidates=[
                        "machine_learning.data.seq_dataset",
                        "tcn.data.seq_dataset",
                        # agrega aquí el path real si es otro
                    ],
                )
                if ok:
                    scaler = joblib.load(scaler_path)
                else:
                    raise
            else:
                raise


    return {
        "model": model,
        "scaler": scaler,
        "config": config,
        "feature_names": feature_names,
        "run_dir": run_dir,
    }



def load_model_artifact_auto(run_dir: Union[str, Path], map_location: str = "gpu") -> Dict[str, Any]:
    """
    Auto-detect loader (arreglado):
    - TCN: solo si config indica explícitamente tcn_regressor (no por "framework == pytorch").
    - PyTorch genérico: si existe model.pt (CNN incluidos).
    - Sklearn/joblib/pkl: si existe algún artefacto joblib/pkl.
    - Sklearn ridge legacy: pipeline.joblib + feature_names.json via load_ridge_artifact.
    """
    run_dir = Path(run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"run_dir does not exist: {run_dir}")

    config = _read_json_if_exists(run_dir / "config.json")
    metrics = _read_json_if_exists(run_dir / "metrics.json")

    model_pt = run_dir / "model.pt"

    # 1) TORCH primero si existe model.pt (porque tu bug venía de aquí)
    if model_pt.exists():
        model_type = str(config.get("model_type") or "").lower()
        # Compatibilidad con configs viejas que guardaban "model": "tcn_regressor"
        legacy_model = str(config.get("model") or "").lower()

        if model_type == "tcn_regressor" or legacy_model == "tcn_regressor":
            art = load_tcn_artifact(run_dir, map_location=map_location)
            # Normaliza un poco por si tu load_tcn_artifact no lo incluye
            if isinstance(art, dict):
                art.setdefault("run_dir", str(run_dir))
                art.setdefault("family", "torch")
                art.setdefault("framework", "pytorch")
                art.setdefault("config", config)
                art.setdefault("metrics", metrics)
            return art

        # Si NO es TCN => loader genérico (CNN, etc.)
        return load_torch_artifact(run_dir, map_location=map_location, config=config, metrics=metrics)

    # 2) SKLEARN legacy (tu caso viejo de ridge)
    if (run_dir / "pipeline.joblib").exists() and (run_dir / "feature_names.json").exists():
        art = load_ridge_artifact(str(run_dir))
        cfg = art.get("config", {}) or {}
        if "model" not in cfg:
            cfg["model"] = "ridge"
        art["config"] = cfg
        art["metrics"] = art.get("metrics", {}) or metrics
        art["family"] = "sklearn"
        art["framework"] = "sklearn"
        art["model_name"] = cfg.get("model", "ridge")
        # alias útil para compatibilidad
        if "model" not in art and "pipeline" in art:
            art["model"] = art["pipeline"]
        return art

    # 3) SKLEARN genérico (pkl/joblib)
    for fname in ("pipeline.pkl", "pipeline.joblib", "model.pkl", "model.joblib", "artifact.pkl", "artifact.joblib"):
        path = run_dir / fname
        if path.exists():
            obj = joblib.load(path)
            out: Dict[str, Any] = {
                "run_dir": str(run_dir),
                "family": "sklearn",
                "framework": "sklearn",
                "model": obj,
                "config": config,
                "metrics": metrics,
            }
            # opcional: feature_names si existe
            feat_path = run_dir / "feature_names.json"
            if feat_path.exists():
                out["feature_names"] = _read_feature_names(feat_path)
            return out

    raise FileNotFoundError(
        f"Could not auto-detect artifact format in {run_dir}. "
        "Found neither model.pt (torch) nor any known sklearn .pkl/.joblib files."
    )

def verify_artifact_roundtrip(
    saved_run_dir: Union[str, Path],
    *,
    model: torch.nn.Module,
    scaler: Any,
    X_unscaled: np.ndarray,
    atol: float = 1e-6,
    rtol: float = 0.0,
    map_location: str = "cpu",
) -> Dict[str, float]:
    """
    Verificación obligatoria:
      - pred_before (CPU, eval)
      - load_model_artifact_auto
      - pred_after  (CPU, eval)
      - compara diffs; falla duro si excede tolerancia
    """
    # pred_before
    model_cpu = model.to("cpu")
    model_cpu.eval()
    X_scaled = scaler.transform(X_unscaled)
    with torch.no_grad():
        pred_before = model_cpu(torch.from_numpy(X_scaled)).cpu().numpy()

    # load -> pred_after
    artifact = load_model_artifact_auto(saved_run_dir, map_location=map_location)
    loaded_model = artifact["model"].to("cpu")
    loaded_model.eval()
    loaded_scaler = artifact["scaler"]
    X_scaled2 = loaded_scaler.transform(X_unscaled)

    with torch.no_grad():
        pred_after = loaded_model(torch.from_numpy(X_scaled2)).cpu().numpy()

    diff = np.abs(pred_before - pred_after)
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))

    if not np.allclose(pred_before, pred_after, rtol=rtol, atol=atol):
        raise RuntimeError(
            f"[ARTIFACT VERIFY FAIL] max_abs_diff={max_abs:.8g} mean_abs_diff={mean_abs:.8g} "
            f"(tolerancia atol={atol}, rtol={rtol})"
        )

    return {"max_abs_diff": max_abs, "mean_abs_diff": mean_abs}

def _atomic_write_text(path: Union[str, Path], text: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _atomic_joblib_dump(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp)
    os.replace(tmp, path)


def _atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)

def finalize_run_dir_torch(
    run_dir: str,
    *,
    config: Dict[str, Any],
    metrics: Dict[str, Any],
    feature_names: List[str],           # IMPORTANT: base features for TCN
    scaler: Any,
    model_name: str,
    last_state_dict: Dict[str, Any],
    best_state_dict: Optional[Dict[str, Any]] = None,
    training_history: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """
    Bulletproof save compatible with load_model_artifact_auto():

    Saves:
      - checkpoints/last.pt ALWAYS
      - checkpoints/best.pt if provided
      - model.pt = best if exists else last
      - config.json, metrics.json, training_history.json
      - scaler.joblib
      - feature_names.json as JSON LIST (required by your loader)
    """
    rd = Path(run_dir)
    rd.mkdir(parents=True, exist_ok=True)

    cfg = dict(config or {})
    cfg.setdefault("model", model_name)
    cfg.setdefault("saved_at", time.strftime("%Y-%m-%d %H:%M:%S"))

    if not feature_names or not isinstance(feature_names, list):
        raise ValueError("feature_names debe ser LIST no vacía (base features para TCN).")
    input_dim = len(feature_names)

    ckpt_dir = rd / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    payload_last = {"state_dict": last_state_dict, "input_dim": input_dim, "model_name": model_name}
    _atomic_torch_save(ckpt_dir / "last.pt", payload_last)

    payload_best = None
    if best_state_dict is not None:
        payload_best = {"state_dict": best_state_dict, "input_dim": input_dim, "model_name": model_name}
        _atomic_torch_save(ckpt_dir / "best.pt", payload_best)

    chosen = payload_best if payload_best is not None else payload_last
    _atomic_torch_save(rd / "model.pt", chosen)

    _atomic_joblib_dump(rd / "scaler.joblib", scaler)

    # IMPORTANT: your load_model_artifact_auto expects JSON LIST here
    _atomic_write_json(rd / "feature_names.json", list(feature_names))

    _atomic_write_json(rd / "config.json", cfg)
    _atomic_write_json(rd / "metrics.json", metrics or {})

    if training_history is not None:
        _atomic_write_json(rd / "training_history.json", training_history)

    return str(rd)


def _import_project_functions():
    """
    Importa funciones del proyecto sin hardcodear demasiado.
    Ajusta estos imports si en tu repo están en otro módulo.
    """
    try:
        from data_collectors import build_ml_dataframe, build_supervised_dataset, get_connection
        from data_collectors import load_model_artifact_auto  # si lo tienes ahí
    except Exception:
        # fallback: algunos proyectos ponen load_model_artifact_auto en machine_learning/artifacts.py viejo
        try:
            from data_collectors import build_ml_dataframe, build_supervised_dataset, get_connection
            from artifacts import load_model_artifact_auto  # fallback común
        except Exception as e:
            raise ImportError(
                "No pude importar build_ml_dataframe/build_supervised_dataset/get_connection/load_model_artifact_auto. "
                "Ajusta _import_project_functions() a tu estructura real."
            ) from e

    return build_ml_dataframe, build_supervised_dataset, get_connection, load_model_artifact_auto

def _call_with_accepted_kwargs(fn, *args, **kwargs):
    """
    Llama fn filtrando kwargs no soportados por su signature.
    Útil porque build_ml_dataframe/build_supervised_dataset suelen cambiar parámetros.
    """
    sig = inspect.signature(fn)
    accepted = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return fn(*args, **accepted)

# --- ADD: TCN builder registration ---

@register_torch_model_builder("tcn_regressor")
def _build_tcn_from_config(config: Dict[str, Any], input_dim: int) -> torch.nn.Module:
    """
    Builder TCN: (config, input_dim) -> nn.Module
    input_dim = n_features base (NO lags).
    """
    import sys
    PROJECT_ROOT = os.path.abspath("..")  # ajusta según tu estructura
    sys.path.append(PROJECT_ROOT)
    from machine_learning.tcn.models.tcn import TCNConfig, TCNRegressor  # ajusta import a tu proyecto

    tcn_cfg = config.get("tcn_config") or {}
    channels = tcn_cfg.get("channels", (64, 64, 64, 64))
    if isinstance(channels, list):
        channels = tuple(channels)
    kernel_size = int(tcn_cfg.get("kernel_size", 3))
    dropout = float(tcn_cfg.get("dropout", 0.10))
    out_dim = int(tcn_cfg.get("out_dim", config.get("output_dim", 1)))

    cfg = TCNConfig(
        num_features=int(input_dim),
        channels=channels,
        kernel_size=kernel_size,
        dropout=dropout,
        output_dim=out_dim,
    )
    return TCNRegressor(cfg)

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------- Timeframe -> target_timestamp estimator ----------
import re
_TIMEFRAME_RE = re.compile(r"^\s*(\d+)\s*([a-zA-Z]+)\s*$")

def estimate_target_timestamp(timestamps, timeframe: str, horizon: int) -> pd.DatetimeIndex:
    """
    - For "1Day": uses BusinessDay (BDay), NOT Timedelta
    - For intraday: uses Timedelta
    Accepts tz-aware timestamps.
    """
    tf = str(timeframe).strip()
    m = _TIMEFRAME_RE.match(tf)
    if m:
        mult = int(m.group(1))
        unit = m.group(2).lower()
    else:
        # fallback: '1d','15m','1h'
        num = "".join([c for c in tf if c.isdigit()])
        unit = "".join([c for c in tf if c.isalpha()]).lower()
        if not num or not unit:
            raise ValueError(f"No puedo parsear timeframe='{timeframe}'")
        mult = int(num)

    idx = pd.DatetimeIndex(pd.to_datetime(timestamps))
    bars = int(horizon) * int(mult)

    if unit in {"d", "day", "days"} or "day" in unit:
        return idx + pd.offsets.BDay(bars)
    if unit in {"h", "hr", "hour", "hours"} or "hour" in unit:
        return idx + pd.Timedelta(hours=bars)
    if unit in {"m", "min", "mins", "minute", "minutes"} or "min" in unit:
        return idx + pd.Timedelta(minutes=bars)

    raise ValueError(f"Unidad timeframe no soportada: {unit}")


# --- machine_learning/artifacts.py (ADD) ---
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


def _resolve_artifact_dir(run_dir: str) -> str:
    """
    Allows passing:
      - the artifact dir itself (contains model.pt or pipeline.joblib)
      - the walk-forward root dir (contains final_model/model.pt)
    """
    p = Path(run_dir)
    if (p / "model.pt").exists() or (p / "pipeline.joblib").exists():
        return str(p)
    if (p / "final_model" / "model.pt").exists():
        return str(p / "final_model")
    if (p / "final_model" / "pipeline.joblib").exists():
        return str(p / "final_model")
    raise FileNotFoundError(
        f"No encuentro artifact en {run_dir}. "
        "Esperaba model.pt/pipeline.joblib aquí o en subdir final_model/."
    )


def _get_horizons_from_cfg(cfg: Dict[str, Any]) -> List[int]:
    hs = _cfg_get(cfg, "horizons_sorted", None, aliases=["horizons"])
    if hs is None:
        h = _cfg_get(cfg, "horizon", None)
        if h is None:
            return []
        return [int(h)]
    return [int(x) for x in list(hs)]


def estimate_target_timestamp(timestamps, timeframe: str, horizon: int) -> pd.DatetimeIndex:
    """
    1Day => BDay, intraday => Timedelta.
    """
    import re
    _RE = re.compile(r"^\s*(\d+)\s*([a-zA-Z]+)\s*$")
    tf = str(timeframe).strip()
    m = _RE.match(tf)
    if m:
        mult = int(m.group(1))
        unit = m.group(2).lower()
    else:
        num = "".join([c for c in tf if c.isdigit()])
        unit = "".join([c for c in tf if c.isalpha()]).lower()
        if not num or not unit:
            raise ValueError(f"No puedo parsear timeframe='{timeframe}'")
        mult = int(num)

    idx = pd.DatetimeIndex(pd.to_datetime(timestamps))
    bars = int(horizon) * int(mult)

    if unit in {"d", "day", "days"} or "day" in unit:
        return idx + pd.offsets.BDay(bars)
    if unit in {"h", "hr", "hour", "hours"} or "hour" in unit:
        return idx + pd.Timedelta(hours=bars)
    if unit in {"m", "min", "mins", "minute", "minutes"} or "min" in unit:
        return idx + pd.Timedelta(minutes=bars)
    raise ValueError(f"Unidad timeframe no soportada: {unit}")

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple

def _build_seq_inference_latest_tail(
    df: pd.DataFrame,
    *,
    base_feature_cols: List[str],
    lookback: int,
    group_col: str,
    timestamp_col: str,
    price_col: str,
    select: str,
    tail_n: int,
    strict: bool,
    fill_value: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    L = int(lookback)
    base_feature_cols = _unique_preserve_order(list(base_feature_cols))
    F = len(base_feature_cols)

    if L <= 0:
        raise ValueError("lookback inválido")
    if select not in {"latest", "tail"}:
        raise ValueError("select debe ser 'latest' o 'tail' para modelos secuenciales.")

    # Seguridad: remover duplicados de columnas UNA vez
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    X_cols = [f"{f}_lag{k}" for f in base_feature_cols for k in range(L)]

    rows = []
    metas = []

    df = df.sort_values([group_col, timestamp_col], kind="mergesort")
    has_price = price_col in df.columns

    for sym, g in df.groupby(group_col, sort=False):
        g = g.reset_index(drop=True)
        n = len(g)

        end_positions = [n - 1] if select == "latest" else list(range(max(0, n - int(tail_n)), n))

        for end_pos in end_positions:
            start_pos = end_pos - (L - 1)
            if start_pos < 0:
                continue

            window = g.iloc[start_pos : end_pos + 1]

            arr = window.loc[:, base_feature_cols].to_numpy(dtype=np.float32, copy=True)

            if arr.shape != (L, F):
                raise ValueError(
                    f"Seq window shape mismatch for symbol={sym}: got {arr.shape}, expected {(L, F)}. "
                    f"base_feature_cols={base_feature_cols}"
                )

            arr[~np.isfinite(arr)] = np.nan
            if strict:
                if np.isnan(arr).any():
                    continue
            else:
                arr = np.nan_to_num(arr, nan=fill_value, posinf=fill_value, neginf=fill_value).astype(np.float32)

            # lag0=newest,... => invert time
            xw = arr[::-1, :].T.reshape(F * L)
            rows.append(xw)

            meta = {group_col: sym, timestamp_col: window[timestamp_col].iloc[-1]}
            if has_price:
                meta[price_col] = window[price_col].iloc[-1]
            metas.append(meta)

    if not rows:
        raise ValueError("No quedaron filas válidas para inferencia (muy poco histórico vs lookback o NaNs).")

    X_live = pd.DataFrame(np.vstack(rows), columns=X_cols, dtype=np.float32)
    meta_live = pd.DataFrame(metas)
    return X_live, meta_live




def build_inference_dataset(
    df: pd.DataFrame,
    *,
    artifact: Dict[str, Any],
    price_col: str = "close",
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    select: str = "latest",   # latest|tail|all
    tail_n: int = 1,
    strict: bool = True,
    fill_value: float = 0.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    cfg = artifact.get("config", {}) or {}
    model_name = artifact.get("model_name", cfg.get("model", "unknown"))
    is_seq = str(model_name) in _SEQ_MODELS

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.sort_values([group_col, timestamp_col], kind="mergesort").reset_index(drop=True)

    # -----------------------------------------
    # SEQ (TCN/CNN): tu lógica con base_feature_cols
    # -----------------------------------------
    if is_seq:
        base_feature_cols = _cfg_get(cfg, "base_feature_cols", None)
        if not base_feature_cols:
            raise ValueError("artifact config missing base_feature_cols (necesario para inferencia seq).")
        base_feature_cols = _unique_preserve_order(list(base_feature_cols))

        lookback = int(_cfg_get(cfg, "lookback", None) or (_parse_lb_h_from_dirname(artifact["run_dir"])[0] or 0))
        if lookback <= 0:
            raise ValueError("No pude inferir lookback para build_inference_dataset.")

        # asegurar base columns exist
        for f in base_feature_cols:
            if f not in df.columns:
                if strict:
                    raise ValueError(f"Falta feature base '{f}' en df_long.")
                df[f] = float(fill_value)

        # recorte mínimo (sin duplicar close)
        keep = [group_col, timestamp_col] + base_feature_cols
        if (price_col in df.columns) and (price_col not in keep):
            keep.append(price_col)
        keep = _unique_preserve_order(keep)
        df = df.loc[:, keep].copy()
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()].copy()

        # fast-path para latest/tail
        if select in {"latest", "tail"}:
            need = lookback if select == "latest" else (lookback + int(tail_n) - 1)
            df_small = (
                df.groupby(group_col, sort=False, group_keys=False)
                  .tail(int(need))
                  .reset_index(drop=True)
            )
            return _build_seq_inference_latest_tail(
                df_small,
                base_feature_cols=base_feature_cols,
                lookback=lookback,
                group_col=group_col,
                timestamp_col=timestamp_col,
                price_col=price_col,
                select=select,
                tail_n=tail_n,
                strict=strict,
                fill_value=fill_value,
            )

        # select=all: aquí puedes dejar tu camino pesado si lo necesitas
        # (no lo reescribo acá para no alargar; tu versión anterior sirve)

        # ... tu camino “pesado” seq ...
        raise ValueError("select='all' en seq no está soportado en esta versión (evita RAM).")

    # -----------------------------------------
    # NO-SEQ (XGB/MLP/tabular): usar feature_names
    # -----------------------------------------
    feature_names = artifact.get("feature_names")
    if not isinstance(feature_names, (list, tuple)) or len(feature_names) == 0:
        raise ValueError("artifact missing feature_names (necesario para modelos no-seq).")
    X_cols = list(map(str, feature_names))

    # Inferir qué columnas base necesitas (incluye indicadores) y qué lags exactos crear
    raw_cols, lags_map = _infer_raw_cols_and_lags_from_feature_names(
        X_cols, group_col=group_col, timestamp_col=timestamp_col
    )

    # asegurar raw cols exist
    for c in raw_cols:
        if c not in df.columns:
            if strict:
                raise ValueError(f"Falta columna base '{c}' requerida por feature_names.")
            df[c] = float(fill_value)

    # recortar a lo necesario (NO solo base_feature_cols)
    keep = [group_col, timestamp_col] + raw_cols
    if (price_col in df.columns) and (price_col not in keep):
        keep.append(price_col)
    keep = _unique_preserve_order(keep)

    df = df.loc[:, keep].copy()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    df = df.sort_values([group_col, timestamp_col], kind="mergesort").reset_index(drop=True)

    # fast-path para latest/tail (evita crear miles de columnas para todo el histórico)
    if select in {"latest", "tail"}:
        # solo necesito hasta max_lag + tail_n filas por símbolo
        max_lag = max((max(v) for v in lags_map.values()), default=0)
        need = (max_lag + 1) if select == "latest" else (max_lag + int(tail_n))
        df_small = (
            df.groupby(group_col, sort=False, group_keys=False)
              .tail(int(need))
              .reset_index(drop=True)
        )
        return _build_wide_inference_latest_tail_nonseq(
            df_small,
            feature_names=X_cols,
            lags_map=lags_map,
            group_col=group_col,
            timestamp_col=timestamp_col,
            price_col=price_col,
            select=select,
            tail_n=tail_n,
            strict=strict,
            fill_value=fill_value,
        )

    # select='all' (camino pesado): crear lags para todos los rows SOLO para lo requerido
    for base, lags in lags_map.items():
        g = df.groupby(group_col, sort=False)[base]
        for k in lags:
            df[f"{base}_lag{k}"] = g.shift(int(k))

    missing = [c for c in X_cols if c not in df.columns]
    if missing and strict:
        raise ValueError(f"Missing {len(missing)} columns for inference. Example: {missing[:10]}")

    # Armar X+meta y filtrar filas inválidas
    X = df.reindex(
        columns=[group_col, timestamp_col] + ([price_col] if price_col in df.columns else []) + X_cols,
        fill_value=fill_value,
    ).copy()

    X_mat = X[X_cols].replace([np.inf, -np.inf], np.nan)
    mask = np.isfinite(X_mat.to_numpy(dtype=np.float64)).all(axis=1)
    X = X.loc[mask].dropna(subset=X_cols).reset_index(drop=True)
    if X.empty:
        raise ValueError("No quedaron filas válidas para inferencia (rango muy corto vs lags).")

    # aplicar select
    if select == "latest":
        idx = X.groupby(group_col, sort=False)[timestamp_col].idxmax()
        X = X.loc[idx].sort_values([group_col, timestamp_col], kind="mergesort").reset_index(drop=True)
    elif select == "tail":
        X = (
            X.sort_values([group_col, timestamp_col], kind="mergesort")
             .groupby(group_col, sort=False)
             .tail(int(tail_n))
             .reset_index(drop=True)
        )
    elif select == "all":
        pass
    else:
        raise ValueError("select debe ser 'latest'|'tail'|'all'")

    meta_cols = [group_col, timestamp_col]
    if price_col in X.columns:
        meta_cols.append(price_col)

    meta_live = X[meta_cols].copy()
    X_live = X[X_cols].copy()
    return X_live, meta_live




def _build_df_long_from_artifact_config(
    artifact: Dict[str, Any],
    *,
    timeframe: Optional[str],
    symbols: Optional[List[str]],
    start: Optional[Union[str, pd.Timestamp]],
    end: Optional[Union[str, pd.Timestamp]],
    conn=None,
) -> pd.DataFrame:
    """
    Uses YOUR build_ml_dataframe with feature families from config.
    """
    cfg = artifact.get("config", {}) or {}
    if conn is None:
        conn = get_connection()

    tf_cfg = _cfg_get(cfg, "timeframe", None)
    tf = timeframe or tf_cfg
    if tf is None:
        raise ValueError("timeframe must be provided (or exist in artifact config).")

    syms = symbols or _cfg_get(cfg, "symbols", None)
    if not syms:
        raise ValueError("symbols must be provided (or exist in artifact config).")

    include_indicators = bool(_cfg_get(cfg, "include_indicators", False))
    include_econ = bool(_cfg_get(cfg, "include_econ", False, aliases=["include_economic_indicators"]))
    include_fmp = bool(_cfg_get(cfg, "include_fmp", False))

    indicator_names = _cfg_get(cfg, "indicator_names", None, aliases=["indicators_used"])
    econ_names = _cfg_get(cfg, "econ_indicator_names", None, aliases=["econ_names"])
    fmp_feature_names = _cfg_get(cfg, "fmp_feature_names", None, aliases=["fmp_names"])

    if include_indicators and not indicator_names:
        raise ValueError("include_indicators=True pero falta indicator_names en config.")
    if include_econ and not econ_names:
        raise ValueError("include_econ=True pero falta econ_indicator_names en config.")
    if include_fmp and not fmp_feature_names:
        raise ValueError("include_fmp=True pero falta fmp_feature_names en config.")

    df = build_ml_dataframe(
        conn,
        symbols=list(syms),
        timeframe=tf,
        start=start,
        end=end,
        include_indicators=include_indicators,
        indicator_names=indicator_names,
        include_econ=include_econ,
        econ_indicator_names=econ_names if include_econ else None,
        include_fmp=include_fmp,
        fmp_feature_names=fmp_feature_names if include_fmp else None,
        fmp_prefix=_cfg_get(cfg, "fmp_prefix", "fmp_"),
        keep_fmp_asof_date=bool(_cfg_get(cfg, "keep_fmp_asof_date", False)),
    )

    if df is None or df.empty:
        raise ValueError("build_ml_dataframe returned empty df.")
    return df


def predict_artifact_live(
    run_dir: str,
    *,
    timeframe: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    price_col: str = "close",
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    select: str = "latest",          # "latest" | "tail" | "all"
    tail_n: int = 1,
    add_prices: bool = True,
    add_ranks: bool = True,
    q: float = 0.1,
    device=None,
    batch_size: int = 8192,
    strict: bool = True,
    fill_value: float = 0.0,
) -> pd.DataFrame:
    """
    Live/paper inference: returns preds even where y_true doesn't exist yet.
    Output long: one row per (symbol, timestamp, horizon).
    """
    art_dir = _resolve_artifact_dir(run_dir)
    artifact = load_model_artifact_auto(art_dir, map_location="cpu")
    cfg = artifact.get("config", {}) or {}

    tf = timeframe or _cfg_get(cfg, "timeframe", None)
    if tf is None:
        raise ValueError("timeframe requerido (no está en config y no se pasó).")

    horizons = _get_horizons_from_cfg(cfg)
    if not horizons:
        raise ValueError("No pude inferir horizons (config['horizons_sorted'] o 'horizon' faltante).")

    df_long = _build_df_long_from_artifact_config(
        artifact,
        timeframe=tf,
        symbols=symbols,
        start=start,
        end=end,
        conn=get_connection(),
    )

    X_live, meta_live = build_inference_dataset(
        df_long,
        artifact=artifact,
        price_col=price_col,
        group_col=group_col,
        timestamp_col=timestamp_col,
        select=select,
        tail_n=tail_n,
        strict=strict,
        fill_value=fill_value,
    )

    if X_live is None or (hasattr(X_live, "__len__") and len(X_live) == 0):
        raise ValueError(
            f"No hay muestras para inferencia live. "
            f"Revisa timeframe/symbols/start/end y que haya lookback suficiente. "
            f"select={select} tail_n={tail_n}"
        )
    y_pred_raw = predict_with_loaded_artifact(
        artifact,
        X_live,
        device=device,
        batch_size=batch_size,
        strict=strict,
        fill_value=fill_value,
    )
    
    # 2) normaliza retorno raro
    if y_pred_raw is None:
        raise ValueError("predict_with_loaded_artifact devolvió None (artifact path o X_live inválido).")

    if isinstance(y_pred_raw, dict):
        # intenta extraer predicciones
        for k in ("y_pred", "pred", "preds", "yhat"):
            if k in y_pred_raw:
                y_pred_raw = y_pred_raw[k]
                break
        else:
            raise ValueError(f"predict_with_loaded_artifact devolvió dict con keys={list(y_pred_raw.keys())}")

    if isinstance(y_pred_raw, (tuple, list)) and len(y_pred_raw) == 2 and not np.isscalar(y_pred_raw[0]):
        # por si devuelve (preds, meta) o similar
        y_pred_raw = y_pred_raw[0]

    y_pred = np.asarray(y_pred_raw, dtype=np.float64)

    # 3) evita el caso ndim==0 (el que te rompe ahora)
    if y_pred.ndim == 0:
        raise ValueError(f"y_pred inválido (ndim=0). type={type(y_pred_raw)} value={y_pred_raw}")

    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    # 4) valida que el output_dim coincide con horizons (si strict)
    if strict and (y_pred.shape[1] not in (1, len(horizons))):
        raise ValueError(
            f"Dimensión de predicción no cuadra: y_pred.shape={y_pred.shape} "
            f"pero horizons={horizons}. "
            f"Esto suele indicar que cargaste un artifact equivocado o el modelo fue guardado con otro output_dim."
        )


    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    ts = pd.to_datetime(meta_live[timestamp_col])

    # Build long output
    out = []
    for j, h in enumerate(horizons):
        yp = y_pred[:, j] if j < y_pred.shape[1] else y_pred[:, 0]
        tgt = estimate_target_timestamp(ts, tf, int(h))

        df_out = pd.DataFrame({
            group_col: meta_live[group_col].astype(str).values,
            timestamp_col: ts.values,
            "target_timestamp_est": tgt.values,
            "y_pred": yp,
            "horizon": int(h),
            "timeframe": str(tf),
            "run_dir": art_dir,
            "model_name": artifact.get("model_name", cfg.get("model", "unknown")),
        })

        if add_prices:
            if price_col not in meta_live.columns:
                raise ValueError(f"add_prices=True pero meta_live no trae '{price_col}'")
            df_out[price_col] = meta_live[price_col].astype(np.float64).values
            df_out["pred_price"] = df_out[price_col].to_numpy(dtype=np.float64) * np.exp(df_out["y_pred"].to_numpy(dtype=np.float64))

        out.append(df_out)

    pred_df = pd.concat(out, ignore_index=True)

    # ranks/buckets cross-sectional por timestamp+horizon
    if add_ranks:
        q = float(q)
        if not (0.0 < q < 0.5):
            raise ValueError("q debe estar en (0,0.5)")

        def _rank_bucket(g: pd.DataFrame) -> pd.DataFrame:
            g = g.copy()
            g["rank"] = g["y_pred"].rank(method="first", ascending=False).astype(int)
            n = len(g)
            g["rank_pct"] = g["rank"] / float(n)
            g["bucket"] = np.where(
                g["rank_pct"] <= q, "long",
                np.where(g["rank_pct"] >= (1.0 - q), "short", "neutral")
            )
            return g

        pred_df = (
            pred_df.groupby([timestamp_col, "horizon"], sort=True, group_keys=False)
                   .apply(_rank_bucket)
                   .reset_index(drop=True)
        )

    return pred_df






# ---------- predict_artifact_to_compare (multi-horizon aware, no leakage) ----------
def predict_artifact_to_compare(
    run_dir: str,
    *,
    timeframe: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    price_col: str = "close",
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    device: Optional[torch.device] = None,
    batch_size: int = 8192,
    strict: bool = True,
    fill_value: float = 0.0,
    conn=None,
    enforce_timeframe_match: bool = True,
):
    """
    Multi-horizon aware:
      - For single-horizon models: returns one horizon.
      - For TCN multi-horizon: returns long DF with horizon column.

    Returns:
      df_gen: symbol,timestamp,target_timestamp,y_true,y_pred,horizon,...
      meta:   symbol,timestamp,target_timestamp (+ horizon)
    """
    art_dir = _resolve_artifact_dir(run_dir)
    artifact = load_model_artifact_auto(art_dir, map_location="cpu")
    cfg = artifact.get("config", {}) or {}

    tf_cfg = _cfg_get(cfg, "timeframe", None)
    tf = timeframe or tf_cfg
    if tf is None:
        raise ValueError("timeframe must be provided (or exist in artifact config).")
    if enforce_timeframe_match and tf_cfg is not None and str(tf_cfg) != str(tf):
        raise ValueError(f"timeframe mismatch: artifact={tf_cfg} vs requested={tf}")

    if conn is None:
        conn = get_connection()

    horizons = _get_horizons_from_cfg(cfg)
    if not horizons:
        raise ValueError("No pude inferir horizons para compare.")

    base_feature_cols = _cfg_get(cfg, "base_feature_cols", None)
    if not base_feature_cols:
        raise ValueError("artifact config missing base_feature_cols")

    lookback = int(_cfg_get(cfg, "lookback", None) or (_parse_lb_h_from_dirname(art_dir)[0] or 0))
    if lookback <= 0:
        raise ValueError("No pude inferir lookback")

    # build df_long using SAME feature families
    df_long = _build_df_long_from_artifact_config(
        artifact,
        timeframe=tf,
        symbols=symbols,
        start=start,
        end=end,
        conn=conn,
    )

    # Build X/y/meta for max horizon (common X for all horizons, avoids row mismatch)
    h_max = int(max(horizons))
    X_max, y_max, meta_max = build_supervised_dataset(
        df=df_long,
        feature_cols=list(base_feature_cols),
        lookback=lookback,
        horizon=h_max,
        price_col=price_col,
        group_col=group_col,
        timestamp_col=timestamp_col,
        lags_by_feature=_cfg_get(cfg, "lags_by_feature", None),
        default_lags=_cfg_get(cfg, "default_lags", None),
    )

    meta_max = meta_max.copy()
    meta_max[timestamp_col] = pd.to_datetime(meta_max[timestamp_col])
    meta_max["key"] = meta_max[group_col].astype(str) + "||" + meta_max[timestamp_col].astype(str)

    # Predict ONCE: should return (N, out_dim) for TCN, or (N,) for single horizon
    y_pred_all = predict_with_loaded_artifact(
        artifact,
        X_max,
        device=device,
        batch_size=batch_size,
        strict=strict,
        fill_value=fill_value,
    )
    y_pred_all = np.asarray(y_pred_all, dtype=np.float64)
    if y_pred_all.ndim == 1:
        y_pred_all = y_pred_all.reshape(-1, 1)

    # Build targets per horizon aligned to meta_max keys
    rows = []
    metas = []
    for j, h in enumerate(horizons):
        X_h, y_h, meta_h = build_supervised_dataset(
            df=df_long,
            feature_cols=list(base_feature_cols),
            lookback=lookback,
            horizon=int(h),
            price_col=price_col,
            group_col=group_col,
            timestamp_col=timestamp_col,
            lags_by_feature=_cfg_get(cfg, "lags_by_feature", None),
            default_lags=_cfg_get(cfg, "default_lags", None),
        )
        meta_h = meta_h.copy()
        meta_h[timestamp_col] = pd.to_datetime(meta_h[timestamp_col])
        meta_h["target_timestamp"] = pd.to_datetime(meta_h["target_timestamp"])
        meta_h["key"] = meta_h[group_col].astype(str) + "||" + meta_h[timestamp_col].astype(str)
        y_h = np.asarray(y_h, dtype=np.float64).reshape(-1)

        tmp = meta_h[[group_col, timestamp_col, "target_timestamp", "key"]].copy()
        tmp["y_true"] = y_h

        merged = meta_max[[group_col, timestamp_col, "key"]].merge(tmp, on=["key"], how="left", suffixes=("", "_h"))
        if merged["y_true"].isna().any():
            # Should not happen if meta_max (max horizon) is subset of all smaller horizons
            missing_n = int(merged["y_true"].isna().sum())
            raise RuntimeError(
                f"Al alinear horizon={h} con horizon_max={h_max}, faltan {missing_n} y_true. "
                "Esto indica inconsistencia en build_supervised_dataset o en el merge keys."
            )

        yp = y_pred_all[:, j] if j < y_pred_all.shape[1] else y_pred_all[:, 0]

        df_out = pd.DataFrame({
            group_col: merged[group_col].astype(str).values,
            timestamp_col: pd.to_datetime(merged[timestamp_col]).values,
            "target_timestamp": pd.to_datetime(merged["target_timestamp"]).values,
            "y_true": merged["y_true"].to_numpy(dtype=np.float64),
            "y_pred": yp.astype(np.float64),
            "horizon": int(h),
            "timeframe": str(tf),
            "run_dir": art_dir,
            "model_name": artifact.get("model_name", cfg.get("model", "unknown")),
            "family": artifact.get("family", "unknown"),
        })

        rows.append(df_out)
        metas.append(df_out[[group_col, timestamp_col, "target_timestamp", "horizon"]].copy())

    df_gen = pd.concat(rows, ignore_index=True)
    meta_out = pd.concat(metas, ignore_index=True)
    return df_gen, meta_out


# ---------- Smoke test determinism (save/load/infer) ----------
def smoke_test_artifact_determinism(
    run_dir: str,
    *,
    timeframe: Optional[str] = None,
    symbols: Optional[List[str]] = None,
    start: Optional[Union[str, pd.Timestamp]] = None,
    end: Optional[Union[str, pd.Timestamp]] = None,
    n_samples: int = 512,
    atol: float = 1e-6,
    rtol: float = 0.0,
    strict: bool = True,
    fill_value: float = 0.0,
) -> Dict[str, float]:
    """
    1) load artifact
    2) build small X reproducible (via predict_artifact_to_compare)
    3) y_pred_1
    4) reload artifact
    5) y_pred_2 on SAME X
    6) assert within tolerance
    """
    # Build X once (no re-fetch between loads)
    artifact1 = load_model_artifact_auto(run_dir, map_location="cpu")
    cfg = artifact1.get("config", {}) or {}
    tf = timeframe or _cfg_get(cfg, "timeframe", None)
    if tf is None:
        raise ValueError("timeframe requerido para smoke test (no está en config).")

    df_gen, _ = predict_artifact_to_compare(
        run_dir,
        timeframe=tf,
        symbols=symbols,
        start=start,
        end=end,
        strict=strict,
        fill_value=fill_value,
    )

    # reconstruye X exacto desde meta (para usar predict_with_loaded_artifact directamente):
    # tomamos el primer horizonte del df_gen para armar X/y
    h0 = int(df_gen["horizon"].iloc[0])
    # Rebuild X,y,meta (single horizon) to get X DataFrame
    X, _, _ = build_Xy_meta_from_artifact_config(
        artifact1,
        timeframe=tf,
        symbols=symbols,
        start=start,
        end=end,
        conn=get_connection(),
        enforce_timeframe_match=False,
    )
    X_small = X.iloc[: min(int(n_samples), len(X))].copy()

    y1 = predict_with_loaded_artifact(artifact1, X_small, device=torch.device("cpu"), strict=strict, fill_value=fill_value)

    artifact2 = load_model_artifact_auto(run_dir, map_location="cpu")
    y2 = predict_with_loaded_artifact(artifact2, X_small, device=torch.device("cpu"), strict=strict, fill_value=fill_value)

    y1 = np.asarray(y1, dtype=np.float64)
    y2 = np.asarray(y2, dtype=np.float64)
    diff = np.abs(y1 - y2)

    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))

    if not np.allclose(y1, y2, atol=float(atol), rtol=float(rtol)):
        raise RuntimeError(
            f"[SMOKE TEST FAIL] run_dir={run_dir}\n"
            f"max_abs_diff={max_abs:.8g} mean_abs_diff={mean_abs:.8g} atol={atol} rtol={rtol}\n"
            "Causas típicas:\n"
            " - model.pt no corresponde a config.json (arquitectura/output_dim)\n"
            " - scaler.joblib diferente o no guardado\n"
            " - feature_names.json inconsistente\n"
        )

    return {"max_abs_diff": max_abs, "mean_abs_diff": mean_abs}

import re
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional

_LAG_RE = re.compile(r"^(?P<base>.+)_lag(?P<k>\d+)$")

def _split_lag(name: str) -> Optional[Tuple[str, int]]:
    m = _LAG_RE.match(str(name))
    if not m:
        return None
    return m.group("base"), int(m.group("k"))

def _infer_raw_cols_and_lags_from_feature_names(
    feature_names: List[str],
    *,
    group_col: str,
    timestamp_col: str,
) -> Tuple[List[str], Dict[str, List[int]]]:
    """
    A partir de feature_names (ej: RSI_14_lag3) infiere:
      - raw_cols: columnas base necesarias en df (ej: RSI_14)
      - lags_map: base -> lista de lags requeridos
    """
    raw_cols: List[str] = []
    lags_map: Dict[str, set] = {}

    seen = set()
    for col in feature_names:
        s = str(col)
        sp = _split_lag(s)
        if sp is not None:
            base, k = sp
            lags_map.setdefault(base, set()).add(int(k))
            if base not in seen and base not in {group_col, timestamp_col}:
                seen.add(base)
                raw_cols.append(base)
        else:
            # feature no laggeada (se toma del "current" timestep)
            if s not in seen and s not in {group_col, timestamp_col}:
                seen.add(s)
                raw_cols.append(s)

    lags_map2 = {b: sorted(list(ks)) for b, ks in lags_map.items()}
    return raw_cols, lags_map2


def _build_wide_inference_latest_tail_nonseq(
    df: pd.DataFrame,
    *,
    feature_names: List[str],
    lags_map: Dict[str, List[int]],
    group_col: str,
    timestamp_col: str,
    price_col: str,
    select: str,
    tail_n: int,
    strict: bool,
    fill_value: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye X_live (2D) para modelos NO-secuenciales (XGB, MLP tabular) evitando
    crear lags para todo el histórico. Solo arma:
      - latest: 1 fila por símbolo
      - tail: tail_n filas por símbolo
    """
    X_cols = list(map(str, feature_names))
    col_idx = {c: i for i, c in enumerate(X_cols)}
    P = len(X_cols)

    # Targets lag y no-lag
    lag_targets: Dict[str, List[Tuple[int, int]]] = {}
    direct_targets: List[Tuple[str, int]] = []

    max_lag = 0
    for c, i in col_idx.items():
        sp = _split_lag(c)
        if sp is not None:
            base, k = sp
            lag_targets.setdefault(base, []).append((int(k), i))
            if k > max_lag:
                max_lag = k
        else:
            direct_targets.append((c, i))

    rows = []
    metas = []
    has_price = price_col in df.columns

    df = df.sort_values([group_col, timestamp_col], kind="mergesort")

    for sym, g in df.groupby(group_col, sort=False):
        g = g.reset_index(drop=True)
        n = len(g)

        if select == "latest":
            end_positions = [n - 1]
        else:
            t = int(tail_n)
            end_positions = list(range(max(0, n - t), n))

        for end_pos in end_positions:
            start_pos = end_pos - max_lag
            if start_pos < 0:
                continue

            window = g.iloc[start_pos : end_pos + 1]
            current = window.iloc[-1]

            row = np.empty(P, dtype=np.float32)

            # lag features
            # window length = max_lag+1, index for lag k: (-1-k) => (len-1-k) => (max_lag-k)
            for base, pairs in lag_targets.items():
                vals = window[base].to_numpy(dtype=np.float32, copy=False)
                for k, j in pairs:
                    idx_in_window = (len(vals) - 1) - int(k)
                    row[j] = vals[idx_in_window]

            # direct features (no lag suffix)
            for c, j in direct_targets:
                row[j] = np.float32(current[c])

            if strict:
                if not np.isfinite(row).all():
                    continue
            else:
                row = np.nan_to_num(row, nan=fill_value, posinf=fill_value, neginf=fill_value).astype(np.float32)

            rows.append(row)
            meta = {group_col: sym, timestamp_col: current[timestamp_col]}
            if has_price:
                meta[price_col] = current[price_col]
            metas.append(meta)

    if not rows:
        raise ValueError("No quedaron filas válidas para inferencia (muy poco histórico vs max_lag o NaNs).")

    X_live = pd.DataFrame(np.vstack(rows), columns=X_cols, dtype=np.float32)
    meta_live = pd.DataFrame(metas)
    return X_live, meta_live
