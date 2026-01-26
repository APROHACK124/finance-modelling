# evaluators.py
from math import sqrt
import numpy as np
from sklearn.metrics import(
    mean_squared_error, mean_absolute_error, median_absolute_error,
    r2_score, roc_auc_score,
)
from typing import Optional, Dict, Any, Tuple

def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if y_true.size == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "HitRate(sign)": np.nan, "PearsonCorr(IC)": np.nan, "N": 0}

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # hit rate: opcionalmente ignora casi-ceros (si quieres)
    eps = 1e-8
    hit = np.mean(np.sign(y_true[np.abs(y_true)>eps]) == np.sign(y_pred[np.abs(y_true)>eps]))

    # corr: requiere >=2 y varianza > 0
    if y_true.size < 2 or np.std(y_true) == 0.0 or np.std(y_pred) == 0.0:
        corr = np.nan
    else:
        corr = float(np.corrcoef(y_true, y_pred)[0, 1])


    residual = np.abs(y_true - y_pred)
    q = np.quantile(residual, 0.99)

    return {"MAE": mae, "RMSE": rmse, "HitRate(sign)": hit, "PearsonCorr(IC)": corr, "CalibrationInterval": q, "N": int(y_true.size)}

def _safe_pearson_corr(a: np.ndarray, b:np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 2:
        return float("nan")
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


import pandas as pd
def _safe_spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 2:
        return float("nan")
    ar = pd.Series(a).rank(method="average").to_numpy(dtype=np.float64)
    br = pd.Series(b).rank(method="average").to_numpy(dtype=np.float64)
    return _safe_pearson_corr(ar, br)

def _conformal_qhat_abs(scores: np.ndarray, alpha: float) -> float:
    """
    Split conformal: qhat is the (1-alpha) quantile of calibration
    absolute residuals with finite-sample correction:
        k = ceil((n + 1)*(1 - alpha))
        qhat = sorted_scores[k - 1]
    This yields conservative coverage under exchangeability
    """

    scores = np.asarray(scores).reshape(-1)
    scores = scores[np.isfinite(scores)]
    n = int(scores.size)
    if n == 0:
        return float("nan")
    
    scores_sorted = np.sort(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    k = min(max(k, 1), n)
    return float(scores_sorted[k - 1])


def _summary_stats(x: np.ndarray) -> Dict[str, Any]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return {"N": 0, "mean": np.nan, "std": np.nan, "tsat": np.nan, "frac_pos": np.nan}
    
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1)) if n > 1 else np.nan
    frac_pos = float(np.mean(x > 0))

    if n > 1 and np.isfinite(std) and std > 0:
        tstat = float(mean / (std / np.sqrt(n)))
    else:
        tstat = np.nan
    return {"N": n, "mean": mean, "std": std, "tstat": tstat, "frac_pos": frac_pos}

def eval_regression_extended(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        *,
        deadzone: float = 0.0,
        meta: Optional[pd.DataFrame] = None,
        time_col: str = "timestamp",
        group_col: str = "symbol",
        periods_per_year: int = 252,
        quantile: float = 0.1,
        min_group_size: int = 20,
        conformal_calib: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        conformal_alphas: Tuple[float, ...] = (0.1,),
) -> Dict[str, Any]:
    """
    Extended evaluation for regression models predicting future returns.

    Base metrics (global):
        - MAE, MedianAE, RMSE, R2
        - HItRate(sign), HitRate(sign, deadzone)
        - PearsonCorr(IC), SpearmanCorr(RankIC)
        - AUC(Sign) where label=(y_true>0), score=y_pred

    Optional cross-sectinoal metrics if meta provided:
        - DailyIC_*: per timestamp Pearson corr across symbols
        - DailyRankIC_*: per timestamp Spearman corr across symbols
        - QuantileSpread_* per timestamp mean(y_true top q) - mean(y_true bottom q)

    Optional slit conformal intervals if conformal_calib provided:
        - qhat from calibration abs residuals
        - coverage and width on evaluated set for each alpha
    """

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_true and y_pred must have same lenght")
    
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    # Align meta
    meta_f: Optional[pd.DataFrame] = None
    if meta is not None:
        if len(meta) == len(mask):
            meta_f = meta.loc[mask].reset_index(drop=True).copy()
        elif len(meta) == int(mask.sum()):
            meta_f = meta.reset_index(drop=True).copy()
        else:
            raise ValueError(
                f"meta length must be either original length ({len(mask)}) or filtered length"
                f"Got {len(meta)}."
            )
        
    # Base output with stable keys
    out: Dict[str, Any] = {
        "MAE": np.nan,
        "MedianAE": np.nan,
        "RMSE": np.nan,
        "R2": np.nan,
        "HitRate(sign)": np.nan,
        "HitRate(sign,deadzone)": np.nan,
        "PearsonCorr(IC)": np.nan,
        "SpearmanCorr(RankIC)": np.nan,
        "AUC(Sign)": np.nan,
        "N": int(y_true_f.size),
        "N_deadzone": 0,
        # Cross-sectional summaries (stable keys)
        "DailyIC_mean": np.nan,
        "DailyIC_std": np.nan,
        "DailyIC_tstat": np.nan,
        "DailyIC_frac_pos": np.nan,
        "DailyIC_N": 0,
        "DailyRankIC_mean": np.nan,
        "DailyRankIC_std": np.nan,
        "DailyRankIC_tstat": np.nan,
        "DailyRankIC_frac_pos": np.nan,
        "DailyRankIC_N": 0,
        "QuantileSpread_mean": np.nan,
        "QuantileSpread_std": np.nan,
        "QuantileSpread_sharpe": np.nan,
        "QuantileSpread_N": 0,
    }

    for a in conformal_alphas:
        out[f"Conformal_qhat(alpha={a})"] = np.nan
        out[f"Conformal_coverage(alpha={a})"] = np.nan
        out[f"Conformal_avg_width(alpha={a})"] = np.nan
        out[f"Conformal_width_over_std(alpha={a})"] = np.nan

    if y_true_f.size == 0:
        return out
    
    # Base global metrics
    out["MAE"] = float(mean_absolute_error(y_true_f, y_pred_f))
    out["MedianAE"] = float(median_absolute_error(y_true_f, y_pred_f))
    out["RMSE"] = float(np.sqrt(mean_squared_error(y_true_f, y_pred_f)))
    out["R2"] = float(r2_score(y_true_f, y_pred_f)) if y_true_f.size >= 2 else np.nan

    out["HitRate(sign)"] = float(np.mean(np.sign(y_true_f) == np.sign(y_pred_f)))
    out["PearsonCorr(IC)"] = _safe_pearson_corr(y_true_f, y_pred_f)
    out["SpearmanCorr(RankIC)"] = _safe_spearman_corr(y_true_f, y_pred_f)

    if deadzone > 0.0:
        dz_mask = np.abs(y_true_f) > float(deadzone)
        out["N_deadzone"] = int(dz_mask.sum())
        if dz_mask.sum() > 0:
            out["HitRate(sign,deadzone)"] = float(np.mean(np.sign(y_true_f[dz_mask]) == np.sign(y_pred_f[dz_mask])))
        else:
            out["HitRate(sign,deadzone)"] = np.nan

    else:
        out["N_deadzone"] = int(y_true_f.size)
        out["HitRate(sign,deadzone)"] = out["HitRate(sign)"]

    # AUC(Sign): ranking ability for up/down
    auc_mask = np.ones_like(y_true_f, dtype=bool)
    if deadzone > 0.0:
        auc_mask = np.abs(y_true_f) > float(deadzone)

    y_label = (y_true_f[auc_mask] > 0).astype(int)
    if y_label.size >= 2 and np.unique(y_label).size == 2:
        out["AUC(Sign)"] = float(roc_auc_score(y_label, y_pred_f[auc_mask]))
    else:
        out["AUC(Sign)"] = np.nan

    if meta_f is not None and (time_col in meta_f.columns):
        meta_cs = meta_f.copy()
        meta_cs[time_col] = pd.to_datetime(meta_cs[time_col], errors="coerce")

        valid_ts = meta_cs[time_col].notna().to_numpy()

        y_true_cs = y_true_f
        y_pred_cs = y_pred_f
        if valid_ts.sum() < len(meta_cs):
            meta_cs = meta_cs.loc[valid_ts].reset_index(drop=True)
            y_true_cs = y_true_f[valid_ts]
            y_pred_cs = y_pred_f[valid_ts]

        else:
            y_true_cs = y_true_f
            y_pred_cs = y_pred_f

        ic_list: list[float] = []
        ric_list: list[float] = []
        spread_list: list[float] = []

        # group by signal timestamp
        for _, g in meta_cs.groupby(time_col, sort=True):
            idx = g.index.to_numpy()
            if idx.size < 2:
                continue

            yt_g = y_true_cs[idx]
            yp_g = y_pred_cs[idx]

            ic = _safe_pearson_corr(yt_g, yp_g)
            if np.isfinite(ic):
                ic_list.append(float(ic))

            ric = _safe_spearman_corr(yt_g, yp_g)
            if np.isfinite(ric):
                ric_list.append(float(ric))

            # Quantile spread (top - bottom)
            n = int(idx.size)
            if quantile is not None and quantile > 0 and n >= int(min_group_size):
                k = int(np.floor(n * float(quantile)))
                if k >= 1:
                    order = np.argsort(yp_g)
                    bot = order[:k]
                    top = order[-k:]
                    spread = float(np.mean(yt_g[top]) - np.mean(yt_g[bot]))
                    if np.isfinite(spread):
                        spread_list.append(spread)

        ic_stats = _summary_stats(np.asarray(ic_list, dtype=np.float64))

        out["DailyIC_N"] = ic_stats["N"]
        out["DailyIC_mean"] = ic_stats["mean"]
        out["DailyIC_std"] = ic_stats["std"]
        out["DailyIC_tstat"] = ic_stats["tstat"]
        out["DailyIC_frac_pos"] = ic_stats["frac_pos"]

        ric_stats = _summary_stats(np.asarray(ric_list, dtype=np.float64))
        out["DailyRankIC_N"] = ric_stats["N"]
        out["DailyRankIC_mean"] = ric_stats["mean"]
        out["DailyRankIC_std"] = ric_stats["std"]
        out["DailyRankIC_tstat"] = ric_stats["tstat"]
        out["DailyRankIC_frac_pos"] = ric_stats["frac_pos"]

        spread_stats = _summary_stats(np.asarray(spread_list, dtype=np.float64))
        out["QuantileSpread_N"] = spread_stats["N"]
        out["QuantileSpread_mean"] = spread_stats["mean"]
        out["QuantileSpread_std"] = spread_stats["std"]

        if spread_stats["N"] > 1 and np.isfinite(spread_stats["std"]) and spread_stats["std"] > 0:
            out["QuantileSpread_sharpe"] = float(
                spread_stats["mean"] / spread_stats["std"] * np.sqrt(int(periods_per_year))
            )            
        else:
            out["QuantileSpread_sharpe"] = np.nan


    # Conformal intervals (split conformal)

    if conformal_calib is not None:
        y_cal_true, y_cal_pred = conformal_calib
        y_cal_true = np.asarray(y_cal_true).reshape(-1)
        y_cal_pred = np.asarray(y_cal_pred).reshape(-1)

        if y_cal_true.shape[0] != y_cal_pred.shape[0]:
            raise ValueError(
                f"conformal_calib arrays must have same length."
            )
        
        cal_mask = np.isfinite(y_cal_true) & np.isfinite(y_cal_pred)
        cal_scores = np.abs(y_cal_true[cal_mask] - y_cal_pred[cal_mask])
        y_std = float(np.std(y_true_f)) if y_true_f.size > 1 else np.nan

        for a in conformal_alphas:
            a = float(a)
            if not (0.0 < a < 1.0):
                continue

            qhat = _conformal_qhat_abs(cal_scores, alpha=a)

            out[f"Conformal_qhat(alpha={a})"] = qhat

            if np.isfinite(qhat):
                lower = y_pred_f - qhat
                upper = y_pred_f + qhat

                coverage = float(np.mean((y_true_f >= lower) & (y_true_f <= upper)))
                width = float(2.0 * qhat)

                out[f"Conformal_coverage(alpha={a})"] = coverage
                out[f"Conformal_avg_width(alpha={a})"] = width
                out[f"Conformal_width_over_std(alpha={a})"] = float(width / y_std) if np.isfinite(y_std) and y_std > 0 else np.nan
            else:
                out[f"Conformal_coverage(alpha={a})"] = np.nan
                out[f"Conformal_avg_width(alpha={a})"] = np.nan
                out[f"Conformal_width_over_std(alpha={a})"] = np.nan

    return out

def calculate_deadzone(horizon: int, k: float = 0.15):
    daily_volatility = 0.012
    return k * daily_volatility * sqrt(horizon)
