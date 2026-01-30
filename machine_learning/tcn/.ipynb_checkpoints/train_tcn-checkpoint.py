from __future__ import annotations

import random
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import os

# PROJECT_ROOT = os.path.abspath('../..')
# sys.path.append(PROJECT_ROOT)

from data.seq_dataset import (
    SequenceDataset,
    SequenceStandardScaler,
    build_multi_horizon_supervised_dataset,
    split_by_target_timestamp,
)

import sys
import os
PROJECT_ROOT = os.path.abspath('..')
sys.path.append(PROJECT_ROOT)

from models.tcn import TCNConfig, TCNRegressor
from artifacts import make_run_dir, save_tcn_artifact


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device: Optional[str] = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _find_metric_recursive(metrics: Any, predicate: Callable[[str], bool]) -> Optional[float]:
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            if isinstance(k, str) and predicate(k):
                try:
                    return float(v)
                except Exception:
                    pass
        for v in metrics.values():
            m = _find_metric_recursive(v, predicate)
            if m is not None:
                return m
    return None


def _extract_rmse_from_metrics(metrics: Dict[str, Any], fallback_rmse: float) -> float:
    rmse = _find_metric_recursive(metrics, lambda k: k.lower() == "rmse")
    return rmse if rmse is not None else fallback_rmse


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      y_true: (N,H)
      y_pred: (N,H)
      idx:    (N,) indices to restore order
    """
    model.eval()
    all_true: List[np.ndarray] = []
    all_pred: List[np.ndarray] = []
    all_idx: List[np.ndarray] = []

    for Xb, yb, idx in loader:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        pred = model(Xb)
        all_true.append(yb.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())
        all_idx.append(np.asarray(idx))

    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    idx = np.concatenate(all_idx, axis=0)

    order = np.argsort(idx)
    return y_true[order], y_pred[order], idx[order]


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    grad_clip: float = 1.0,
) -> float:
    model.train()
    losses: List[float] = []
    for Xb, yb, _ in loader:
        Xb = Xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        pred = model(Xb)
        loss = loss_fn(pred, yb)
        loss.backward()

        if grad_clip is not None and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    return float(np.mean(losses)) if losses else float("nan")


def calibrate_conformal_qhat(
    y_true_val: np.ndarray,
    y_pred_val: np.ndarray,
    horizons: Sequence[int],
    alphas: Sequence[float] = (0.1,),
) -> Dict[int, Dict[float, float]]:
    """
    Symmetric split conformal calibration using absolute residual quantiles.
    Returns qhat per horizon per alpha.
    """
    y_true_val = np.asarray(y_true_val)
    y_pred_val = np.asarray(y_pred_val)
    H = y_true_val.shape[1]
    if H != len(horizons):
        raise ValueError("Mismatch between y shape and horizons length")

    out: Dict[int, Dict[float, float]] = {}
    for j, h in enumerate(horizons):
        resid = np.abs(y_true_val[:, j] - y_pred_val[:, j])
        out[h] = {}
        for a in alphas:
            a = float(a)
            if not (0 < a < 1):
                raise ValueError("alphas must be in (0,1)")
            qhat = float(np.quantile(resid, 1.0 - a))
            out[h][a] = qhat
    return out


def eval_conformal(
    y_true_test: np.ndarray,
    y_pred_test: np.ndarray,
    horizons: Sequence[int],
    qhat: Dict[int, Dict[float, float]],
    alphas: Sequence[float] = (0.1,),
) -> Dict[int, Dict[float, Dict[str, float]]]:
    """
    Evaluate symmetric conformal intervals on test set.
    Returns coverage and avg_width per horizon per alpha.
    """
    y_true_test = np.asarray(y_true_test)
    y_pred_test = np.asarray(y_pred_test)

    out: Dict[int, Dict[float, Dict[str, float]]] = {}
    for j, h in enumerate(horizons):
        out[h] = {}
        for a in alphas:
            q = float(qhat[h][float(a)])
            lo = y_pred_test[:, j] - q
            hi = y_pred_test[:, j] + q
            covered = (y_true_test[:, j] >= lo) & (y_true_test[:, j] <= hi)
            out[h][float(a)] = {
                "qhat": q,
                "coverage": float(np.mean(covered)),
                "avg_width": float(np.mean(hi - lo)),
            }
    return out


def compute_metrics_multi_horizon(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    meta: pd.DataFrame,
    horizons: Sequence[int],
    *,
    eval_fn: Callable[..., Dict[str, Any]],
    time_col: str = "timestamp",
    group_col: str = "symbol",
    periods_per_year: int = 252,
) -> Dict[int, Dict[str, Any]]:
    """
    Runs eval_regression_extended per horizon output.
    Returns dict: horizon -> metrics dict.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    H = y_true.shape[1]
    if H != len(horizons) or y_pred.shape[1] != H:
        raise ValueError("Mismatch between shapes and horizons")

    out: Dict[int, Dict[str, Any]] = {}
    for j, h in enumerate(horizons):
        out[h] = eval_fn(
            y_true[:, j],
            y_pred[:, j],
            meta=meta,
            time_col=time_col,
            group_col=group_col,
            periods_per_year=periods_per_year,
        )
    return out


def summarize_rmse(metrics_by_h: Dict[int, Dict[str, Any]], y_true: np.ndarray, y_pred: np.ndarray, horizons: Sequence[int]) -> Dict[str, float]:
    rmses = []
    per_h = {}
    for j, h in enumerate(horizons):
        fallback = _rmse(y_true[:, j], y_pred[:, j])
        rmse = _extract_rmse_from_metrics(metrics_by_h[h], fallback)
        per_h[f"rmse_h{h}"] = rmse
        rmses.append(rmse)
    per_h["rmse_mean"] = float(np.mean(rmses)) if rmses else float("nan")
    return per_h


def _print_final_metrics(title: str, metrics_by_h: Dict[int, Dict[str, Any]], horizons: Sequence[int]) -> None:
    """
    Best-effort pretty print of a few common metrics, without assuming a fixed schema.
    """
    def pick(m: Dict[str, Any], name: str) -> Optional[float]:
        name_l = name.lower()
        return _find_metric_recursive(m, lambda k: k.lower() == name_l)

    def pick_contains(m: Dict[str, Any], token: str) -> Optional[float]:
        token_l = token.lower()
        return _find_metric_recursive(m, lambda k: token_l in k.lower())

    print(f"\n=== {title} ===")
    for h in horizons:
        m = metrics_by_h[h]
        rmse = pick(m, "RMSE")
        mae = pick(m, "MAE")
        r2 = pick(m, "R2")
        ic = pick(m, "IC") or pick_contains(m, "pearson")
        rankic = pick(m, "RankIC") or pick_contains(m, "spearman")
        daily_ic = pick_contains(m, "dailyic")
        daily_rankic = pick_contains(m, "dailyrankic")
        qspread = pick_contains(m, "quantilespread")

        parts = []
        if rmse is not None: parts.append(f"RMSE={rmse:.6f}")
        if mae is not None: parts.append(f"MAE={mae:.6f}")
        if r2 is not None: parts.append(f"R2={r2:.4f}")
        if ic is not None: parts.append(f"IC={ic:.4f}")
        if rankic is not None: parts.append(f"RankIC={rankic:.4f}")
        if daily_ic is not None: parts.append(f"DailyIC*={daily_ic:.4f}")
        if daily_rankic is not None: parts.append(f"DailyRankIC*={daily_rankic:.4f}")
        if qspread is not None: parts.append(f"QuantileSpread*={qspread:.6f}")

        extra = " | ".join(parts) if parts else "(métricas disponibles en metrics.json)"
        print(f"H{h}: {extra}")
    print("(* = se busca por substring; depende del schema exacto de eval_regression_extended)\n")


@dataclass
class TrainTCNConfig:
    # Data
    lookback: int = 60
    horizons: Sequence[int] = (5, 20, 60)
    price_col: str = "close"
    group_col: str = "symbol"
    timestamp_col: str = "timestamp"
    periods_per_year: int = 252

    # Split
    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    # Model
    channels: Sequence[int] = (64, 64, 64, 64)
    kernel_size: int = 3
    dropout: float = 0.10

    # Train
    loss: str = "smoothl1"  # "mse" or "smoothl1"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 50
    patience: int = 10
    grad_clip: float = 1.0
    num_workers: int = 0

    # Repro
    seed: int = 42
    device: Optional[str] = None

    # Conformal
    conformal_alphas: Sequence[float] = (0.1,)

    # Artifact
    run_base_dir: str = "runs"
    run_name: Optional[str] = None


def train_eval_tcn(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    build_supervised_dataset_fn: Callable[..., Tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
    eval_fn: Callable[..., Dict[str, Any]],
    cfg: TrainTCNConfig = TrainTCNConfig(),
) -> Dict[str, Any]:
    """
    End-to-end pipeline.

    IMPORTANT ANTI-LEAKAGE GUARANTEES:
    - Split is done by meta['target_timestamp_max'] (== target_timestamp of MAX horizon),
      never by feature timestamp.
    - StandardScaler is fit ONLY on train (flattened across time) and then applied to val/test.
    """
    set_global_seed(cfg.seed, deterministic=True)
    device = get_device(cfg.device)

    # 1) Build supervised dataset (multi-horizon)
    X_all, y_all, meta_all, horizons_sorted = build_multi_horizon_supervised_dataset(
        df,
        feature_cols=list(feature_cols),
        lookback=cfg.lookback,
        horizons=cfg.horizons,
        build_supervised_dataset_fn=build_supervised_dataset_fn,
        price_col=cfg.price_col,
        group_col=cfg.group_col,
        timestamp_col=cfg.timestamp_col,
        dtype=np.float32, # type: ignore
    )

    # 2) Split by target timestamp (max horizon) to avoid leakage
    split = split_by_target_timestamp(
        meta_all,
        target_col="target_timestamp_max",
        train_frac=cfg.train_frac,
        val_frac=cfg.val_frac,
        test_frac=cfg.test_frac,
    )

    X_train, y_train, meta_train = X_all[split.train_idx], y_all[split.train_idx], meta_all.iloc[split.train_idx].reset_index(drop=True)
    X_val, y_val, meta_val = X_all[split.val_idx], y_all[split.val_idx], meta_all.iloc[split.val_idx].reset_index(drop=True)
    X_test, y_test, meta_test = X_all[split.test_idx], y_all[split.test_idx], meta_all.iloc[split.test_idx].reset_index(drop=True)

    # 3) Scaling (fit only on train)
    scaler = SequenceStandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # 4) DataLoaders
    train_ds = SequenceDataset(X_train_s, y_train, meta_train)
    val_ds = SequenceDataset(X_val_s, y_val, meta_val)
    test_ds = SequenceDataset(X_test_s, y_test, meta_test)

    pin_memory = (device.type == "cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,   # SOLO train
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    # 5) Model
    model_cfg = TCNConfig(
        num_features=len(feature_cols),
        channels=tuple(cfg.channels),
        kernel_size=int(cfg.kernel_size),
        dropout=float(cfg.dropout),
        output_dim=len(horizons_sorted),
    )
    model = TCNRegressor(model_cfg).to(device)

    # 6) Loss
    if cfg.loss.lower() == "mse":
        loss_fn = nn.MSELoss()
        loss_name = "mse"
    elif cfg.loss.lower() == "smoothl1":
        # Huber-like: robust to fat-tailed return outliers
        loss_fn = nn.SmoothL1Loss(beta=1.0)
        loss_name = "smoothl1"
    else:
        raise ValueError("cfg.loss must be 'mse' or 'smoothl1'")

    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    # 7) Training loop with early stopping on mean val RMSE
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_val_rmse = float("inf")
    best_epoch = -1
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, cfg.max_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, grad_clip=cfg.grad_clip)

        # Validate
        y_val_true, y_val_pred, _ = predict(model, val_loader, device=device)

        val_metrics_by_h = compute_metrics_multi_horizon(
            y_val_true,
            y_val_pred,
            meta_val,
            horizons_sorted,
            eval_fn=eval_fn,
            time_col=cfg.timestamp_col,
            group_col=cfg.group_col,
            periods_per_year=cfg.periods_per_year,
        )
        val_rmse_summary = summarize_rmse(val_metrics_by_h, y_val_true, y_val_pred, horizons_sorted)
        val_rmse_mean = val_rmse_summary["rmse_mean"]

        improved = val_rmse_mean < (best_val_rmse - 1e-8)
        if improved:
            best_val_rmse = val_rmse_mean
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        epoch_rec: Dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_rmse_mean": val_rmse_mean,
            "val_rmse_by_horizon": {k: v for k, v in val_rmse_summary.items() if k.startswith("rmse_h")},
            "elapsed_sec": float(time.time() - t0),
        }
        history.append(epoch_rec)

        rmse_parts = ", ".join([f"h{h}:{val_rmse_summary[f'rmse_h{h}']:.6f}" for h in horizons_sorted])
        print(
            f"[epoch {epoch:03d}] train_loss={train_loss:.6f} | val_rmse_mean={val_rmse_mean:.6f} "
            f"({rmse_parts}) | best={best_val_rmse:.6f} @epoch {best_epoch} | no_improve={epochs_no_improve}"
        )

        if epochs_no_improve >= cfg.patience:
            print(f"Early stopping: no improvement for {cfg.patience} epochs.")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best_state (unexpected).")

    # Restore best model
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # 8) Final eval on val + test with best model
    y_val_true, y_val_pred, _ = predict(model, val_loader, device=device)
    y_test_true, y_test_pred, _ = predict(model, test_loader, device=device)

    val_metrics_by_h = compute_metrics_multi_horizon(
        y_val_true,
        y_val_pred,
        meta_val,
        horizons_sorted,
        eval_fn=eval_fn,
        time_col=cfg.timestamp_col,
        group_col=cfg.group_col,
        periods_per_year=cfg.periods_per_year,
    )
    test_metrics_by_h = compute_metrics_multi_horizon(
        y_test_true,
        y_test_pred,
        meta_test,
        horizons_sorted,
        eval_fn=eval_fn,
        time_col=cfg.timestamp_col,
        group_col=cfg.group_col,
        periods_per_year=cfg.periods_per_year,
    )

    _print_final_metrics("VALIDACIÓN (best epoch)", val_metrics_by_h, horizons_sorted)
    _print_final_metrics("TEST (best epoch)", test_metrics_by_h, horizons_sorted)

    # Optional conformal calibration on val, evaluation on test
    conformal_qhat = calibrate_conformal_qhat(
        y_val_true,
        y_val_pred,
        horizons_sorted,
        alphas=cfg.conformal_alphas,
    )
    conformal_test = eval_conformal(
        y_test_true,
        y_test_pred,
        horizons_sorted,
        qhat=conformal_qhat,
        alphas=cfg.conformal_alphas,
    )

    # 9) Save artifact
    run_dir = make_run_dir(cfg.run_base_dir, cfg.run_name)

    artifact_config: Dict[str, Any] = {
        **asdict(cfg),
        "num_features": len(feature_cols),
        "feature_cols": list(feature_cols),
        "horizons_sorted": horizons_sorted,
        "output_dim": len(horizons_sorted),
        "channels": list(cfg.channels),
        "kernel_size": cfg.kernel_size,
        "dropout": cfg.dropout,
        "loss_name": loss_name,
        "best_epoch": best_epoch,
        "best_val_rmse_mean": best_val_rmse,
        "n_samples": {
            "train": int(len(train_ds)),
            "val": int(len(val_ds)),
            "test": int(len(test_ds)),
        },
        "n_unique_target_timestamps": {
            "train": int(len(split.train_targets)),
            "val": int(len(split.val_targets)),
            "test": int(len(split.test_targets)),
        },
    }

    metrics_out: Dict[str, Any] = {
        "val": val_metrics_by_h,
        "test": test_metrics_by_h,
        "conformal": {
            "alphas": list(map(float, cfg.conformal_alphas)),
            "qhat_by_horizon": conformal_qhat,
            "test": conformal_test,
        },
        "best_val_rmse_mean": best_val_rmse,
        "best_epoch": best_epoch,
    }

    save_tcn_artifact(
        run_dir,
        config=artifact_config,
        model=model.cpu(),  # save on CPU for portability
        scaler=scaler,
        feature_names=list(feature_cols),
        horizons=horizons_sorted,
        metrics=metrics_out,
        training_history=history,
    )

    model.to(device)

    return {
        "run_dir": str(run_dir),
        "model": model,
        "scaler": scaler,
        "config": artifact_config,
        "metrics": metrics_out,
        "history": history,
        "splits": {
            "train_idx": split.train_idx,
            "val_idx": split.val_idx,
            "test_idx": split.test_idx,
            "train_targets": split.train_targets.astype(str).tolist(),
            "val_targets": split.val_targets.astype(str).tolist(),
            "test_targets": split.test_targets.astype(str).tolist(),
        },
        "horizons": horizons_sorted,
        "meta": {
            "train": meta_train,
            "val": meta_val,
            "test": meta_test,
        },
    }


if __name__ == "__main__":
    print("Entrenamiento TCN: importa train_eval_tcn() desde tu pipeline. (Ver ejemplo de uso en la respuesta.)")
