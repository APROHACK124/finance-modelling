from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, replace, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from artifacts import make_run_dir, save_tcn_artifact, verify_artifact_roundtrip
from data.seq_dataset import (
    SequenceDataset,
    SequenceStandardScaler,
    build_multi_horizon_supervised_dataset,
)
from data.timestamp_sampler import TimestampBatchSampler, TimestampBatchSamplerConfig
from evaluation import BacktestConfig, daily_cross_sectional_summary, toy_long_short_backtest
from losses import CombinedRegICLoss, LossConfig
from models.tcn import TCNConfig, TCNRegressor
from walk_forward import FoldSpec, WalkForwardConfig, indices_for_fold, make_target_timestamp_walk_forward_folds

from time import perf_counter


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


@torch.no_grad()
def predict_array(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    model.eval()
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        pb = model(xb).detach().cpu().numpy()
        preds.append(pb)
    return np.concatenate(preds, axis=0)


def _horizon_index(horizons: Sequence[int], trade_horizon: int) -> int:
    if trade_horizon not in horizons:
        raise ValueError(f"trade_horizon={trade_horizon} no está en horizons={list(horizons)}")
    return list(horizons).index(trade_horizon)


@dataclass
class ModelHP:
    channels: Sequence[int] = (64, 64, 64, 64)
    kernel_size: int = 3
    dropout: float = 0.10


@dataclass
class TrainHP:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 50
    patience: int = 10
    grad_clip: float = 1.0
    train_max_batch_size: Optional[int] = None  # max cs batch; None => todo el día
    num_workers: int = 0
    pin_memory: bool = True


@dataclass
class LossHP:
    ic_lambda: float = 0.2
    min_cs_size: int = 20
    smoothl1_beta: float = 1.0


@dataclass
class EvalHP:
    eval_batch_size: int = 4096
    quantile: float = 0.1
    periods_per_year: int = 252
    cost_bps: float = 0.0
    trade_horizon: int = 5  # horizonte usado para toy backtest


@dataclass
class ExperimentConfig:
    timeframe: Optional[str] = None
    lookback: int = 60
    horizons: Sequence[int] = (5, 20, 60)
    price_col: str = "close"
    group_col: str = "symbol"
    timestamp_col: str = "timestamp"

    # Walk-forward por TARGET timestamp
    wf: WalkForwardConfig = WalkForwardConfig()

    # HPs
    model_hp: ModelHP = field(default_factory=ModelHP)
    
    train_hp: TrainHP = field(default_factory=TrainHP)
    loss_hp: LossHP = field(default_factory=LossHP)
    eval_hp: EvalHP = field(default_factory=EvalHP)

    # Repro/Device
    seed: int = 42
    device: Optional[str] = None

    # Artifact
    run_base_dir: str = "runs"
    run_name: Optional[str] = None

    # Verificación
    verify_atol: float = 1e-6
    verify_each_fold: bool = True


from datetime import datetime, timezone
from dataclasses import asdict
from typing import Any, Dict, Sequence

def build_artifact_cfg_base(
    *,
    cfg: "ExperimentConfig",
    feature_cols: Sequence[str],
    horizons: Sequence[int],
    scaler: Any,
    sampler_cfg: "TimestampBatchSamplerConfig",
) -> Dict[str, Any]:
    """
    Base config para artifacts (fold y final).
    - Incluye TODO lo necesario para re-crear el modelo + entender cómo fue entrenado.
    - Mantiene claves "planas" (backward compat) y además agrega snapshots anidados.
    """
    horizons_list = [int(h) for h in horizons]
    channels_list = [int(c) for c in cfg.model_hp.channels]

    base = {
        # --- Identidad / versionado (no afecta reload, pero ayuda)
        "artifact_version": 1,
        "artifact_family": "tcn_regressor",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),

        # --- Lo mínimo necesario para re-instanciar arquitectura + saber orden de inputs/outputs
        "num_features": int(len(feature_cols)),
        "feature_cols": list(feature_cols),
        "base_feature_cols": list(feature_cols),

        # IMPORTANTE: este orden debe reflejar EXACTAMENTE el orden del output del modelo
        "horizons": horizons_list,
        # Mantengo tu clave original por compatibilidad
        "horizons_sorted": list(horizons_list),

        "output_dim": int(len(horizons_list)),
        "channels": list(channels_list),
        "kernel_size": int(cfg.model_hp.kernel_size),
        "dropout": float(cfg.model_hp.dropout),

        "scaler_type": scaler.__class__.__name__,

        # --- Especificación de dataset/meta (para poder re-evaluar/backtest sin cfg externo)
        "lookback": int(cfg.lookback),
        "price_col": cfg.price_col,
        "group_col": cfg.group_col,
        "timestamp_col": cfg.timestamp_col,
        "target_col": cfg.wf.target_col,

        # --- HPs (snapshots)
        "train_hp": {
            "lr": float(cfg.train_hp.lr),
            "weight_decay": float(cfg.train_hp.weight_decay),
            "max_epochs": int(cfg.train_hp.max_epochs),
            "patience": int(cfg.train_hp.patience),
            "grad_clip": float(cfg.train_hp.grad_clip),
            "train_max_batch_size": None if cfg.train_hp.train_max_batch_size is None else int(cfg.train_hp.train_max_batch_size),
            "num_workers": int(cfg.train_hp.num_workers),
            "pin_memory": bool(cfg.train_hp.pin_memory),
        },
        "loss_hp": {
            "ic_lambda": float(cfg.loss_hp.ic_lambda),
            "min_cs_size": int(cfg.loss_hp.min_cs_size),
            "smoothl1_beta": float(cfg.loss_hp.smoothl1_beta),
        },
        "eval_hp": {
            "eval_batch_size": int(cfg.eval_hp.eval_batch_size),
            "quantile": float(cfg.eval_hp.quantile),
            "periods_per_year": int(cfg.eval_hp.periods_per_year),
            "cost_bps": float(cfg.eval_hp.cost_bps),
            "trade_horizon": int(cfg.eval_hp.trade_horizon),
        },

        # --- Sampler (importante porque afecta IC loss y reproducibilidad)
        "sampler_cfg": asdict(sampler_cfg),

        # --- Repro info
        "repro": {
            "seed": int(cfg.seed),
            "device": cfg.device,
            "run_base_dir": cfg.run_base_dir,
            "run_name": cfg.run_name,
        },

        # --- Backward compatibility: dejo también estas claves planas como tú ya las tenías
        "lr": float(cfg.train_hp.lr),
        "weight_decay": float(cfg.train_hp.weight_decay),
        "max_epochs": int(cfg.train_hp.max_epochs),
        "patience": int(cfg.train_hp.patience),
        "grad_clip": float(cfg.train_hp.grad_clip),
        "ic_lambda": float(cfg.loss_hp.ic_lambda),
        "min_cs_size": int(cfg.loss_hp.min_cs_size),
        "smoothl1_beta": float(cfg.loss_hp.smoothl1_beta),
    }
    return base



def train_single_fold(
    *,
    fold: FoldSpec,
    fold_dir: Path,
    X_all: np.ndarray,
    y_all: np.ndarray,
    meta_all: pd.DataFrame,
    feature_cols: Sequence[str],
    horizons: Sequence[int],
    eval_fn: Callable[..., Dict[str, Any]],
    cfg: ExperimentConfig,
    fold_seed: int,
) -> Dict[str, Any]:
    """
    Entrena desde cero en un fold:
      - split por target_timestamp (ya viene en fold)
      - scaler fit SOLO en train
      - train batches por timestamp (decision time) => IC_loss correcto
      - early stopping: maximiza mean DailyIC en val (OOS)
      - eval + toy backtest en test
      - guarda artifact fold + verificación save/load/infer
    """
    device = get_device(cfg.device)
    set_global_seed(fold_seed, deterministic=True)

    train_idx, val_idx, test_idx = indices_for_fold(meta_all, fold, target_col=cfg.wf.target_col)
    import gc

    X_tr, y_tr, m_tr = X_all[train_idx], y_all[train_idx], meta_all.iloc[train_idx].reset_index(drop=True)
    X_va, y_va, m_va = X_all[val_idx],   y_all[val_idx],   meta_all.iloc[val_idx].reset_index(drop=True)
    X_te, y_te, m_te = X_all[test_idx],  y_all[test_idx],  meta_all.iloc[test_idx].reset_index(drop=True)

    # Guarda SOLO lo necesario para verify (no guardes todo el X_te unscaled)
    n_verify = int(min(1024, len(X_te)))
    X_te_verify = X_te[:n_verify].copy()

    scaler = SequenceStandardScaler()

    X_tr_s = scaler.fit_transform(X_tr)
    del X_tr
    X_va_s = scaler.transform(X_va)
    del X_va
    X_te_s = scaler.transform(X_te)
    del X_te

    gc.collect()

    # Datasets
    ds_tr = SequenceDataset(X_tr_s, y_tr, m_tr)
    ds_va = SequenceDataset(X_va_s, y_va, m_va)
    ds_te = SequenceDataset(X_te_s, y_te, m_te)

    # Train loader por timestamp (decision time)
    sampler_cfg = TimestampBatchSamplerConfig(
        time_col=cfg.timestamp_col,
        min_cs_size=cfg.loss_hp.min_cs_size,
        shuffle=True,
        seed=fold_seed,
        max_batch_size=cfg.train_hp.train_max_batch_size,
        drop_last=False,
    )
    batch_sampler = TimestampBatchSampler(ds_tr.meta, cfg=sampler_cfg)

    train_loader = DataLoader(
        ds_tr,
        batch_sampler=batch_sampler,
        num_workers=cfg.train_hp.num_workers,
        pin_memory=(cfg.train_hp.pin_memory and device.type == "cuda"),
    )

    # Val/Test loaders para predicción
    val_loader = DataLoader(
        ds_va,
        batch_size=cfg.eval_hp.eval_batch_size,
        shuffle=False,
        num_workers=cfg.train_hp.num_workers,
        pin_memory=(cfg.train_hp.pin_memory and device.type == "cuda"),
    )
    test_loader = DataLoader(
        ds_te,
        batch_size=cfg.eval_hp.eval_batch_size,
        shuffle=False,
        num_workers=cfg.train_hp.num_workers,
        pin_memory=(cfg.train_hp.pin_memory and device.type == "cuda"),
    )

    # Model
    model_cfg = TCNConfig(
        num_features=len(feature_cols),
        channels=tuple(cfg.model_hp.channels),
        kernel_size=int(cfg.model_hp.kernel_size),
        dropout=float(cfg.model_hp.dropout),
        output_dim=len(horizons),
    )
    model = TCNRegressor(model_cfg).to(device)

    # Loss
    loss_fn = CombinedRegICLoss(
        LossConfig(ic_lambda=cfg.loss_hp.ic_lambda, min_cs_size=cfg.loss_hp.min_cs_size),
        beta=cfg.loss_hp.smoothl1_beta,
    )
    optim = AdamW(model.parameters(), lr=cfg.train_hp.lr, weight_decay=cfg.train_hp.weight_decay)

    # Early stopping
    best_score = -np.inf
    best_epoch = -1
    best_state = None
    no_improve = 0
    history: List[Dict[str, Any]] = []

    def _val_score(y_true, y_pred):
        scores = []
        for j, h in enumerate(horizons):
            cs = daily_cross_sectional_summary(
                y_true[:, j], y_pred[:, j], m_va,
                time_col=cfg.timestamp_col,
                min_cs_size=cfg.loss_hp.min_cs_size,
                quantile=cfg.eval_hp.quantile,
            )
            s = float(cs.daily_rankic_mean)
            w = 2.0 if h == cfg.eval_hp.trade_horizon else 1.0
            scores.append(w * s)
        return float(np.sum(scores) / np.sum([2.0 if h==cfg.eval_hp.trade_horizon else 1.0 for h in horizons]))



    t_start = time.time()
    for epoch in range(1, cfg.train_hp.max_epochs + 1):
        t0 = perf_counter()
        t_train0 = perf_counter()
        model.train()
        batch_sampler.set_epoch(epoch)

        loss_sum = torch.zeros((), device=device)
        n_batches = 0

        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
        
            optim.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
        
            if cfg.train_hp.grad_clip and cfg.train_hp.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train_hp.grad_clip)
        
            optim.step()
        
            loss_sum += loss.detach()
            n_batches += 1

        if device.type == "cuda":
            torch.cuda.synchronize()
        train_loss = (loss_sum / max(1, n_batches)).item()
        t_train = perf_counter() - t_train0

        # Val
        t_valpred0 = perf_counter()
        model.eval()
        y_va_pred = []
        y_va_true = []
        with torch.inference_mode():
            for xb, yb, _ in val_loader:
                xb = xb.to(device, non_blocking=True)
                pb = model(xb).detach().cpu().numpy()
                y_va_pred.append(pb)
                y_va_true.append(yb.detach().cpu().numpy())

        if device.type == "cuda":
            torch.cuda.synchronize()
        t_valpred = perf_counter() - t_valpred0
        
        y_va_pred = np.concatenate(y_va_pred, axis=0)
        y_va_true = np.concatenate(y_va_true, axis=0)

        t_score0 = perf_counter()
        score = _val_score(y_va_true, y_va_pred)
        t_score = perf_counter() - t_score0

        t_total = perf_counter() - t0
        
        improved = score > (best_score + 1e-10)

        if improved:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_mean_daily_ic": score,
            "best_val_mean_daily_ic": best_score,
            "no_improve": no_improve,
            "elapsed_sec": float(time.time() - t_start),
        })

        print(
            f"[fold {fold.fold:02d}][epoch {epoch:03d}] "
            f"train_loss={train_loss:.6f} | val_mean_dailyIC={score:.6f} | "
            f"best={best_score:.6f}@{best_epoch} | no_improve={no_improve}"
            f"time_train={t_train:.1f}s time_valpred={t_valpred:.1f}s time_score={t_score:.1f}s total={t_total:.1f}s"
        )

        if no_improve >= cfg.train_hp.patience:
            print(f"[fold {fold.fold:02d}] Early stopping: {cfg.train_hp.patience} epochs sin mejora.")
            break

    if best_state is None:
        raise RuntimeError(f"Fold {fold.fold}: no best_state (inesperado).")

    # Restore best
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    # Predicciones finales val/test
    y_va_pred = predict_array(model, X_va_s, device=device, batch_size=cfg.eval_hp.eval_batch_size)
    y_te_pred = predict_array(model, X_te_s, device=device, batch_size=cfg.eval_hp.eval_batch_size)

    # Eval por horizonte
    val_metrics_by_h: Dict[int, Dict[str, Any]] = {}
    test_metrics_by_h: Dict[int, Dict[str, Any]] = {}
    val_cs_summary_by_h: Dict[int, Dict[str, Any]] = {}
    test_cs_summary_by_h: Dict[int, Dict[str, Any]] = {}

    for j, h in enumerate(horizons):
        val_metrics_by_h[h] = eval_fn(
            y_va[:, j], y_va_pred[:, j],
            meta=m_va,
            time_col=cfg.timestamp_col,
            group_col=cfg.group_col,
            periods_per_year=cfg.eval_hp.periods_per_year,
        )
        test_metrics_by_h[h] = eval_fn(
            y_te[:, j], y_te_pred[:, j],
            meta=m_te,
            time_col=cfg.timestamp_col,
            group_col=cfg.group_col,
            periods_per_year=cfg.eval_hp.periods_per_year,
        )

        cs_va = daily_cross_sectional_summary(
            y_va[:, j], y_va_pred[:, j], m_va,
            time_col=cfg.timestamp_col,
            min_cs_size=cfg.loss_hp.min_cs_size,
            quantile=cfg.eval_hp.quantile,
        )
        cs_te = daily_cross_sectional_summary(
            y_te[:, j], y_te_pred[:, j], m_te,
            time_col=cfg.timestamp_col,
            min_cs_size=cfg.loss_hp.min_cs_size,
            quantile=cfg.eval_hp.quantile,
        )
        val_cs_summary_by_h[h] = asdict(cs_va)
        test_cs_summary_by_h[h] = asdict(cs_te)

    # Toy backtest en test
    trade_h = cfg.eval_hp.trade_horizon
    j_trade = _horizon_index(horizons, trade_h)
    bt = toy_long_short_backtest(
        y_true=y_te[:, j_trade],
        y_pred=y_te_pred[:, j_trade],
        meta=m_te,
        cfg=BacktestConfig(
            time_col=cfg.timestamp_col,
            group_col=cfg.group_col,
            quantile=cfg.eval_hp.quantile,
            min_cs_size=cfg.loss_hp.min_cs_size,
            cost_bps=cfg.eval_hp.cost_bps,
            periods_per_year=cfg.eval_hp.periods_per_year,
        ),
    )
    bt_metrics = {k: v for k, v in bt.items() if k != "series"}

    # Guardar artifact del fold (REUSABLE)
    fold_dir.mkdir(parents=True, exist_ok=True)

    artifact_cfg = build_artifact_cfg_base(
        cfg=cfg,
        feature_cols=feature_cols,
        horizons=horizons,
        scaler=scaler,
        sampler_cfg=sampler_cfg,
    )
    artifact_cfg.update({
        "artifact_kind": "fold",
        "fold": int(fold.fold),
        "fold_seed": int(fold_seed),
        "fold_boundaries": {
            "train_start_target": str(fold.train_start),
            "train_end_target": str(fold.train_end),
            "val_start_target": str(fold.val_start),
            "val_end_target": str(fold.val_end),
            "test_start_target": str(fold.test_start),
            "test_end_target": str(fold.test_end),
        },
        "best_epoch": int(best_epoch),
        "best_val_mean_daily_ic": float(best_score),
        "n_samples": {"train": int(len(ds_tr)), "val": int(len(ds_va)), "test": int(len(ds_te))},
        "training_strategy": "early_stopping",
    })

    metrics_out = {
        "val": val_metrics_by_h,
        "test": test_metrics_by_h,
        "val_cs_summary": val_cs_summary_by_h,
        "test_cs_summary": test_cs_summary_by_h,
        "toy_backtest_test": bt_metrics,
        "best_epoch": int(best_epoch),
        "best_val_mean_daily_ic": float(best_score),
        "n_samples": {"train": int(len(ds_tr)), "val": int(len(ds_va)), "test": int(len(ds_te))},
    }

    save_tcn_artifact(
        fold_dir,
        config=artifact_cfg,
        model=model.to("cpu"),
        scaler=scaler,
        feature_names=list(feature_cols),
        horizons=list(horizons),
        metrics=metrics_out,
        training_history=history,
    )
    model.to(device)

    # Verificación save→load→infer (por fold)
    verify_info = None
    if cfg.verify_each_fold:
        verify_info = verify_artifact_roundtrip(
            fold_dir,
            model=model,
            scaler=scaler,
            X_unscaled=X_te_verify,   # <- aquí
            atol=cfg.verify_atol,
            rtol=0.0,
            map_location="cpu",
        )

        print(f"[fold {fold.fold:02d}] artifact verify: {verify_info}")

    return {
        "fold": fold.fold,
        "fold_boundaries": artifact_cfg["fold_boundaries"],
        "best_epoch": best_epoch,
        "best_val_mean_daily_ic": best_score,
        "val_cs_summary": val_cs_summary_by_h,
        "test_cs_summary": test_cs_summary_by_h,
        "toy_backtest_test": bt_metrics,
        "artifact_dir": str(fold_dir),
        "verify": verify_info,
    }



def aggregate_fold_results(fold_results: List[Dict[str, Any]], horizons: Sequence[int], trade_horizon: int) -> pd.DataFrame:
    rows = []
    for r in fold_results:
        row = {
            "fold": r["fold"],
            "best_epoch": r["best_epoch"],
            "best_val_mean_daily_ic": r["best_val_mean_daily_ic"],
            "bt_sharpe": r["toy_backtest_test"].get("sharpe_ann", np.nan),
            "bt_max_dd": r["toy_backtest_test"].get("max_drawdown", np.nan),
            "bt_turnover": r["toy_backtest_test"].get("turnover_avg", np.nan),
            "bt_cum_ret": r["toy_backtest_test"].get("cum_return", np.nan),
        }
        for h in horizons:
            va = r["val_cs_summary"].get(h, {})
            te = r["test_cs_summary"].get(h, {})
            row[f"val_dailyIC_h{h}"] = va.get("daily_ic_mean", np.nan)
            row[f"val_dailyRankIC_h{h}"] = va.get("daily_rankic_mean", np.nan)
            row[f"val_qspread_h{h}"] = va.get("quantile_spread_mean", np.nan)
            row[f"test_dailyIC_h{h}"] = te.get("daily_ic_mean", np.nan)
            row[f"test_dailyRankIC_h{h}"] = te.get("daily_rankic_mean", np.nan)
            row[f"test_qspread_h{h}"] = te.get("quantile_spread_mean", np.nan)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("fold").reset_index(drop=True)
    return df




def train_final_model(
    *,
    final_dir: Path,
    X_all: np.ndarray,
    y_all: np.ndarray,
    meta_all: pd.DataFrame,
    feature_cols: Sequence[str],
    horizons: Sequence[int],
    last_fold: FoldSpec,
    eval_fn: Callable[..., Dict[str, Any]],
    cfg: ExperimentConfig,
    final_epochs: int,
) -> Dict[str, Any]:
    """
    Modelo final:
    - entrenado SOLO con datos anteriores al test del último fold (train+val del último fold),
      evitando tocar el test.
    - nº epochs fijo = mediana(best_epoch por fold) (robusto, usa solo validaciones OOS).
    """
    device = get_device(cfg.device)

    import gc

    # train_final targets = train_targets U val_targets del último fold
    train_targets = last_fold.train_targets.union(last_fold.val_targets)

    tgt = pd.to_datetime(meta_all[cfg.wf.target_col])
    mask_tr = tgt.isin(train_targets)
    mask_te = tgt.isin(last_fold.test_targets)

    tr_idx = np.where(mask_tr.to_numpy())[0]
    te_idx = np.where(mask_te.to_numpy())[0]

    X_tr, y_tr, m_tr = X_all[tr_idx], y_all[tr_idx], meta_all.iloc[tr_idx].reset_index(drop=True)
    X_te, y_te, m_te = X_all[te_idx], y_all[te_idx], meta_all.iloc[te_idx].reset_index(drop=True)

    n_verify = int(min(1024, len(X_te)))
    X_te_verify = X_te[:n_verify].copy()

    scaler = SequenceStandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    del X_tr
    X_te_s = scaler.transform(X_te)
    del X_te
    gc.collect()

    ds_tr = SequenceDataset(X_tr_s, y_tr, m_tr)

    sampler_cfg = TimestampBatchSamplerConfig(
        time_col=cfg.timestamp_col,
        min_cs_size=cfg.loss_hp.min_cs_size,
        shuffle=True,
        seed=cfg.seed + 999,
        max_batch_size=cfg.train_hp.train_max_batch_size,
        drop_last=False,
    )
    batch_sampler = TimestampBatchSampler(ds_tr.meta, cfg=sampler_cfg)

    train_loader = DataLoader(
        ds_tr,
        batch_sampler=batch_sampler,
        num_workers=cfg.train_hp.num_workers,
        pin_memory=(cfg.train_hp.pin_memory and device.type == "cuda"),
    )

    model_cfg = TCNConfig(
        num_features=len(feature_cols),
        channels=tuple(cfg.model_hp.channels),
        kernel_size=int(cfg.model_hp.kernel_size),
        dropout=float(cfg.model_hp.dropout),
        output_dim=len(horizons),
    )
    model = TCNRegressor(model_cfg).to(device)

    loss_fn = CombinedRegICLoss(
        LossConfig(ic_lambda=cfg.loss_hp.ic_lambda, min_cs_size=cfg.loss_hp.min_cs_size),
        beta=cfg.loss_hp.smoothl1_beta,
    )
    optim = AdamW(model.parameters(), lr=cfg.train_hp.lr, weight_decay=cfg.train_hp.weight_decay)

    # Train fijo (guardamos history para inspección/repro)
    set_global_seed(cfg.seed + 12345, deterministic=True)
    history: List[Dict[str, Any]] = []

    for epoch in range(1, final_epochs + 1):
        model.train()
        batch_sampler.set_epoch(epoch)
        losses = []
        for xb, yb, _ in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()

            if cfg.train_hp.grad_clip and cfg.train_hp.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train_hp.grad_clip)

            optim.step()
            losses.append(float(loss.detach().cpu().item()))

        train_loss = float(np.mean(losses)) if losses else float("nan")
        history.append({"epoch": int(epoch), "train_loss": float(train_loss)})

        print(f"[final][epoch {epoch:03d}/{final_epochs}] train_loss={train_loss:.6f}")

    # Eval en el test del último fold (solo reporte; no se usó para training)
    model.eval()
    y_te_pred = predict_array(model, X_te_s, device=device, batch_size=cfg.eval_hp.eval_batch_size)

    test_metrics_by_h: Dict[int, Dict[str, Any]] = {}
    test_cs_summary_by_h: Dict[int, Dict[str, Any]] = {}
    for j, h in enumerate(horizons):
        test_metrics_by_h[h] = eval_fn(
            y_te[:, j], y_te_pred[:, j],
            meta=m_te,
            time_col=cfg.timestamp_col,
            group_col=cfg.group_col,
            periods_per_year=cfg.eval_hp.periods_per_year,
        )
        cs = daily_cross_sectional_summary(
            y_te[:, j], y_te_pred[:, j], m_te,
            time_col=cfg.timestamp_col,
            min_cs_size=cfg.loss_hp.min_cs_size,
            quantile=cfg.eval_hp.quantile,
        )
        test_cs_summary_by_h[h] = asdict(cs)

    # Backtest en ese test (horizonte elegido)
    j_trade = _horizon_index(horizons, cfg.eval_hp.trade_horizon)
    bt = toy_long_short_backtest(
        y_true=y_te[:, j_trade],
        y_pred=y_te_pred[:, j_trade],
        meta=m_te,
        cfg=BacktestConfig(
            time_col=cfg.timestamp_col,
            group_col=cfg.group_col,
            quantile=cfg.eval_hp.quantile,
            min_cs_size=cfg.loss_hp.min_cs_size,
            cost_bps=cfg.eval_hp.cost_bps,
            periods_per_year=cfg.eval_hp.periods_per_year,
        ),
    )
    bt_metrics = {k: v for k, v in bt.items() if k != "series"}

    # Save artifact final (REUSABLE)
    final_dir.mkdir(parents=True, exist_ok=True)

    artifact_cfg = build_artifact_cfg_base(
        cfg=cfg,
        feature_cols=feature_cols,
        horizons=horizons,
        scaler=scaler,
        sampler_cfg=sampler_cfg,
    )

    n_test = int(len(y_te))

    train_symbols = sorted(m_tr[cfg.group_col].astype(str).unique().tolist())

    artifact_cfg.update({
        "timeframe": getattr(cfg, "timeframe", None),  # si lo agregas a ExperimentConfig
        "symbols": train_symbols,
        "n_symbols": int(len(train_symbols)),
        "artifact_kind": "final",
        "training_strategy": "fixed_epochs",
        "final_epochs": int(final_epochs),

        # Info del split usado (para auditar leakage)
        "trained_on": {
            "train_start_target": str(last_fold.train_start),
            "train_end_target": str(last_fold.val_end),   # train+val
            "excluded_test_start_target": str(last_fold.test_start),
            "excluded_test_end_target": str(last_fold.test_end),
        },
        "model": "tcn_regressor",
        "n_samples": {"train": int(len(ds_tr)), "test": n_test}
    })

    metrics_out = {
        "test": test_metrics_by_h,
        "test_cs_summary": test_cs_summary_by_h,
        "toy_backtest_test": bt_metrics,
        "n_samples": {"train": int(len(ds_tr)), "test": n_test},
    }

    save_tcn_artifact(
        final_dir,
        config=artifact_cfg,
        model=model.to("cpu"),
        scaler=scaler,
        feature_names=list(feature_cols),
        horizons=list(horizons),
        metrics=metrics_out,
        training_history=history,   # ahora sí queda registro
    )
    model.to(device)

    # Verificación OBLIGATORIA (subset fijo del test del último fold)
    verify_info = verify_artifact_roundtrip(
        final_dir,
        model=model,
        scaler=scaler,
        X_unscaled=X_te_verify,
        atol=cfg.verify_atol,
        rtol=0.0,
        map_location="cpu",
    )
    print(f"[final] artifact verify: {verify_info}")

    return {
        "final_dir": str(final_dir),
        "final_epochs": final_epochs,
        "verify": verify_info,
        "toy_backtest_test": bt_metrics,
        "test_cs_summary": test_cs_summary_by_h,
    }



def run_walk_forward_tcn(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    build_supervised_dataset_fn: Callable[..., Tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
    eval_fn: Callable[..., Dict[str, Any]],
    cfg: ExperimentConfig = ExperimentConfig(),
    hp_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Pipeline completo:
      1) build dataset multi-horizon (wide->3D) + meta con target_timestamp
      2) genera folds por target_timestamp
      3) (opcional) selección simple de hiperparámetros por mean val DailyIC promedio en folds
      4) entrena/eval por fold, guarda artifacts por fold + verificación
      5) agrega resultados
      6) entrena modelo final (train+val último fold) + guarda + verificación obligatoria
    """
    set_global_seed(cfg.seed, deterministic=True)
    run_dir = make_run_dir(cfg.run_base_dir, cfg.run_name)
    folds_dir = run_dir / "folds"
    final_dir = run_dir / "final_model"
    folds_dir.mkdir(parents=True, exist_ok=True)

    # 1) Dataset (multi-horizon)
    X_all, y_all, meta_all, horizons_sorted = build_multi_horizon_supervised_dataset(
        df,
        feature_cols=list(feature_cols),
        lookback=cfg.lookback,
        horizons=cfg.horizons,
        build_supervised_dataset_fn=build_supervised_dataset_fn,
        price_col=cfg.price_col,
        group_col=cfg.group_col,
        timestamp_col=cfg.timestamp_col,
        lags_by_feature=None,  # tu modo clásico
        dtype=np.float32, # type: ignore
    )

    # 2) Folds por TARGET timestamp
    wf_cfg = cfg.wf
    folds = make_target_timestamp_walk_forward_folds(meta_all, wf_cfg)

    # Guardar config global del run
    global_cfg = asdict(cfg)
    global_cfg["horizons_sorted"] = list(horizons_sorted)
    global_cfg["n_samples"] = int(len(X_all))
    global_cfg["n_features"] = int(len(feature_cols))
    global_cfg["run_dir"] = str(run_dir)
    (run_dir / "config.json").write_text(json.dumps(global_cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # 3) Selección simple de HP (opcional)
    #    hp_candidates es lista de overrides, ej:
    #      [{"loss_hp": {"ic_lambda": 0.0}}, {"loss_hp": {"ic_lambda": 0.2}}]
    #    Para mantenerlo ejecutable sin magia, solo soportamos overrides superficiales de dataclasses.
    chosen_cfg = cfg
    chosen_overrides = None
    if hp_candidates and len(hp_candidates) > 1:
        print("\n=== Hyperparam selection (val mean DailyIC promedio en folds) ===")
        cand_scores = []
        for ci, ov in enumerate(hp_candidates, start=1):
            # aplica overrides soportados: model_hp/train_hp/loss_hp/eval_hp
            tmp = cfg
            if "model_hp" in ov:
                tmp = replace(tmp, model_hp=replace(tmp.model_hp, **ov["model_hp"]))
            if "train_hp" in ov:
                tmp = replace(tmp, train_hp=replace(tmp.train_hp, **ov["train_hp"]))
            if "loss_hp" in ov:
                tmp = replace(tmp, loss_hp=replace(tmp.loss_hp, **ov["loss_hp"]))
            if "eval_hp" in ov:
                tmp = replace(tmp, eval_hp=replace(tmp.eval_hp, **ov["eval_hp"]))

            fold_scores = []
            for f in folds:
                # entreno rápido por fold solo para score (sin guardar)
                # Para no duplicar código, entreno pero no guardo artifact: guardo a un subdir temporal y lo borro.
                tmp_dir = folds_dir / f"_hpsearch_tmp_fold{f.fold:02d}_cand{ci:02d}"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                res = train_single_fold(
                    fold=f,
                    fold_dir=tmp_dir,
                    X_all=X_all,
                    y_all=y_all,
                    meta_all=meta_all,
                    feature_cols=feature_cols,
                    horizons=horizons_sorted,
                    eval_fn=eval_fn,
                    cfg=replace(tmp, verify_each_fold=False),  # no verificar en búsqueda
                    fold_seed=cfg.seed + 1000 * f.fold + 10 * ci,
                )
                fold_scores.append(float(res["best_val_mean_daily_ic"]))
                # Limpieza best-effort (no es crítica; puedes comentar si quieres inspeccionar)
                try:
                    for p in tmp_dir.rglob("*"):
                        if p.is_file():
                            p.unlink()
                    tmp_dir.rmdir()
                except Exception:
                    pass

            avg_score = float(np.mean(fold_scores)) if fold_scores else -np.inf
            cand_scores.append((avg_score, ov))
            print(f"Candidate {ci:02d} avg_val_mean_dailyIC={avg_score:.6f} overrides={ov}")

        cand_scores.sort(key=lambda x: x[0], reverse=True)
        best_score, best_ov = cand_scores[0]
        chosen_overrides = best_ov
        print(f"==> Seleccionado overrides={best_ov} con avg_val_mean_dailyIC={best_score:.6f}")

        # aplica al cfg final
        tmp = cfg
        if "model_hp" in best_ov:
            tmp = replace(tmp, model_hp=replace(tmp.model_hp, **best_ov["model_hp"]))
        if "train_hp" in best_ov:
            tmp = replace(tmp, train_hp=replace(tmp.train_hp, **best_ov["train_hp"]))
        if "loss_hp" in best_ov:
            tmp = replace(tmp, loss_hp=replace(tmp.loss_hp, **best_ov["loss_hp"]))
        if "eval_hp" in best_ov:
            tmp = replace(tmp, eval_hp=replace(tmp.eval_hp, **best_ov["eval_hp"]))
        chosen_cfg = tmp

        (run_dir / "hp_selected.json").write_text(json.dumps({"overrides": best_ov, "score": best_score}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # 4) Entrenamiento/eval por fold (con chosen_cfg) + artifacts + verify
    print("\n=== Walk-forward training (por fold) ===")
    fold_results: List[Dict[str, Any]] = []
    for f in folds:
        fold_dir = folds_dir / f"fold_{f.fold:02d}"
        res = train_single_fold(
            fold=f,
            fold_dir=fold_dir,
            X_all=X_all,
            y_all=y_all,
            meta_all=meta_all,
            feature_cols=feature_cols,
            horizons=horizons_sorted,
            eval_fn=eval_fn,
            cfg=chosen_cfg,
            fold_seed=chosen_cfg.seed + 1000 * f.fold,
        )
        fold_results.append(res)

    # 5) Agregación
    agg_df = aggregate_fold_results(fold_results, horizons_sorted, chosen_cfg.eval_hp.trade_horizon)
    print("\n=== Resultados por fold (tabla) ===")
    print(agg_df.to_string(index=False))

    summary = {}
    for col in agg_df.columns:
        if col == "fold":
            continue
        vals = pd.to_numeric(agg_df[col], errors="coerce")
        summary[col] = {"mean": float(np.nanmean(vals)), "std": float(np.nanstd(vals))}

    (run_dir / "fold_results.csv").write_text(agg_df.to_csv(index=False), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    # 6) Final model (train+val del último fold; sin tocar test)
    best_epochs = [int(r["best_epoch"]) for r in fold_results if r.get("best_epoch", -1) > 0]
    final_epochs = int(np.median(best_epochs)) if best_epochs else int(chosen_cfg.train_hp.max_epochs)
    last_fold = folds[-1]

    print("\n=== Entrenando modelo final (train+val del último fold) ===")
    final_res = train_final_model(
        final_dir=final_dir,
        X_all=X_all,
        y_all=y_all,
        meta_all=meta_all,
        feature_cols=feature_cols,
        horizons=horizons_sorted,
        last_fold=last_fold,
        eval_fn=eval_fn,
        cfg=chosen_cfg,
        final_epochs=final_epochs,
    )

    report = {
        "run_dir": str(run_dir),
        "n_folds": len(folds),
        "horizons": list(horizons_sorted),
        "chosen_overrides": chosen_overrides,
        "final_epochs": final_epochs,
        "fold_artifacts": [r["artifact_dir"] for r in fold_results],
        "final_artifact": final_res["final_dir"],
        "final_verify": final_res["verify"],
        "summary": summary,
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    print("\n=== DONE ===")
    print(f"run_dir: {run_dir}")
    print(f"final_model: {final_dir}")
    print(f"final verify: {final_res['verify']}")

    return {
        "run_dir": str(run_dir),
        "fold_results": fold_results,
        "agg_table": agg_df,
        "summary": summary,
        "final": final_res,
        "chosen_cfg": asdict(chosen_cfg),
    }
