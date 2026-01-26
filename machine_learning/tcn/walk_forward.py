from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class WalkForwardConfig:
    target_col: str = "target_timestamp"  # OBLIGATORIO: split por target_timestamp
    train_span: Optional[int] = 252 * 3   # en unidades de nº target_timestamps únicos; None => expanding
    val_span: int = 126
    test_span: int = 126
    step_span: int = 126
    embargo_span: int = 0                 # gap opcional (en target_timestamps)
    min_train_span: int = 252             # mínimo de targets en train para fold válido


@dataclass(frozen=True)
class FoldSpec:
    fold: int
    train_targets: pd.Index
    val_targets: pd.Index
    test_targets: pd.Index
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_target_timestamp_walk_forward_folds(meta: pd.DataFrame, cfg: WalkForwardConfig) -> List[FoldSpec]:
    """
    Genera folds walk-forward basados EXCLUSIVAMENTE en target_timestamp únicos ordenados.
    - Train -> Val -> Test contiguos (salvo embargo_span opcional).
    - Rolling si train_span != None; expanding si train_span=None.
    """
    if cfg.target_col not in meta.columns:
        raise KeyError(f"meta debe tener '{cfg.target_col}'")

    targets = pd.to_datetime(meta[cfg.target_col])
    unique_targets = pd.Index(sorted(targets.unique()))
    n = len(unique_targets)
    if n < (cfg.val_span + cfg.test_span + 5):
        raise ValueError(f"No hay suficientes target_timestamps únicos: n={n}")

    folds: List[FoldSpec] = []
    fold_id = 1

    # índice base del primer val_start
    if cfg.train_span is None:
        base_train_end = max(cfg.min_train_span, 1)
    else:
        base_train_end = int(cfg.train_span)
        if base_train_end < cfg.min_train_span:
            raise ValueError("train_span < min_train_span")

    i = 0
    while True:
        if cfg.train_span is None:
            # expanding: train_start=0, train_end = base_train_end + i*step
            train_start_idx = 0
            train_end_idx_base = base_train_end + i * cfg.step_span
        else:
            # rolling: train window fijo
            train_start_idx = i * cfg.step_span
            train_end_idx_base = train_start_idx + int(cfg.train_span)

        # embargo: recorta train_end efectivo, pero val sigue pegado al train_end_base
        train_end_idx = train_end_idx_base - int(cfg.embargo_span)
        val_start_idx = train_end_idx_base
        val_end_idx = val_start_idx + int(cfg.val_span)
        test_start_idx = val_end_idx
        test_end_idx = test_start_idx + int(cfg.test_span)

        if test_end_idx > n:
            break
        if train_end_idx <= train_start_idx:
            break
        if (train_end_idx - train_start_idx) < cfg.min_train_span:
            break
        if val_start_idx < 0 or val_end_idx <= val_start_idx:
            break

        train_targets = unique_targets[train_start_idx:train_end_idx]
        val_targets = unique_targets[val_start_idx:val_end_idx]
        test_targets = unique_targets[test_start_idx:test_end_idx]

        f = FoldSpec(
            fold=fold_id,
            train_targets=train_targets,
            val_targets=val_targets,
            test_targets=test_targets,
            train_start=pd.Timestamp(train_targets[0]),
            train_end=pd.Timestamp(train_targets[-1]),
            val_start=pd.Timestamp(val_targets[0]),
            val_end=pd.Timestamp(val_targets[-1]),
            test_start=pd.Timestamp(test_targets[0]),
            test_end=pd.Timestamp(test_targets[-1]),
        )
        folds.append(f)

        fold_id += 1
        i += 1

    if len(folds) == 0:
        raise ValueError(
            "No se pudo construir ningún fold con esos parámetros. "
            "Prueba bajar val_span/test_span/step_span o bajar min_train_span."
        )

    return folds


def indices_for_fold(meta: pd.DataFrame, fold: FoldSpec, *, target_col: str = "target_timestamp") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Devuelve índices (np arrays) para train/val/test según membership en target_timestamps.
    """
    if target_col not in meta.columns:
        raise KeyError(f"meta no tiene '{target_col}'")

    tgt = pd.to_datetime(meta[target_col])

    train_mask = tgt.isin(fold.train_targets)
    val_mask = tgt.isin(fold.val_targets)
    test_mask = tgt.isin(fold.test_targets)

    train_idx = np.where(train_mask.to_numpy())[0]
    val_idx = np.where(val_mask.to_numpy())[0]
    test_idx = np.where(test_mask.to_numpy())[0]

    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise ValueError(
            f"Fold {fold.fold}: split vacío. "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}. "
            "Revisa spans o tu dataset."
        )

    return train_idx, val_idx, test_idx
