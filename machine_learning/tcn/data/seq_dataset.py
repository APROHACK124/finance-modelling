from __future__ import annotations

import inspect
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset




def _call_build_supervised_dataset(
    fn: Callable[..., Tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
    df: pd.DataFrame,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Llama build_supervised_dataset pasando SOLO kwargs aceptados (para tolerar firmas distintas).
    df va posicional.
    """
    sig = inspect.signature(fn)
    accepted: Dict[str, object] = {}
    for k, v in kwargs.items():
        if k in sig.parameters:
            accepted[k] = v
    return fn(df, **accepted)  # type: ignore[misc]


# def build_multi_horizon_supervised_dataset(
#     df: pd.DataFrame,
#     *,
#     feature_cols: Sequence[str],
#     lookback: int,
#     horizons: Sequence[int],
#     build_supervised_dataset_fn: Callable[..., Tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
#     price_col: str = "close",
#     group_col: str = "symbol",
#     timestamp_col: str = "timestamp",
#     lags_by_feature=None,  # tu modo "clásico" suele ser None
#     dtype: np.dtype = np.float32, # type: ignore
# ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[int]]:
#     """
#     Construye dataset multi-horizonte (un único modelo con H outputs).
#     Alinea por (symbol, timestamp) en la intersección de horizontes.

#     Importante:
#     - meta tendrá SIEMPRE meta["target_timestamp"] para split.
#     - En multi-horizon, meta["target_timestamp"] == target_timestamp del horizonte MÁXIMO,
#       lo cual es una regla conservadora que evita leakage para todos los horizontes.
#     """
#     horizons_sorted = sorted({int(h) for h in horizons})
#     if len(horizons_sorted) < 1:
#         raise ValueError("horizons must be non-empty")
#     max_h = max(horizons_sorted)

#     per_h: Dict[int, Dict[str, object]] = {}
#     common_index: Optional[pd.MultiIndex] = None

#     for h in horizons_sorted:
#         X_wide, y_out, meta = _call_build_supervised_dataset(
#             build_supervised_dataset_fn,
#             df,
#             feature_cols=list(feature_cols),
#             lookback=int(lookback),
#             horizon=int(h),
#             price_col=price_col,
#             group_col=group_col,
#             timestamp_col=timestamp_col,
#             lags_by_feature=lags_by_feature,
#         )
#         if "target_timestamp" not in meta.columns:
#             raise KeyError("meta debe incluir 'target_timestamp' para split sin leakage")

#         meta = meta.copy()
#         meta[timestamp_col] = pd.to_datetime(meta[timestamp_col])
#         meta["target_timestamp"] = pd.to_datetime(meta["target_timestamp"])

#         idx = pd.MultiIndex.from_frame(meta[[group_col, timestamp_col]])

#         Xw = X_wide.copy()
#         Xw.index = idx

#         y_ser = pd.Series(np.asarray(y_out, dtype=np.float32), index=idx, name=f"y_{h}")

#         met = meta.copy()
#         met.index = idx

#         per_h[h] = {"X_wide": Xw, "y": y_ser, "meta": met}
#         common_index = idx if common_index is None else common_index.intersection(idx)

#     if common_index is None or len(common_index) == 0:
#         raise ValueError("No hay intersección de muestras entre horizontes (symbol,timestamp).")

#     common_index = common_index.sort_values()

#     # Usamos X del max horizon (idéntico en features si build_supervised_dataset es consistente).
#     X_wide_common = per_h[max_h]["X_wide"].loc[common_index]  # type: ignore[index]

#     # y: apilado (N,H)
#     y_mat = np.stack([per_h[h]["y"].loc[common_index].to_numpy(dtype=np.float32) for h in horizons_sorted], axis=1)  # type: ignore[index]

#     # meta base: symbol, timestamp y target_timestamp por horizonte
#     meta_base = per_h[max_h]["meta"].loc[common_index][[group_col, timestamp_col]].copy()  # type: ignore[index]
#     for h in horizons_sorted:
#         meta_base[f"target_timestamp_{h}"] = per_h[h]["meta"].loc[common_index]["target_timestamp"].values  # type: ignore[index]

#     # Convención: meta["target_timestamp"] = target del horizonte máximo (split conservador)
#     meta_base["target_timestamp"] = meta_base[f"target_timestamp_{max_h}"]
#     meta_base["target_timestamp"] = pd.to_datetime(meta_base["target_timestamp"])

#     # Limpieza non-finite
#     X_num = X_wide_common.to_numpy(dtype=np.float32, copy=False)
#     mask_finite = np.isfinite(X_num).all(axis=1) & np.isfinite(y_mat).all(axis=1)
#     if not np.all(mask_finite):
#         X_wide_common = X_wide_common.iloc[mask_finite]
#         y_mat = y_mat[mask_finite]
#         meta_base = meta_base.iloc[mask_finite]

#     # Orden determinista: por target_timestamp (label time), luego timestamp (decision), luego symbol
#     meta_ord = meta_base.copy()
#     meta_ord[group_col] = meta_ord[group_col].astype(str)

#     # Fix
#     sort_keys = ["target_timestamp", timestamp_col, group_col]

#     # Si alguna key está a la vez en columnas y en niveles del índice, pandas lo considera ambiguo.
#     # Solución: crear columnas temporales para ordenar (tomando explícitamente la columna),
#     # y ordenar usando esos nombres temporales.
#     tmp_cols = []
#     sort_by = []
#     for k in sort_keys:
#         if (k in meta_ord.columns) and (k in meta_ord.index.names):
#             tmp = f"__sort_{k}"
#             meta_ord[tmp] = meta_ord[k]      # <- fuerza usar la COLUMNA, no el index level
#             sort_by.append(tmp)
#             tmp_cols.append(tmp)
#         else:
#             sort_by.append(k)

#     meta_ord = meta_ord.sort_values(sort_by, kind="mergesort")

#     if tmp_cols:
#         meta_ord = meta_ord.drop(columns=tmp_cols)

#     order_idx = meta_ord.index  # MultiIndex

#     X_wide_common = X_wide_common.loc[order_idx]
#     y_mat = pd.DataFrame(y_mat, index=meta_base.index).loc[order_idx].to_numpy(dtype=np.float32)
#     meta_ord = meta_ord.reset_index(drop=True)

#     # wide -> 3D
#     X_seq = wide_lagged_df_to_3d(
#         X_wide_common.reset_index(drop=True),
#         feature_cols=list(feature_cols),
#         lookback=int(lookback),
#         dtype=dtype,
#         time_order="oldest_to_newest",
#     )

#     return X_seq, y_mat, meta_ord, horizons_sorted

import gc
from typing import Dict, List, Optional, Sequence, Tuple

def build_multi_horizon_supervised_dataset(
    df: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    lookback: int,
    horizons: Sequence[int],
    build_supervised_dataset_fn: Callable[..., Tuple[pd.DataFrame, pd.Series, pd.DataFrame]],
    price_col: str = "close",
    group_col: str = "symbol",
    timestamp_col: str = "timestamp",
    lags_by_feature=None,
    dtype: np.dtype = np.float32,  # type: ignore
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[int]]:
    horizons_sorted = sorted({int(h) for h in horizons})
    if not horizons_sorted:
        raise ValueError("horizons must be non-empty")
    max_h = max(horizons_sorted)

    common_index: Optional[pd.MultiIndex] = None

    # Guardamos SOLO lo necesario
    y_by_h: Dict[int, pd.Series] = {}
    tgt_by_h: Dict[int, pd.Series] = {}

    X_wide_max: Optional[pd.DataFrame] = None

    for h in horizons_sorted:
        X_wide, y_out, meta = _call_build_supervised_dataset(
            build_supervised_dataset_fn,
            df,
            feature_cols=list(feature_cols),
            lookback=int(lookback),
            horizon=int(h),
            price_col=price_col,
            group_col=group_col,
            timestamp_col=timestamp_col,
            lags_by_feature=lags_by_feature,
        )
        if "target_timestamp" not in meta.columns:
            raise KeyError("meta debe incluir 'target_timestamp' para split sin leakage")

        # Normaliza fechas
        meta = meta.copy()
        meta[timestamp_col] = pd.to_datetime(meta[timestamp_col])
        meta["target_timestamp"] = pd.to_datetime(meta["target_timestamp"])

        idx = pd.MultiIndex.from_frame(meta[[group_col, timestamp_col]])
        idx = idx.set_names([group_col, timestamp_col])

        # y y target_timestamp (ligeros)
        y_by_h[h] = pd.Series(np.asarray(y_out, dtype=np.float32), index=idx, name=f"y_{h}")
        tgt_by_h[h] = pd.Series(meta["target_timestamp"].to_numpy(), index=idx, name=f"target_timestamp_{h}")

        common_index = idx if common_index is None else common_index.intersection(idx)

        # Features: SOLO guardamos X del max horizon
        if h == max_h:
            # Evita float64 si se coló
            # (si ya es float32, copy=False no duplica)
            X_wide = X_wide.astype(np.float32, copy=False)
            X_wide.index = idx  # NO .copy()
            X_wide_max = X_wide

        # libera lo grande en horizontes no-max
        if h != max_h:
            del X_wide
        del y_out, meta
        gc.collect()

    if common_index is None or len(common_index) == 0:
        raise ValueError("No hay intersección de muestras entre horizontes (symbol,timestamp).")
    if X_wide_max is None:
        raise RuntimeError("X_wide_max is None (inesperado).")

    common_index = common_index.sort_values()

    # X del max horizon alineado
    X_wide_common = X_wide_max.loc[common_index]

    # y (N,H) alineado
    y_mat = np.stack(
        [y_by_h[h].loc[common_index].to_numpy(dtype=np.float32) for h in horizons_sorted],
        axis=1,
    )

    # meta base desde el índice (evita guardar meta gigante por horizonte)
    meta_base = pd.DataFrame({
        group_col: common_index.get_level_values(0).astype(str),
        timestamp_col: common_index.get_level_values(1),
    }, index=common_index)

    for h in horizons_sorted:
        meta_base[f"target_timestamp_{h}"] = tgt_by_h[h].loc[common_index].values

    meta_base["target_timestamp"] = meta_base[f"target_timestamp_{max_h}"]
    meta_base["target_timestamp"] = pd.to_datetime(meta_base["target_timestamp"])

    # Limpieza non-finite (sin materializar 3D todavía)
    X_num = X_wide_common.to_numpy(dtype=np.float32, copy=False)
    mask_finite = np.isfinite(X_num).all(axis=1) & np.isfinite(y_mat).all(axis=1)
    if not np.all(mask_finite):
        X_wide_common = X_wide_common.iloc[mask_finite]
        y_mat = y_mat[mask_finite]
        meta_base = meta_base.iloc[mask_finite]

    # Orden determinista
    meta_ord = meta_base.copy()
    sort_keys = ["target_timestamp", timestamp_col, group_col]

    tmp_cols = []
    sort_by = []
    for k in sort_keys:
        if (k in meta_ord.columns) and (k in meta_ord.index.names):
            tmp = f"__sort_{k}"
            meta_ord[tmp] = meta_ord[k]
            sort_by.append(tmp)
            tmp_cols.append(tmp)
        else:
            sort_by.append(k)

    meta_ord = meta_ord.sort_values(sort_by, kind="mergesort")
    if tmp_cols:
        meta_ord = meta_ord.drop(columns=tmp_cols)

    order_idx = meta_ord.index
    X_wide_common = X_wide_common.loc[order_idx]
    y_mat = pd.DataFrame(y_mat, index=meta_base.index).loc[order_idx].to_numpy(dtype=np.float32)

    # IMPORTANTÍSIMO: evita reset_index(drop=True) (puede copiar)
    X_wide_common.index = range(len(X_wide_common))

    meta_ord = meta_ord.reset_index(drop=True)

    # wide -> 3D
    X_seq = wide_lagged_df_to_3d(
        X_wide_common,
        feature_cols=list(feature_cols),
        lookback=int(lookback),
        dtype=dtype,
        time_order="oldest_to_newest",
    )

    # libera DataFrames grandes cuanto antes
    del X_wide_common, X_wide_max, X_num
    gc.collect()

    return X_seq, y_mat, meta_ord, horizons_sorted


from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset


def _as_sorted_unique(values: pd.Series) -> pd.Index:
    idx = pd.Index(values)
    return idx.sort_values().unique()


def wide_lagged_df_to_3d(
    X_wide: pd.DataFrame,
    feature_cols: Sequence[str],
    lookback: int,
    *,
    dtype: np.dtype = np.float32, # type: ignore
    time_order: str = "oldest_to_newest",
) -> np.ndarray:
    """
    Convert a wide lagged-feature DataFrame to a 3D numpy array for Conv1d.

    Assumes columns: {feature}_lag{k} for k=0..lookback-1.

    Returns:
        X_seq: (N, F, L) where L=lookback and time increases along last axis.
               If time_order == "oldest_to_newest": t=0 oldest (lag=lookback-1), t=L-1 newest (lag=0).
    """
    if time_order not in {"oldest_to_newest", "newest_to_oldest"}:
        raise ValueError("time_order must be 'oldest_to_newest' or 'newest_to_oldest'")

    feature_cols = list(feature_cols)
    N = len(X_wide)
    F = len(feature_cols)
    L = int(lookback)
    if L <= 0:
        raise ValueError("lookback must be > 0")

    if time_order == "oldest_to_newest":
        lags = list(range(L - 1, -1, -1))
    else:
        lags = list(range(0, L))

    cols: List[str] = []
    missing: List[str] = []
    for lag in lags:
        for feat in feature_cols:
            col = f"{feat}_lag{lag}"
            cols.append(col)
            if col not in X_wide.columns:
                missing.append(col)

    if missing:
        missing_preview = ", ".join(missing[:10])
        raise KeyError(
            f"Missing {len(missing)} expected lag columns (showing up to 10): {missing_preview}. "
            "Make sure build_supervised_dataset created ALL lags 0..lookback-1 for every feature."
        )

    mat = X_wide[cols].to_numpy(dtype=dtype, copy=False)  # (N, L*F)
    mat = mat.reshape(N, L, F)                            # (N, L, F)
    X_seq = np.transpose(mat, (0, 2, 1)).astype(dtype, copy=False)  # (N, F, L)
    return X_seq


class SequenceStandardScaler:
    """
    StandardScaler for 3D tensors shaped as (N, F, L).
    Fits on train only; transforms any split without refit.
    """
    def __init__(self, with_mean: bool = True, with_std: bool = True):
        self.scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
        self._fitted = False

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, X: np.ndarray) -> "SequenceStandardScaler":
        if X.ndim != 3:
            raise ValueError(f"Expected X with ndim=3 (N,F,L). Got shape {X.shape}")
        X2 = np.transpose(X, (0, 2, 1)).reshape(-1, X.shape[1])  # (N*L, F)
        self.scaler.fit(X2)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Scaler not fitted. Call fit() on train split first.")
        if X.ndim != 3:
            raise ValueError(f"Expected X with ndim=3 (N,F,L). Got shape {X.shape}")
        N, F, L = X.shape
        X2 = np.transpose(X, (0, 2, 1)).reshape(-1, F)       # (N*L, F)
        X2t = self.scaler.transform(X2)
        Xt = X2t.reshape(N, L, F)
        Xt = np.transpose(Xt, (0, 2, 1)).astype(np.float32, copy=False)
        return Xt

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class SequenceDataset(Dataset):
    """
    PyTorch Dataset for sequence regression.

    X: (N, F, L) float32
    y: (N, H) float32  (H = number of horizons/outputs)
    meta: DataFrame length N
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, meta: pd.DataFrame):
        if X.ndim != 3:
            raise ValueError(f"X must be 3D (N,F,L). Got shape={X.shape}")
        if y.ndim != 2:
            raise ValueError(f"y must be 2D (N,H). Got shape={y.shape}")
        if len(X) != len(y) or len(X) != len(meta):
            raise ValueError(f"Length mismatch: len(X)={len(X)}, len(y)={len(y)}, len(meta)={len(meta)}")

        self.X = torch.from_numpy(np.asarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.asarray(y, dtype=np.float32))
        self.meta = meta.reset_index(drop=True)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx], idx


@dataclass(frozen=True)
class SplitIndices:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_targets: pd.Index
    val_targets: pd.Index
    test_targets: pd.Index


def split_by_target_timestamp(
    meta: pd.DataFrame,
    *,
    target_col: str = "target_timestamp_max",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
) -> SplitIndices:
    """
    Split indices into train/val/test by UNIQUE, ORDERED target timestamps.

    IMPORTANT: This avoids leakage across horizons because the split is based on the label time,
    not the feature time.
    """
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")
    if target_col not in meta.columns:
        raise KeyError(f"meta is missing required column '{target_col}'")

    targets = _as_sorted_unique(meta[target_col])
    n = len(targets)
    if n < 3:
        raise ValueError(
            f"Need at least 3 unique target timestamps to split. Got {n}. "
            "Try expanding the date range or reducing horizon/lookback."
        )

    n_train = int(np.floor(train_frac * n))
    n_val = int(np.floor(val_frac * n))

    n_train = max(n_train, 1)
    n_val = max(n_val, 1)
    n_test = n - n_train - n_val
    if n_test < 1:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        else:
            n_val -= 1
        if n_val < 1:
            raise ValueError("Not enough target timestamps to create non-empty train/val/test splits.")

    train_targets = targets[:n_train]
    val_targets = targets[n_train:n_train + n_val]
    test_targets = targets[n_train + n_val:]

    train_mask = meta[target_col].isin(train_targets)
    val_mask = meta[target_col].isin(val_targets)
    test_mask = meta[target_col].isin(test_targets)

    train_idx = np.where(train_mask.to_numpy())[0]
    val_idx = np.where(val_mask.to_numpy())[0]
    test_idx = np.where(test_mask.to_numpy())[0]

    return SplitIndices(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_targets=train_targets,
        val_targets=val_targets,
        test_targets=test_targets,
    )
