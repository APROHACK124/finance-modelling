from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler


@dataclass
class TimestampBatchSamplerConfig:
    time_col: str = "timestamp"
    min_cs_size: int = 20
    shuffle: bool = True
    seed: int = 42
    max_batch_size: Optional[int] = None  # si un día tiene 5000 activos y revienta la VRAM
    drop_last: bool = False


class TimestampBatchSampler(Sampler[List[int]]):
    """
    BatchSampler que devuelve lotes donde TODOS los samples comparten el mismo meta[time_col].
    Eso permite computar IC_loss cross-sectional dentro del batch.

    - Filtra timestamps con cs_size < min_cs_size.
    - Si max_batch_size está seteado, parte un timestamp grande en varios batches (mismo timestamp).
      (Trade-off: IC_loss se estima sobre un subset por batch si se parte.)
    """
    def __init__(
        self,
        meta: pd.DataFrame,
        *,
        cfg: TimestampBatchSamplerConfig = TimestampBatchSamplerConfig(),
        indices: Optional[Sequence[int]] = None,
    ):
        self.meta = meta.reset_index(drop=True)
        self.cfg = cfg
        self.indices = np.asarray(indices if indices is not None else np.arange(len(self.meta)), dtype=np.int64)

        if cfg.time_col not in self.meta.columns:
            raise KeyError(f"meta no tiene columna '{cfg.time_col}'")

        ts = pd.to_datetime(self.meta.loc[self.indices, cfg.time_col]).to_numpy()
        self._groups: List[List[int]] = []
        by_ts: Dict[np.datetime64, List[int]] = {}

        for i, t in zip(self.indices, ts):
            by_ts.setdefault(t, []).append(int(i))

        # Filtra por tamaño mínimo
        groups = [idxs for idxs in by_ts.values() if len(idxs) >= int(cfg.min_cs_size)]
        groups.sort(key=lambda g: pd.to_datetime(self.meta.loc[g[0], cfg.time_col]))  # orden temporal

        # Split si hay max_batch_size
        if cfg.max_batch_size is not None and cfg.max_batch_size > 0:
            split_groups: List[List[int]] = []
            m = int(cfg.max_batch_size)
            for g in groups:
                if len(g) <= m:
                    split_groups.append(g)
                else:
                    for j in range(0, len(g), m):
                        split_groups.append(g[j:j + m])
            groups = split_groups

        self._groups = groups
        if len(self._groups) == 0:
            raise ValueError(
                "TimestampBatchSampler: no quedaron batches tras filtrar min_cs_size. "
                "Baja min_cs_size o revisa tu universo."
            )

        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = int(epoch)

    def __iter__(self) -> Iterator[List[int]]:
        groups = list(self._groups)
        if self.cfg.shuffle:
            rng = np.random.default_rng(self.cfg.seed + self._epoch)
            rng.shuffle(groups)

        if self.cfg.drop_last:
            for g in groups:
                yield g
        else:
            for g in groups:
                yield g

    def __len__(self) -> int:
        return len(self._groups)
