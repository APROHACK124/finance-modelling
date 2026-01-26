from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


def pearson_corr_batch(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Pearson corr por columna.
    x,y: (B,H)
    retorna: (H,)
    """
    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    cov = (x * y).mean(dim=0)
    std_x = torch.sqrt((x * x).mean(dim=0) + eps)
    std_y = torch.sqrt((y * y).mean(dim=0) + eps)
    corr = cov / (std_x * std_y + eps)
    return corr


def ic_loss(pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Loss para maximizar IC (Pearson) cross-sectional.
    - pred,y_true: (B,H)
    - retorna escalar: -mean(corr)  (minimizar => maximizar corr)
    """
    corr = pearson_corr_batch(pred, y_true, eps=eps)  # (H,)
    return -corr.mean()


@dataclass
class LossConfig:
    ic_lambda: float = 0.2
    min_cs_size: int = 20
    eps: float = 1e-8


class CombinedRegICLoss(nn.Module):
    """
    SmoothL1 (regresión) + lambda * IC_loss.
    IC_loss solo se aplica si batch_size >= min_cs_size (fallback: solo regresión).
    """
    def __init__(self, cfg: LossConfig, beta: float = 1.0):
        super().__init__()
        self.cfg = cfg
        self.reg = nn.SmoothL1Loss(beta=beta)

    def forward(self, pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        loss_reg = self.reg(pred, y_true)
        if pred.shape[0] < self.cfg.min_cs_size:
            return loss_reg

        # Evitar NaNs si std ~ 0
        with torch.no_grad():
            y_std = y_true.std(dim=0).mean().item()
            p_std = pred.std(dim=0).mean().item()
        if y_std < self.cfg.eps or p_std < self.cfg.eps:
            return loss_reg

        loss_ic = ic_loss(pred, y_true, eps=self.cfg.eps)
        return loss_reg + float(self.cfg.ic_lambda) * loss_ic
