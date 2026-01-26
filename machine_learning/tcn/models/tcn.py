from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    """
    Recorta los últimos `chomp_size` pasos para convertir padding simétrico
    en padding puramente causal (izquierda).
    Entrada/Salida: (N, C, L)
    """
    def __init__(self, chomp_size: int):
        super().__init__()
        if chomp_size < 0:
            raise ValueError("chomp_size must be >= 0")
        self.chomp_size = int(chomp_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Bloque residual: Conv1d causal dilatada x2 + ReLU + Dropout.
    """
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        if kernel_size < 2:
            raise ValueError("kernel_size should be >= 2")
        if dilation < 1:
            raise ValueError("dilation must be >= 1")

        padding = (kernel_size - 1) * dilation  # padding total; luego se chompea para causalidad

        self.conv1 = weight_norm(
            nn.Conv1d(
                in_channels=n_inputs,
                out_channels=n_outputs,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                in_channels=n_outputs,
                out_channels=n_outputs,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.drop1,
            self.conv2, self.chomp2, self.relu2, self.drop2,
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, kernel_size=1) if n_inputs != n_outputs else None
        self.out_relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity="linear")
            if self.downsample.bias is not None:
                nn.init.zeros_(self.downsample.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.out_relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Stack de TemporalBlocks con dilataciones 1,2,4,8,... (exponencial).
    """
    def __init__(
        self,
        num_inputs: int,
        channels: Sequence[int],
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if len(channels) == 0:
            raise ValueError("channels must have at least one element")
        layers: List[nn.Module] = []
        n_in = num_inputs
        for i, n_out in enumerate(channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    n_inputs=n_in,
                    n_outputs=n_out,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            n_in = n_out
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class TCNConfig:
    num_features: int
    channels: Sequence[int] = (64, 64, 64, 64)
    kernel_size: int = 3
    dropout: float = 0.1
    output_dim: int = 1  # nº de horizontes (outputs)


class TCNRegressor(nn.Module):
    """
    TCN -> toma el último timestep (causal) -> Linear -> outputs.
    Entrada: (N, F, L)
    Salida: (N, H)
    """
    def __init__(self, cfg: TCNConfig):
        super().__init__()
        self.cfg = cfg
        self.tcn = TemporalConvNet(
            num_inputs=cfg.num_features,
            channels=cfg.channels,
            kernel_size=cfg.kernel_size,
            dropout=cfg.dropout,
        )
        self.head = nn.Linear(cfg.channels[-1], cfg.output_dim)

        nn.init.kaiming_normal_(self.head.weight, nonlinearity="linear")
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.tcn(x)        # (N, C, L)
        last = feats[:, :, -1]     # (N, C)
        out = self.head(last)      # (N, H)
        return out
