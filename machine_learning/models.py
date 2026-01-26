from torch import nn
from typing import Iterable

# MLP 1
class MLPRegressor(nn.Module):
    def __init__(self,
                 input_dim: int,
                 hidden_sizes: Iterable[int],
                 dropout: float = 0.1,
                 batch_norm: bool = True,):
        
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers.append(
                nn.Linear(prev, h)
            )
            if batch_norm:
                layers.append(
                    nn.BatchNorm1d(h)
                )
            layers.append(
                nn.ReLU()
            )
            layers.append(
                nn.Dropout(dropout)
            )
            prev = h

        layers.append(
            nn.Linear(prev, 1)
        )

        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X).squeeze(-1)
    
# CNN 1
import torch
from typing import List, Dict, Tuple, Optional

class ConvBlock1D(nn.Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            *,
            kernel_size: int = 3,
            dilation: int = 0,
            dropout: float = 0.1,
            use_bn: bool = True,
    ):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation

        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=not use_bn
        )

        self.bn = nn.BatchNorm1d(out_ch) if use_bn else nn.Identity()
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.drop(x)
        return x

class CNN1DRegressor(nn.Module):
    def __init__(
            self, *,
            input_dim: int,
            n_channels: int,
            seq_len: int,
            perm: List[int],
            conv_channels: Tuple[int, ...] = (32, 64, 64),
            kernel_size: int = 5,
            dilations: Optional[Tuple[int, ...]] = None,
            dropout: float = 0.1,
            use_bn: bool = True,
            head_hidden: int = 64,
            out_dim: int = 1
    ):
        super().__init__()
        assert input_dim == n_channels * seq_len, "input_dim must be n_channels*seq_len"

        self.input_dim = input_dim
        self.n_channels = n_channels
        self.seq_len = seq_len
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.dropout = dropout
        self.use_bn = use_bn
        self.head_hidden = head_hidden
        self.out_dim = out_dim

        # store perm as buffer (non-trainable)
        self.register_buffer("perm", torch.tensor(perm, dtype=torch.long), persistent=True)

        # backbone conv
        layers = []
        in_ch = n_channels

        if dilations is None:
            dilations = tuple([1] * len(conv_channels))
        assert len(dilations) == len(conv_channels), "dilations must have the same length as conv_channels"

        for out_ch, dil in zip(conv_channels, dilations):
            layers.append(
                ConvBlock1D(
                    in_ch, out_ch,
                    kernel_size=kernel_size,
                    dilation=dil,
                    # dropout=dropout,
                    use_bn=use_bn,
                )
            )
            in_ch = out_ch
        self.backbone = nn.Sequential(*layers)

        # head
        self.head = nn.Sequential(
            nn.Linear(in_ch, head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, input_dim) where input_dim = C * T
        """

        if x.dim() != 2:
            raise ValueError(f"Expected x dim=2 (B, input_dim), got {tuple(x.shape)}")
        
        # Reorder columns -> (B, C*T) consistent
        x = x.index_select(1, self.perm) # type: ignore

        b = x.shape[0]
        x = x.view(b, self.n_channels, self.seq_len) # (B, C, T)

        x = self.backbone(x)        # (B, hidden, T)

        x = x.mean(dim=-1)     # (B, hidden)

        y = self.head(x)
        return y.squeeze(-1)