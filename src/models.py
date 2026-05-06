"""
Model architectures for multi-horizon electricity consumption forecasting.

Proposed TCN family:
  - LightTCN:      Single-path dilated causal TCN with LayerNorm + GELU
  - ConcatTCN:     LightTCN + exogenous MLP concatenation (ablation baseline)
  - DualTCN:       Dual-path (consumption + exogenous) TCN with gated fusion
  - DualTCN_Attn:  DualTCN with cross-attention between paths

Reference models:
  - LSTM, BiLSTM, CNN-LSTM

All models accept (batch, seq_len, num_features) input and produce (batch, horizon) output.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Block: Causal Dilated Convolution
# =============================================================================

class LightCausalBlock(nn.Module):
    """
    Single causal dilated convolution block with residual connection.

    Architecture: CausalPad → Conv1d → LayerNorm → GELU → Dropout → Residual

    LayerNorm is used instead of BatchNorm for FP16 AMP compatibility.
    GELU activation provides smoother gradients than ReLU.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dilation: int = 1, dropout: float = 0.15):
        super().__init__()
        self.pad_len = (kernel_size - 1) * dilation  # causal: only left padding

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              dilation=dilation, padding=0)
        self.norm = nn.LayerNorm(out_channels)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

        # 1x1 projection for residual when channel dims differ
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, seq_len)
        res = self.residual(x)

        # Causal padding: pad only the left side
        x = F.pad(x, (self.pad_len, 0))
        x = self.conv(x)

        # LayerNorm expects (batch, seq_len, channels)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)

        x = self.drop(self.act(x))
        return x + res


def _compute_n_blocks(input_window: int) -> int:
    """
    Compute number of TCN blocks to guarantee receptive field >= input_window.

    n_blocks = ceil(log2(w))
    Receptive field R = 2^(n+1) - 1 >= w

    Examples:
        w=168  -> n=8  blocks, R=511
        w=336  -> n=9  blocks, R=1023
        w=720  -> n=10 blocks, R=2047
        w=2160 -> n=12 blocks, R=8191
    """
    return max(3, math.ceil(math.log2(max(input_window, 2))))


# =============================================================================
# LightTCN: Base Architecture
# =============================================================================

class LightTCN(nn.Module):
    """
    Lightweight Temporal Convolutional Network with automatic receptive field.

    Architecture:
        Input (batch, seq_len, features)
          -> transpose -> (batch, features, seq_len)
          -> [LightCausalBlock x n_blocks]
          -> last time step x[:, :, -1]
          -> Linear -> (batch, horizon)

    The number of blocks is determined automatically from input_window
    to guarantee full receptive field coverage.
    """

    def __init__(self, input_size: int = 7, base_channels: int = 64,
                 kernel_size: int = 3, dropout: float = 0.15,
                 output_size: int = 1, input_window: int = 168):
        super().__init__()
        self.model_type = "LightTCN"

        n_blocks = _compute_n_blocks(input_window)

        blocks = []
        for i in range(n_blocks):
            in_ch = input_size if i == 0 else base_channels
            blocks.append(LightCausalBlock(in_ch, base_channels,
                                           kernel_size, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)
        self.fc = nn.Linear(base_channels, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)      # (batch, features, seq_len)
        x = self.tcn(x)             # (batch, channels, seq_len)
        x = x[:, :, -1]             # last step: most recent representation
        return self.fc(x)


# =============================================================================
# ConcatTCN: Exogenous Concatenation (Ablation Baseline)
# =============================================================================

class ConcatTCN(nn.Module):
    """
    LightTCN with naive exogenous feature concatenation at the final step.

    Architecture:
        TCN last step -> (batch, C)
        Exo MLP       -> (batch, C//2)
        Concat        -> (batch, C + C//2)
        LayerNorm + FC -> (batch, horizon)

    Serves as ablation baseline: tests whether simple concatenation
    is sufficient for exogenous integration.
    """

    def __init__(self, input_size: int = 7, base_channels: int = 64,
                 kernel_size: int = 3, dropout: float = 0.15,
                 output_size: int = 1, input_window: int = 168):
        super().__init__()
        self.model_type = "ConcatTCN"

        num_exo = input_size - 1
        exo_dim = base_channels // 2
        n_blocks = _compute_n_blocks(input_window)

        # TCN backbone
        blocks = []
        for i in range(n_blocks):
            in_ch = input_size if i == 0 else base_channels
            blocks.append(LightCausalBlock(in_ch, base_channels,
                                           kernel_size, dilation=2**i, dropout=dropout))
        self.tcn = nn.Sequential(*blocks)

        # Exogenous MLP (last time step features)
        self.exo_mlp = nn.Sequential(
            nn.Linear(num_exo, exo_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(exo_dim * 2, exo_dim),
            nn.LayerNorm(exo_dim),
        )

        fusion_dim = base_channels + exo_dim
        self.norm = nn.LayerNorm(fusion_dim)
        self.fc = nn.Linear(fusion_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Consumption column index = last column
        cons_idx = x.shape[-1] - 1

        tcn_out = self.tcn(x.transpose(1, 2))
        last = tcn_out[:, :, -1]                    # (batch, C)

        exo = x[:, -1, :cons_idx]                    # (batch, num_exo)
        exo_repr = self.exo_mlp(exo)                 # (batch, C//2)

        fused = torch.cat([last, exo_repr], dim=-1)
        return self.fc(self.norm(fused))


# =============================================================================
# DualTCN: Dual-Path with Gated Fusion
# =============================================================================

class DualTCN(nn.Module):
    """
    Dual-path TCN with gated fusion for consumption and exogenous features.

    Architecture:
        Consumption TCN: x[:,:,-1:] -> T (batch, C)    [univariate path]
        Exogenous TCN:   x[:,:,:-1] -> E (batch, C)    [calendar path]

        Gated Fusion:
            g = sigmoid(W_g * [T; E])      (batch, C)
            out = g * T + (1-g) * E        (batch, C)

        LayerNorm -> FC -> (batch, horizon)

    The gate learns to dynamically balance consumption history vs.
    calendar context for each channel dimension.
    """

    def __init__(self, input_size: int = 7, base_channels: int = 64,
                 kernel_size: int = 3, dropout: float = 0.15,
                 output_size: int = 1, input_window: int = 168):
        super().__init__()
        self.model_type = "DualTCN"

        cons_size = 1
        exo_size = input_size - 1
        n_blocks = _compute_n_blocks(input_window)

        # Consumption path
        cons_blocks = []
        for i in range(n_blocks):
            in_ch = cons_size if i == 0 else base_channels
            cons_blocks.append(LightCausalBlock(in_ch, base_channels,
                                                kernel_size, dilation=2**i, dropout=dropout))
        self.cons_tcn = nn.Sequential(*cons_blocks)

        # Exogenous path
        exo_blocks = []
        for i in range(n_blocks):
            in_ch = exo_size if i == 0 else base_channels
            exo_blocks.append(LightCausalBlock(in_ch, base_channels,
                                               kernel_size, dilation=2**i, dropout=dropout))
        self.exo_tcn = nn.Sequential(*exo_blocks)

        # Gated fusion: g in (0,1)^C
        self.gate_proj = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(base_channels)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cons_idx = x.shape[-1] - 1
        cons = x[:, :, cons_idx:cons_idx + 1]    # (batch, seq, 1)
        exo = x[:, :, :cons_idx]                  # (batch, seq, num_exo)

        T = self.cons_tcn(cons.transpose(1, 2))[:, :, -1]   # (batch, C)
        E = self.exo_tcn(exo.transpose(1, 2))[:, :, -1]     # (batch, C)

        gate = self.gate_proj(torch.cat([T, E], dim=-1))     # (batch, C)
        out = gate * T + (1 - gate) * E

        return self.fc(self.norm(self.drop(out)))


# =============================================================================
# DualTCN_Attn: Cross-Attention Enhanced Dual-Path
# =============================================================================

class DualTCNAttn(nn.Module):
    """
    DualTCN with cross-attention between consumption and exogenous paths.

    Architecture:
        Consumption TCN -> T_seq (batch, seq, C)
        Exogenous TCN   -> E_seq (batch, seq, C)

        Cross-Attention:
            Q = T_seq[:, -1, :].unsqueeze(1)    (batch, 1, C)
            K = V = E_seq                        (batch, seq, C)
            attn = softmax(Q * K^T / sqrt(C)) * V
            E_ctx = attn.squeeze(1)              (batch, C)

        Gated Fusion:
            g = sigmoid(W * [T_last, E_ctx])
            out = g * T_last + (1-g) * E_ctx

        LayerNorm -> FC -> (batch, horizon)

    The consumption path queries the exogenous path to find which
    historical calendar contexts are most relevant for the current
    prediction — enabling dynamic attention to holidays, weekends, etc.
    """

    def __init__(self, input_size: int = 7, base_channels: int = 64,
                 kernel_size: int = 3, dropout: float = 0.15,
                 output_size: int = 1, input_window: int = 168):
        super().__init__()
        self.model_type = "DualTCN_Attn"

        cons_size = 1
        exo_size = input_size - 1
        n_blocks = _compute_n_blocks(input_window)

        # Dual TCN paths
        cons_blocks, exo_blocks = [], []
        for i in range(n_blocks):
            d = 2 ** i
            cons_blocks.append(LightCausalBlock(
                cons_size if i == 0 else base_channels,
                base_channels, kernel_size, d, dropout))
            exo_blocks.append(LightCausalBlock(
                exo_size if i == 0 else base_channels,
                base_channels, kernel_size, d, dropout))
        self.cons_tcn = nn.Sequential(*cons_blocks)
        self.exo_tcn = nn.Sequential(*exo_blocks)

        # Cross-attention
        self.scale = base_channels ** -0.5

        # Gated fusion
        self.gate_proj = nn.Sequential(
            nn.Linear(base_channels * 2, base_channels),
            nn.Sigmoid(),
        )

        self.attn_norm = nn.LayerNorm(base_channels)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(base_channels, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cons_idx = x.shape[-1] - 1
        cons = x[:, :, cons_idx:cons_idx + 1]
        exo = x[:, :, :cons_idx]

        # Full sequence outputs for attention
        T_seq = self.cons_tcn(cons.transpose(1, 2)).transpose(1, 2)  # (batch, seq, C)
        E_seq = self.exo_tcn(exo.transpose(1, 2)).transpose(1, 2)    # (batch, seq, C)

        # Query from consumption last step
        T_last = T_seq[:, -1, :]                # (batch, C)
        query = T_last.unsqueeze(1)             # (batch, 1, C)

        # Cross-attention: Q=consumption, K=V=exogenous
        attn_scores = torch.bmm(query, E_seq.transpose(1, 2)) * self.scale
        attn_weights = torch.softmax(attn_scores, dim=-1)
        E_ctx = torch.bmm(attn_weights, E_seq).squeeze(1)   # (batch, C)

        # Gated fusion
        gate = self.gate_proj(torch.cat([T_last, E_ctx], dim=-1))
        out = gate * T_last + (1 - gate) * E_ctx

        return self.fc(self.attn_norm(self.drop(out)))


# =============================================================================
# Reference Models: LSTM, BiLSTM, CNN-LSTM
# =============================================================================

class LSTMModel(nn.Module):
    """Standard LSTM for time series forecasting."""

    def __init__(self, input_size: int = 7, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.15,
                 output_size: int = 1):
        super().__init__()
        self.model_type = "LSTM"
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class BiLSTMModel(nn.Module):
    """Bidirectional LSTM for time series forecasting."""

    def __init__(self, input_size: int = 7, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.15,
                 output_size: int = 1):
        super().__init__()
        self.model_type = "BiLSTM"
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class CNNLSTMModel(nn.Module):
    """CNN-LSTM: two Conv1D layers for local patterns + LSTM for temporal modeling."""

    def __init__(self, input_size: int = 7, cnn_filters: int = 64,
                 lstm_hidden: int = 64, dropout: float = 0.15,
                 output_size: int = 1):
        super().__init__()
        self.model_type = "CNN_LSTM"
        self.conv1 = nn.Conv1d(input_size, cnn_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_filters)
        self.conv2 = nn.Conv1d(cnn_filters, cnn_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_filters)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(cnn_filters, lstm_hidden, num_layers=1,
                            batch_first=True)
        self.fc = nn.Linear(lstm_hidden, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # (batch, features, seq)
        x = self.drop(self.relu(self.bn1(self.conv1(x))))
        x = self.drop(self.relu(self.bn2(self.conv2(x))))
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# =============================================================================
# Model Factory
# =============================================================================

MODEL_REGISTRY = {
    "LightTCN": LightTCN,
    "ConcatTCN": ConcatTCN,
    "DualTCN": DualTCN,
    "DualTCN_Attn": DualTCNAttn,
    "LSTM": LSTMModel,
    "BiLSTM": BiLSTMModel,
    "CNN_LSTM": CNNLSTMModel,
}


def create_model(model_name: str, input_size: int, output_size: int,
                 config: dict, input_window: int = 168) -> nn.Module:
    """
    Create model by name and configuration.

    Args:
        model_name: e.g., "LightTCN_medium", "DualTCN_Attn_large"
        input_size: number of input features (7 for multivariate)
        output_size: forecast horizon H
        config: hyperparameter dict from MODEL_CONFIGS
        input_window: w, used for TCN receptive field computation

    Returns:
        nn.Module instance
    """
    # Extract base architecture from name (e.g., "DualTCN_Attn_large" -> "DualTCN_Attn")
    for key in sorted(MODEL_REGISTRY.keys(), key=len, reverse=True):
        if model_name == key or model_name.startswith(key + "_"):
            cls = MODEL_REGISTRY[key]

            # TCN-based models need input_window
            if key in ("LightTCN", "ConcatTCN", "DualTCN", "DualTCN_Attn"):
                return cls(input_size=input_size,
                           base_channels=config["base_channels"],
                           kernel_size=config["kernel_size"],
                           dropout=config["dropout"],
                           output_size=output_size,
                           input_window=input_window)

            # LSTM/BiLSTM
            elif key in ("LSTM", "BiLSTM"):
                return cls(input_size=input_size,
                           hidden_size=config["hidden_size"],
                           num_layers=config["num_layers"],
                           dropout=config["dropout"],
                           output_size=output_size)

            # CNN-LSTM
            elif key == "CNN_LSTM":
                return cls(input_size=input_size,
                           cnn_filters=config["cnn_filters"],
                           lstm_hidden=config["lstm_hidden"],
                           dropout=config["dropout"],
                           output_size=output_size)

    raise ValueError(f"Unknown model: {model_name}")
