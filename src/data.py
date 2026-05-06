"""
Data loading, feature engineering, and leakage-safe splitting.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional


# Feature column order — consumption MUST be the last column
FEATURE_COLS = [
    'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
    'is_weekend', 'is_holiday', 'Consumption (MWh)',
]
NUM_FEATURES = len(FEATURE_COLS)
CONSUMPTION_COL_IDX = NUM_FEATURES - 1  # = 6


def build_feature_matrix(df: pd.DataFrame,
                         target_col: str = "Consumption (MWh)") -> np.ndarray:
    """
    Build 7-dimensional feature matrix from raw DataFrame.

    Cyclic encoding preserves hour=23 <-> hour=0 adjacency.
    Only consumption is normalized later; sin/cos are already in [-1,1].
    """
    n = len(df)
    X = np.zeros((n, NUM_FEATURES), dtype=np.float32)

    hour = df['hour'].values
    X[:, 0] = np.sin(2 * np.pi * hour / 24)
    X[:, 1] = np.cos(2 * np.pi * hour / 24)

    dow = df['day_of_week'].values
    X[:, 2] = np.sin(2 * np.pi * dow / 7)
    X[:, 3] = np.cos(2 * np.pi * dow / 7)

    X[:, 4] = df['is_weekend'].values.astype(np.float32)
    X[:, 5] = df['is_holiday'].values.astype(np.float32)
    X[:, 6] = df[target_col].values.astype(np.float32)

    return X


def split_raw_series(data: np.ndarray,
                     train_ratio: float = 0.80,
                     val_ratio: float = 0.10
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological split into train/val/test segments.

    CRITICAL: Split raw series BEFORE creating sequences to prevent
    data leakage between partitions.
    """
    n = len(data)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return data[:train_end], data[train_end:val_end], data[val_end:]


def create_sequences(segment: np.ndarray, seq_len: int, horizon: int
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding window sequences from a single segment.

    Args:
        segment: (T, num_features) array
        seq_len: input window length w
        horizon: forecast horizon H

    Returns:
        X: (N, seq_len, num_features) input windows
        y: (N, horizon) target consumption values
    """
    T = len(segment)
    total_len = seq_len + horizon
    if T < total_len:
        return np.empty((0, seq_len, segment.shape[1])), np.empty((0, horizon))

    N = T - total_len + 1
    X = np.zeros((N, seq_len, segment.shape[1]), dtype=np.float32)
    y = np.zeros((N, horizon), dtype=np.float32)

    for i in range(N):
        X[i] = segment[i:i + seq_len]
        y[i] = segment[i + seq_len:i + total_len, CONSUMPTION_COL_IDX]

    return X, y


class ZScoreNormalizer:
    """Z-score normalization for the consumption column only."""

    def __init__(self):
        self.mean: Optional[float] = None
        self.std: Optional[float] = None

    def fit(self, train_segment: np.ndarray) -> 'ZScoreNormalizer':
        """Compute mean/std from training data consumption column."""
        cons = train_segment[:, CONSUMPTION_COL_IDX]
        self.mean = float(np.mean(cons))
        self.std = float(np.std(cons)) + 1e-8
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Normalize consumption column in-place (returns same array)."""
        data = data.copy()
        data[:, CONSUMPTION_COL_IDX] = (
            (data[:, CONSUMPTION_COL_IDX] - self.mean) / self.std
        )
        return data

    def inverse_transform_targets(self, y: np.ndarray) -> np.ndarray:
        """Convert normalized predictions back to original MWh scale."""
        return y * self.std + self.mean


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for sliding window sequences."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
