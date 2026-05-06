"""
Evaluation metrics for electricity consumption forecasting.

Primary metric: MAPE (scale-independent, interpretable)
Secondary: RMSE, MAE, R², MASE, MASE_rel (horizon-adaptive)
"""

import numpy as np
from typing import Dict, Optional


SEASONAL_PERIOD_BY_HORIZON = {
    1: 24,      # daily cycle
    24: 24,     # daily cycle
    168: 168,   # weekly cycle
    720: 168,   # weekly cycle (for monthly forecast)
}


def get_seasonal_period(horizon: int) -> int:
    """Return seasonal period m for the given forecast horizon."""
    return SEASONAL_PERIOD_BY_HORIZON.get(horizon, 24)


def naive_forecast(X: np.ndarray, horizon: int, cons_idx: int = -1) -> np.ndarray:
    """Naive (persistence): repeat last observed consumption value."""
    if cons_idx == -1:
        cons_idx = X.shape[-1] - 1
    last = X[:, -1, cons_idx]
    return np.repeat(last[:, np.newaxis], horizon, axis=1)


def seasonal_naive_forecast(X: np.ndarray, horizon: int,
                            seasonal_period: int, cons_idx: int = -1) -> np.ndarray:
    """
    Seasonal naive: use value from m steps ago.

    For h < m: take from input sequence
    For h >= m: recursive (use own prediction)
    """
    if cons_idx == -1:
        cons_idx = X.shape[-1] - 1

    n, seq_len, _ = X.shape
    preds = np.zeros((n, horizon), dtype=np.float32)
    cons_seq = X[:, :, cons_idx]

    for i in range(n):
        for h in range(horizon):
            if seasonal_period <= 0:
                preds[i, h] = cons_seq[i, -1]
            elif h < seasonal_period:
                idx = seq_len - seasonal_period + h
                preds[i, h] = cons_seq[i, idx] if 0 <= idx < seq_len else cons_seq[i, -1]
            else:
                preds[i, h] = preds[i, h - seasonal_period]
    return preds


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      mase_scale: Optional[float] = None) -> Dict[str, float]:
    """
    Compute all metrics on original MWh scale.

    Returns: dict with RMSE, MAE, MAPE, R2, MASE
    """
    yt = y_true.flatten()
    yp = y_pred.flatten()
    err = yt - yp

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    mape = float(np.mean(np.abs(err) / (np.abs(yt) + 1e-8)) * 100.0)
    r2 = float(1 - np.sum(err ** 2) / (np.sum((yt - np.mean(yt)) ** 2) + 1e-8))
    mase = (float("nan")
            if (mase_scale is None or not np.isfinite(mase_scale) or mase_scale < 1e-12)
            else float(mae / mase_scale))

    return {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2, "MASE": mase}


def mase_scale_from_train(train_cons: np.ndarray, seasonal_period: int) -> float:
    """
    Standard MASE scale factor (Hyndman & Koehler, 2006).

    scale = mean|y_t - y_{t-m}| computed on training set only.
    """
    y = train_cons.astype(np.float32)
    if len(y) <= seasonal_period:
        return float("nan")
    diffs = np.abs(y[seasonal_period:] - y[:-seasonal_period])
    return float(np.mean(diffs))


def compute_mase_rel(y_true: np.ndarray, y_pred: np.ndarray,
                     X_test: np.ndarray, horizon: int) -> float:
    """
    MASE_rel: horizon-adaptive relative MASE.

    MASE_rel = MAE_model / MAE_seasonal_naive(same H, test set)

    This uses the test set for the denominator but does NOT fit any model,
    so it is not data leakage.
    """
    try:
        m = get_seasonal_period(horizon)
        sn_preds = seasonal_naive_forecast(X_test, horizon, m)
        mae_model = float(np.mean(np.abs(y_true - y_pred)))
        mae_sn = float(np.mean(np.abs(y_true - sn_preds)))
        if mae_sn < 1e-8:
            return float("nan")
        return float(mae_model / mae_sn)
    except Exception:
        return float("nan")
