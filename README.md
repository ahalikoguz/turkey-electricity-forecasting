# Multi-Horizon Electricity Consumption Forecasting with TCN Architectures

This repository contains the code, data, and trained model weights for the paper:

> **Multi-Horizon Deep Learning Forecasting of Turkey's Hourly Electricity Consumption: A Comprehensive Comparison of Temporal Convolutional Network Families on a Decade of National Data**

## Overview

We propose four TCN-based architectures for electricity consumption forecasting and systematically evaluate them across four forecast horizons (H ∈ {1, 24, 168, 720} hours) using 87,671 hourly observations from Turkey's national grid (2016–2025).

| Model | Description | Best Horizon |
|-------|-------------|-------------|
| **LightTCN** | Dilated causal TCN with LayerNorm + GELU | H=1 (0.93%), H=24 (4.11%), H=720 (7.35%) |
| **ConcatTCN** | LightTCN + exogenous MLP concatenation (ablation baseline) | — |
| **DualTCN** | Dual-path (consumption + exogenous) with gated fusion | — |
| **DualTCN_Attn** | DualTCN with cross-attention between paths | H=168 (6.92%) |

Reference models: LSTM, BiLSTM, CNN-LSTM, N-HiTS, PatchTST, Naive, Seasonal Naive.

## Key Results

- **197 experiment scenarios** (153 main + 8 long-horizon + 36 feature ablation)
- **Feature ablation** reveals horizon-dependent exogenous feature contribution: negligible at H=1, systematic benefit at H=24 (6/6 multivariate wins), implicit learning at H=168 with large input windows
- **Error cancellation**: Weekly forecasts aggregated to monthly GWh reduce MAPE from 6.88% (hourly) to 1.91% (monthly)

## Repository Structure

```
├── train.py                  # Main training entry point
├── configs/
│   └── experiments.json      # All experiment configurations
├── src/
│   ├── models.py             # Model architectures (LightTCN, ConcatTCN, DualTCN, DualTCN_Attn, LSTM, BiLSTM, CNN-LSTM)
│   ├── data.py               # Data loading, feature engineering, leakage-safe splitting
│   ├── metrics.py            # MAPE, RMSE, MAE, R², MASE, MASE_rel
│   └── train_utils.py        # Training loop, seed management, inference
├── data/
│   └── electricity_consumption_with_features.csv  # EPİAŞ dataset (87,671 hours)
└── results/                  # Experiment outputs (CSV)
```

## Quick Start

### Requirements

```bash
pip install torch numpy pandas openpyxl
```

Full requirements: see `requirements.txt`.

### Reproduce Main Results (H=1, 24, 168)

```bash
python train.py --experiment main --data data/electricity_consumption_with_features.csv
```

### Reproduce Long-Horizon Results (H=720)

```bash
python train.py --experiment long_horizon --data data/electricity_consumption_with_features.csv
```

### Reproduce Feature Ablation (Univariate)

```bash
python train.py --experiment ablation_univariate --data data/electricity_consumption_with_features.csv
```

### Demo Mode (Quick Test)

```bash
python train.py --experiment main --demo
```

Uses 1% of data with 3 epochs to verify the pipeline works.

## Dataset

The dataset is sourced from the [EPİAŞ Transparency Platform](https://seffaflik.epias.com.tr) (publicly available) and contains hourly electricity consumption for Turkey (2016–2025).

| Feature | Description | Range |
|---------|-------------|-------|
| `hour_sin`, `hour_cos` | Cyclic hour encoding | [-1, 1] |
| `dow_sin`, `dow_cos` | Cyclic day-of-week encoding | [-1, 1] |
| `is_weekend` | Weekend binary flag | {0, 1} |
| `is_holiday` | Turkey national holiday flag | {0, 1} |
| `Consumption (MWh)` | Target variable (z-score normalized) | [15,333; 59,504] |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss | Huber (δ=1.0) |
| Optimizer | Adam (lr=1e-3, weight_decay=1e-5) |
| Scheduler | ReduceLROnPlateau (factor=0.7, patience=4) |
| Early Stopping | patience=10 epochs |
| Precision | FP16 AMP (LayerNorm for stability) |
| Hardware | NVIDIA RTX 4090 Laptop GPU |
| Seed | 42 |

## Model Architectures

### LightTCN
Base architecture: stack of causal dilated convolution blocks with automatic receptive field guarantee. Block count is computed as `n = ⌈log₂(w)⌉` ensuring `R = 2^(n+1) - 1 ≥ w`.

### DualTCN
Separates consumption and exogenous features into independent TCN paths. Outputs are combined via learned gated fusion: `g = σ(W·[T;E])`, `out = g⊙T + (1-g)⊙E`.

### DualTCN_Attn
Extends DualTCN with cross-attention: the consumption path queries the exogenous path to dynamically weight historical calendar contexts. `Attention(Q, K, V) = softmax(Q·Kᵀ/√d)·V` where Q=consumption last step, K=V=exogenous sequence.

## License

This project is released under the MIT License. The EPİAŞ dataset is publicly available under the terms of the EPİAŞ Transparency Platform.
