#!/usr/bin/env python3
"""
Main training script for Turkey electricity consumption forecasting.

Usage:
    python train.py --experiment main                  # H=1,24,168 (153 scenarios)
    python train.py --experiment long_horizon           # H=720 (8 scenarios)
    python train.py --experiment ablation_univariate    # Feature ablation (36 scenarios)
    python train.py --experiment main --demo            # Quick test with 1% data

All experiment configurations are defined in configs/experiments.json.
Results are saved to results/<experiment_name>/ as CSV.
"""

import os
import sys
import json
import time
import gc
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import (build_feature_matrix, split_raw_series, create_sequences,
                      ZScoreNormalizer, TimeSeriesDataset, NUM_FEATURES, CONSUMPTION_COL_IDX)
from src.models import create_model
from src.metrics import (calculate_metrics, mase_scale_from_train,
                         get_seasonal_period, compute_mase_rel,
                         naive_forecast, seasonal_naive_forecast)
from src.train_utils import set_seed, count_parameters, train_model, run_inference


def parse_args():
    parser = argparse.ArgumentParser(description="Train forecasting models")
    parser.add_argument("--experiment", type=str, default="main",
                        choices=["main", "long_horizon", "ablation_univariate"],
                        help="Experiment name from configs/experiments.json")
    parser.add_argument("--data", type=str, default="data/electricity_consumption_with_features.csv",
                        help="Path to data CSV file")
    parser.add_argument("--config", type=str, default="configs/experiments.json",
                        help="Path to experiment config JSON")
    parser.add_argument("--output", type=str, default="results",
                        help="Output directory for results")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: use 1%% of data, 3 epochs")
    parser.add_argument("--verbose", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        full_config = json.load(f)

    exp_config = full_config["experiments"][args.experiment]
    train_config = full_config["training"].copy()
    model_configs = full_config["model_configs"]
    data_config = full_config["data"]

    if args.demo:
        train_config["num_epochs"] = 3
        train_config["early_stopping_patience"] = 2

    train_config["verbose"] = args.verbose
    train_config["print_every"] = 5

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print(f"\nLoading data: {args.data}")
    if args.data.endswith(".xlsx"):
        df = pd.read_excel(args.data)
    else:
        df = pd.read_csv(args.data)

    if args.demo:
        n_demo = max(5000, int(len(df) * 0.01))
        df = df.tail(n_demo).reset_index(drop=True)
        print(f"  DEMO MODE: using last {n_demo} rows")

    # Determine number of features
    num_features = exp_config.get("num_features", NUM_FEATURES)
    feature_matrix = build_feature_matrix(df, data_config["target_column"])

    if num_features == 1:
        # Univariate: consumption only
        feature_matrix = feature_matrix[:, CONSUMPTION_COL_IDX:CONSUMPTION_COL_IDX + 1]
        print(f"  Univariate mode: 1 feature (consumption only)")
    else:
        print(f"  Multivariate mode: {num_features} features")

    print(f"  Total observations: {len(feature_matrix):,}")

    # Split
    train_seg, val_seg, test_seg = split_raw_series(
        feature_matrix, data_config["train_ratio"], data_config["val_ratio"]
    )
    print(f"  Train: {len(train_seg):,} | Val: {len(val_seg):,} | Test: {len(test_seg):,}")

    # Normalize (consumption column only)
    cons_idx = 0 if num_features == 1 else CONSUMPTION_COL_IDX
    normalizer = ZScoreNormalizer()
    # For univariate, adjust normalizer to use column 0
    if num_features == 1:
        normalizer.mean = float(np.mean(train_seg[:, 0]))
        normalizer.std = float(np.std(train_seg[:, 0])) + 1e-8
        train_norm = train_seg.copy()
        val_norm = val_seg.copy()
        test_norm = test_seg.copy()
        train_norm[:, 0] = (train_norm[:, 0] - normalizer.mean) / normalizer.std
        val_norm[:, 0] = (val_norm[:, 0] - normalizer.mean) / normalizer.std
        test_norm[:, 0] = (test_norm[:, 0] - normalizer.mean) / normalizer.std
    else:
        normalizer.fit(train_seg)
        train_norm = normalizer.transform(train_seg)
        val_norm = normalizer.transform(val_seg)
        test_norm = normalizer.transform(test_seg)

    # Output directory
    out_dir = os.path.join(args.output, args.experiment)
    os.makedirs(out_dir, exist_ok=True)

    # Results accumulator
    all_results = []

    horizons = exp_config["horizons"]
    windows = exp_config["input_windows"]
    models_to_run = exp_config["models"]
    seeds = exp_config["seeds"]
    run_baselines = exp_config.get("run_baselines", True)

    total_scenarios = len(horizons) * len(windows) * (len(models_to_run) * len(seeds) + (2 if run_baselines else 0))
    print(f"\n{'='*60}")
    print(f"Experiment: {args.experiment}")
    print(f"Total scenarios: ~{total_scenarios}")
    print(f"Horizons: {horizons}, Windows: {windows}")
    print(f"Models: {len(models_to_run)}, Seeds: {seeds}")
    print(f"{'='*60}\n")

    scenario_idx = 0

    for w in windows:
        for h in horizons:
            print(f"\n--- H={h}, w={w} ---")

            # Create sequences
            X_tr, y_tr = create_sequences(train_norm, w, h)
            X_va, y_va = create_sequences(val_norm, w, h)
            X_te, y_te = create_sequences(test_norm, w, h)

            if len(X_tr) == 0 or len(X_te) == 0:
                print(f"  SKIP: insufficient data for w={w}, H={h}")
                continue

            print(f"  Samples: train={len(X_tr)}, val={len(X_va)}, test={len(X_te)}")

            # DataLoaders
            bs = train_config["batch_size"]
            tr_loader = DataLoader(TimeSeriesDataset(X_tr, y_tr), batch_size=bs, shuffle=True,
                                   num_workers=4, pin_memory=True)
            va_loader = DataLoader(TimeSeriesDataset(X_va, y_va), batch_size=bs * 2,
                                   num_workers=4, pin_memory=True)
            te_loader = DataLoader(TimeSeriesDataset(X_te, y_te), batch_size=bs * 2,
                                   num_workers=4, pin_memory=True)

            # Original-scale test targets (for metrics)
            X_te_orig = create_sequences(test_seg, w, h)[0]  # unnormalized
            y_te_orig = create_sequences(test_seg, w, h)[1]

            # MASE scale from training set
            m = get_seasonal_period(h)
            train_cons = train_seg[:, cons_idx]
            mase_sc = mase_scale_from_train(train_cons, m)

            # --- Baselines ---
            if run_baselines:
                for bl_name, bl_func in [("Naive", naive_forecast),
                                          ("Seasonal_Naive", seasonal_naive_forecast)]:
                    if bl_name == "Seasonal_Naive":
                        bl_preds = bl_func(X_te_orig, h, m, cons_idx=cons_idx if num_features > 1 else 0)
                    else:
                        bl_preds = bl_func(X_te_orig, h, cons_idx=cons_idx if num_features > 1 else 0)

                    metrics = calculate_metrics(y_te_orig, bl_preds, mase_sc)
                    metrics["MASE_rel"] = compute_mase_rel(y_te_orig, bl_preds, X_te_orig, h)

                    result = {
                        "model_name": bl_name, "input_window": w, "horizon": h,
                        "seed": None, "num_params": 0, "num_features": num_features,
                        "train_samples": len(X_tr), "val_samples": len(X_va),
                        "test_samples": len(X_te), "train_time_sec": 0,
                        "best_epoch": 0, "best_val_loss": None,
                        **metrics
                    }
                    all_results.append(result)
                    scenario_idx += 1
                    print(f"  [{scenario_idx}] {bl_name}: MAPE={metrics['MAPE']:.2f}%")

            # --- Deep Learning Models ---
            for model_name in models_to_run:
                cfg = model_configs[model_name]

                for seed in seeds:
                    set_seed(seed)
                    scenario_idx += 1

                    print(f"  [{scenario_idx}] {model_name} (seed={seed})...")

                    try:
                        model = create_model(model_name, num_features, h, cfg, w)
                        model = model.to(device)
                        n_params = count_parameters(model)

                        # Train
                        train_result = train_model(model, tr_loader, va_loader,
                                                   device, train_config)

                        # Inference
                        t_infer = time.time()
                        preds_norm = run_inference(model, te_loader, device)
                        infer_time = time.time() - t_infer

                        # Inverse transform
                        preds_real = normalizer.inverse_transform_targets(preds_norm)

                        # Metrics
                        metrics = calculate_metrics(y_te_orig, preds_real, mase_sc)
                        metrics["MASE_rel"] = compute_mase_rel(y_te_orig, preds_real, X_te_orig, h)

                        result = {
                            "model_name": model_name, "input_window": w, "horizon": h,
                            "seed": seed, "num_params": n_params,
                            "num_features": num_features,
                            "train_samples": len(X_tr), "val_samples": len(X_va),
                            "test_samples": len(X_te),
                            "train_time_sec": train_result["train_time_sec"],
                            "inference_time_sec": infer_time,
                            "best_epoch": train_result["best_epoch"],
                            "best_val_loss": train_result["best_val_loss"],
                            **metrics
                        }
                        all_results.append(result)
                        print(f"       MAPE={metrics['MAPE']:.2f}% | "
                              f"params={n_params:,} | "
                              f"train={train_result['train_time_sec']:.0f}s | "
                              f"epoch={train_result['best_epoch']}")

                    except Exception as e:
                        print(f"       ERROR: {e}")
                        all_results.append({
                            "model_name": model_name, "input_window": w, "horizon": h,
                            "seed": seed, "error": str(e),
                        })

                    # Cleanup
                    del model
                    gc.collect()
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            # Save intermediate results
            pd.DataFrame(all_results).to_csv(
                os.path.join(out_dir, "results.csv"), index=False
            )

    # Final save
    results_df = pd.DataFrame(all_results)
    csv_path = os.path.join(out_dir, "results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n{'='*60}")
    print(f"Done! {len(all_results)} scenarios completed.")
    print(f"Results saved to: {csv_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
