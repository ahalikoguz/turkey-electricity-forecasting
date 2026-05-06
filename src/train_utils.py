"""
Training utilities: training loop, seed management, model I/O.
"""

import os
import time
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Dict, Any, Optional

try:
    from torch.amp import autocast, GradScaler
    AMP_DEVICE_TYPE = "cuda"
except ImportError:
    from torch.cuda.amp import autocast, GradScaler
    AMP_DEVICE_TYPE = None


def set_seed(seed: int) -> None:
    """Set seed for full reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model: nn.Module,
                train_loader: DataLoader,
                val_loader: DataLoader,
                device: torch.device,
                config: dict) -> Dict[str, Any]:
    """
    Train a model with Adam, ReduceLROnPlateau, early stopping, AMP.

    Args:
        model: nn.Module
        train_loader, val_loader: DataLoaders
        device: torch.device
        config: dict with keys:
            learning_rate, weight_decay, num_epochs, early_stopping_patience,
            scheduler_factor, scheduler_patience, scheduler_min_lr,
            gradient_clip_norm, use_amp, huber_delta, verbose, print_every

    Returns:
        dict with best_val_loss, best_epoch, train_time_sec, history
    """
    lr = config.get("learning_rate", 1e-3)
    wd = config.get("weight_decay", 1e-5)
    epochs = config.get("num_epochs", 90)
    es_patience = config.get("early_stopping_patience", 10)
    sf = config.get("scheduler_factor", 0.7)
    sp = config.get("scheduler_patience", 4)
    min_lr = config.get("scheduler_min_lr", 1e-6)
    clip = config.get("gradient_clip_norm", 1.0)
    use_amp = config.get("use_amp", True) and device.type == "cuda"
    delta = config.get("huber_delta", 1.0)
    verbose = config.get("verbose", True)
    print_every = config.get("print_every", 5)

    criterion = nn.HuberLoss(delta=delta)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=sf,
                                  patience=sp, min_lr=min_lr)
    scaler = GradScaler() if use_amp else None

    best_val, best_state, best_epoch = float("inf"), None, 0
    patience_ctr = 0
    history = {"train_loss": [], "val_loss": [], "lr": []}
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                ctx = (autocast(device_type=AMP_DEVICE_TYPE)
                       if AMP_DEVICE_TYPE else autocast())
                with ctx:
                    loss = criterion(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step()

            tr_loss += loss.item()
        tr_loss /= max(1, len(train_loader))

        # --- Validate ---
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                va_loss += criterion(
                    model(xb.to(device, non_blocking=True)),
                    yb.to(device, non_blocking=True)
                ).item()
        va_loss /= max(1, len(val_loader))

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["lr"].append(current_lr)
        scheduler.step(va_loss)

        if va_loss < best_val:
            best_val = va_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1

        if verbose and (epoch == 1 or epoch % print_every == 0):
            print(f"    Epoch {epoch:3d}/{epochs} | "
                  f"Train: {tr_loss:.6f} | Val: {va_loss:.6f} | LR: {current_lr:.2e}")

        if patience_ctr >= es_patience:
            if verbose:
                print(f"    Early stopping @ epoch {epoch}")
            break

    if best_state:
        model.load_state_dict(best_state)

    train_time = time.time() - t_start

    return {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "train_time_sec": train_time,
        "history": history,
    }


def run_inference(model: nn.Module, dataloader: DataLoader,
                  device: torch.device) -> np.ndarray:
    """Run inference and return predictions as numpy array."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for xb, _ in dataloader:
            preds = model(xb.to(device, non_blocking=True))
            all_preds.append(preds.cpu().numpy())
    return np.concatenate(all_preds, axis=0)
