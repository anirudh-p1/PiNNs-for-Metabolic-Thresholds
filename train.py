# Training pipeline with AdamW + cosine LR decay, early stopping and checkpointing.
# Run python train.py for synthetic data or python train.py -- data your_file.csv for real measurements.

"""
MetabolicPINN — Training Pipeline
====================================
Run:
    python train.py                     # train on synthetic data
    python train.py --data real_data.csv --epochs 300
"""

import argparse
import os
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from pinn_model import MetabolicPINN, OUTPUT_NAMES, INPUT_FEATURES
from physics_loss import PhysicsLoss
from data_utils import (
    generate_synthetic_population, make_dataloaders,
    normalise, denormalise_tensor, TARGET_COLS
)

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration defaults
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    "epochs":          200,
    "batch_size":      128,
    "lr":              3e-4,
    "weight_decay":    1e-4,
    "hidden_dim":      256,
    "n_blocks":        6,
    "dropout":         0.1,
    "n_synthetic":     20_000,
    "lambda_data":     1.0,
    "lambda_ode1":     0.5,
    "lambda_ode2":     0.5,
    "lambda_power":    0.3,
    "lambda_emp":      0.8,
    "lambda_phys":     0.2,
    "checkpoint_dir":  "checkpoints",
    "patience":        20,          # early stopping patience
}


# ─────────────────────────────────────────────────────────────────────────────
#  Utility functions
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("  CUDA GPU detected — using GPU.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("  Apple MPS detected — using MPS.")
        return torch.device("mps")
    else:
        print("  No GPU found — training on CPU (may be slow for large datasets).")
        return torch.device("cpu")


def evaluate(model, loader, loss_fn, device) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    comps_accum = {}
    n = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            params = model(x)
            loss, comps = loss_fn(params, x, y, lambda xn: denormalise_tensor(xn))
            total_loss += loss.item() * len(x)
            for k, v in comps.items():
                comps_accum[k] = comps_accum.get(k, 0.0) + v * len(x)
            n += len(x)

    avg_loss = total_loss / n
    avg_comps = {k: v / n for k, v in comps_accum.items()}
    return avg_loss, avg_comps


def compute_metrics(model, loader, device) -> dict:
    """Compute MAE on VO2max and threshold percentages."""
    model.eval()
    all_pred = []
    all_true = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            params = model(x)
            all_pred.append(params.cpu())
            all_true.append(y.cpu())

    pred = torch.cat(all_pred, dim=0)   # (N, 9)
    true = torch.cat(all_true, dim=0)   # (N, 9) — may have NaNs

    metrics = {}
    for i, name in enumerate(TARGET_COLS):
        mask = ~torch.isnan(true[:, i])
        if mask.sum() > 0:
            mae = (pred[mask, i] - true[mask, i]).abs().mean().item()
            metrics[f"MAE_{name}"] = mae

    return metrics


def save_checkpoint(state: dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model: MetabolicPINN, optimizer=None) -> int:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    epoch = ckpt.get("epoch", 0)
    print(f"  Loaded checkpoint from epoch {epoch}")
    return epoch


# ─────────────────────────────────────────────────────────────────────────────
#  Main training function
# ─────────────────────────────────────────────────────────────────────────────

def train(config: dict, data_path: Optional[str] = None):
    device = get_device()
    print(f"\n{'='*60}")
    print("  MetabolicPINN — Training")
    print(f"{'='*60}\n")

    # ── Data ────────────────────────────────────────────────────────────────
    if data_path and os.path.exists(data_path):
        import pandas as pd
        print(f"  Loading real data from {data_path}")
        df = pd.read_csv(data_path)
        # Merge with synthetic to augment small real datasets
        df_synth = generate_synthetic_population(config["n_synthetic"] // 4)
        # Mark synthetic vs real (model doesn't need this, but useful for analysis)
        df["source"] = "real"
        df_synth["source"] = "synthetic"
        import pandas as pd2; df = pd2.concat([df, df_synth], ignore_index=True)
        print(f"  Real: {(df['source']=='real').sum():,}  +  Synthetic: {(df['source']=='synthetic').sum():,}")
    else:
        print(f"  No real data found — training on {config['n_synthetic']:,} synthetic samples.")
        print("  (Run with --data your_data.csv to add real measurements)")
        df = generate_synthetic_population(config["n_synthetic"])

    train_dl, val_dl, test_dl = make_dataloaders(df, batch_size=config["batch_size"])

    # ── Model ────────────────────────────────────────────────────────────────
    model = MetabolicPINN(
        hidden_dim=config["hidden_dim"],
        n_blocks=config["n_blocks"],
        dropout=config["dropout"],
    ).to(device)
    print(f"\n  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Loss ─────────────────────────────────────────────────────────────────
    loss_fn = PhysicsLoss(
        lambda_data  = config["lambda_data"],
        lambda_ode1  = config["lambda_ode1"],
        lambda_ode2  = config["lambda_ode2"],
        lambda_power = config["lambda_power"],
        lambda_emp   = config["lambda_emp"],
        lambda_phys  = config["lambda_phys"],
    ).to(device)

    # ── Optimiser ────────────────────────────────────────────────────────────
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)

    # ── Training loop ────────────────────────────────────────────────────────
    history = {"train_loss": [], "val_loss": [], "metrics": []}
    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    print(f"\n  Training for up to {config['epochs']} epochs  (patience={config['patience']})\n")
    print(f"  {'Epoch':>6}  {'Train':>10}  {'Val':>10}  {'ODE1':>8}  {'ODE2':>8}  {'EmpVO2':>8}  {'LR':>10}")
    print(f"  {'-'*70}")

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            params = model(x)
            loss, comps = loss_fn(params, x, y, lambda xn: denormalise_tensor(xn.to(device)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * len(x)

        train_loss /= len(train_dl.dataset)
        val_loss, val_comps = evaluate(model, val_dl, loss_fn, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        lr = scheduler.get_last_lr()[0]

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  {epoch:>6}  {train_loss:>10.5f}  {val_loss:>10.5f}  "
                f"{val_comps.get('ode1',0):>8.5f}  {val_comps.get('ode2',0):>8.5f}  "
                f"{val_comps.get('emp_vo2',0):>8.5f}  {lr:>10.2e}"
            )

        # Early stopping & checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            save_checkpoint(
                {"model_state": model.state_dict(),
                 "optimizer_state": optimizer.state_dict(),
                 "epoch": epoch, "config": config,
                 "val_loss": val_loss},
                f"{config['checkpoint_dir']}/best_model.pt"
            )
        else:
            patience_counter += 1
            if patience_counter >= config["patience"]:
                print(f"\n  Early stopping at epoch {epoch} (best: {best_epoch})")
                break

    # ── Final evaluation ─────────────────────────────────────────────────────
    print(f"\n  Loading best model (epoch {best_epoch})...")
    load_checkpoint(f"{config['checkpoint_dir']}/best_model.pt", model)
    metrics = compute_metrics(model, test_dl, device)

    print(f"\n{'='*60}")
    print("  Test Set Metrics (MAE):")
    for k, v in metrics.items():
        unit = "ml/kg/min" if "vo2" in k.lower() else ""
        print(f"    {k:25s}: {v:.4f} {unit}")
    print(f"{'='*60}\n")

    # Save history & config
    with open(f"{config['checkpoint_dir']}/training_history.json", "w") as f:
        json.dump({"history": history, "metrics": metrics, "config": config}, f, indent=2)

    print(f"  Training complete. Checkpoint saved to: {config['checkpoint_dir']}/best_model.pt")
    return model, history, metrics


# ─────────────────────────────────────────────────────────────────────────────
#  CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────

from typing import Optional

def parse_args():
    p = argparse.ArgumentParser(description="Train MetabolicPINN")
    p.add_argument("--data",       type=str,   default=None,       help="Path to real CSV data")
    p.add_argument("--epochs",     type=int,   default=200)
    p.add_argument("--batch_size", type=int,   default=128)
    p.add_argument("--lr",         type=float, default=3e-4)
    p.add_argument("--hidden_dim", type=int,   default=256)
    p.add_argument("--n_blocks",   type=int,   default=6)
    p.add_argument("--synthetic",  type=int,   default=20_000,     help="Synthetic samples")
    p.add_argument("--out_dir",    type=str,   default="checkpoints")
    
    # Use parse_known_args and only return the 'args' part (the first item in the tuple)
    args, _ = p.parse_known_args()
    return args


if __name__ == "__main__":
    # FIX: Call the function you just defined to get the args
    args = parse_args()
    
    config = DEFAULT_CONFIG.copy()
    config.update({
        "epochs":         args.epochs,
        "batch_size":     args.batch_size,
        "lr":             args.lr,
        "hidden_dim":     args.hidden_dim,
        "n_blocks":       args.n_blocks,
        "n_synthetic":    args.synthetic,
        "checkpoint_dir": args.out_dir,
    })
    train(config, data_path=args.data)
