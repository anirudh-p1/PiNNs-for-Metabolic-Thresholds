# Synthetic data generator 
# Creates 20,000 physiologically realistic 'people'

"""
MetabolicPINN — Data Utilities
================================
Covers:
  • Input normalisation / denormalisation
  • Synthetic population generator  (used for pre-training and testing)
  • Real dataset loaders            (FRIEND, PhysioNet, Mendeley stubs)
  • PyTorch Dataset & DataLoader wrappers
"""

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
#  Normalisation statistics (mean, std for each input feature)
#  Computed from a large synthetic population — update after collecting real data.
# ─────────────────────────────────────────────────────────────────────────────

NORM_STATS = {
    #              mean    std
    "age":        (35.0,   12.0),
    "weight_kg":  (75.0,   15.0),
    "height_cm":  (172.0,  10.0),
    "resting_hr": (65.0,   10.0),
    "exercise_hr":(150.0,  25.0),
    "power_watts":(200.0,  100.0),
    "duration_min":(45.0,  25.0),
    "race_speed": (3.5,    1.5),
}

FEATURE_ORDER = [
    "age", "weight_kg", "height_cm", "resting_hr",
    "exercise_hr", "power_watts", "duration_min", "race_speed"
]


def normalise(df: pd.DataFrame) -> np.ndarray:
    """Normalise a DataFrame of raw features using z-score statistics."""
    out = np.zeros((len(df), len(FEATURE_ORDER)), dtype=np.float32)
    for i, feat in enumerate(FEATURE_ORDER):
        mean, std = NORM_STATS[feat]
        vals = df[feat].values.astype(np.float32) if feat in df.columns else np.zeros(len(df))
        out[:, i] = (vals - mean) / (std + 1e-8)
    return out


def denormalise_tensor(x: torch.Tensor) -> dict:
    """Convert a normalised input tensor back to real-unit dict."""
    device = x.device
    result = {}
    for i, feat in enumerate(FEATURE_ORDER):
        mean, std = NORM_STATS[feat]
        result[feat] = x[:, i] * std + mean
    return result


def normalise_single(features: dict) -> torch.Tensor:
    """Normalise a single dict of features to a (1, 8) tensor."""
    arr = np.zeros((1, len(FEATURE_ORDER)), dtype=np.float32)
    for i, feat in enumerate(FEATURE_ORDER):
        mean, std = NORM_STATS[feat]
        arr[0, i] = (features.get(feat, 0.0) - mean) / (std + 1e-8)
    return torch.tensor(arr)


# ─────────────────────────────────────────────────────────────────────────────
#  Synthetic Population Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_population(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a physiologically realistic synthetic population.

    Ground-truth VO2max is computed from empirical equations, then
    realistic noise and individual variation are added.

    This data is used for PINN pre-training when real labelled data is absent.
    """
    rng = np.random.default_rng(seed)

    # Demographics
    age        = rng.integers(18, 70, n).astype(float)
    weight_kg  = rng.normal(75, 15, n).clip(45, 150)
    height_cm  = rng.normal(172, 10, n).clip(145, 210)
    sex_male   = rng.binomial(1, 0.5, n)  # 1=male, 0=female
    fitness    = rng.choice(["sedentary", "recreational", "trained", "elite"],
                            n, p=[0.30, 0.40, 0.25, 0.05])

    # ── VO2max (ground truth, ml/kg/min) ──────────────────────────────────
    # Base by fitness category and sex
    vo2_base = np.where(fitness == "sedentary",   30,
               np.where(fitness == "recreational", 42,
               np.where(fitness == "trained",       55, 70)))
    vo2_base += sex_male * 8.0                          # males ~8 ml/kg/min higher
    vo2_base -= (age - 25) * 0.25                       # age-related decline
    # Weight penalty for obese
    bmi = weight_kg / (height_cm / 100) ** 2
    vo2_base -= np.maximum(0, (bmi - 25) * 0.8)
    vo2_base += rng.normal(0, 4, n)                     # individual variation
    vo2max = vo2_base.clip(18, 88)

    # ── Heart rate ───────────────────────────────────────────────────────
    resting_hr  = rng.normal(65, 10, n).clip(35, 100)
    # Fitter people have lower RHR
    resting_hr -= (vo2max - 40) * 0.3
    resting_hr = resting_hr.clip(35, 100)
    hr_max = (220 - age + rng.normal(0, 5, n)).clip(120, 210)
    # Exercise HR at ~80% intensity for those who provided workout data
    has_workout = rng.binomial(1, 0.6, n)
    exercise_hr = hr_max * rng.uniform(0.65, 0.90, n) * has_workout
    exercise_hr += rng.normal(0, 5, n) * has_workout

    # ── Power output (cycling) or 0 ──────────────────────────────────────
    has_power  = rng.binomial(1, 0.35, n)
    # FTP ≈ VO2max * weight * 0.0135  (rough estimate in Watts)
    ftp = vo2max * weight_kg * 0.0135
    power_watts = (ftp * rng.uniform(0.75, 1.05, n) * has_power).clip(0, 600)
    duration_min = (rng.normal(45, 20, n) * has_workout).clip(0, 180)

    # ── Race data (5K / 10K speed) ────────────────────────────────────────
    has_race  = rng.binomial(1, 0.30, n)
    # Jack Daniels: VDOT from vo2max, race pace ≈ f(VDOT)
    # Approximate running speed at 5K: v ≈ (vo2max + 4.60) / (0.182258 * 60)  m/s
    race_speed_ideal = (vo2max + 4.60) / (0.182258 * 60)            # m/s
    race_speed = (race_speed_ideal * rng.uniform(0.88, 1.05, n) * has_race).clip(0, 10)

    # ── Physiological parameters (ground truth for supervised training) ───
    tau = (55 - (vo2max - 35) * 0.4 + rng.normal(0, 5, n)).clip(20, 120)
    C   = (0.3 + (vo2max - 35) * 0.005 + rng.normal(0, 0.05, n)).clip(0.05, 1.5)
    P_w = (0.8 + (power_watts / 200) * 0.4 + rng.normal(0, 0.1, n)).clip(0.1, 5.0)
    L0  = rng.normal(1.2, 0.3, n).clip(0.5, 2.5)
    eta = rng.normal(0.28, 0.04, n).clip(0.15, 0.45)
    alpha = rng.uniform(0.40, 0.80, n)
    beta  = 1.0 - alpha + rng.normal(0, 0.05, n)
    beta  = beta.clip(0.10, 0.70)
    gamma = rng.normal(30, 10, n).clip(5, 80)

    # ── AeT / AnT (for evaluation) ─────────────────────────────────────
    # Steady state lactate at threshold:  AeT → 2mmol/L, AnT → 4mmol/L
    # Production rate at threshold = C * L_threshold
    # Power at AeT ≈ (2*C) * reference_factor
    aet_pct = (60 - (age - 25) * 0.15 + rng.normal(0, 3, n)).clip(50, 75)   # % VO2max
    ant_pct = (82 - (age - 25) * 0.10 + rng.normal(0, 3, n)).clip(70, 92)   # % VO2max
    vo2_aet = vo2max * aet_pct / 100
    vo2_ant = vo2max * ant_pct / 100

    df = pd.DataFrame({
        # Inputs
        "age": age, "weight_kg": weight_kg, "height_cm": height_cm,
        "resting_hr": resting_hr, "exercise_hr": exercise_hr,
        "power_watts": power_watts, "duration_min": duration_min,
        "race_speed": race_speed,
        # Targets
        "vo2max": vo2max, "tau": tau, "C": C, "P_w": P_w,
        "L0": L0, "eta": eta, "alpha": alpha, "beta": beta, "gamma": gamma,
        "vo2_aet": vo2_aet, "vo2_ant": vo2_ant,
        "aet_pct": aet_pct, "ant_pct": ant_pct,
        # Auxiliary
        "sex_male": sex_male, "fitness_level": fitness,
        "bmi": bmi, "hr_max": hr_max,
    })
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  Real dataset loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_friend_registry(path: str) -> pd.DataFrame:
    """
    Load FRIEND Registry data (Kaminsky et al., MSSE 2015).
    Download from:  https://www.exerciseismedicine.org/friend-registry/
    Expected CSV columns: age, sex, weight_kg, height_cm, vo2max, resting_hr, ...

    The FRIEND (Fitness Registry and the Importance of Exercise National Database)
    contains >100,000 cardiopulmonary exercise tests with VO2max from the US population.
    """
    df = pd.read_csv(path)
    # Normalise column names (FRIEND uses specific naming conventions)
    col_map = {
        "Age": "age", "Weight": "weight_kg", "Height": "height_cm",
        "RestHR": "resting_hr", "PeakVO2": "vo2max",
        "MaxHR": "hr_max",
    }
    df = df.rename(columns=col_map)
    # Add missing columns as zeros
    for col in FEATURE_ORDER:
        if col not in df.columns:
            df[col] = 0.0
    return df[FEATURE_ORDER + ["vo2max"]].dropna(subset=["vo2max"])


def load_mendeley_lactate(path: str) -> pd.DataFrame:
    """
    Load lactate threshold datasets from Mendeley Data.
    Several datasets available at: https://data.mendeley.com
    Search: "lactate threshold" or "MLSS" (maximal lactate steady state)

    Good datasets to look for:
      • Faude et al. (2009) "Lactate threshold concepts" — Mendeley
      • Machado et al. lactate kinetics dataset
    """
    df = pd.read_csv(path)
    return df


def load_physionet_cpet(path: str) -> pd.DataFrame:
    """
    Load CPET (Cardiopulmonary Exercise Test) data from PhysioNet.
    See: https://physionet.org/content/
    Search "CPET" or "cardiopulmonary exercise"
    """
    df = pd.read_csv(path)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  PyTorch Dataset
# ─────────────────────────────────────────────────────────────────────────────

TARGET_COLS = ["vo2max", "tau", "C", "P_w", "L0", "eta", "alpha", "beta", "gamma"]


class MetabolicDataset(Dataset):
    """
    Dataset that returns (x, y) pairs where:
      x: normalised input features  (8,)
      y: target parameters          (9,)  — NaN where unavailable
    """

    def __init__(self, df: pd.DataFrame, augment: bool = False, augment_noise: float = 0.02):
        self.x = torch.tensor(normalise(df), dtype=torch.float32)

        # Build targets with NaN for missing columns
        y_arr = np.full((len(df), len(TARGET_COLS)), np.nan, dtype=np.float32)
        for i, col in enumerate(TARGET_COLS):
            if col in df.columns:
                y_arr[:, i] = df[col].values.astype(np.float32)
        self.y = torch.tensor(y_arr, dtype=torch.float32)

        self.augment = augment
        self.noise   = augment_noise

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x, y = self.x[idx].clone(), self.y[idx].clone()
        if self.augment:
            x += torch.randn_like(x) * self.noise
        return x, y


def make_dataloaders(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.05,
    batch_size: int = 128,
    augment: bool = True,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Split df into train/val/test DataLoaders."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_val  = int(len(df) * val_frac)
    n_test = int(len(df) * test_frac)

    df_train = df.iloc[idx[n_val + n_test:]]
    df_val   = df.iloc[idx[:n_val]]
    df_test  = df.iloc[idx[n_val:n_val + n_test]]

    train_ds = MetabolicDataset(df_train, augment=augment)
    val_ds   = MetabolicDataset(df_val,   augment=False)
    test_ds  = MetabolicDataset(df_test,  augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"Dataset splits — Train: {len(train_ds):,}  Val: {len(val_ds):,}  Test: {len(test_ds):,}")
    return train_dl, val_dl, test_dl


if __name__ == "__main__":
    print("Generating synthetic population...")
    df = generate_synthetic_population(5000)
    print(df.describe().T[["mean", "std", "min", "max"]].to_string())
    train_dl, val_dl, _ = make_dataloaders(df, batch_size=64)
    x, y = next(iter(train_dl))
    print(f"\nBatch shapes: x={x.shape}, y={y.shape}")
    print(f"VO2max stats (batch): mean={y[:,0].nanmean().item():.1f} ml/kg/min")
