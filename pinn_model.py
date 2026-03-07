"""
MetabolicPINN - Physics-Informed Neural Network for Metabolic Threshold Prediction
Core model architecture.

Equations governed:
  1. VO2 Kinetics:       tau * dVO2/dt + VO2(t) = VO2_ss
  2. Lactate Dynamics:   dL/dt = P(w) - C * L(t)
  3. Metabolic Power:    W_total = eta * (alpha * VO2 + beta * dL/dt + gamma)
"""

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
#  INPUT FEATURE NAMES  (order matters — matches data_utils normalisation)
# ─────────────────────────────────────────────────────────────────────────────
INPUT_FEATURES = [
    "age",          # years
    "weight_kg",    # kg
    "height_cm",    # cm
    "resting_hr",   # bpm
    "exercise_hr",  # bpm  (mean HR during effort; 0 if unknown)
    "power_watts",  # W    (mean power output; 0 if unknown)
    "duration_min", # min  (exercise duration; 0 if unknown)
    "race_speed",   # m/s  (derived from race distance & time; 0 if no race)
]

# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT NAMES
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_NAMES = [
    "vo2max",       # ml·kg⁻¹·min⁻¹  — maximal oxygen uptake
    "tau",          # s               — VO2 time constant
    "C",            # min⁻¹           — lactate clearance rate
    "P_w",          # mmol·L⁻¹·min⁻¹ — lactate production rate at given power
    "L0",           # mmol·L⁻¹       — resting lactate
    "eta",          #                 — metabolic efficiency (0-1)
    "alpha",        #                 — VO2 weighting in power equation
    "beta",         #                 — lactate weighting in power equation
    "gamma",        # W               — baseline metabolic power (offset)
]

# Physiological bounds  (min, max)
PARAM_BOUNDS = {
    "vo2max":  (20.0,  90.0),
    "tau":     (20.0,  120.0),
    "C":       (0.05,  1.50),
    "P_w":     (0.10,  5.00),
    "L0":      (0.50,  2.50),
    "eta":     (0.15,  0.45),
    "alpha":   (0.30,  0.90),
    "beta":    (0.10,  0.70),
    "gamma":   (5.0,   80.0),
}


class ResidualBlock(nn.Module):
    """Pre-activation residual block with skip connection (improves gradient flow)."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MetabolicPINN(nn.Module):
    """
    Physics-Informed Neural Network for metabolic parameter estimation.

    Architecture:
        Input layer  →  Embedding projection  →  N × ResidualBlock  →  Output head
        Output head applies sigmoid-bounded activations to enforce physiological ranges.
    """

    def __init__(
        self,
        input_dim: int = len(INPUT_FEATURES),
        hidden_dim: int = 256,
        n_blocks: int = 6,
        output_dim: int = len(OUTPUT_NAMES),
        dropout: float = 0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_names = OUTPUT_NAMES

        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )

        # Residual trunk
        self.trunk = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout) for _ in range(n_blocks)]
        )

        # Output head — raw, unbounded
        self.output_head = nn.Linear(hidden_dim, output_dim)

        # Bounds tensors (registered as buffers so they move with .to(device))
        bounds_min = torch.tensor([PARAM_BOUNDS[n][0] for n in OUTPUT_NAMES], dtype=torch.float32)
        bounds_max = torch.tensor([PARAM_BOUNDS[n][1] for n in OUTPUT_NAMES], dtype=torch.float32)
        self.register_buffer("bounds_min", bounds_min)
        self.register_buffer("bounds_max", bounds_max)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, input_dim) — normalised input features
        Returns:
            params: (batch, output_dim) — physiologically bounded parameters
        """
        h = self.input_proj(x)
        h = self.trunk(h)
        raw = self.output_head(h)

        # Sigmoid-bounded outputs → physiological range
        params = self.bounds_min + (self.bounds_max - self.bounds_min) * torch.sigmoid(raw)
        return params

    def get_named_params(self, x: torch.Tensor) -> dict:
        """Returns a dict of predicted parameters, keyed by name."""
        params = self.forward(x)
        return {name: params[:, i] for i, name in enumerate(OUTPUT_NAMES)}

    def predict_thresholds(self, x: torch.Tensor) -> dict:
        """
        Derives aerobic/anaerobic thresholds from the raw predicted parameters.

        Biology:
            Steady-state lactate at a given power = P_w / C
            AeT  ≈ power at which L_ss = 2 mmol/L  → P_w_aet = 2 * C
            AnT  ≈ power at which L_ss = 4 mmol/L  → P_w_ant = 4 * C
            In VO2 units, we scale by the VO2/power relationship.
        """
        p = self.get_named_params(x)

        vo2max = p["vo2max"]
        tau    = p["tau"]
        C      = p["C"]
        P_w    = p["P_w"]
        eta    = p["eta"]

        # Steady-state lactate at the tested power (used for scaling)
        L_ss = P_w / C.clamp(min=1e-6)

        # Threshold powers (relative to current P_w)
        aet_scale = (2.0 / L_ss.clamp(min=1e-6)).clamp(0.0, 1.0)
        ant_scale = (4.0 / L_ss.clamp(min=1e-6)).clamp(0.0, 1.0)

        # VO2 at thresholds — assume linear relationship between power and VO2
        vo2_aet = vo2max * aet_scale.clamp(0.50, 0.75)
        vo2_ant = vo2max * ant_scale.clamp(0.70, 0.92)

        # Fractional intensities
        aet_pct = (vo2_aet / vo2max.clamp(min=1.0)) * 100.0
        ant_pct = (vo2_ant / vo2max.clamp(min=1.0)) * 100.0

        # Approximate HR at thresholds (Karvonen-derived)
        hr_reserve = p.get("exercise_hr", torch.zeros_like(vo2max)) - torch.zeros_like(vo2max)
        # We use VO2-HR linearity assumption:  HR_threshold ≈ HRrest + (HRmax-HRrest) * (VO2_t/VO2max)
        # These are estimated without HR reserve here — full version needs resting/max HR
        # (computed properly in app.py after denormalisation)

        return {
            "vo2max":    vo2max,
            "tau_s":     tau,
            "C_lactate": C,
            "eta":       eta,
            "vo2_aet":   vo2_aet,
            "vo2_ant":   vo2_ant,
            "aet_pct":   aet_pct,
            "ant_pct":   ant_pct,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Analytical ODE solutions  (used in physics loss and inference visualisation)
# ─────────────────────────────────────────────────────────────────────────────

def vo2_trajectory(t: torch.Tensor, vo2max: torch.Tensor, vo2_rest: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """
    Analytical solution to:   tau * dVO2/dt + VO2(t) = VO2max
        VO2(t) = VO2max - (VO2max - VO2_rest) * exp(-t / tau)

    Args:
        t:        (T,)  — time points in seconds
        vo2max:   (B,)  — maximal VO2
        vo2_rest: (B,)  — resting VO2 (~3.5 ml/kg/min)
        tau:      (B,)  — time constant in seconds
    Returns:
        (B, T) tensor of VO2 values
    """
    t = t.unsqueeze(0)                    # (1, T)
    vo2max   = vo2max.unsqueeze(1)        # (B, 1)
    vo2_rest = vo2_rest.unsqueeze(1)      # (B, 1)
    tau      = tau.unsqueeze(1)           # (B, 1)
    return vo2max - (vo2max - vo2_rest) * torch.exp(-t / tau.clamp(min=1e-6))


def lactate_trajectory(t: torch.Tensor, P_w: torch.Tensor, C: torch.Tensor, L0: torch.Tensor) -> torch.Tensor:
    """
    Analytical solution to:   dL/dt = P(w) - C * L(t)
        L(t) = P_w/C + (L0 - P_w/C) * exp(-C * t)

    Args:
        t:    (T,)  — time in minutes
        P_w:  (B,)  — lactate production rate
        C:    (B,)  — clearance rate
        L0:   (B,)  — initial (resting) lactate
    Returns:
        (B, T) tensor
    """
    t  = t.unsqueeze(0)
    P_w = P_w.unsqueeze(1)
    C   = C.unsqueeze(1).clamp(min=1e-6)
    L0  = L0.unsqueeze(1)
    L_ss = P_w / C
    return L_ss + (L0 - L_ss) * torch.exp(-C * t)


if __name__ == "__main__":
    # Quick sanity check
    model = MetabolicPINN()
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    x_dummy = torch.randn(4, len(INPUT_FEATURES))
    out = model(x_dummy)
    print(f"Output shape: {out.shape}")
    print("Named params sample (batch[0]):")
    named = model.get_named_params(x_dummy)
    for k, v in named.items():
        print(f"  {k:12s}: {v[0].item():.4f}")
