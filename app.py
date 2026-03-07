# Gradio UI with 5 plots
# 1. Vo2, 2. Lactate, 3. Zone, 4. Hr/VO2 and 5. VO2 gauge

"""
MetabolicPINN — Gradio User Interface
======================================
Run:  python app.py
Opens a browser at http://localhost:7860
"""

import os
import numpy as np
import torch
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

from pinn_model import MetabolicPINN, INPUT_FEATURES
from data_utils import normalise_single, denormalise_tensor
from pinn_model import vo2_trajectory, lactate_trajectory

# ─────────────────────────────────────────────────────────────────────────────
#  Load model
# ─────────────────────────────────────────────────────────────────────────────

CHECKPOINT = "checkpoints/best_model.pt"
_model = None

def get_model() -> MetabolicPINN:
    global _model
    if _model is not None:
        return _model

    _model = MetabolicPINN()

    if Path(CHECKPOINT).exists():
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        _model.load_state_dict(ckpt["model_state"])
        print(f"  Loaded trained model from {CHECKPOINT}")
    else:
        print("  WARNING: No checkpoint found — running with random weights.")
        print("  Run `python train.py` first to train the model.")

    _model.eval()
    return _model


# ─────────────────────────────────────────────────────────────────────────────
#  Derived quantities helpers
# ─────────────────────────────────────────────────────────────────────────────

def hr_at_intensity(intensity_pct: float, hr_rest: float, hr_max: float) -> float:
    """Karvonen formula: HR = HRrest + pct * (HRmax - HRrest)"""
    return hr_rest + (intensity_pct / 100.0) * (hr_max - hr_rest)


def pace_from_vo2(vo2_target: float) -> str:
    """Approximate running pace (min/km) from VO2 using ACSM running equation."""
    # VO2 (ml/kg/min) = 3.5 + 0.2 * speed_m_min
    speed_m_min = max(1.0, (vo2_target - 3.5) / 0.2)
    speed_km_h  = speed_m_min * 60 / 1000
    min_per_km  = 60.0 / speed_km_h
    mins        = int(min_per_km)
    secs        = int((min_per_km - mins) * 60)
    return f"{mins}:{secs:02d} /km"


# ─────────────────────────────────────────────────────────────────────────────
#  Core prediction function
# ─────────────────────────────────────────────────────────────────────────────

def predict(
    age, weight_kg, height_cm, resting_hr,
    has_workout, exercise_hr, power_watts, duration_min,
    has_race, race_distance_km, race_time_min,
):
    # ── Input validation ────────────────────────────────────────────────────
    errors = []
    if age < 10 or age > 100:    errors.append("Age must be between 10 and 100.")
    if weight_kg < 30:           errors.append("Weight seems too low.")
    if height_cm < 100:          errors.append("Height seems too low.")
    if resting_hr < 30:          errors.append("Resting HR seems too low.")
    if errors:
        return "\n".join(errors), None, None

    # ── Race speed ─────────────────────────────────────────────────────────
    race_speed = 0.0
    if has_workout and race_distance_km > 0 and race_time_min > 0:
        race_speed = (race_distance_km * 1000) / (race_time_min * 60)  # m/s

    # ── Build feature dict ─────────────────────────────────────────────────
    features = {
        "age":          age,
        "weight_kg":    weight_kg,
        "height_cm":    height_cm,
        "resting_hr":   resting_hr,
        "exercise_hr":  exercise_hr if has_workout else 0.0,
        "power_watts":  power_watts if has_workout else 0.0,
        "duration_min": duration_min if has_workout else 0.0,
        "race_speed":   race_speed,
    }

    x = normalise_single(features)  # (1, 8)
    model = get_model()

    with torch.no_grad():
        params = model(x)

    # Unpack predictions
    vo2max   = params[0, 0].item()
    tau      = params[0, 1].item()
    C        = params[0, 2].item()
    P_w      = params[0, 3].item()
    L0       = params[0, 4].item()
    eta      = params[0, 5].item()
    alpha    = params[0, 6].item()
    beta     = params[0, 7].item()
    gamma    = params[0, 8].item()

    # ── Derive thresholds ──────────────────────────────────────────────────
    # Steady-state lactate = P_w / C
    L_ss = P_w / max(C, 1e-6)

    # AeT: L_ss = 2 mmol/L → scale factor
    aet_scale = min(max(2.0 / max(L_ss, 0.1), 0.50), 0.75)
    ant_scale = min(max(4.0 / max(L_ss, 0.1), 0.70), 0.92)

    vo2_aet = vo2max * aet_scale
    vo2_ant = vo2max * ant_scale
    aet_pct = (vo2_aet / max(vo2max, 1.0)) * 100
    ant_pct = (vo2_ant / max(vo2max, 1.0)) * 100

    # HR at thresholds
    hr_max = max(220 - age, 120)
    hr_aet = hr_at_intensity(aet_pct, resting_hr, hr_max)
    hr_ant = hr_at_intensity(ant_pct, resting_hr, hr_max)

    # Paces
    pace_aet = pace_from_vo2(vo2_aet)
    pace_ant = pace_from_vo2(vo2_ant)

    # FTP estimate (power at AnT ≈ 95% of functional threshold)
    # Using W = eta * (alpha * VO2 + beta * dL/dt + gamma)
    dL_aet = P_w * aet_scale - C * 2.0
    dL_ant = P_w * ant_scale - C * 4.0
    w_ant  = eta * (alpha * (vo2_ant * weight_kg / 1000 * 20.9) + beta * max(dL_ant, 0) + gamma)
    w_aet  = eta * (alpha * (vo2_aet * weight_kg / 1000 * 20.9) + beta * max(dL_aet, 0) + gamma)

    # ── Summary text ──────────────────────────────────────────────────────
    summary = f"""
╔══════════════════════════════════════════════════════════════╗
║             METABOLIC THRESHOLD PREDICTION REPORT            ║
╠══════════════════════════════════════════════════════════════╣
║  AEROBIC CAPACITY                                            ║
║  ─────────────────────────────────────────────────────────   ║
║  VO₂max                 : {vo2max:>6.1f} ml/kg/min           ║
║  VO₂ time constant (τ)  : {tau:>6.1f} s   (kinetic response) ║
║                                                              ║
║  AEROBIC THRESHOLD (AeT)  ≈ {aet_pct:.0f}% VO₂max            ║
║  ─────────────────────────────────────────────────────────   ║
║  VO₂ at AeT             : {vo2_aet:>6.1f} ml/kg/min          ║
║  Heart Rate at AeT      : {hr_aet:>6.0f} bpm                 ║
║  Running pace at AeT    : {pace_aet:>10s}                    ║
║  Lactate ~2 mmol/L      ← first threshold                    ║
║                                                              ║
║  ANAEROBIC THRESHOLD (AnT)  ≈ {ant_pct:.0f}% VO₂max          ║
║  ─────────────────────────────────────────────────────────   ║
║  VO₂ at AnT             : {vo2_ant:>6.1f} ml/kg/min          ║
║  Heart Rate at AnT      : {hr_ant:>6.0f} bpm                 ║
║  Running pace at AnT    : {pace_ant:>10s}                    ║
║  Lactate ~4 mmol/L      ← MLSS / OBLA                        ║
║                                                              ║
║  LACTATE KINETICS                                            ║
║  ─────────────────────────────────────────────────────────   ║
║  Clearance rate (C)     : {C:>6.3f} min⁻¹                    ║
║  Production rate (P_w)  : {P_w:>6.3f} mmol/L/min             ║
║  Resting lactate (L₀)   : {L0:>6.2f} mmol/L                  ║
║                                                              ║
║  METABOLIC EFFICIENCY                                        ║
║  ─────────────────────────────────────────────────────────   ║
║  Efficiency (η)         : {eta*100:>5.1f}%                   ║
╚══════════════════════════════════════════════════════════════╝
"""
    # Return values for plots
    plot_data = {
        "vo2max": vo2max, "tau": tau, "C": C, "P_w": P_w, "L0": L0,
        "eta": eta, "alpha": alpha, "beta": beta, "gamma": gamma,
        "vo2_aet": vo2_aet, "vo2_ant": vo2_ant,
        "aet_pct": aet_pct, "ant_pct": ant_pct,
        "hr_aet": hr_aet, "hr_ant": hr_ant,
        "hr_max": hr_max, "resting_hr": resting_hr,
        "weight_kg": weight_kg,
    }

    fig = make_plots(plot_data)
    return summary, fig, make_training_zone_table(vo2max, vo2_aet, vo2_ant, resting_hr, hr_max)


# ─────────────────────────────────────────────────────────────────────────────
#  Visualisations
# ─────────────────────────────────────────────────────────────────────────────

DARK_BG   = "#0d1117"
CARD_BG   = "#161b22"
ACCENT1   = "#58a6ff"   # blue
ACCENT2   = "#f78166"   # red/orange
ACCENT3   = "#3fb950"   # green
ACCENT4   = "#d2a8ff"   # purple
GRID_COL  = "#30363d"
TEXT_COL  = "#e6edf3"

plt.rcParams.update({
    "figure.facecolor":  DARK_BG,
    "axes.facecolor":    CARD_BG,
    "axes.edgecolor":    GRID_COL,
    "axes.labelcolor":   TEXT_COL,
    "xtick.color":       TEXT_COL,
    "ytick.color":       TEXT_COL,
    "grid.color":        GRID_COL,
    "text.color":        TEXT_COL,
    "font.family":       "DejaVu Sans",
    "axes.titlepad":     12,
})


def make_plots(d: dict) -> plt.Figure:
    fig = plt.figure(figsize=(16, 10), facecolor=DARK_BG)
    gs  = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35,
                   left=0.07, right=0.97, top=0.92, bottom=0.08)

    _plot_vo2_kinetics(fig.add_subplot(gs[0, 0]), d)
    _plot_lactate_curve(fig.add_subplot(gs[0, 1]), d)
    _plot_zone_bar(fig.add_subplot(gs[0, 2]), d)
    _plot_threshold_intensity(fig.add_subplot(gs[1, 0:2]), d)
    _plot_metabolic_gauge(fig.add_subplot(gs[1, 2]), d)

    fig.suptitle("MetabolicPINN — Physiological Analysis", color=TEXT_COL,
                 fontsize=15, fontweight="bold", y=0.97)
    return fig


def _plot_vo2_kinetics(ax, d):
    t = torch.linspace(0, 360, 200)
    vo2max   = torch.tensor([d["vo2max"]])
    vo2_rest = torch.tensor([3.5])
    tau      = torch.tensor([d["tau"]])
    vo2_t    = vo2_trajectory(t, vo2max, vo2_rest, tau)[0].numpy()

    ax.plot(t.numpy(), vo2_t, color=ACCENT1, lw=2.5, label="VO₂(t)")
    ax.axhline(d["vo2max"],   color=ACCENT2, lw=1.2, ls="--", alpha=0.8, label=f"VO₂max = {d['vo2max']:.1f}")
    ax.axhline(d["vo2_aet"],  color=ACCENT3, lw=1.0, ls=":",  alpha=0.8, label=f"AeT = {d['vo2_aet']:.1f}")
    ax.axhline(d["vo2_ant"],  color=ACCENT4, lw=1.0, ls=":",  alpha=0.8, label=f"AnT = {d['vo2_ant']:.1f}")
    ax.axvline(d["tau"],      color=ACCENT1, lw=0.8, ls=":",  alpha=0.5)
    ax.text(d["tau"] + 5, 4.5, f"τ = {d['tau']:.0f}s", color=ACCENT1, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("VO₂ (ml/kg/min)")
    ax.set_title("Eq 1 — VO₂ Kinetics", fontweight="bold")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(True, alpha=0.3)


def _plot_lactate_curve(ax, d):
    t = torch.linspace(0, 60, 200)
    P_w = torch.tensor([d["P_w"]])
    C   = torch.tensor([d["C"]])
    L0  = torch.tensor([d["L0"]])
    L_t = lactate_trajectory(t, P_w, C, L0)[0].numpy()

    ax.plot(t.numpy(), L_t, color=ACCENT2, lw=2.5, label="Lactate(t)")
    ax.axhline(2.0, color=ACCENT3, lw=1.2, ls="--", alpha=0.8, label="AeT ≈ 2 mmol/L")
    ax.axhline(4.0, color=ACCENT4, lw=1.2, ls="--", alpha=0.8, label="AnT ≈ 4 mmol/L")
    L_ss = d["P_w"] / max(d["C"], 1e-6)
    ax.axhline(L_ss, color=ACCENT2, lw=0.8, ls=":",  alpha=0.5, label=f"L_ss = {L_ss:.2f}")
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Blood Lactate (mmol/L)")
    ax.set_title("Eq 2 — Lactate Dynamics", fontweight="bold")
    ax.legend(fontsize=7.5)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(L_t.max() * 1.2, 5.0))


def _plot_zone_bar(ax, d):
    categories = ["Zone 1\n(Recovery)", "Zone 2\n(Aerobic)", "Zone 3\n(Tempo)",
                  "Zone 4\n(Threshold)", "Zone 5\n(VO₂max)"]
    colors = ["#2ea043", "#58a6ff", "#e3b341", "#f78166", "#db6d28"]

    aet_pct = d["aet_pct"]
    ant_pct = d["ant_pct"]
    boundaries = [0, aet_pct * 0.75, aet_pct, ant_pct, ant_pct * 1.04, 100]
    widths     = [boundaries[i+1] - boundaries[i] for i in range(5)]
    lefts      = boundaries[:-1]

    bars = ax.barh([0]*5, widths, left=lefts, color=colors, height=0.5, edgecolor=DARK_BG, linewidth=1.5)
    ax.axvline(aet_pct, color=ACCENT3, lw=2, ls="--")
    ax.axvline(ant_pct, color=ACCENT4, lw=2, ls="--")
    ax.text(aet_pct + 1, 0.3, f"AeT\n{aet_pct:.0f}%", color=ACCENT3, fontsize=8, va="center")
    ax.text(ant_pct + 1, 0.3, f"AnT\n{ant_pct:.0f}%", color=ACCENT4, fontsize=8, va="center")

    ax.set_xlim(0, 110)
    ax.set_yticks([])
    ax.set_xlabel("% VO₂max")
    ax.set_title("Training Zone Distribution", fontweight="bold")

    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, categories)]
    ax.legend(handles=patches, fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, axis="x", alpha=0.3)


def _plot_threshold_intensity(ax, d):
    hr_rest = d["resting_hr"]
    hr_max  = d["hr_max"]
    pcts    = np.linspace(0, 100, 100)
    hrs     = hr_rest + pcts / 100 * (hr_max - hr_rest)
    vo2s    = d["vo2max"] * pcts / 100

    ax.plot(hrs, vo2s, color=ACCENT1, lw=2.5, label="VO₂ vs HR")

    # AeT marker
    ax.axvline(d["hr_aet"], color=ACCENT3, lw=1.5, ls="--", alpha=0.9)
    ax.axhline(d["vo2_aet"], color=ACCENT3, lw=1.5, ls="--", alpha=0.9)
    ax.scatter(d["hr_aet"], d["vo2_aet"], color=ACCENT3, s=80, zorder=5,
               label=f"AeT — {d['hr_aet']:.0f} bpm / {d['vo2_aet']:.1f} ml/kg/min")

    # AnT marker
    ax.axvline(d["hr_ant"], color=ACCENT4, lw=1.5, ls="--", alpha=0.9)
    ax.axhline(d["vo2_ant"], color=ACCENT4, lw=1.5, ls="--", alpha=0.9)
    ax.scatter(d["hr_ant"], d["vo2_ant"], color=ACCENT4, s=80, zorder=5,
               label=f"AnT — {d['hr_ant']:.0f} bpm / {d['vo2_ant']:.1f} ml/kg/min")

    ax.set_xlabel("Heart Rate (bpm)")
    ax.set_ylabel("VO₂ (ml/kg/min)")
    ax.set_title("Eq 3 — Threshold Intensity Map (HR vs VO₂)", fontweight="bold")
    ax.legend(fontsize=8.5)
    ax.grid(True, alpha=0.3)


def _plot_metabolic_gauge(ax, d):
    # Radial gauge for VO2max fitness category
    vo2 = d["vo2max"]

    # Fitness categories (ml/kg/min, approximate mixed sex)
    thresholds = [20, 30, 40, 50, 60, 75, 90]
    labels     = ["Poor", "Fair", "Average", "Good", "Excellent", "Elite", ""]
    colors_g   = ["#f85149", "#f78166", "#e3b341", "#58a6ff", "#3fb950", "#d2a8ff"]

    theta_start, theta_end = np.pi, 0  # half-circle
    n_seg = len(colors_g)
    thetas = np.linspace(theta_start, theta_end, n_seg + 1)

    for i in range(n_seg):
        ax.bar(x=(thetas[i] + thetas[i+1]) / 2, height=0.3,
               width=thetas[i+1] - thetas[i],
               bottom=0.65, color=colors_g[i], alpha=0.7,
               edgecolor=DARK_BG, linewidth=1.5)

    # Needle
    pct_along = (vo2 - thresholds[0]) / (thresholds[-2] - thresholds[0])
    pct_along = np.clip(pct_along, 0, 1)
    needle_theta = theta_start + pct_along * (theta_end - theta_start)
    ax.annotate("", xy=(np.cos(needle_theta) * 0.8, np.sin(needle_theta) * 0.8),
                xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color=TEXT_COL, lw=2))
    ax.plot(0, 0, "o", color=TEXT_COL, ms=8, zorder=10)

    # Category text
    cat_idx = np.searchsorted(thresholds[1:-1], vo2)
    cat_label = labels[min(cat_idx, len(labels)-2)]
    ax.text(0, 0.25, f"{vo2:.1f}", ha="center", va="center",
            fontsize=20, fontweight="bold", color=TEXT_COL)
    ax.text(0, -0.15, "ml/kg/min", ha="center", va="center", fontsize=9, color=TEXT_COL, alpha=0.8)
    ax.text(0, -0.40, cat_label, ha="center", va="center",
            fontsize=13, fontweight="bold", color=colors_g[min(cat_idx, len(colors_g)-1)])

    # Labels
    for i, lab in enumerate(labels[:-1]):
        theta = (thetas[i] + thetas[i+1]) / 2
        ax.text(np.cos(theta) * 1.05, np.sin(theta) * 1.05, lab,
                ha="center", va="center", fontsize=6.5, color=TEXT_COL, alpha=0.7)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.6, 1.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("VO₂max Fitness Category", fontweight="bold")


def make_training_zone_table(vo2max, vo2_aet, vo2_ant, hr_rest, hr_max) -> str:
    zones = [
        ("Zone 1", "Recovery",   0.60 * vo2max, 0.65 * vo2max),
        ("Zone 2", "Aerobic",    vo2_aet * 0.75, vo2_aet),
        ("Zone 3", "Tempo",      vo2_aet,         vo2_ant * 0.92),
        ("Zone 4", "Threshold",  vo2_ant * 0.92,  vo2_ant),
        ("Zone 5", "VO₂max",     vo2_ant,          vo2max),
    ]

    lines = ["PERSONALISED TRAINING ZONES\n"]
    lines.append(f"{'Zone':<8}  {'Name':<12}  {'VO₂ range':>18}  {'HR range':>16}  {'Purpose'}")
    lines.append("─" * 82)

    purposes = [
        "Active recovery, fat metabolism",
        "Base fitness, primary aerobic zone",
        "Race pace, lactate clearance",
        "Threshold development",
        "Max power, anaerobic capacity",
    ]

    for (z, name, v_lo, v_hi), purpose in zip(zones, purposes):
        hr_lo = hr_at_intensity((v_lo / max(vo2max, 1)) * 100, hr_rest, hr_max)
        hr_hi = hr_at_intensity((v_hi / max(vo2max, 1)) * 100, hr_rest, hr_max)
        lines.append(
            f"{z:<8}  {name:<12}  {v_lo:6.1f}–{v_hi:5.1f} ml/kg/min  "
            f"{hr_lo:5.0f}–{hr_hi:4.0f} bpm  {purpose}"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
#  Gradio Interface
# ─────────────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
body { background: #0d1117 !important; }
.gradio-container { max-width: 1100px; margin: auto; }
.gr-box { border-radius: 10px !important; border: 1px solid #30363d !important; }
h1, h2, h3 { color: #58a6ff !important; }
.gr-button-primary { background: #238636 !important; border: 1px solid #2ea043 !important; }
.gr-button-primary:hover { background: #2ea043 !important; }
#title-banner {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    margin-bottom: 20px;
}
"""

DESCRIPTION = """
<div id="title-banner">
<h1 style="color:#58a6ff; font-size:2em; margin:0;">⚡ MetabolicPINN</h1>
<p style="color:#8b949e; font-size:1.05em; margin-top:8px;">
Physics-Informed Neural Network for Metabolic Threshold Prediction<br>
<em>Governing equations: VO₂ kinetics · Lactate dynamics · Metabolic power constraint</em>
</p>
</div>
"""

def toggle_workout(has_workout):
    return gr.update(visible=has_workout)

def toggle_race(has_race):
    return gr.update(visible=has_race)


def build_interface():
    with gr.Blocks(css=CUSTOM_CSS, title="MetabolicPINN") as demo:
        gr.HTML(DESCRIPTION)

        with gr.Row():
            # ── LEFT COLUMN: Inputs ─────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### 👤 Personal Profile")
                with gr.Group():
                    age        = gr.Slider(15, 80, value=30, step=1,  label="Age (years)")
                    weight_kg  = gr.Slider(40, 150, value=75, step=0.5, label="Weight (kg)")
                    height_cm  = gr.Slider(140, 220, value=175, step=1, label="Height (cm)")
                    resting_hr = gr.Slider(35, 100, value=65, step=1,  label="Resting Heart Rate (bpm)")

                gr.Markdown("### 🏋️ Exercise Data (optional)")
                has_workout = gr.Checkbox(label="I have workout data", value=False)
                with gr.Group(visible=False) as workout_group:
                    exercise_hr  = gr.Slider(60, 220, value=150, step=1,  label="Average Exercise HR (bpm)")
                    power_watts  = gr.Slider(0, 600,  value=200, step=5,  label="Average Power (watts)  — set 0 if running/unknown")
                    duration_min = gr.Slider(1, 300,  value=45,  step=1,  label="Exercise Duration (min)")

                gr.Markdown("### 🏃 Race / Time Trial (optional)")
                has_race = gr.Checkbox(label="I have race data", value=False)
                with gr.Group(visible=False) as race_group:
                    race_distance = gr.Dropdown(
                        choices=["1K (1 km)", "5K (5 km)", "10K (10 km)", "Half Marathon (21.1 km)", "Marathon (42.2 km)"],
                        value="5K (5 km)", label="Race Distance"
                    )
                    race_time_min = gr.Slider(5, 300, value=25, step=0.5, label="Race Time (minutes)")

                predict_btn = gr.Button("🔬 Predict Metabolic Thresholds", variant="primary", size="lg")

            # ── RIGHT COLUMN: Outputs ───────────────────────────────────────
            with gr.Column(scale=2):
                gr.Markdown("### 📊 Results")
                summary_text = gr.Textbox(
                    label="Metabolic Report",
                    lines=28, max_lines=35,
                    show_copy_button=True,
                )
                gr.Markdown("### 📈 Physiological Plots")
                plot_output = gr.Plot(label="Analysis Charts")
                gr.Markdown("### 🏅 Training Zones")
                zones_text = gr.Textbox(
                    label="Personalised Training Zones",
                    lines=12, max_lines=15,
                    show_copy_button=True,
                )

        # ── Toggle visibility ──────────────────────────────────────────────
        has_workout.change(toggle_workout, has_workout, workout_group)
        has_race.change(toggle_race, has_race, race_group)

        # ── Wire up predict ────────────────────────────────────────────────
        DIST_MAP = {
            "1K (1 km)": 1.0, "5K (5 km)": 5.0, "10K (10 km)": 10.0,
            "Half Marathon (21.1 km)": 21.1, "Marathon (42.2 km)": 42.2,
        }

        def run_prediction(age, wt, ht, rhr, hw, ehr, pw, dm, hr, rd, rtm):
            dist_km = DIST_MAP.get(rd, 5.0)
            return predict(age, wt, ht, rhr, hw, ehr, pw, dm, hr, dist_km, rtm)

        predict_btn.click(
            run_prediction,
            inputs=[age, weight_kg, height_cm, resting_hr,
                    has_workout, exercise_hr, power_watts, duration_min,
                    has_race, race_distance, race_time_min],
            outputs=[summary_text, plot_output, zones_text],
        )

        gr.Markdown("""
---
> **Disclaimer:** Predictions are based on physics-informed modelling and empirical equations.
> For clinical or competitive use, validate with a professional CPET (cardiopulmonary exercise test).
> Model accuracy improves significantly after training on real lactate/VO₂max datasets.
        """)

    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port",  type=int, default=7860)
    
    args, _ = parser.parse_known_args()

    demo = build_interface()
    
    demo.launch(server_port=args.port, share=args.share, show_error=True)
