# PiNNs-for-Metabolic-Thresholds
Metabolic Threshold Prediction via Physics-Informed Neural Networks (PiNNs) [MetabolicPINN] is an ongoing research project I am doing, aimed at enhancing the accuracy of physiological modeling through the integration of deep learning and physical laws. I am developing a PINN-based framework to model physiological transitions during high-intensity exercise. Whilst typical models often hallucinate, my research mitigates this by using ODEs that govern oxygen uptake kinetics, and lactate accumulation directly into the neural network's loss function. It predicts **VO₂max**, **Aerobic Threshold (AeT)**, and **Anaerobic Threshold (AnT)** from user-accessible measurements (age, weight, heart rate, exercise data, race results), using the ODE framework.

### Eq 1 — VO₂ Kinetics
$$\tau \frac{dVO_2(t)}{dt} + VO_2(t) = VO_{2,ss}$$

**Analytical solution:** $VO_2(t) = VO_{2max} - (VO_{2max} - VO_{2,rest}) \cdot e^{-t/\tau}$

- **τ (tau)** — time constant of VO₂ response (seconds). Fit individuals have lower τ (~20–40s); sedentary adults ~60–90s.
- **VO₂max** — maximal oxygen uptake (ml/kg/min), the gold standard of aerobic fitness.

### Eq 2 — Lactate Accumulation & Clearance
$$\frac{dL}{dt} = P(w) - C \cdot L(t)$$

**Analytical solution:** $L(t) = \frac{P_w}{C} + \left(L_0 - \frac{P_w}{C}\right) e^{-Ct}$

- **P(w)** — lactate production rate (depends on work rate)
- **C** — lactate clearance rate (min⁻¹). Trained athletes have higher C.
- Steady-state: $L_{ss} = P_w / C$
  - **Aerobic Threshold (AeT)**: $L_{ss} \approx 2$ mmol/L → $P_w^{AeT} = 2C$
  - **Anaerobic Threshold (AnT)**: $L_{ss} \approx 4$ mmol/L → $P_w^{AnT} = 4C$

### Eq 3 — Metabolic Power Constraint
$$W_{total} = \eta \left( \alpha \cdot VO_2 + \beta \cdot \frac{dL}{dt} + \gamma \right)$$

- **η (eta)** — gross mechanical efficiency (typically 20–45%)
- **α, β** — weighting of aerobic vs. anaerobic energy pathways
- **γ** — baseline metabolic power offset

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  INPUT  [age, weight, height, RHR, HR_ex, power, duration, speed]   │
│                        (normalised, dim=8)                           │
└────────────────────────────┬─────────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   Input Projection  256-d   │
              │        + GELU               │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   6 × Residual Block        │
              │  (LayerNorm → Linear        │
              │    → GELU → Dropout)        │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │     Output Head (Linear)    │
              │  + Sigmoid-bounded outputs  │
              └──────────────┬──────────────┘
                             │
     ┌───────────────────────▼────────────────────────┐
     │  [VO₂max | τ | C | P_w | L₀ | η | α | β | γ] │
     └────────────────────────────────────────────────┘
```

**Physics-informed loss:**

```
L_total = λ_data  × L_supervised        (labelled VO₂max, τ, C ... if available)
        + λ_ode1  × ||τ·dVO₂/dt + VO₂ - VO₂max||²
        + λ_ode2  × ||dL/dt - P_w + C·L||²
        + λ_power × ||W_total - η(α·VO₂ + β·dL/dt + γ)||²
        + λ_emp   × ||VO₂max - VO₂max_empirical||²    (Uth / ACSM / Daniels)
        + λ_phys  × physiological_bound_violations

```

**Requirements**

- torch>=2.1.0
- numpy>=1.24.0
- pandas>=2.0.0
- matplotlib>=3.7.0
- gradio>=4.0.0
- scikit-learn>=1.3.0
- scipy>=1.11.0
- tqdm>=4.65.0
