# PiNNs-for-Metabolic-Thresholds
Metabolic Threshold Prediction via Physics-Informed Neural Networks (PiNNs) is an ongoing research project I am doing, aimed at enhancing the accuracy of physiological modeling through the integration of deep learning and physical laws. I am developing a PINN-based framework to model physiological transitions during high-intensity exercise. Whilst typical models often hallucinate, my research mitigates this by using ODEs that govern oxygen uptake kinetics, and lactate accumulation directly into the neural network's loss function. It predicts **VO₂max**, **Aerobic Threshold (AeT)**, and **Anaerobic Threshold (AnT)** from user-accessible measurements (age, weight, heart rate, exercise data, race results), using the ODE framework.

Equation 1 - Oxygen Uptake (VO2​) Kinetics
$$\frac{d{V}O_2(t)}{dt} = \frac{{V}O_{2max} - {V}O_2(t)}{\tau}$$
or $$\tau \frac{dVO_2(t)}{dt} + VO_2(t) = VO_{2,ss}$$

Equation 2 - Lactate Accumulation & Clearance
$$\frac{dL}{dt} = P(w) - C \cdot L(t)$$

Equation 3 - Metabolic Power Constraint
$$W_{\text{total}} = \eta \left( \alpha \cdot VO_2 + \beta \cdot \frac{dL\text{actate}}{dt} + \gamma \right)$$

### Eq 1 — VO₂ Kinetics
$$\tau \frac{dVO_2(t)}{dt} + VO_2(t) = VO_{2,ss}$$

**Analytical solution:** $VO_2(t) = VO_{2max} - (VO_{2max} - VO_{2,rest}) \cdot e^{-t/\tau}$

- **τ (tau)** — time constant of VO₂ response (seconds). Fit individuals have lower τ (~20–40s); sedentary adults ~60–90s.
- **VO₂max** — maximal oxygen uptake (ml/kg/min), the "gold standard" of aerobic fitness.

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

