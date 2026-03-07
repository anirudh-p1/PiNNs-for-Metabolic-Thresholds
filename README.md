# PiNNs-for-Metabolic-Thresholds
Metabolic Threshold Prediction via Physics-Informed Neural Networks (PiNNs) is an ongoing research project I am doing, aimed at enhancing the accuracy of physiological modeling through the integration of deep learning and physical laws. I am developing a PINN-based framework to model physiological transitions during high-intensity exercise. Whilst typical models often hallucinate, my research mitigates this by using ODEs that govern oxygen uptake kinetics, and lactate accumulation directly into the neural network's loss function.

Equation 1 - Oxygen Uptake (VO2​) Kinetics
$$\frac{d\dot{V}O_2(t)}{dt} = \frac{\dot{V}O_{2max} - \dot{V}O_2(t)}{\tau}$$
​
Equation 2 - Lactate Accumulation & Clearance
$$\frac{dL}{dt} = P(w) - C \cdot L(t)$$
