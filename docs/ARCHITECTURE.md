# Architecture

```
            ┌──────────────────────── Streamlit dashboard (app.py) ───────────────────────┐
 sliders ─▶ │ Overview │ Simulator │ EC Lab │ Noise Sweep │ Regime Map │ Repro │ Learn     │
            └───────┬──────────────────────────────────────────────────────▲──────────────┘
                    │ call_in_worker (single Qiskit thread)                │ tables, figs, text
                    ▼                                                      │
   NoiseConfig ─▶ noise_models.py ──▶ Aer NoiseModel                       │
   (p_x,p_z,                         (Pauli on `id`, CX depolarizing, readout)
    p_gate,p_meas)                          │                              │
   strategies.py ─▶ circuits.py ──▶ transpiled circuits (id,h,x,sx,rz,cx)  │
                                            ▼                              │
                          benchmarks.py: AerSimulator(density_matrix)      │
                          counts → failure rate, Wilson CI, p-values,      │
                          qubits / CX / depth, calibration|validation split│
                                            ▼                              │
                          adaptive.py: budget filter → z-test vs baseline ─┘
                          → objective J → tie-break → held-out check → explanation
```
