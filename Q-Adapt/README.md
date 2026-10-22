# ⚛️ Q-Adapt — Noise-Aware Adaptive Quantum Error Correction

Qiskit Fall Fest 2026 · Track 3 (Quantum Hardware & Error Correction)

Q-Adapt **simulates** noisy quantum memories with Qiskit Aer, **benchmarks** several error-correction
strategies, and an **adaptive engine** recommends the best one for the noise and the qubit budget you
choose — or recommends *no correction* when the data show it doesn't help.

## Quick start
```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                      # 35 tests (analytic checks, engine logic, dashboard smoke tests)
streamlit run app.py           # opens http://localhost:8501
python run_benchmark.py        # optional: regenerates results/ (CSV, PNG, report)
```
Tested with Python 3.13, qiskit 2.6.0, qiskit-aer 0.18.0, streamlit 1.66.1.

## What it does
| Area | Implementation |
|---|---|
| Noise Lab | Bit-flip `p_x`, phase-flip `p_z`, CX-gate depolarizing `p_gate`, read-out `p_meas` → a real Aer `NoiseModel` |
| Simulator | Aer `density_matrix`, ideal vs noisy runs, circuit drawings, expected-vs-observed table |
| Strategies | Uncorrected · 3q/5q **bit-flip** code · 3q/5q **phase-flip** code (coherent decoders) · 3q/5q **majority-vote** (simplified, classical-bit only) |
| Benchmarks | Failure rate, success prob., Wilson 95% CI, trial-to-trial σ, improvement vs baseline, p-value, qubits, CX count, gate count, depth |
| Adaptive engine | Budget filter → significance test vs baseline → objective `J = failure + λ(qubits−1)` → tie-break to cheapest → held-out validation |
| Dashboard | Dark Streamlit UI, 7 tabs, sweeps, noise-regime map, CSV/JSON/Markdown export |

## Experiment definition (scientific details)
* **Memory experiment:** prepare logical |0⟩ (Z test) and, for the *quantum* task, |+⟩ (X test) → encode → noisy storage (`id` gates carry the Pauli noise) → decode → measure. Expected result is 0; a **logical failure** is a decoded 1.
* **Failure rate** = failures / shots pooled over the tested states. **Success** = 1 − failure. **CI** = Wilson score 95%.
* **Qubit overhead** = physical qubits per logical qubit. **Gate count / depth** come from the transpiled circuit (basis `id,h,x,sx,rz,cx`, no optimisation) of the Z-readout circuit.
* **Calibration / validation split:** trials alternate between *calibration* (engine decides) and *held-out* (reports real performance of the choice) → no selection bias in the adaptive curve.
* **Quantum vs classical task:** the *quantum* task needs a real code (|0⟩ and |+⟩ both tested). The *classical* task only needs a stored bit, so majority-vote is allowed — it is **simplified logical-bit correction, not full QEC**.

## Decision algorithm (no ML)
1. Exclude strategies with qubits > budget. 2. Keep a strategy as *candidate* only if it beats the baseline (one-sided two-proportion z-test, p < α). 3. Minimise `J = failure + λ·(qubits−1)`. 4. Strategies within 1.96·σ of the best J are *tied* → pick fewest qubits, then CX, then depth. 5. If nothing qualifies → recommend no correction and say why. 6. Check the choice on held-out data and flag unconfirmed improvements.

## Real results from this project (`results/`, seed 2026, 5000 shots × 6 trials)
* Bit-flip noise: engine picks **3q bit-flip code** for p_x ≤ 0.10 and **5q** for p_x ≥ 0.125 (e.g. p_x=0.10: 5.15% → 1.39%; p_x=0.20: 9.8% → 2.8%, held-out).
* Matches theory: 3-qubit failure ≈ 3p²−2p³ per readout basis; unit tests check this with statistical tolerances.
* Mixed X+Z noise (p_x, p_z both > 0): **no single repetition code helps** → engine recommends *no correction*.
* Gate noise: with p_x = 0.05, even **p_gate = 0.006** makes the 10-CX decoder lose to doing nothing; the 5q decoder (188 CX) is far worse.
* Regime map, sweeps and per-point tables: `results/BENCHMARK_REPORT.md`, `results/*.csv`, `results/benchmark_overview.png`.

## Honest limitations
* **Noise is simulated, user-chosen** — not estimated from hardware. Real devices have amplitude damping, crosstalk, leakage, correlated errors.
* Repetition codes protect against **one Pauli type only**; nothing here corrects arbitrary errors. No Shor/Steane/surface code, no ancilla syndrome extraction, no mid-circuit measurement, no fault-tolerance.
* Encode/decode gates are ideal unless `p_gate>0`; storage noise is a single Pauli channel per qubit.
* 5q decoder is an ancilla-free multi-controlled-X decomposition → very expensive (188 CX).
* Failure is averaged over |0⟩/|+⟩, not a full process fidelity.
* The adaptive engine does **not** always win; it can only pick among the strategies implemented, and borderline regime-map cells vary with the seed.

## Troubleshooting note (important)
In the tested environment, creating a Qiskit circuit in a *fresh Streamlit script thread* segfaulted on the second rerun. All Qiskit work therefore runs on one long-lived worker thread (`qadapt/worker.py`). Don't call Qiskit directly from `app.py`; use `call_in_worker`. If the app ever crashes after a click on your machine, please report your Python/Qiskit/Streamlit versions.

## Future work
Syndrome-extraction codes with ancillas and mid-circuit measurement; Steane/Shor/surface-code strategies; correlated noise; importing noise from real backends (`NoiseModel.from_backend`) clearly labelled as hardware-derived; non-uniform strategy weights; Bayesian/sequential decision rules.

## Files
`app.py` dashboard · `qadapt/` (`noise_models`, `strategies`, `circuits`, `benchmarks`, `adaptive`, `stats`, `worker`) · `tests/` · `run_benchmark.py` · `results/` · `docs/`
