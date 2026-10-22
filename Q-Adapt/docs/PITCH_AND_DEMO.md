# Pitch (60 seconds)
Quantum computers are noisy, and error correction is not free: more qubits and more gates mean more places to fail.
**Q-Adapt tests before it trusts.** It simulates noise in Qiskit Aer, benchmarks seven strategies on real counts,
and recommends the one with the best reliability *within your qubit budget* — validated on held-out runs.
Our results show why adaptivity matters: bit-flip noise → bit-flip code (3q, then 5q as noise grows); phase-flip noise → phase code;
mixed noise or imperfect gates → **no correction at all**, because a single repetition code makes the other error worse and big decoders add their own errors.
We never fake a win: every number is computed, every claim carries a confidence interval, and we state exactly what is simplified.

# Live demo (5 minutes)
1. **Overview** (0:30): preset *Bit-flip dominated* → recommendation card, gold-outlined bar, "improvement CONFIRMED on held-out data". Point to the CI and p-value.
2. **Budget** (0:45): drag qubit budget 5→3→1. Decision updates instantly (5q excluded, then fallback to no correction). Show the reliability-vs-resources chart.
3. **Wrong code** (0:45): preset *Phase-flip dominated* → Run → engine switches to phase code; EC Lab tab shows bit code *amplifies* phase errors.
4. **Honest failure** (0:45): preset *Noisy gates* → Run → "no error correction", with the reason listed per strategy.
5. **Simulator** (0:45): Circuit Simulator tab: 3-qubit code circuit, ideal vs noisy histogram, expected vs observed.
6. **Sweep + Regime map** (1:00): Noise Sweep (gold adaptive line) → Regime Map shows where each strategy wins.
7. **Reproducibility** (0:20): seed, versions, CSV/Markdown export. Close with limitations.

# Likely judge questions
* *Is the noise real hardware noise?* No — simulated, user-set; labelled in the UI. Hardware import is future work.
* *Is this full QEC?* No. Repetition codes protect one Pauli type; majority-vote is explicitly "simplified classical-bit".
* *Why not always use the big code?* Resource weight λ, budget, and gate noise; results show it can lose.
* *Selection bias?* Calibration trials choose, held-out trials report.
