"""Tiny demo: Q-Adapt picks a strategy for several noise conditions.

    python scripts/demo.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import q_adapt                                     # noqa: E402
from q_adapt.noise import make_profile             # noqa: E402

CASES = [("bit-flip noise, p=3e-4", "bitflip", 3e-4),
         ("phase-flip noise, p=3e-4", "phaseflip", 3e-4),
         ("amplitude damping, p=1e-4", "amp_damping", 1e-4),
         ("depolarizing, p=1e-2 (very noisy)", "depolarizing", 1e-2),
         ("readout-dominated, p=1e-2", "readout", 1e-2)]

for title, prof, p in CASES:
    out = q_adapt.run(make_profile(prof, p), idle=100, max_qubits=9)
    r, b = out["result"], out["uncorrected"]
    print(f"{title:38s} -> {out['strategy']:<19s} "
          f"LER {r['logical_error_rate']:.4f} (uncorrected {b['logical_error_rate']:.4f}), "
          f"{r['n_qubits']} qubits, depth {r['depth']}")

print("\nWith a 3-qubit budget under depolarizing noise at p=1e-4:")
out = q_adapt.run(make_profile("depolarizing", 1e-4), max_qubits=3)
print(" ->", out["strategy"], out["result"]["logical_error_rate"])
