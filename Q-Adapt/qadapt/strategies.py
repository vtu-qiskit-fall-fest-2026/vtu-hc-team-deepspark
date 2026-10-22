"""Definitions of the error-correction strategies compared by Q-Adapt.

Naming
------
bit3_q / bit5_q   : n-qubit bit-flip repetition code with a COHERENT (unitary) decoder.
                    Works for any logical qubit state (true quantum code for X errors).
phase3_q / phase5_q: the same codes in the Hadamard basis -> protect against Z errors.
bit3_v / bit5_v   : *simplified classical-bit* repetition code: encode a classical bit in
                    n qubits, measure ALL qubits, majority-vote classically. This only
                    preserves a Z-basis bit (no quantum superpositions) and is NOT full QEC.

Assumptions (also shown in the dashboard): encoding/decoding gates are ideal unless
p_gate > 0; no ancilla-based syndrome extraction or mid-circuit measurement.
"""
from __future__ import annotations
from dataclasses import dataclass

TASK_QUANTUM = "quantum"      # preserve an arbitrary qubit: test |0> and |+> (Z & X readout)
TASK_CLASSICAL = "classical"  # preserve a classical bit only: test |0> (Z readout)

TASKS = {
    TASK_QUANTUM: "Quantum memory (|0> and |+> averaged; needs true QEC)",
    TASK_CLASSICAL: "Classical-bit memory (|0> only; majority-vote allowed)",
}


@dataclass(frozen=True)
class Strategy:
    key: str
    label: str
    code: str       # 'none' | 'bit' | 'phase'
    n: int          # physical qubits
    decoder: str    # 'none' | 'quantum' | 'vote'
    tasks: tuple
    blurb: str

    @property
    def bases(self) -> tuple:
        return ("Z",) if self.decoder == "vote" else ("Z", "X")


STRATEGIES: dict[str, Strategy] = {s.key: s for s in [
    Strategy("none", "Uncorrected (1 qubit)", "none", 1, "none",
             (TASK_QUANTUM, TASK_CLASSICAL),
             "One physical qubit, no protection. The baseline for every comparison."),
    Strategy("bit3_q", "Bit-flip code, 3q (quantum decode)", "bit", 3, "quantum",
             (TASK_QUANTUM, TASK_CLASSICAL),
             "3-qubit repetition code, Toffoli majority correction. Fixes any single X error."),
    Strategy("phase3_q", "Phase-flip code, 3q (quantum decode)", "phase", 3, "quantum",
             (TASK_QUANTUM,),
             "Hadamard-basis repetition code. Fixes any single Z error."),
    Strategy("bit5_q", "Bit-flip code, 5q (quantum decode)", "bit", 5, "quantum",
             (TASK_QUANTUM, TASK_CLASSICAL),
             "5-qubit repetition code, threshold majority decode. Fixes up to 2 X errors."),
    Strategy("phase5_q", "Phase-flip code, 5q (quantum decode)", "phase", 5, "quantum",
             (TASK_QUANTUM,),
             "5-qubit Hadamard-basis repetition code. Fixes up to 2 Z errors."),
    Strategy("bit3_v", "Majority vote, 3q (classical bit only)", "bit", 3, "vote",
             (TASK_CLASSICAL,),
             "SIMPLIFIED: store a classical bit in 3 qubits, read all, vote. Not full QEC."),
    Strategy("bit5_v", "Majority vote, 5q (classical bit only)", "bit", 5, "vote",
             (TASK_CLASSICAL,),
             "SIMPLIFIED: store a classical bit in 5 qubits, read all, vote. Not full QEC."),
]}

BASELINE_KEY = "none"


def strategies_for_task(task: str) -> list[Strategy]:
    if task not in TASKS:
        raise ValueError(f"unknown task {task}")
    return [s for s in STRATEGIES.values() if task in s.tasks]
