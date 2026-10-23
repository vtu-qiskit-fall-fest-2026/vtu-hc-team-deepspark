"""Error-correction strategies for Q-Adapt.

Workload: *logical memory* -- prepare a logical state, let it sit for ``idle_steps`` noisy
storage steps, decode, read out.  This is the standard first benchmark for any QEC scheme.

Strategies
----------
none                 : bare qubit, no protection (baseline)
readout_mitigation   : bare qubit + confusion-matrix inversion of the readout (error
                       *mitigation*, not correction; zero qubit overhead)
bitflip3             : 3-qubit repetition code, corrects one X error
phaseflip3           : Hadamard-conjugated repetition code, corrects one Z error
shor9                : Shor's 9-qubit code, corrects any single-qubit error

Encoding/decoding are *coherent* (CNOTs + Toffoli majority vote), so no mid-circuit
measurement or classical feed-forward is needed.  Every encode/decode gate is noisy, so the
circuit overhead is paid for honestly: at high noise a code can do worse than no code.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from qiskit import QuantumCircuit, transpile

# Gate set the circuits are compiled to (these are the gates carrying noise, see noise.py)
BASIS_GATES = ["id", "x", "y", "z", "h", "t", "tdg", "cx", "ry"]

STRATEGIES: List[str] = ["none", "readout_mitigation", "bitflip3", "phaseflip3", "shor9"]
N_PHYS: Dict[str, int] = {"none": 1, "readout_mitigation": 1, "bitflip3": 3, "phaseflip3": 3, "shor9": 9}
DESCRIPTION: Dict[str, str] = {
    "none": "No protection",
    "readout_mitigation": "Readout-error mitigation (no extra qubits)",
    "bitflip3": "3-qubit bit-flip repetition code",
    "phaseflip3": "3-qubit phase-flip code",
    "shor9": "Shor 9-qubit code",
}

# logical test states and the measurement outcome expected for each (after basis rotation)
STATES: Dict[str, int] = {"0": 0, "1": 1, "+": 0, "-": 1}


# --------------------------------------------------------------------------------------
# encoders / decoders
# --------------------------------------------------------------------------------------
def _encode(qc: QuantumCircuit, strategy: str) -> None:
    if strategy == "bitflip3":
        qc.cx(0, 1)
        qc.cx(0, 2)
    elif strategy == "phaseflip3":
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.h([0, 1, 2])
    elif strategy == "shor9":
        qc.cx(0, 3)
        qc.cx(0, 6)
        qc.h([0, 3, 6])
        for a in (0, 3, 6):
            qc.cx(a, a + 1)
            qc.cx(a, a + 2)


def _decode(qc: QuantumCircuit, strategy: str) -> None:
    if strategy == "bitflip3":
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.ccx(1, 2, 0)
    elif strategy == "phaseflip3":
        qc.h([0, 1, 2])
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.ccx(1, 2, 0)
    elif strategy == "shor9":
        for a in (0, 3, 6):  # inner bit-flip decoding of each block
            qc.cx(a, a + 1)
            qc.cx(a, a + 2)
            qc.ccx(a + 1, a + 2, a)
        qc.h([0, 3, 6])      # outer phase-flip decoding
        qc.cx(0, 3)
        qc.cx(0, 6)
        qc.ccx(3, 6, 0)


# --------------------------------------------------------------------------------------
# circuit builders
# --------------------------------------------------------------------------------------
def build_core(strategy: str, idle_steps: int,
               inject: Optional[Iterable[Tuple[int, str]]] = None) -> QuantumCircuit:
    """encode -> ``idle_steps`` noisy idle steps -> decode.  No state prep, no measurement.

    ``inject`` = [(qubit, 'X'|'Y'|'Z'), ...] inserts *ideal* Pauli errors after the idle
    block; used only by the unit tests to check each code's correction capability.
    """
    n = N_PHYS[strategy]
    qc = QuantumCircuit(n)
    _encode(qc, strategy)
    qc.barrier()
    for _ in range(idle_steps):
        for q in range(n):
            qc.id(q)
        qc.barrier()
    if inject:
        for q, pauli in inject:
            getattr(qc, pauli.lower())(q)
        qc.barrier()
    _decode(qc, strategy)
    return qc


def build_full(strategy: str, state: str, idle_steps: int,
               inject: Optional[Iterable[Tuple[int, str]]] = None) -> QuantumCircuit:
    """Complete experiment: prepare ``state``, run the core, measure logical qubit 0."""
    if state not in STATES:
        raise ValueError(state)
    n = N_PHYS[strategy]
    qc = QuantumCircuit(n, 1)
    if state == "1":
        qc.x(0)
    elif state == "+":
        qc.ry(math.pi / 2, 0)
    elif state == "-":
        qc.ry(-math.pi / 2, 0)
    qc.barrier()
    qc.compose(build_core(strategy, idle_steps, inject), inplace=True)
    if state in ("+", "-"):
        qc.ry(-math.pi / 2, 0)  # rotate X basis -> Z basis
    qc.measure(0, 0)
    return qc


def compile_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    """Compile to the noisy basis without optimisation (so noise slots are preserved)."""
    return transpile(qc, basis_gates=BASIS_GATES, optimization_level=0)


# --------------------------------------------------------------------------------------
# resource cost
# --------------------------------------------------------------------------------------
_COST_CACHE: Dict[Tuple[str, int], Dict[str, int]] = {}


def strategy_cost(strategy: str, idle_steps: int) -> Dict[str, int]:
    """Physical-resource cost of protecting a memory of ``idle_steps`` steps."""
    key = (strategy, idle_steps)
    if key not in _COST_CACHE:
        core = compile_circuit(build_core(strategy, idle_steps))
        ops = core.count_ops()
        _COST_CACHE[key] = {
            "n_qubits": N_PHYS[strategy],
            "depth": core.depth(),
            "cx_count": int(ops.get("cx", 0)),
            "gate_count": int(sum(v for k, v in ops.items() if k != "barrier")),
        }
    return dict(_COST_CACHE[key])
