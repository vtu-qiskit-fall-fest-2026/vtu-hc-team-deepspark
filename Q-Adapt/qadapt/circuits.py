"""Qiskit circuit construction for every Q-Adapt strategy.

Logical memory experiment (one 'cycle'):
    prepare logical |0> (Z test) or |+> (X test)  ->  encode  ->  STORAGE (noisy `id`
    gates on every physical qubit)  ->  decode / correct  ->  measure  ->  expect 0.

A run FAILS when the decoded logical bit is 1.
"""
from __future__ import annotations
from functools import lru_cache
from itertools import combinations
from qiskit import QuantumCircuit, transpile
from .strategies import STRATEGIES, Strategy
from .noise_models import STORAGE_GATE

BASIS_GATES = ["id", "h", "x", "sx", "rz", "cx"]  # common gate set => fair gate counting


def _encode_bit(qc: QuantumCircuit, n: int) -> None:
    for i in range(1, n):
        qc.cx(0, i)


def _majority_flip(qc: QuantumCircuit, n: int) -> None:
    """After CX(0,i) the qubits 1..n-1 hold (q0 XOR qi). Flip q0 iff a majority of
    them is 1 (i.e. q0 disagrees with the majority of the block)."""
    if n == 3:
        qc.ccx(1, 2, 0)
    elif n == 5:
        others = [1, 2, 3, 4]
        for trio in combinations(others, 3):          # exactly three 1s
            rest = [q for q in others if q not in trio][0]
            qc.x(rest)
            qc.mcx(others, 0)
            qc.x(rest)
        qc.mcx(others, 0)                             # all four 1s
    else:
        raise NotImplementedError(f"n={n}")


def build_logical_circuit(strategy: Strategy, basis: str) -> QuantumCircuit:
    """Un-transpiled, human-readable circuit. basis: 'Z' (test |0>) or 'X' (test |+>)."""
    if basis not in strategy.bases:
        raise ValueError(f"{strategy.key} does not support basis {basis}")
    n = strategy.n
    n_clbits = n if strategy.decoder == "vote" else 1
    qc = QuantumCircuit(n, n_clbits, name=f"{strategy.key}_{basis}")
    if basis == "X":
        qc.h(0)                                     # logical |+>
    if strategy.code != "none":
        _encode_bit(qc, n)
        if strategy.code == "phase":
            qc.h(range(n))
    qc.barrier()
    for q in range(n):
        qc.id(q)                                    # <- noisy storage window
    qc.barrier()
    if strategy.decoder == "quantum":
        if strategy.code == "phase":
            qc.h(range(n))
        for i in range(1, n):
            qc.cx(0, i)
        _majority_flip(qc, n)
        if basis == "X":
            qc.h(0)
        qc.measure(0, 0)
    elif strategy.decoder == "vote":
        qc.measure(range(n), range(n))
    else:  # none
        if basis == "X":
            qc.h(0)
        qc.measure(0, 0)
    return qc


@lru_cache(maxsize=None)
def transpiled_circuit(key: str, basis: str) -> QuantumCircuit:
    """Circuit rewritten into BASIS_GATES (noise-independent, so cached)."""
    qc = build_logical_circuit(STRATEGIES[key], basis)
    return transpile(qc, basis_gates=BASIS_GATES,
                     optimization_level=0, seed_transpiler=7)


def circuit_metrics(key: str) -> dict:
    """Physical-qubit overhead, gate count, CX count, depth (Z-readout circuit)."""
    tc = transpiled_circuit(key, "Z")
    ops = dict(tc.count_ops())
    cx = ops.get("cx", 0)
    gates = sum(v for k, v in ops.items() if k not in ("id", "measure", "barrier"))
    return {"qubits": STRATEGIES[key].n, "cx_count": int(cx),
            "gate_count": int(gates), "depth": int(tc.depth())}


def logical_outcome(strategy: Strategy, bitstring: str) -> int:
    """Map a measured bitstring to the decoded logical bit (0 or 1)."""
    bits = bitstring.replace(" ", "")
    if strategy.decoder == "vote":
        return 1 if bits.count("1") > len(bits) / 2 else 0
    return int(bits[-1])
