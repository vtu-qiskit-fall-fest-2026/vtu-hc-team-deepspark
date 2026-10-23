"""Correctness tests for the QEC circuits.

Run:  python -m pytest -q tests      (or)      python tests/test_codes.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qiskit_aer import AerSimulator

from q_adapt.codes import STATES, N_PHYS, build_full, compile_circuit

SIM = AerSimulator(method="statevector")


def _success_prob(strategy, state, inject=None, idle=2):
    qc = compile_circuit(build_full(strategy, state, idle, inject))
    counts = SIM.run(qc, shots=256, seed_simulator=7).result().get_counts()
    return counts.get(str(STATES[state]), 0) / 256


def test_noiseless_all_strategies_perfect():
    for s in N_PHYS:
        for st in STATES:
            assert _success_prob(s, st) == 1.0, (s, st)


def test_bitflip3_corrects_every_single_X():
    for q in range(3):
        for st in STATES:
            assert _success_prob("bitflip3", st, [(q, "X")]) == 1.0, (q, st)


def test_phaseflip3_corrects_every_single_Z():
    for q in range(3):
        for st in STATES:
            assert _success_prob("phaseflip3", st, [(q, "Z")]) == 1.0, (q, st)


def test_shor9_corrects_any_single_pauli():
    for q in range(9):
        for pauli in "XYZ":
            for st in STATES:
                assert _success_prob("shor9", st, [(q, pauli)]) == 1.0, (q, pauli, st)


def test_codes_do_not_correct_what_they_should_not():
    # bit-flip code cannot fix a Z error -> some logical state must fail
    assert any(_success_prob("bitflip3", st, [(1, "Z")]) < 1.0 for st in STATES)
    # phase-flip code cannot fix an X error
    assert any(_success_prob("phaseflip3", st, [(1, "X")]) < 1.0 for st in STATES)
    # bare qubit has no protection at all
    assert any(_success_prob("none", st, [(0, "X")]) < 1.0 for st in STATES)


def test_two_errors_exceed_distance():
    # two X errors on the 3-qubit code are beyond its distance -> must fail for |1>
    assert _success_prob("bitflip3", "1", [(0, "X"), (1, "X")]) == 0.0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
