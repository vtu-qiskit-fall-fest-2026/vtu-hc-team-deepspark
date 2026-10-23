"""Q-Adapt interactive demo.   Run:  python -m streamlit run app.py"""
import os
import sys
import threading

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from q_adapt.analyzer import NoiseProfile, characterize                       # noqa: E402
from q_adapt.codes import DESCRIPTION, STRATEGIES                             # noqa: E402
from q_adapt.metrics import evaluate                                          # noqa: E402
from q_adapt.noise import PROFILE_NAMES, NoiseSpec, make_profile              # noqa: E402
from q_adapt.selector import Constraints, LookupSelector, RuleBasedSelector, features   # noqa: E402

st.set_page_config(page_title="Q-Adapt", layout="wide")
st.title("Q-Adapt: adaptive quantum error-correction")
st.caption("Measure the noise, then pick the strategy with the lowest logical error for your budget.")

tab_demo, tab_results = st.tabs(["Live demo", "Benchmark results"])


@st.cache_resource
def load_lookup():
    path = os.path.join(ROOT, "results", "selector_training_table.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    fields = list(NoiseProfile.__dataclass_fields__)
    X = np.array([features(NoiseProfile(**{k: r[k] for k in fields})) for _, r in df.iterrows()])
    table = [{s: r[f"ler_{s}"] for s in STRATEGIES} for _, r in df.iterrows()]
    return LookupSelector(k=3).fit(X, table)


def _big_stack(fn, *args):
    """Run fn in a thread with a large stack (Qiskit/Aer can overflow Streamlit's small default)."""
    out = {}

    def work():
        try:
            out["v"] = fn(*args)
        except BaseException as e:      # re-raised in the caller
            out["e"] = e

    old = None
    for size in (64 * 1024 * 1024, 16 * 1024 * 1024, 8 * 1024 * 1024):
        try:
            old = threading.stack_size(size)
            break
        except (ValueError, OverflowError):
            continue

    try:
        t = threading.Thread(target=work)
        t.start()
        t.join()
    finally:
        if old is not None:
            try:
                threading.stack_size(old)
            except (ValueError, OverflowError):
                pass
    if "e" in out:
        raise out["e"]
    return out["v"]


@st.cache_data(show_spinner=False)
def run_all(spec_dict, idle, shots):
    return _big_stack(_run_all, spec_dict, idle, shots)


def _run_all(spec_dict, idle, shots):
    spec = NoiseSpec(**spec_dict)
    prof = characterize(spec, idle=idle)
    rows = []
    for s in STRATEGIES:
        m = evaluate(s, spec, idle=idle, shots=shots, ro_estimate=(prof.ro0, prof.ro1))
        rows.append({"strategy": s, **{k: m[k] for k in
                     ("logical_error_rate", "success_probability", "n_qubits", "depth", "cx_count")}})
    return prof.as_dict(), pd.DataFrame(rows)


with tab_demo:
    with st.sidebar:
        st.header("Noise condition")
        mode = st.radio("Source", ["Named profile", "Custom Pauli noise"])
        if mode == "Named profile":
            prof_name = st.selectbox("Profile", PROFILE_NAMES, index=0)
            logp = st.slider("log10 of error probability p per step", -4.0, -1.5, -3.5, 0.1)
            spec = make_profile(prof_name, 10 ** logp)
            st.write(f"p = {10 ** logp:.2e}")
        else:
            px = st.number_input("X (bit-flip) prob.", 0.0, 0.2, 0.0003, format="%.5f")
            py = st.number_input("Y prob.", 0.0, 0.2, 0.0, format="%.5f")
            pz = st.number_input("Z (phase-flip) prob.", 0.0, 0.2, 0.0, format="%.5f")
            gm = st.number_input("Amplitude damping (T1)", 0.0, 0.2, 0.0, format="%.5f")
            ro = st.number_input("Readout error", 0.0, 0.3, 0.0, format="%.4f")
            spec = NoiseSpec(px=px, py=py, pz=pz, gamma=gm, ro0=ro, ro1=ro, label="custom")
        st.header("Workload & budget")
        idle = st.select_slider("Logical memory length (idle steps)", [10, 30, 100, 300], value=100)
        max_q = st.select_slider("Max physical qubits", [1, 3, 9], value=9)
        shots = st.select_slider("Shots per logical state", [500, 1000, 3000], value=1000)
        policy = st.radio("Selector", ["Rule-based", "Lookup (benchmark-trained)"])
        go = st.button("Analyze & select", type="primary")

    if go:
        with st.spinner("Calibrating and simulating all strategies..."):
            prof_d, table = run_all(spec.as_dict(), idle, shots)
        prof = NoiseProfile(**prof_d)
        cons = Constraints(max_qubits=max_q)
        sel = RuleBasedSelector()
        if policy.startswith("Lookup"):
            lk = load_lookup()
            if lk is None:
                st.warning("results/selector_training_table.csv not found - using rule-based.")
            elif idle != 100:
                st.warning("Lookup table was trained for 100 idle steps - using rule-based.")
            else:
                sel = lk
        choice = sel.select(prof, idle, cons)
        base = table.set_index("strategy").loc["none", "logical_error_rate"]
        got = table.set_index("strategy").loc[choice, "logical_error_rate"]
        best_ok = table[table.n_qubits <= max_q].set_index("strategy").logical_error_rate

        c1, c2, c3 = st.columns(3)
        c1.metric("Chosen strategy", choice, DESCRIPTION[choice])
        c2.metric("Logical error rate", f"{got:.4f}", f"{got - base:+.4f} vs uncorrected", delta_color="inverse")
        c3.metric("Best possible within budget", best_ok.idxmin(), f"{best_ok.min():.4f}", delta_color="off")

        st.subheader("What the analyzer measured (from calibration circuits only)")
        st.dataframe(pd.DataFrame([prof_d]).round(5), hide_index=True)

        st.subheader("All strategies")
        show = table.copy()
        show["within_budget"] = show.n_qubits <= max_q
        show["chosen"] = show.strategy == choice
        st.dataframe(show.round(4), hide_index=True)
        st.bar_chart(show.set_index("strategy")["logical_error_rate"])
    else:
        st.info("Pick a noise condition in the sidebar and press **Analyze & select**.")

with tab_results:
    figdir = os.path.join(ROOT, "figures")
    names = [("fig1_logical_error_vs_noise.png", "Logical error vs noise strength"),
             ("fig2_best_strategy_heatmap.png", "Best strategy per noise condition"),
             ("fig3_adaptive_vs_fixed.png", "Adaptive vs fixed vs uncorrected (unseen noise)"),
             ("fig4_resource_overhead.png", "Resource overhead"),
             ("fig5_workload_length_breakeven.png", "Break-even vs workload length"),
             ("fig6_resource_budget.png", "Error vs qubit budget")]
    for fn, cap in names:
        p = os.path.join(figdir, fn)
        if os.path.exists(p):
            st.subheader(cap)
            st.image(p)
    summ = os.path.join(ROOT, "results", "adaptive_summary.csv")
    if os.path.exists(summ):
        st.subheader("Summary table")
        st.dataframe(pd.read_csv(summ).round(4), hide_index=True)
