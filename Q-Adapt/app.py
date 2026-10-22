"""Q-Adapt — Noise-Aware Adaptive Quantum Error Correction  (Streamlit dashboard).

Run:  streamlit run app.py
Every number shown is computed by real Qiskit-Aer simulations (density-matrix method with
a user-configured *simulated* noise model). Nothing is estimated from real hardware.
"""
from __future__ import annotations
import json
import platform
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import qiskit
import qiskit_aer
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qadapt import __version__
from qadapt.noise_models import NoiseConfig, NOISE_PARAMS, describe_noise
from qadapt.strategies import (STRATEGIES, TASKS, TASK_QUANTUM, TASK_CLASSICAL, BASELINE_KEY,
                               strategies_for_task)
from qadapt.circuits import build_logical_circuit
from qadapt.benchmarks import (run_point, results_to_df, run_sweep_raw, sweep_tables, run_map_raw,
                               map_choices, make_simulator, count_failures)
from qadapt.circuits import transpiled_circuit, logical_outcome
from qadapt.adaptive import recommend, pareto_frontier
from qadapt.stats import wilson_interval
from qadapt.worker import call_in_worker

st.set_page_config(page_title="Q-Adapt · Adaptive Quantum Error Correction", page_icon="⚛️",
                   layout="wide", initial_sidebar_state="expanded")

# ------------------------------------------------------------------ look & feel
PALETTE = {"none": "#94a3b8", "bit3_q": "#38bdf8", "phase3_q": "#a78bfa", "bit5_q": "#34d399",
           "phase5_q": "#fbbf24", "bit3_v": "#f472b6", "bit5_v": "#fb7185"}
ADAPT_COLOR = "#fde047"
BG, PANEL, GRID, TXT = "#0b1020", "#131a30", "#27324f", "#e6ecff"

st.markdown(f"""
<style>
.block-container {{padding-top: 3.4rem; max-width: 1400px;}}
.hero {{background: radial-gradient(1200px 300px at 10% -20%, #1d4ed855, transparent),
        linear-gradient(135deg, #111a38 0%, #0b1020 70%); border:1px solid {GRID};
        border-radius:18px; padding:26px 30px; margin-bottom:14px;}}
.hero h1 {{margin:0; font-size:2.3rem; letter-spacing:.5px;}}
.hero .tag {{color:#7dd3fc; font-weight:600; letter-spacing:2px; font-size:.78rem;}}
.hero p {{color:#b8c4e6; margin:.4rem 0 0 0; font-size:1.02rem;}}
.card {{background:{PANEL}; border:1px solid {GRID}; border-radius:16px; padding:18px 22px;}}
.rec {{background: linear-gradient(135deg,#0f2a3d,#131a30); border:1px solid #38bdf8;
       border-radius:16px; padding:20px 24px;}}
.rec h3 {{margin:0 0 6px 0; color:#7dd3fc;}}
.badge {{display:inline-block; padding:3px 10px; border-radius:999px; font-size:.75rem; font-weight:700;}}
.ok {{background:#064e3b; color:#6ee7b7;}} .warn {{background:#78350f; color:#fcd34d;}}
.sim {{background:#312e81; color:#c7d2fe;}}
.small {{color:#93a4cf; font-size:.85rem;}}
div[data-testid="stMetric"] {{background:{PANEL}; border:1px solid {GRID}; border-radius:12px; padding:10px 14px;}}
</style>""", unsafe_allow_html=True)


def style_fig(fig: go.Figure, title: str = "", height: int = 430) -> go.Figure:
    fig.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=PANEL, height=height,
                      title=dict(text=title, x=0.01, font=dict(size=16)),
                      font=dict(color=TXT), margin=dict(l=50, r=20, t=50, b=50),
                      legend=dict(bgcolor="rgba(0,0,0,0)"))
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig



def simulate_view(skey: str, basis: str, cfg_tuple: tuple, sim_shots: int, seed: int):
    """Runs on the Qiskit worker thread: draw the circuit (PNG bytes) + ideal/noisy counts."""
    import io
    strat_ = STRATEGIES[skey]
    qc = build_logical_circuit(strat_, basis)
    tc = transpiled_circuit(skey, basis)
    ideal_ = make_simulator(NoiseConfig()).run(tc, shots=sim_shots, seed_simulator=seed).result().get_counts()
    noisy_ = make_simulator(NoiseConfig(*cfg_tuple)).run(tc, shots=sim_shots, seed_simulator=seed + 1).result().get_counts()
    buf = io.BytesIO()
    try:
        fig_c = qc.draw("mpl", style="iqp-dark", fold=30)
        fig_c.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor=fig_c.get_facecolor())
        plt.close(fig_c)
    except Exception:  # drawing is cosmetic; never break the experiment
        fig_c = plt.figure(figsize=(8, 2)); fig_c.patch.set_facecolor(PANEL)
        fig_c.text(0.01, 0.5, str(qc.draw("text", fold=100))[:1500], family="monospace", color=TXT, fontsize=5, va="center")
        fig_c.savefig(buf, format="png", dpi=130, facecolor=PANEL); plt.close(fig_c)
    return buf.getvalue(), dict(ideal_), dict(noisy_)


# ------------------------------------------------------------------ state & controls
DEFAULTS = dict(p_x=0.08, p_z=0.0, p_gate=0.0, p_meas=0.0, shots=2000, trials=4, seed=42,
                task=TASK_QUANTUM, budget=5, lam=0.005, alpha=0.05)
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

PRESETS = {
    "🔵 Bit-flip dominated": dict(p_x=0.10, p_z=0.0, p_gate=0.0, p_meas=0.0),
    "🟣 Phase-flip dominated": dict(p_x=0.0, p_z=0.10, p_gate=0.0, p_meas=0.0),
    "🟠 Mixed X + Z": dict(p_x=0.06, p_z=0.06, p_gate=0.0, p_meas=0.0),
    "🔴 Noisy gates": dict(p_x=0.05, p_z=0.0, p_gate=0.04, p_meas=0.0),
    "⚪ Very low noise": dict(p_x=0.005, p_z=0.0, p_gate=0.0, p_meas=0.0),
}


def apply_preset(name: str):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


with st.sidebar:
    st.markdown("## ⚛️ Q-Adapt controls")
    st.markdown('<span class="badge sim">SIMULATED NOISE — not hardware data</span>', unsafe_allow_html=True)
    st.markdown("**Quick noise presets**")
    for name in PRESETS:
        st.button(name, key=f"preset_{name}", on_click=apply_preset, args=(name,), width="stretch")
    st.markdown("---")
    st.markdown("### 🧪 Noise Lab")
    st.slider("Bit-flip probability  p_x", 0.0, 0.30, step=0.005, key="p_x", format="%.3f",
              help="Chance that an X (bit-flip) error hits each qubit while it is stored.")
    st.slider("Phase-flip probability  p_z", 0.0, 0.30, step=0.005, key="p_z", format="%.3f",
              help="Chance that a Z (phase-flip) error hits each qubit while it is stored.")
    st.slider("Gate error  p_gate (CX)", 0.0, 0.10, step=0.0025, key="p_gate", format="%.4f",
              help="Depolarizing error on every two-qubit CX gate. Makes encoding/decoding circuits imperfect.")
    st.slider("Measurement error  p_meas", 0.0, 0.30, step=0.005, key="p_meas", format="%.3f",
              help="Probability that a measured bit is read incorrectly.")
    st.select_slider("Shots per circuit per trial", options=[500, 1000, 2000, 5000, 10000], key="shots")
    st.slider("Repeated trials", 2, 10, key="trials",
              help="Even trials = calibration data (engine decides); odd trials = held-out validation.")
    st.number_input("Random seed", 0, 2**31 - 1, key="seed", step=1)
    st.markdown("---")
    st.markdown("### 🎯 Task & resources")
    st.radio("What must be preserved?", list(TASKS), key="task", format_func=lambda k: TASKS[k])
    st.slider("Physical-qubit budget", 1, 5, key="budget",
              help="Strategies needing more physical qubits than this are excluded.")
    st.slider("Resource weight λ (failure traded per extra qubit)", 0.0, 0.05, step=0.001,
              key="lam", format="%.3f")
    st.select_slider("Significance level α", options=[0.10, 0.05, 0.01, 0.001], key="alpha")
    st.caption("Budget, λ and α re-run the decision instantly on stored benchmark data. "
               "Noise / shots / trials need a new run.")

cfg = NoiseConfig(st.session_state.p_x, st.session_state.p_z, st.session_state.p_gate, st.session_state.p_meas)
task, shots, trials, seed = st.session_state.task, st.session_state.shots, st.session_state.trials, st.session_state.seed
budget, lam, alpha = st.session_state.budget, st.session_state.lam, st.session_state.alpha
SIG = (cfg.as_tuple(), task, shots, trials, seed)

# ------------------------------------------------------------------ hero
st.markdown(f"""
<div class="hero">
  <div class="tag">QISKIT FALL FEST 2026 · TRACK 3 · HARDWARE & ERROR CORRECTION</div>
  <h1>⚛️ Q-Adapt <span style="font-weight:300;color:#7dd3fc">— Noise-Aware Adaptive Quantum Error Correction</span></h1>
  <p>Quantum bits are fragile. Q-Adapt <b>measures</b> how well different error-correction strategies
  survive a given noise condition (real Qiskit-Aer simulations), then <b>recommends</b> the one with the best
  reliability for the qubits you can afford — and tells you honestly when correction does <i>not</i> help.</p>
</div>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ helpers
def run_current_point():
    with st.spinner("Running Qiskit-Aer simulations…"):
        t0 = time.time()
        res = call_in_worker(run_point, cfg, task, shots, trials, seed)
        st.session_state["point"] = dict(sig=SIG, results=res, secs=time.time() - t0)


def stale(entry) -> bool:
    return entry is not None and entry["sig"] != SIG


if "point" not in st.session_state:
    run_current_point()
point = st.session_state["point"]
results = point["results"]
rec = recommend(results, budget, lam, alpha)
by = {r.key: r for r in results}
df_point = results_to_df(results)

tabs = st.tabs(["🏠 Overview & Recommendation", "⚛️ Circuit Simulator", "🛡️ Error-Correction Lab",
                "📊 Noise Sweep", "🗺️ Regime Map", "🔬 Reproducibility", "📚 Learn"])

# =================================================================== TAB 1: recommendation
with tabs[0]:
    c1, c2 = st.columns([1, 3])
    with c1:
        if st.button("▶ Run experiment", type="primary", width="stretch"):
            run_current_point()
            st.rerun()
    with c2:
        if stale(point):
            st.warning("Noise / shots / trials / task changed since the last run — press **Run experiment** "
                       "to refresh the numbers below. (Budget, λ, α already update live.)")
        else:
            st.caption(f"Current noise: **{describe_noise(cfg)}** · results from {sum(r.shots for r in results):,} "
                       f"simulated shots · ran in {point['secs']:.1f}s")

    badge = ('<span class="badge ok">IMPROVEMENT CONFIRMED ON HELD-OUT DATA</span>' if rec.confident and rec.chosen != BASELINE_KEY
             else '<span class="badge ok">BASELINE SUPPORTED BY EVIDENCE</span>' if rec.confident
             else '<span class="badge warn">LOW CONFIDENCE — see notes</span>')
    bullets = "".join(f"<li>{line}</li>" for line in rec.explanation)
    st.markdown(f'<div class="rec"><h3>🤖 {rec.headline}</h3>{badge}<ul style="margin-top:10px">{bullets}</ul></div>',
                unsafe_allow_html=True)
    for w in rec.warnings:
        st.warning(w)

    ch = by[rec.chosen]
    st.markdown("#### Chosen strategy at a glance (calibration + held-out data pooled)")
    m = st.columns(6)
    lo, hi = ch.ci
    m[0].metric("Logical failure rate", f"{ch.rate:.2%}", f"{(ch.rate - by[BASELINE_KEY].rate):+.2%} vs baseline",
                delta_color="inverse")
    m[1].metric("Success probability", f"{ch.success:.2%}")
    m[2].metric("95% CI (failure)", f"{lo:.2%}–{hi:.2%}")
    m[3].metric("Physical qubits", ch.n_qubits, f"budget {budget}", delta_color="off")
    m[4].metric("CX gates", ch.cx_count)
    m[5].metric("Circuit depth", ch.depth)

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for r in results:
            lo_, hi_ = r.ci
            fig.add_trace(go.Bar(x=[f"{r.key}<br>{r.n_qubits}q"], y=[r.rate], name=r.label,
                                 marker_color=PALETTE[r.key], opacity=1.0 if r.n_qubits <= budget else 0.3,
                                 marker_line=dict(color=ADAPT_COLOR, width=3) if r.key == rec.chosen else None,
                                 error_y=dict(type="data", symmetric=False, array=[hi_ - r.rate],
                                              arrayminus=[r.rate - lo_], color="#e6ecff"),
                                 hovertemplate=f"{r.label}<br>failure %{{y:.3%}}<extra></extra>"))
        fig.add_hline(y=by[BASELINE_KEY].rate, line_dash="dash", line_color="#94a3b8",
                      annotation_text="baseline", annotation_position="top left")
        fig.update_layout(showlegend=False, yaxis_title="logical failure rate", yaxis_tickformat=".1%")
        st.plotly_chart(style_fig(fig, "Failure rate by strategy (gold outline = recommended; faded = over budget)"),
                        width="stretch")
    with right:
        front = pareto_frontier(results)
        fig = go.Figure()
        xs = [r.n_qubits for r in results]
        fig.add_trace(go.Scatter(x=[by[k].n_qubits for k in front], y=[by[k].rate for k in front],
                                 mode="lines", line=dict(color="#475569", dash="dot"), name="Pareto frontier"))
        for r in results:
            fig.add_trace(go.Scatter(x=[r.n_qubits], y=[r.rate], mode="markers+text", text=[r.key],
                                     textposition="top center", showlegend=False,
                                     marker=dict(size=14 + 22 * np.sqrt(r.cx_count / max(1, max(x.cx_count for x in results))),
                                                 color=PALETTE[r.key], opacity=0.85,
                                                 line=dict(color=ADAPT_COLOR if r.key == rec.chosen else "#0b1020", width=3)),
                                     hovertemplate=f"{r.label}<br>{r.n_qubits} qubits · {r.cx_count} CX · depth {r.depth}"
                                                   f"<br>failure %{{y:.3%}}<extra></extra>"))
        fig.add_vline(x=budget + 0.5, line_dash="dash", line_color="#f87171",
                      annotation_text=f"budget = {budget}", annotation_position="top right")
        fig.update_layout(xaxis=dict(title="physical qubits", dtick=1), yaxis=dict(title="logical failure rate", tickformat=".1%"))
        st.plotly_chart(style_fig(fig, "Reliability vs resources (bubble size ∝ CX gates)"), width="stretch")

    st.markdown("#### Decision table (what the engine saw — calibration data only)")
    dt = pd.DataFrame(rec.table)
    dt["decision"] = np.where(dt["chosen"], "✅ RECOMMENDED", dt["status"])
    show = dt[["label", "qubits", "cx", "depth", "rate", "ci_lo", "ci_hi", "p", "J", "decision"]].rename(columns={
        "label": "strategy", "cx": "CX", "rate": "failure", "ci_lo": "CI low", "ci_hi": "CI high",
        "p": "p (better than baseline)", "J": "objective J"})
    st.dataframe(show.style.format({"failure": "{:.3%}", "CI low": "{:.3%}", "CI high": "{:.3%}",
                                    "p (better than baseline)": "{:.2g}", "objective J": "{:.4f}"}, na_rep="—"),
                 width="stretch", hide_index=True)
    st.caption("Objective  J = failure rate + λ · (physical qubits − 1).  A strategy is only a candidate if it beats the "
               "baseline with a one-sided two-proportion z-test (p < α) and fits the qubit budget. Statistically tied "
               "options → the cheapest wins.")

# =================================================================== TAB 2: simulator
with tabs[1]:
    st.markdown("### ⚛️ Quantum Circuit Simulator — ideal vs noisy")
    st.caption("Pick one strategy and one logical test state. The circuit is simulated twice with Qiskit Aer: "
               "once ideal, once with the noise from the sidebar. Expected logical result: **0**.")
    avail = strategies_for_task(task)
    sc1, sc2, sc3 = st.columns([2, 1, 1])
    skey = sc1.selectbox("Strategy", [s.key for s in avail], format_func=lambda k: STRATEGIES[k].label, key="sim_strat")
    strat = STRATEGIES[skey]
    basis = sc2.selectbox("Logical test state", list(strat.bases),
                          format_func=lambda b: "|0⟩ (read in Z)" if b == "Z" else "|+⟩ (read in X)", key="sim_basis")
    sim_shots = sc3.select_slider("Shots", [500, 1000, 2000, 5000, 10000], value=2000, key="sim_shots")
    st.info(strat.blurb)
    with st.container():
        png, ideal, noisy = call_in_worker(simulate_view, skey, basis, cfg.as_tuple(), sim_shots, seed)
        cc1, cc2 = st.columns([3, 2])
        with cc1:
            st.markdown("**Circuit** (prepare → encode → noisy storage `id` → decode → measure)")
            st.image(png, width="stretch")
        with cc2:
            keys = sorted(set(ideal) | set(noisy))
            fig = go.Figure()
            fig.add_trace(go.Bar(x=keys, y=[ideal.get(k, 0) / sim_shots for k in keys], name="ideal", marker_color="#64748b"))
            fig.add_trace(go.Bar(x=keys, y=[noisy.get(k, 0) / sim_shots for k in keys], name="noisy", marker_color="#38bdf8"))
            fig.update_layout(barmode="group", yaxis=dict(title="probability", tickformat=".0%"),
                              xaxis=dict(title="measured bitstring" if strat.decoder == "vote" else "decoded logical bit", type="category"))
            st.plotly_chart(style_fig(fig, "Measurement outcomes", 340), width="stretch")
        f_ideal, f_noisy = count_failures(strat, ideal), count_failures(strat, noisy)
        tvd = 0.5 * sum(abs(ideal.get(k, 0) - noisy.get(k, 0)) for k in keys) / sim_shots
        lo_, hi_ = wilson_interval(f_noisy, sim_shots)
        tab = pd.DataFrame({"": ["Expected logical bit", "Observed (ideal sim)", "Observed (noisy sim)"],
                            "P(logical = 0)": ["100%", f"{1 - f_ideal / sim_shots:.2%}", f"{1 - f_noisy / sim_shots:.2%}"],
                            "P(logical = 1) = failure": ["0%", f"{f_ideal / sim_shots:.2%}",
                                                         f"{f_noisy / sim_shots:.2%}  (95% CI {lo_:.2%}–{hi_:.2%})"]})
        st.dataframe(tab, hide_index=True, width="stretch")
        st.caption(f"Total-variation distance between ideal and noisy raw outcome distributions: **{tvd:.3f}**. "
                   + ("Majority vote over all measured qubits gives the logical bit (classical-bit strategy)."
                      if strat.decoder == "vote" else "The decoded logical bit is read from qubit 0."))

# =================================================================== TAB 3: EC lab
with tabs[2]:
    st.markdown("### 🛡️ Error-Correction Lab")
    st.caption(f"Task: **{TASKS[task]}** · noise: {describe_noise(cfg)}")
    show = df_point[["label", "physical_qubits", "cx_count", "gate_count", "depth", "logical_failure_rate", "ci95_low",
                     "ci95_high", "success_probability", "Z_basis_failure", "X_basis_failure",
                     "abs_improvement_vs_baseline", "rel_reduction_vs_baseline", "p_value_better_than_baseline",
                     "trial_std"]].copy()
    st.dataframe(show.style.format({c: "{:.3%}" for c in ["logical_failure_rate", "ci95_low", "ci95_high", "success_probability",
                                                          "Z_basis_failure", "X_basis_failure", "abs_improvement_vs_baseline",
                                                          "rel_reduction_vs_baseline", "trial_std"]}
                                   | {"p_value_better_than_baseline": "{:.2g}"}, na_rep="—"),
                 width="stretch", hide_index=True)
    st.download_button("⬇ Download this table (CSV)", df_point.to_csv(index=False), "qadapt_point_benchmark.csv", "text/csv")
    a, b = st.columns(2)
    with a:
        fig = go.Figure()
        for lbl, col, ycol in [("Z-readout (|0⟩ test)", "#38bdf8", "Z_basis_failure"), ("X-readout (|+⟩ test)", "#a78bfa", "X_basis_failure")]:
            fig.add_trace(go.Bar(x=df_point["strategy"], y=df_point[ycol], name=lbl, marker_color=col))
        fig.update_layout(barmode="group", yaxis=dict(title="failure rate", tickformat=".1%"))
        st.plotly_chart(style_fig(fig, "Where does each strategy fail? (per logical test state)", 380), width="stretch")
    with b:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=df_point["strategy"], y=df_point["cx_count"], name="CX gates", marker_color="#f97316"))
        fig.add_trace(go.Bar(x=df_point["strategy"], y=df_point["depth"], name="depth", marker_color="#14b8a6"))
        fig.update_layout(barmode="group", yaxis_title="count", yaxis_type="log")
        st.plotly_chart(style_fig(fig, "Circuit cost (log scale): CX gates & depth", 380), width="stretch")
    st.markdown("""
**How to read this.** A *logical* failure means the information we wanted to keep came out wrong after correction.
* **Quantum memory** tests both |0⟩ and |+⟩: a code must protect a real qubit, not just a bit. A bit-flip code protects |0⟩/|1⟩ but
  **amplifies phase errors**; a phase-flip code does the reverse. Neither protects against both X *and* Z errors (that needs e.g. Shor/Steane/surface codes — future work).
* **Classical-bit memory** only checks |0⟩; the *majority-vote* strategies are allowed here. They are a **simplified logical-bit
  correction**, not full quantum error correction, because they measure every qubit and therefore cannot keep a superposition.
* Gate errors make the encode/decode circuits themselves noisy — that is why bigger codes sometimes lose.""")

# =================================================================== TAB 4: sweep
with tabs[3]:
    st.markdown("### 📊 Benchmark across noise levels")
    sc = st.columns([1.3, 1, 1, 1, 1])
    sweep_param = sc[0].selectbox("Parameter to sweep", list(NOISE_PARAMS), format_func=lambda k: f"{k} — {NOISE_PARAMS[k]}", key="sw_param")
    default_max = {"p_x": 0.25, "p_z": 0.25, "p_gate": 0.08, "p_meas": 0.25}[sweep_param]
    sw_lo = sc[1].number_input("from", 0.0, 0.5, 0.0, 0.01, key=f"sw_lo_{sweep_param}", format="%.3f")
    sw_hi = sc[2].number_input("to", 0.001, 0.5, default_max, 0.01, key=f"sw_hi_{sweep_param}", format="%.3f")
    sw_n = sc[3].slider("points", 4, 16, 9, key="sw_n")
    sc[4].markdown("<br>", unsafe_allow_html=True)
    if sc[4].button("▶ Run sweep", type="primary", key="go_sweep"):
        vals = np.linspace(sw_lo, max(sw_hi, sw_lo + 1e-3), sw_n)
        bar = st.progress(0.0, text="Sweeping noise levels with Qiskit Aer…")
        raw = run_sweep_raw(sweep_param, vals, cfg, task, shots, trials, seed, progress=lambda f: bar.progress(f), runner=call_in_worker)
        bar.empty()
        st.session_state["sweep"] = dict(sig=SIG, param=sweep_param, raw=raw)
    sw = st.session_state.get("sweep")
    if sw is None:
        st.info("Press **Run sweep** to benchmark every strategy across a range of noise levels. "
                "Other noise parameters stay at the sidebar values.")
    else:
        if stale(sw):
            st.warning("Sidebar settings changed since this sweep was run (the sweep holds its own settings).")
        param = sw["param"]
        df_all, df_ad = sweep_tables(sw["raw"], param, budget, lam, alpha)
        logy = st.checkbox("log y-axis (zero-failure points are hidden)", value=False)
        fig = go.Figure()
        for k in df_all["strategy"].unique():
            d = df_all[df_all["strategy"] == k]
            if STRATEGIES[k].n > budget:
                continue
            fig.add_trace(go.Scatter(x=d[param], y=d["logical_failure_rate"], mode="lines+markers", name=STRATEGIES[k].label,
                                     line=dict(color=PALETTE[k], width=2.5 if k != BASELINE_KEY else 2, dash="dash" if k == BASELINE_KEY else "solid"),
                                     error_y=dict(type="data", symmetric=False, array=d["ci95_high"] - d["logical_failure_rate"],
                                                  arrayminus=d["logical_failure_rate"] - d["ci95_low"], thickness=1, width=2)))
        fig.add_trace(go.Scatter(x=df_ad[param], y=df_ad["heldout_failure_rate"], mode="lines+markers", name="⭐ Adaptive (held-out data)",
                                 line=dict(color=ADAPT_COLOR, width=4),
                                 marker=dict(size=11, symbol="star", color=ADAPT_COLOR),
                                 customdata=df_ad["adaptive_choice"], hovertemplate="adaptive → %{customdata}<br>failure %{y:.3%}<extra></extra>"))
        fig.update_layout(xaxis_title=f"{param}  ({NOISE_PARAMS[param]})", yaxis=dict(title="logical failure rate (95% CI bars)", tickformat=".1%",
                                                                                      type="log" if logy else "linear"))
        st.plotly_chart(style_fig(fig, f"Failure rate vs {param} — strategies within the {budget}-qubit budget", 520), width="stretch")
        st.caption("The adaptive curve uses, at every noise level, the strategy the engine picked from **calibration** trials, and plots that "
                   "strategy's failure on **separate held-out** trials — so it is not inflated by selection bias. It is *not* guaranteed to be the lowest line everywhere.")
        st.markdown("**Which strategy did the engine choose at each noise level?**")
        st.dataframe(df_ad.style.format({c: "{:.3%}" for c in ["heldout_failure_rate", "heldout_ci95_low", "heldout_ci95_high", "heldout_baseline_rate"]}),
                     width="stretch", hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("⬇ All strategies (CSV)", df_all.to_csv(index=False), "qadapt_sweep_all.csv", "text/csv")
        d2.download_button("⬇ Adaptive choices (CSV)", df_ad.to_csv(index=False), "qadapt_sweep_adaptive.csv", "text/csv")
        with st.expander("Full sweep table"):
            st.dataframe(df_all, width="stretch", hide_index=True)

# =================================================================== TAB 5: regime map
with tabs[4]:
    st.markdown("### 🗺️ Noise-regime map — which strategy wins where?")
    st.caption("Every cell is an independent benchmark (real simulations) followed by the same adaptive decision, "
               "respecting the budget / λ / α in the sidebar.")
    mc = st.columns([1.2, 1.2, 1, 1])
    xp = mc[0].selectbox("X axis", list(NOISE_PARAMS), index=0, key="map_x")
    yp = mc[1].selectbox("Y axis", [p for p in NOISE_PARAMS if p != xp], index=0 if xp != "p_z" else 1, key="map_y")
    gsz = mc[2].slider("grid size", 3, 8, 6, key="map_n")
    mc[3].markdown("<br>", unsafe_allow_html=True)
    rng = {"p_x": 0.25, "p_z": 0.25, "p_gate": 0.08, "p_meas": 0.25}
    if mc[3].button("▶ Build map", type="primary", key="go_map"):
        xs = np.linspace(0, rng[xp], gsz)
        ys = np.linspace(0, rng[yp], gsz)
        bar = st.progress(0.0, text="Benchmarking grid cells…")
        raw = run_map_raw(xp, xs, yp, ys, cfg, task, max(500, shots // 2), max(2, trials // 2 * 2), seed, progress=lambda f: bar.progress(f), runner=call_in_worker)
        bar.empty()
        st.session_state["map"] = dict(sig=SIG, raw=raw)
    mp = st.session_state.get("map")
    if mp is None:
        st.info("Press **Build map**. Other noise parameters stay at the sidebar values.")
    else:
        raw = mp["raw"]
        dfm = map_choices(raw, budget, lam, alpha)
        keys = [k for k in PALETTE if k in set(dfm["best_strategy"])]
        idx = {k: i for i, k in enumerate(keys)}
        Z = np.full((len(raw["ys"]), len(raw["xs"])), np.nan)
        T = [["" for _ in raw["xs"]] for _ in raw["ys"]]
        H = [["" for _ in raw["xs"]] for _ in raw["ys"]]
        for r in dfm.itertuples():
            Z[r.j][r.i] = idx[r.best_strategy]
            T[r.j][r.i] = r.best_strategy
            H[r.j][r.i] = (f"{r.label}<br>held-out failure {r.heldout_failure_rate:.2%} vs baseline "
                           f"{r.heldout_baseline_rate:.2%}<br>{'confident' if r.confident else 'low confidence'}")
        n = len(keys)
        cs = []
        for i, k in enumerate(keys):
            cs += [[i / n, PALETTE[k]], [(i + 1) / n, PALETTE[k]]]
        fig = go.Figure(go.Heatmap(z=Z, x=[f"{v:g}" for v in raw["xs"]], y=[f"{v:g}" for v in raw["ys"]], text=T,
                                   texttemplate="%{text}", textfont=dict(color="#0b1020", size=11), customdata=H,
                                   hovertemplate="%{customdata}<extra></extra>", colorscale=cs, zmin=-0.5, zmax=n - 0.5,
                                   showscale=False, xgap=3, ygap=3))
        fig.update_layout(xaxis_title=f"{raw['x_param']}", yaxis_title=f"{raw['y_param']}")
        st.plotly_chart(style_fig(fig, f"Recommended strategy (budget {budget} qubits)", 540), width="stretch")
        st.markdown(" ".join(f'<span class="badge" style="background:{PALETTE[k]};color:#0b1020">{k}: {STRATEGIES[k].label}</span>' for k in keys),
                    unsafe_allow_html=True)
        st.caption("Cells where `none` wins are regimes where correction was not demonstrated to help (too little noise, wrong code type, "
                   "or gate noise outweighing the protection). Note the map uses fewer shots per cell than the main benchmark, so borderline cells can flip with the seed.")
        st.download_button("⬇ Regime map (CSV)", dfm.drop(columns=["i", "j"]).to_csv(index=False), "qadapt_regime_map.csv", "text/csv")

# =================================================================== TAB 6: reproducibility
with tabs[5]:
    st.markdown("### 🔬 Experiment summary & reproducibility")
    settings = dict(noise=cfg.as_dict(), task=task, shots_per_circuit_per_trial=shots, trials=trials, seed=seed,
                    qubit_budget=budget, lambda_resource_weight=lam, alpha=alpha,
                    simulator="qiskit_aer.AerSimulator(method='density_matrix') + NoiseModel",
                    storage_noise="X then/independent Z Pauli channel on every qubit via the `id` gate",
                    gate_set="id,h,x,sx,rz,cx (optimization_level=0)")
    versions = dict(q_adapt=__version__, qiskit=qiskit.__version__, qiskit_aer=qiskit_aer.__version__,
                    python=platform.python_version())
    c1, c2 = st.columns(2)
    c1.markdown("**Settings (JSON)**"); c1.json(settings)
    c2.markdown("**Software versions**"); c2.json(versions)
    summary = f"""# Q-Adapt experiment summary
* Noise (SIMULATED): {json.dumps(cfg.as_dict())}  → {describe_noise(cfg)}
* Task: {TASKS[task]}
* Shots/circuit/trial: {shots}; trials: {trials} (even = calibration, odd = held-out); seed: {seed}
* Budget: {budget} physical qubits; λ = {lam}; α = {alpha}
* **{rec.headline}**
{chr(10).join('  - ' + l for l in rec.explanation)}

| strategy | qubits | CX | depth | failure | 95% CI |
|---|---|---|---|---|---|
""" + "\n".join(f"| {r.label} | {r.n_qubits} | {r.cx_count} | {r.depth} | {r.rate:.3%} | {r.ci[0]:.3%}–{r.ci[1]:.3%} |" for r in results)
    st.markdown("**Experiment summary**")
    st.code(summary, language="markdown")
    e1, e2, e3 = st.columns(3)
    e1.download_button("⬇ Summary (Markdown)", summary, "qadapt_summary.md", "text/markdown")
    e2.download_button("⬇ Settings (JSON)", json.dumps(dict(settings=settings, versions=versions), indent=2), "qadapt_settings.json", "application/json")
    e3.download_button("⬇ Benchmark (CSV)", df_point.to_csv(index=False), "qadapt_point_benchmark.csv", "text/csv")
    st.markdown("""#### Model assumptions & limitations (please read)
* **Simulated noise only.** All noise values are user-chosen settings. Q-Adapt does *not* read or estimate real-hardware noise.
* **Storage noise is a Pauli (X/Z) channel** applied once per qubit during an idle window; X and Z are independent. Real devices also have amplitude damping, crosstalk, leakage and correlated errors.
* **Encode/decode gates are ideal unless p_gate > 0**, and p_gate is a simple CX depolarizing error. There is **no ancilla-based syndrome extraction, no mid-circuit measurement, no fault-tolerant gadgets**.
* **Repetition codes protect only one Pauli type.** No strategy here corrects arbitrary quantum errors.
* **Majority-vote strategies are simplified classical-bit correction**, valid only for the classical-bit task.
* Gate counts come from an *unoptimised* decomposition into {id,h,x,sx,rz,cx}; the 5-qubit coherent decoder uses multi-controlled-X gates without ancillas, so it is deliberately expensive.
* Failure rate is averaged over the tested logical states (|0⟩ and |+⟩ for the quantum task); it is not a full process fidelity.
* The adaptive engine can and does recommend *no correction*; it is not guaranteed to beat every fixed strategy at every noise level.""")

# =================================================================== TAB 7: learn
with tabs[6]:
    st.markdown("### 📚 Quantum error correction in plain English")
    g = st.columns(2)
    with g[0]:
        st.markdown("""
**Qubit** – a quantum bit. Like a classical bit it can be 0 or 1, but it can also be in a *superposition* of both.

**Quantum gate** – an operation on qubits (like a logic gate). **X** flips 0↔1. **H** (Hadamard) creates a superposition. **CX** (CNOT) flips a target qubit if a control qubit is 1.

**Noise** – unwanted disturbances. A *bit-flip* swaps 0↔1 (an X error); a *phase-flip* (Z error) changes the sign of a superposition, which you can't see in a plain 0/1 measurement.

**Error correction** – spread one *logical* qubit over several *physical* qubits so that a single error can be detected and undone by a majority vote.""")
    with g[1]:
        st.markdown("""
**Logical error rate** – how often the information you wanted to store comes out wrong *after* correction. Lower is better.

**Resource overhead** – the price of protection: extra physical qubits, extra gates (CX), longer depth. Correction only pays off when it removes more errors than its own extra gates add.

**Why "adaptive"?** The best strategy depends on the noise. Phase-flip noise → phase code; bit-flip noise → bit code; weak noise or noisy gates → maybe no code at all; few qubits available → smaller code. Q-Adapt tests, compares, and picks — using data, not guesses.

**3-qubit repetition code:** encode |ψ⟩ → copy the basis state to 3 qubits with CX gates, let noise strike, then undo the copies and use a Toffoli gate to apply a majority-vote fix. One flip is corrected; two flips fool it, giving failure ≈ 3p² − 2p³ instead of p.""")
    st.info("Try this: pick the *Phase-flip dominated* preset, press **Run experiment**, and watch the engine switch from the bit-flip code to the phase-flip code. Then set the budget to 1 and see it fall back to *no correction*.")
