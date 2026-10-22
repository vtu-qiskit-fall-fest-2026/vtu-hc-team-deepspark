"""Headless end-to-end test of the Streamlit dashboard (Streamlit's AppTest harness)."""
from pathlib import Path
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")

TIMEOUT = 180


def fresh():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_app_loads_and_recommends():
    at = fresh()
    html = " ".join(m.value for m in at.markdown)
    assert "Recommendation" in html and "Q-Adapt" in html


def test_preset_and_budget_change_the_decision():
    at = fresh()
    # phase-flip preset -> engine must pick the phase code
    next(b for b in at.sidebar.button if "Phase-flip" in b.label).click().run()
    next(b for b in at.button if "Run experiment" in b.label).click().run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    assert "Phase-flip code" in html.split("Recommendation:")[1][:80]
    # budget 1 -> falls back to no correction (live, no re-simulation)
    at.sidebar.slider(key="budget").set_value(1).run()
    html = " ".join(m.value for m in at.markdown)
    assert "no error correction" in html


def test_sweep_and_map_run():
    at = fresh()
    at.slider(key="sw_n").set_value(4)
    next(b for b in at.button if b.key == "go_sweep").click().run()
    assert not at.exception
    next(b for b in at.button if b.key == "go_map").click().run()
    assert not at.exception
