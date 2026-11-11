"""Headless tests: every sidebar section renders without exceptions and the predictor works.

    python -m pytest tests -q
"""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "cricupbeat.py")
SECTIONS = ["Win Predictor", "Batsman Analysis", "Bowler Analysis", "Fielder Analysis",
            "Match Analysis", "Score Analysis", "Powerplay Analysis", "Average Score Analysis",
            "Fantasy team best players"]


def fresh():
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_app_loads():
    fresh()


@pytest.mark.parametrize("section", SECTIONS)
def test_sidebar_section(section):
    at = fresh()
    btn = next(b for b in at.sidebar.button if section in b.label)
    btn.click().run()
    assert not at.exception, f"{section}: {at.exception}"


def test_win_predictor_outputs_probability():
    at = fresh()
    at.selectbox[0].select("MI")
    at.run()
    at.selectbox[1].select("CSK")
    at.run()
    next(b for b in at.button if b.label == "Predict").click().run()
    assert not at.exception
    msg = at.success[0].value
    import re
    m = re.search(r"MI (\d+)% vs CSK (\d+)%", msg)
    assert m, msg
    assert int(m.group(1)) + int(m.group(2)) in (99, 100, 101)
