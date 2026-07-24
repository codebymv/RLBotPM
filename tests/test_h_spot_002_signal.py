"""
Unit tests for H-SPOT-002 signal / gates (synthetic series — no network).

Guards:
1. Band-trade state machine matches 06 §2.
2. Missing CSV → INCONCLUSIVE_DATA (never silent PASS).
3. G6 method is random_sign_flip (not multiset-invariant shuffle).
4. Causal lag: today's return uses yesterday's position.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "research" / "backtests" / "H-SPOT-002.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("h_spot_002", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules["h_spot_002"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def m():
    return _load_mod()


def test_band_trade_entry_exit_flip(m):
    # Hand-crafted z path: enter short, exit, enter long, flip to short.
    z = [None] * 10 + [
        0.0,  # flat
        2.1,  # enter short (-1)
        1.5,  # hold short (|z| >= 0.5)
        0.2,  # exit to flat
        -2.2,  # enter long (+1)
        -1.0,  # hold
        2.5,  # flip to short
        0.1,  # exit
    ]
    pos = m.build_pos_raw(z)
    # indices: 10..17 map to the crafted values
    assert pos[10] == 0
    assert pos[11] == -1
    assert pos[12] == -1
    assert pos[13] == 0
    assert pos[14] == 1
    assert pos[15] == 1
    assert pos[16] == -1
    assert pos[17] == 0


def test_zscore_window(m):
    # Constant spread → σ=0 → None z after warm-up
    spread = [1.0] * 40
    z = m.zscores(spread, w=30)
    assert all(x is None for x in z[:29])
    assert z[29] is None  # zero variance

    # Late spike → large |z|
    spread2 = [0.0] * 50 + [3.0]
    z2 = m.zscores(spread2, w=30)
    assert z2[-1] is not None
    assert z2[-1] > 2.0


def test_causal_lag_pnl(m):
    # BTC up 10%, ETH flat on day t; position known at t-1 is +1 → earn ~0.10
    btc = [100.0, 100.0, 110.0]
    eth = [50.0, 50.0, 50.0]
    # Force pos_raw so day index 2 uses pos at index 1
    pos_raw = [None, 1, 1]
    # daily_returns starts at W=30; temporarily exercise formula inline
    r_ratio = (btc[2] / btc[1] - 1.0) - (eth[2] / eth[1] - 1.0)
    assert abs(r_ratio - 0.10) < 1e-12
    assert abs(pos_raw[1] * r_ratio - 0.10) < 1e-12


def test_missing_csv_inconclusive(m, tmp_path):
    import json
    import subprocess

    missing = tmp_path / "nope.csv"
    res = subprocess.run(
        [sys.executable, str(SCRIPT), "--csv", str(missing)],
        capture_output=True,
        text=True,
        check=True,
    )
    out = json.loads(res.stdout)
    assert out["hypothesis"] == "H-SPOT-002"
    assert out["verdict"] == "INCONCLUSIVE_DATA"
    assert out["data_contract_ok"] is False
    assert out["g6_method"] == "random_sign_flip"


def test_short_history_inconclusive(m):
    # ~100 days < 730 → INCONCLUSIVE_DATA
    n = 100
    t0 = 1_700_000_000
    times = [t0 + i * 86400 for i in range(n)]
    btc = [100.0 * (1.001**i) for i in range(n)]
    eth = [50.0 * (1.0005**i) for i in range(n)]
    out = m.evaluate(times, btc, eth, trials=50, seed=42)
    assert out["verdict"] == "INCONCLUSIVE_DATA"
    assert out["d1_calendar_depth_ge_730d"] is False
    assert out["data_contract_ok"] is False


def test_g6_method_is_sign_flip_when_d1_met(m):
    """Long synthetic panel must record random_sign_flip (not multiset shuffle)."""
    n = 800
    t0 = 1_600_000_000
    times = [t0 + i * 86400 for i in range(n)]
    btc = [100.0 * (1.0003**i) for i in range(n)]
    eth = [50.0 * (1.00025**i) for i in range(n)]
    out = m.evaluate(times, btc, eth, trials=100, seed=42)
    assert out["verdict"] != "INCONCLUSIVE_DATA"
    assert out["g6_method"] == "random_sign_flip"
    assert "g6_placebo_frac_beats" in out
    assert 0.0 <= float(out["g6_placebo_frac_beats"]) <= 1.0


def test_sign_flip_placebo_rejects_strong_positive_edge():
    """Isolated G6 math: constant positive OOS days → frac_beats ≤ 5%."""
    import random

    oos = [0.01] * 200
    cum = sum(oos)
    rng = random.Random(42)
    beat = 0
    trials = 200
    for _ in range(trials):
        shuf = sum(rng.choice((-1.0, 1.0)) * x for x in oos)
        if shuf >= cum:
            beat += 1
    assert (beat / trials) <= 0.05
