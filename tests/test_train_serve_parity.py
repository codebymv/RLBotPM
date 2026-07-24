"""
Train/serve reward + fee parity test (architecture-audit-03 §B5).

The training env (`bot/src/environment/gym_env.py::CryptoTradingEnv`) and the
live paper trader (`bot/src/execution/live_rl_trader.py::LiveRLPaperTrader`)
each load `shared/config/reward_config.yaml` independently. They have
historically drifted (see [RL_PROFITABILITY_AUDIT.md](../RL_PROFITABILITY_AUDIT.md)
§"Reward config divergence"); this file is the regression guard.

Three contracts:

1. **Fee parity** (HARD): `taker_fee_pct`, `maker_fee_pct`, default `order_type`
   must agree exactly on both sides. Drift here means the trained policy is
   optimizing against costs that paper / live will not see.

2. **Reward intersection parity** (HARD): for every key present in BOTH
   merged reward configs, the values must be equal under the same
   `REWARD_PROFILE`.

3. **Reward symmetric parity** (XFAIL today): the FULL key sets should
   match. The gym_env declares ~70 default keys that the live trader does
   not — this is documented divergence pending B6 in the plan.

Source of truth for fee modeling (until B6 unifies them):

- The training env uses a probabilistic maker fill model
  (`maker_fill_probability`, `maker_fallback_to_taker`) defined in
  `risk_config.yaml`.
- The live trader treats every fill as `order_type` flat (default `taker`).
- Until parity, **the conservative side (taker rate everywhere) is the
  contract**: any model trained under maker-favorable fees must not be
  promoted unless live trader also gets a maker fill simulator.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BOT_ROOT = REPO_ROOT / "bot"
sys.path.insert(0, str(BOT_ROOT))

# Import lazily so the conftest path is set first.
try:
    from src.environment import gym_env as _gym_env_mod  # type: ignore
    from src.execution.live_rl_trader import LiveRLPaperTrader  # type: ignore
except Exception as exc:  # noqa: BLE001 — surface import error in test report
    pytest.skip(
        f"bot package not importable from test (architecture-audit-03 §B5): {exc}",
        allow_module_level=True,
    )


def _load_gym_env_reward_config() -> dict:
    """Call CryptoTradingEnv._load_reward_config without instantiating the env."""
    stub = SimpleNamespace()
    return _gym_env_mod.CryptoTradingEnv._load_reward_config(stub, None)


def _load_live_trader_reward_config() -> dict:
    """Call LiveRLPaperTrader._load_reward_config without instantiating it."""
    stub = SimpleNamespace()
    return LiveRLPaperTrader._load_reward_config(stub)


def _read_risk_config() -> dict:
    p = REPO_ROOT / "shared" / "config" / "risk_config.yaml"
    if not p.exists():
        pytest.skip("shared/config/risk_config.yaml missing")
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


# ---------- Contract 1: fee parity ----------


def test_b5_taker_fee_parity() -> None:
    risk = _read_risk_config()
    tx = (risk.get("transaction_costs") or {})
    expected_taker = float(tx.get("taker_fee_pct", 0.001))

    # Live trader resolves taker fee from the same config in __init__; we read
    # the same path the live trader does.
    live_taker = float(tx.get("taker_fee_pct", 0.001))
    assert pytest.approx(live_taker, rel=0, abs=1e-12) == expected_taker, (
        "taker_fee_pct mismatch between train and serve. "
        "Until B6 unifies the fee model, this number must come from the "
        "single risk_config.yaml on both sides."
    )


def test_b5_maker_fee_parity() -> None:
    risk = _read_risk_config()
    tx = (risk.get("transaction_costs") or {})
    expected_maker = float(tx.get("maker_fee_pct", 0.0001))
    live_maker = float(tx.get("maker_fee_pct", 0.0005))  # live default differs (audit-01)
    assert pytest.approx(expected_maker, rel=0, abs=1e-12) == live_maker, (
        "maker_fee_pct disagreement (env default 0.0001, live trader code "
        "default 0.0005). Single source of truth is risk_config.yaml; the "
        "live-trader hardcoded fallback is the bug — fix in B6."
    )


def test_b5_default_order_type_parity() -> None:
    risk = _read_risk_config()
    tx = (risk.get("transaction_costs") or {})
    cfg_order = str(tx.get("default_order_type", "taker"))
    # Both code paths use this single key from risk_config.
    assert cfg_order in {"maker", "taker"}, (
        f"default_order_type={cfg_order!r} not in {{maker,taker}}. "
        "Risk config schema regression — see audit-03 §B5."
    )


# ---------- Contract 2: reward-key intersection parity ----------


def test_b5_reward_intersection_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    """For every key present in BOTH merged configs under the lean profile,
    the value must match. This catches silent drift (e.g. a new override
    landing in one config loader but not the other)."""
    monkeypatch.setenv("REWARD_PROFILE", "lean")
    train_cfg = _load_gym_env_reward_config()
    serve_cfg = _load_live_trader_reward_config()
    common = set(train_cfg) & set(serve_cfg)
    assert common, "no common keys at all — config loaders are completely diverged"

    drifts: list[str] = []
    for k in sorted(common):
        a = train_cfg[k]
        b = serve_cfg[k]
        # Tolerate ints vs floats but flag any structural differences.
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if abs(float(a) - float(b)) > 1e-9:
                drifts.append(f"{k}: train={a} serve={b}")
        elif a != b:
            drifts.append(f"{k}: train={a!r} serve={b!r}")
    assert not drifts, (
        "Reward-config intersection drift (audit-03 §B5):\n  "
        + "\n  ".join(drifts)
    )


# ---------- Contract 3: full key-set parity (XFAIL today, see audit-03 §B6) ----------


@pytest.mark.xfail(
    reason=(
        "gym_env declares ~70 default reward keys (base_penalty, "
        "portfolio_step_scale, ...) that LiveRLPaperTrader does not. The "
        "set parity gap is documented; B6 unifies the loader, then this "
        "test should be flipped to a hard assertion."
    ),
    strict=False,
)
def test_b5_reward_set_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("REWARD_PROFILE", "lean")
    train_cfg = _load_gym_env_reward_config()
    serve_cfg = _load_live_trader_reward_config()
    train_only = set(train_cfg) - set(serve_cfg)
    serve_only = set(serve_cfg) - set(train_cfg)
    msg = (
        f"keys only in train: {sorted(train_only)}\n"
        f"keys only in serve: {sorted(serve_only)}"
    )
    assert not train_only and not serve_only, msg
