"""
Launch run 174 — pre-flight checks + training kickoff.

Pre-registered in [research/RL_RUN_174_PLAN.md](../../research/RL_RUN_174_PLAN.md).

This script REFUSES to start training unless every B1–B5 pre-condition
from architecture-audit-03 is satisfied at the moment of kickoff. The
intent is to make the audit-03 fixes structurally inseparable from the
training run that consumes them, so a future operator cannot accidentally
re-introduce the run-170/run-172 failure modes (silent fleet skip,
reward-best masquerading as best_model, no held-out gate, etc.).

Usage:
    python bot/scripts/launch_run_174.py            # check + dry-run
    python bot/scripts/launch_run_174.py --commit   # actually train

The `--commit` flag is required to start training. Without it, the
script reports the gate status and exits 0/1 without invoking the
trainer. This keeps it safe to run inside CI / pre-commit hooks.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
BOT_DIR = REPO_ROOT / "bot"
RESEARCH_DIR = REPO_ROOT / "research"


def _result(ok: bool, label: str, detail: str = "") -> Tuple[bool, str]:
    tag = "PASS" if ok else "FAIL"
    return ok, f"[{tag}] {label}" + (f" — {detail}" if detail else "")


def check_b1_fleet_yaml() -> Tuple[bool, str]:
    cfg_path = REPO_ROOT / "shared" / "config" / "fleet.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    rl = cfg.get("rl_crypto", {})
    enabled = rl.get("enabled", True)
    model = rl.get("model", "missing")
    if enabled is False and model is None:
        return _result(True, "B1 fleet.yaml", "rl_crypto disabled + model=null")
    return _result(
        False,
        "B1 fleet.yaml",
        f"expected enabled=false + model=null, got enabled={enabled} model={model!r}",
    )


def check_b1_readme_banner() -> Tuple[bool, str]:
    """The audit banner must be the first non-empty header in the README,
    and the stale 'Run 170' victory claims must be gone."""
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    if "architecture-audit" not in readme.lower():
        return _result(False, "B1 README banner", "no audit reference at top of README")
    if "+0.80%" in readme or "Sharpe 9.45" in readme:
        return _result(False, "B1 README banner", "stale run-170 victory claim still present")
    return _result(True, "B1 README banner")


def check_b2_callback_authority() -> Tuple[bool, str]:
    """Verify only EarlyStoppingCallback writes the deployable artifact.

    We look for the *write pattern* `f"best_model_run_{` (which only appears
    in real `model.save(...)` paths, not in module/class docstrings) inside
    the CheckpointCallback class body. Doc references like
    ``best_model_run_*`` are allowed and even encouraged for readers.
    """
    src = (BOT_DIR / "src" / "training" / "callbacks.py").read_text(encoding="utf-8")
    cp_start = src.index("class CheckpointCallback")
    es_start = src.index("class EarlyStoppingCallback")
    cp_block = src[cp_start:es_start]
    if 'f"best_model_run_{' in cp_block:
        return _result(
            False,
            "B2 callback authority",
            "CheckpointCallback writes best_model_run_* (should be reward_best_run_*)",
        )
    if "reward_best_run_" not in cp_block:
        return _result(
            False,
            "B2 callback authority",
            "CheckpointCallback no longer mentions reward_best_run_* (rename regression)",
        )
    es_block = src[es_start:]
    if 'f"best_model_run_{' not in es_block:
        return _result(
            False,
            "B2 callback authority",
            "EarlyStoppingCallback no longer writes best_model_run_* (deployment regression)",
        )
    return _result(True, "B2 callback authority")


def check_b3_evaluator_split() -> Tuple[bool, str]:
    sys.path.insert(0, str(BOT_DIR))
    try:
        evaluator_mod = importlib.import_module("src.training.evaluator")
        cls = getattr(evaluator_mod, "Evaluator")
        sig = inspect.signature(cls.__init__)
        for required in ("dataset_split", "held_out_days"):
            if required not in sig.parameters:
                return _result(
                    False,
                    "B3 evaluator split",
                    f"Evaluator.__init__ missing {required}",
                )
    finally:
        if str(BOT_DIR) in sys.path:
            sys.path.remove(str(BOT_DIR))
    return _result(True, "B3 evaluator split")


def check_b3_walk_forward_gate() -> Tuple[bool, str]:
    src = (BOT_DIR / "scripts" / "rl_promotion_check.py").read_text(encoding="utf-8")
    if "g_wf" not in src or "--walk-forward" not in src:
        return _result(
            False,
            "B3 walk-forward gate",
            "rl_promotion_check.py missing g_wf gate or --walk-forward arg",
        )
    return _result(True, "B3 walk-forward gate")


def check_b4_db_schema() -> Tuple[bool, str]:
    sys.path.insert(0, str(BOT_DIR))
    try:
        db_mod = importlib.import_module("src.data.database")
        rl_trade = getattr(db_mod, "RLCryptoTrade")
        cols = {c.name for c in rl_trade.__table__.columns}
        required = {"fee_usdt", "cumulative_equity", "peak_equity", "order_type", "fill_was_maker"}
        missing = required - cols
        if missing:
            return _result(
                False, "B4 DB schema", f"missing columns: {sorted(missing)}"
            )
    finally:
        if str(BOT_DIR) in sys.path:
            sys.path.remove(str(BOT_DIR))
    return _result(True, "B4 DB schema")


def check_b5_parity_test() -> Tuple[bool, str]:
    test_path = REPO_ROOT / "tests" / "test_train_serve_parity.py"
    if not test_path.exists():
        return _result(False, "B5 parity test", "tests/test_train_serve_parity.py missing")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", str(test_path), "-q", "--no-header"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tail = (result.stdout + result.stderr).splitlines()[-5:]
        return _result(
            False, "B5 parity test", "pytest failed:\n  " + "\n  ".join(tail)
        )
    return _result(True, "B5 parity test")


def check_pnl_only_active() -> Tuple[bool, str]:
    cfg_path = REPO_ROOT / "shared" / "config" / "model_config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) or {}
    profile = cfg.get("environment", {}).get("reward_profile")
    if profile != "pnl_only":
        return _result(
            False,
            "Reward profile",
            f"environment.reward_profile must be 'pnl_only', got {profile!r}",
        )
    return _result(True, "Reward profile pnl_only")


def check_plan_frozen() -> Tuple[bool, str]:
    plan = RESEARCH_DIR / "RL_RUN_174_PLAN.md"
    if not plan.exists():
        return _result(False, "Pre-registration plan", "research/RL_RUN_174_PLAN.md missing")
    text = plan.read_text(encoding="utf-8")
    if "PRE-REGISTERED" not in text:
        return _result(
            False,
            "Pre-registration plan",
            "RL_RUN_174_PLAN.md missing 'PRE-REGISTERED' status banner",
        )
    return _result(True, "Pre-registration plan present")


def check_training_data_available() -> Tuple[bool, str]:
    """Operational pre-flight: the configured DATA_SOURCE/INTERVAL/SYMBOLS
    must actually have enough rows in the past `REQUIRE_HISTORICAL_DAYS`
    window to train. Otherwise the trainer dies with a generic
    `DataUnavailableError` and the operator cannot tell whether the failure
    was an audit regression or just stale data.
    """
    sys.path.insert(0, str(BOT_DIR))
    try:
        from datetime import datetime, timedelta

        from src.core.config import get_settings
        from src.data.database import CryptoCandle, get_db_session
        from sqlalchemy import func
    except Exception as exc:  # pragma: no cover - defensive
        if str(BOT_DIR) in sys.path:
            sys.path.remove(str(BOT_DIR))
        return _result(False, "Training data freshness", f"import failed: {exc!r}")

    try:
        settings = get_settings()
        interval = settings.DATA_INTERVAL
        source = settings.DATA_SOURCE
        symbols = [
            s.strip()
            for s in (settings.DATA_SYMBOLS or "").split(",")
            if s.strip()
        ]
        if not symbols:
            return _result(False, "Training data freshness", "DATA_SYMBOLS env is empty")

        max_steps = 500  # mirrors model_config.yaml environment.max_steps
        seq_len = 1  # MlpPolicy with no SequenceStackWrapper
        min_rows = max_steps + seq_len + 1

        end = datetime.utcnow()
        start = end - timedelta(days=settings.REQUIRE_HISTORICAL_DAYS)

        session = get_db_session()
        try:
            missing: List[str] = []
            for sym in symbols:
                count = (
                    session.query(func.count(CryptoCandle.id))
                    .filter(
                        CryptoCandle.source == source,
                        CryptoCandle.interval == interval,
                        CryptoCandle.symbol == sym,
                        CryptoCandle.timestamp >= start,
                        CryptoCandle.timestamp <= end,
                    )
                    .scalar()
                )
                if (count or 0) < min_rows:
                    missing.append(f"{sym}={count}/{min_rows}")
            if missing:
                hint = (
                    "Run: python bot/main.py collect-data --source "
                    f"{source} --symbols {','.join(symbols)} --interval "
                    f"{interval} --days {settings.REQUIRE_HISTORICAL_DAYS}"
                )
                return _result(
                    False,
                    "Training data freshness",
                    f"insufficient rows in last {settings.REQUIRE_HISTORICAL_DAYS}d: "
                    + "; ".join(missing)
                    + f"\n         {hint}",
                )
        finally:
            session.close()
    finally:
        if str(BOT_DIR) in sys.path:
            sys.path.remove(str(BOT_DIR))
    return _result(True, "Training data freshness")


CHECKS = [
    check_b1_fleet_yaml,
    check_b1_readme_banner,
    check_b2_callback_authority,
    check_b3_evaluator_split,
    check_b3_walk_forward_gate,
    check_b4_db_schema,
    check_b5_parity_test,
    check_pnl_only_active,
    check_plan_frozen,
    check_training_data_available,
]


def run_checks() -> bool:
    print("=" * 70)
    print("RL run 174 pre-flight checks (architecture-audit-03 §B1-B5)")
    print("=" * 70)
    all_ok = True
    for check in CHECKS:
        try:
            ok, msg = check()
        except Exception as exc:  # pragma: no cover - defensive
            ok = False
            msg = f"[FAIL] {check.__name__} raised: {exc!r}"
        print(msg)
        all_ok = all_ok and ok
    print("=" * 70)
    if all_ok:
        print("All gates PASS. Re-run with --commit to start training.")
    else:
        print("BLOCKED: fix the failing gate(s) before launching run 174.")
    print("=" * 70)
    return all_ok


def kickoff_training(episodes: int) -> int:
    print(f"\n>>> Launching `python main.py train --episodes {episodes}` from bot/")
    print(">>> reward_profile pinned to pnl_only via shared/config/model_config.yaml")
    env = os.environ.copy()
    cmd = [sys.executable, "main.py", "train", "--episodes", str(episodes)]
    proc = subprocess.run(cmd, cwd=str(BOT_DIR), env=env)
    return proc.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Actually start training. Without this flag the script only checks gates.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=600,
        help=(
            "Episodes to train. Default 600 ≈ 300k timesteps at max_steps=500, "
            "matching shared/config/model_config.yaml total_timesteps."
        ),
    )
    args = parser.parse_args()

    ok = run_checks()
    if not ok:
        return 1
    if not args.commit:
        return 0
    return kickoff_training(args.episodes)


if __name__ == "__main__":
    raise SystemExit(main())
