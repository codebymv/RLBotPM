"""
Verify Kalshi API credentials, balance, and readiness for live trading.

Read-only — does not place orders.

Run from repo root:
    python bot/scripts/verify_kalshi_live_readiness.py
    python bot/scripts/verify_kalshi_live_readiness.py --min-balance 25
    python bot/scripts/verify_kalshi_live_readiness.py --include-demo
    python bot/scripts/verify_kalshi_live_readiness.py --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BOT_DIR = _REPO_ROOT / "bot"


def _setup() -> None:
    sys.path.insert(0, str(_BOT_DIR))
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = _REPO_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path)
    else:
        load_dotenv()


def _has_cryptography() -> bool:
    try:
        import cryptography  # noqa: F401
        return True
    except ImportError:
        return False


def _check_env() -> tuple[bool, list[str]]:
    issues: list[str] = []
    key = os.getenv("KALSHI_API_KEY", "").strip()
    secret = os.getenv("KALSHI_API_SECRET", "").strip()
    if not key:
        issues.append("KALSHI_API_KEY is missing or empty")
    if not secret:
        issues.append("KALSHI_API_SECRET is missing or empty")
    if not _has_cryptography():
        issues.append(
            "Python package 'cryptography' is required for Kalshi RSA auth "
            "(pip install cryptography)"
        )
    return len(issues) == 0, issues


def _probe(demo: bool) -> dict:
    from src.execution.kalshi_client import KalshiExecutionClient

    client = KalshiExecutionClient(demo=demo)
    health = client.healthcheck()
    positions: list = []
    active_positions: list = []
    pos_error = None
    if health.get("ok"):
        try:
            positions = client.get_positions()
            active_positions = [p for p in positions if client.is_active_position(p)]
        except Exception as e:
            pos_error = str(e)
    return {
        "demo": demo,
        "health": health,
        "position_count": len(positions) if positions is not None else None,
        "active_position_count": len(active_positions) if active_positions is not None else None,
        "positions_error": pos_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Kalshi account funds and API readiness (read-only)."
    )
    parser.add_argument(
        "--min-balance",
        type=float,
        default=5.0,
        help="Minimum available balance in USD to pass (default: 5)",
    )
    parser.add_argument(
        "--include-demo",
        action="store_true",
        help="Also probe the demo API for comparison",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON only",
    )
    args = parser.parse_args()

    _setup()

    ok_env, env_issues = _check_env()
    result: dict = {
        "env_ok": ok_env,
        "env_issues": env_issues,
        "production": None,
        "demo": None,
        "ready_for_live": False,
    }

    if not ok_env:
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("Kalshi live readiness check\n")
            for i in env_issues:
                print(f"  [FAIL] {i}")
        return 1

    try:
        result["production"] = _probe(demo=False)
    except Exception as e:
        result["production"] = {"error": str(e)}
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(f"[FAIL] Production probe crashed: {e}")
        return 1

    if args.include_demo:
        try:
            result["demo"] = _probe(demo=True)
        except Exception as e:
            result["demo"] = {"error": str(e)}

    prod = result["production"]
    h = prod.get("health") or {}
    avail = float(h.get("balance_available") or 0)
    total = float(h.get("balance_total") or 0)
    api_ok = bool(h.get("ok"))
    authed = bool(h.get("authenticated"))

    result["ready_for_live"] = (
        api_ok
        and authed
        and avail >= args.min_balance
        and int(prod.get("active_position_count") or 0) == 0
    )

    if args.json:
        print(json.dumps(result, indent=2))
        return 0 if result["ready_for_live"] else 1

    print("Kalshi live readiness check (production API)\n")
    print(f"  Credentials:     {'OK' if ok_env else 'MISSING'}")
    print(f"  API reachable:   {'OK' if api_ok else 'FAIL'}")
    print(f"  RSA auth loaded: {'yes' if authed else 'no'}")
    if not api_ok:
        err = h.get("error", "unknown")
        print(f"  Error detail:    {err[:200]}")
    else:
        print(f"  Available cash:  ${avail:.2f}")
        print(f"  Portfolio value: ${total:.2f}")
        pc = prod.get("position_count")
        if pc is not None:
            print(f"  Open positions:  {pc}")
        apc = prod.get("active_position_count")
        if apc is not None:
            print(f"  Active positions:{apc:>4}")
        if prod.get("positions_error"):
            print(f"  Positions note:  {prod['positions_error'][:120]}")

    print()
    if result["ready_for_live"]:
        print(f"  [PASS] Ready for live trading (available >= ${args.min_balance:.2f}).")
    else:
        if api_ok and avail < args.min_balance:
            print(
                f"  [FAIL] Available balance ${avail:.2f} is below "
                f"--min-balance ${args.min_balance:.2f}."
            )
        elif api_ok and int(prod.get("active_position_count") or 0) > 0:
            print(
                f"  [FAIL] Account still has {int(prod.get('active_position_count') or 0)} "
                "active open positions."
            )
        elif not api_ok:
            print("  [FAIL] Production API check did not succeed.")
        elif not authed:
            print("  [FAIL] RSA private key not loaded (check KALSHI_API_SECRET).")

    if args.include_demo and result.get("demo"):
        d = result["demo"]
        print("\nDemo API (reference only)")
        dh = d.get("health") or {}
        if dh.get("ok"):
            print(f"  Available: ${float(dh.get('balance_available', 0)):.2f}")
        else:
            print(f"  Status: {dh.get('error', 'failed')}")

    return 0 if result["ready_for_live"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
