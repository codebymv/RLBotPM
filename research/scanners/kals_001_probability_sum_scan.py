#!/usr/bin/env python3
"""
H-KALS-001 / H-KALS-001b — read-only Kalshi probability-sum scanners (no orders).

- **001** — [research/06_backtest_design_H-KALS-001.md](../06_backtest_design_H-KALS-001.md) (rule set A; H-KALS-001 parked — see 07).
- **001b** — [research/06_backtest_design_H-KALS-001b.md](../06_backtest_design_H-KALS-001b.md) (rule set B; contiguous `between` ladders).

Demo and live logs are **separate** (Option B / `--live` Σp replication):
  python research/scanners/kals_001_probability_sum_scan.py --variant 001b --once
  python research/scanners/kals_001_probability_sum_scan.py --variant 001b --live --once
  python research/scanners/kals_001_probability_sum_scan.py --variant 001b --repeat 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

# RLBotPM/research/scanners -> parents[2] == RLBotPM
_RLPM_ROOT = Path(__file__).resolve().parents[2]
_BOT_DIR = _RLPM_ROOT / "bot"
_DEFAULT_OUT_001 = _RLPM_ROOT / "research" / "datasets" / "H-KALS-001" / "scan_events.jsonl"
_DEFAULT_OUT_001B = _RLPM_ROOT / "research" / "datasets" / "H-KALS-001b" / "scan_events.jsonl"
_DEFAULT_OUT_001_LIVE = (
    _RLPM_ROOT / "research" / "datasets" / "H-KALS-001-live" / "scan_events.jsonl"
)
_DEFAULT_OUT_001B_LIVE = (
    _RLPM_ROOT / "research" / "datasets" / "H-KALS-001b-live" / "scan_events.jsonl"
)

_DEMO_DEFAULT_PATHS = frozenset({_DEFAULT_OUT_001.resolve(), _DEFAULT_OUT_001B.resolve()})
_LIVE_DEFAULT_PATHS = frozenset(
    {_DEFAULT_OUT_001_LIVE.resolve(), _DEFAULT_OUT_001B_LIVE.resolve()}
)

_STRIKE_ALLOWED = frozenset({"greater", "less", "between"})
_MAX_SPREAD_CENTS = 15
_OVER = 1.05
_UNDER = 0.95


class OutputModeConflict(ValueError):
    """Raised when a scan would append into a JSONL reserved for the other API mode."""


def _setup_path() -> None:
    sys.path.insert(0, str(_BOT_DIR))


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = _RLPM_ROOT / ".env"
    if env_path.is_file():
        load_dotenv(env_path)


def default_output_path(variant: str, live: bool) -> Path:
    """Resolve the default append path for a variant × demo/live pair."""
    if variant == "001b":
        return _DEFAULT_OUT_001B_LIVE if live else _DEFAULT_OUT_001B
    return _DEFAULT_OUT_001_LIVE if live else _DEFAULT_OUT_001


def assert_output_mode_compatible(
    out_path: Path,
    *,
    live: bool,
    allow_mixed_output: bool = False,
) -> None:
    """
    Refuse writing live rows into known demo JSONL paths (and vice versa).

    Custom ``--output`` paths outside the reserved defaults are allowed.
    Pass ``allow_mixed_output`` only for explicit operator override (not recommended).
    """
    if allow_mixed_output:
        return
    resolved = out_path.resolve()
    if live and resolved in _DEMO_DEFAULT_PATHS:
        raise OutputModeConflict(
            f"--live refuses demo JSONL path {out_path}. "
            f"Use default live path or a dedicated --output under "
            f"research/datasets/H-KALS-*-live/."
        )
    if (not live) and resolved in _LIVE_DEFAULT_PATHS:
        raise OutputModeConflict(
            f"demo mode refuses live JSONL path {out_path}. "
            f"Use the demo default or a non-live --output."
        )


def audit_jsonl_mode_purity(
    path: Path,
    *,
    expect_live: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Offline audit: every scan line's ``demo`` flag must be consistent.

    Returns a dict with ``ok``, counts, and ``errors`` (empty when clean).
    When ``expect_live`` is set, also require all rows match that mode.
    """
    errors: List[str] = []
    n = 0
    n_demo = 0
    n_live = 0
    if not path.is_file():
        return {
            "ok": True,
            "path": str(path),
            "lines": 0,
            "demo_lines": 0,
            "live_lines": 0,
            "errors": [],
            "note": "missing_file_treated_as_empty",
        }

    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            n += 1
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                errors.append(f"line {i}: invalid JSON ({e})")
                continue
            if "demo" not in row:
                errors.append(f"line {i}: missing demo field")
                continue
            is_demo = bool(row["demo"])
            if is_demo:
                n_demo += 1
            else:
                n_live += 1
            if expect_live is True and is_demo:
                errors.append(f"line {i}: expected live (demo=false), got demo=true")
            if expect_live is False and not is_demo:
                errors.append(f"line {i}: expected demo (demo=true), got demo=false")

    if n_demo and n_live:
        errors.append(
            f"mixed modes in one file: demo_lines={n_demo} live_lines={n_live}"
        )

    return {
        "ok": len(errors) == 0,
        "path": str(path),
        "lines": n,
        "demo_lines": n_demo,
        "live_lines": n_live,
        "errors": errors,
    }


def live_credentials_present() -> Tuple[bool, str]:
    """
    Soft preflight for production scans.

    Public market listing may work without auth; missing keys are reported so
    operators know a live batch may be Blocked until credentials are set.
    """
    key = (os.getenv("KALSHI_API_KEY") or "").strip()
    secret = (os.getenv("KALSHI_API_SECRET") or "").strip()
    if key and secret and "YOUR_KEY" not in secret and key != "your-kalshi-api-key":
        return True, "KALSHI_API_KEY and KALSHI_API_SECRET look set"
    missing = []
    if not key or key == "your-kalshi-api-key":
        missing.append("KALSHI_API_KEY")
    if not secret or "YOUR_KEY" in secret:
        missing.append("KALSHI_API_SECRET")
    return False, "missing or placeholder: " + ", ".join(missing)


def _float_tol(a: float, b: float) -> float:
    return max(1e-4, 1e-6 * max(1.0, abs(a), abs(b)))


def _partition_members_rule_a(markets: List[Any]) -> Optional[List[Any]]:
    """H-KALS-001 rule set A."""
    if len(markets) < 2:
        return None

    close_ref = markets[0].close_time
    for m in markets:
        if m.close_time != close_ref:
            return None
        st = (m.strike_type or "").lower()
        if st not in _STRIKE_ALLOWED:
            return None
        if st == "between" and (m.floor_strike is None or m.cap_strike is None):
            return None

    included: List[Any] = []
    for m in markets:
        spread = float(m.yes_ask) - float(m.yes_bid)
        if spread > _MAX_SPREAD_CENTS or m.yes_ask <= m.yes_bid:
            continue
        included.append(m)

    if len(included) < 2:
        return None
    return included


def _between_spread_ok(m: Any) -> bool:
    spread = float(m.yes_ask) - float(m.yes_bid)
    return spread <= _MAX_SPREAD_CENTS and m.yes_ask > m.yes_bid


def _adjacent_ladder(m1: Any, m2: Any) -> bool:
    """Undirected adjacency for ladder graph (shared boundary on strike axis)."""
    c1, f1 = float(m1.cap_strike), float(m1.floor_strike)
    c2, f2 = float(m2.cap_strike), float(m2.floor_strike)
    t12 = _float_tol(c1, f2)
    t21 = _float_tol(c2, f1)
    return abs(c1 - f2) <= t12 or abs(c2 - f1) <= t21


def _chain_sorted_is_contiguous(sorted_ms: List[Any]) -> bool:
    for i in range(len(sorted_ms) - 1):
        a, b = sorted_ms[i], sorted_ms[i + 1]
        ca, fb = float(a.cap_strike), float(b.floor_strike)
        if abs(ca - fb) > _float_tol(ca, fb):
            return False
    return True


def _connected_components(vertices: List[Any]) -> List[List[Any]]:
    n = len(vertices)
    adj: List[Set[int]] = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if _adjacent_ladder(vertices[i], vertices[j]):
                adj[i].add(j)
                adj[j].add(i)

    seen: Set[int] = set()
    comps: List[List[Any]] = []
    for i in range(n):
        if i in seen:
            continue
        stack = [i]
        seen.add(i)
        comp_idx: List[int] = []
        while stack:
            u = stack.pop()
            comp_idx.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        comps.append([vertices[k] for k in comp_idx])
    return comps


def _iter_rule_b_ladders(group: List[Any]) -> List[List[Any]]:
    """
    H-KALS-001b rule set B: split by close_time, between-only, spread ok;
    connected components; each sorted by floor must be a contiguous chain, len>=2.
    """
    by_close: DefaultDict[Any, List[Any]] = defaultdict(list)
    for m in group:
        if m.event_ticker is None:
            continue
        by_close[m.close_time].append(m)

    ladders: List[List[Any]] = []
    for _, ms in by_close.items():
        b_only: List[Any] = []
        for m in ms:
            if (m.strike_type or "").lower() != "between":
                continue
            if m.floor_strike is None or m.cap_strike is None:
                continue
            if not _between_spread_ok(m):
                continue
            b_only.append(m)
        if len(b_only) < 2:
            continue
        for comp in _connected_components(b_only):
            if len(comp) < 2:
                continue
            comp_sorted = sorted(comp, key=lambda x: float(x.floor_strike))
            if not _chain_sorted_is_contiguous(comp_sorted):
                continue
            ladders.append(comp_sorted)
    return ladders


def _violation_row(
    event_ticker: str,
    members: List[Any],
    variant: str,
) -> Dict[str, Any]:
    ps: List[float] = []
    tickers: List[str] = []
    sum_yes_ask = 0.0
    sum_yes_bid = 0.0
    for m in members:
        mid_c = (float(m.yes_bid) + float(m.yes_ask)) / 2.0
        ps.append(mid_c / 100.0)
        tickers.append(m.ticker)
        sum_yes_ask += float(m.yes_ask) / 100.0
        sum_yes_bid += float(m.yes_bid) / 100.0

    s = sum(ps)
    kind: Optional[str] = None
    if s > _OVER:
        kind = "OVER"
    elif s < _UNDER:
        kind = "UNDER"

    row: Dict[str, Any] = {
        "event_ticker": event_ticker,
        "sum_p": round(s, 6),
        "kind": kind,
        "tickers": tickers,
        "ps": [round(x, 6) for x in ps],
    }
    if variant == "001b":
        n = len(members)
        toy_fee_illustrative = 2 * n * 0.01  # see 06: 1c per contract per side toy
        row["sum_yes_ask_frac"] = round(sum_yes_ask, 6)
        row["sum_yes_bid_frac"] = round(sum_yes_bid, 6)
        row["toy_naive_long_ask_vs_par"] = round(1.0 - sum_yes_ask, 6)
        row["toy_fee_illustrative_roundtrip"] = round(toy_fee_illustrative, 6)
    return row


def build_scan_row(
    markets: List[Any],
    *,
    variant: str,
    demo: bool,
    timestamp: Optional[str] = None,
    api_base_url: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Pure scan aggregation (no I/O). Used by ``_scan_once`` and offline fixtures.
    """
    by_event: Dict[str, List[Any]] = defaultdict(list)
    for m in markets:
        et = m.event_ticker
        if not et:
            continue
        by_event[et].append(m)

    violations: List[Dict[str, Any]] = []
    partition_candidates = 0

    if variant == "001":
        row_type = "kals_001_scan"
        for event_ticker, group in by_event.items():
            members = _partition_members_rule_a(group)
            if members is None:
                continue
            partition_candidates += 1
            vr = _violation_row(event_ticker, members, variant)
            if vr.get("kind"):
                violations.append(vr)
    else:
        row_type = "kals_001b_scan"
        for event_ticker, group in by_event.items():
            for ladder in _iter_rule_b_ladders(group):
                partition_candidates += 1
                vr = _violation_row(event_ticker, ladder, variant)
                if vr.get("kind"):
                    violations.append(vr)

    row: Dict[str, Any] = {
        "type": row_type,
        "variant": variant,
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "demo": bool(demo),
        "api_mode": "demo" if demo else "live",
        "markets_fetched": len(markets),
        "events_partition_candidates": partition_candidates,
        "violation_count": len(violations),
        "violations": violations,
    }
    if api_base_url:
        row["api_base_url"] = api_base_url
    return row


def _scan_once(adapter: Any, out_path: Path, variant: str) -> Dict[str, Any]:
    markets = adapter.list_open_markets_all_pages(limit_per_page=200)
    row = build_scan_row(
        markets,
        variant=variant,
        demo=bool(getattr(adapter, "demo", False)),
        api_base_url=getattr(adapter, "base_url", None),
    )

    # Refuse appending into a file that already mixes / mismatches mode.
    purity = audit_jsonl_mode_purity(out_path, expect_live=not row["demo"])
    if not purity["ok"] and purity["lines"] > 0:
        raise OutputModeConflict(
            "refusing append: existing JSONL fails mode purity audit: "
            + "; ".join(purity["errors"])
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return row


def main() -> int:
    _setup_path()
    _load_env()

    parser = argparse.ArgumentParser(
        description="H-KALS-001 / 001b probability-sum observation scan"
    )
    parser.add_argument(
        "--variant",
        choices=("001", "001b"),
        default="001",
        help="001 = rule set A (legacy); 001b = contiguous between ladders (default output path changes)",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--demo", action="store_true", help="Kalshi demo API (default if neither mode flag)"
    )
    mode.add_argument("--live", action="store_true", help="Kalshi production API")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL append path (defaults by --variant and --demo/--live)",
    )
    parser.add_argument(
        "--allow-mixed-output",
        action="store_true",
        help="Override demo/live path guard (not recommended; breaks Option B provenance)",
    )
    parser.add_argument("--once", action="store_true", help="Force a single scan")
    parser.add_argument("--interval", type=int, default=0, help="Seconds between scans; 0 = single")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        metavar="N",
        help="Run N full scans back-to-back (append N lines); use for 001b G3 batch. Ignored with --interval > 0.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit existing --output (or default) JSONL for demo/live purity; no network.",
    )
    parser.add_argument(
        "--require-live-credentials",
        action="store_true",
        help="With --live, exit 2 if KALSHI_API_KEY/SECRET look unset (offline-safe preflight).",
    )
    args = parser.parse_args()

    use_demo = not args.live
    if args.output is None:
        args.output = default_output_path(args.variant, live=args.live)

    try:
        assert_output_mode_compatible(
            args.output,
            live=args.live,
            allow_mixed_output=args.allow_mixed_output,
        )
    except OutputModeConflict as e:
        print(f"Output mode conflict: {e}", file=sys.stderr)
        return 2

    if args.audit_only:
        if args.live:
            expect_live: Optional[bool] = True
        elif args.demo:
            expect_live = False
        else:
            expect_live = None  # internal purity only
        report = audit_jsonl_mode_purity(args.output, expect_live=expect_live)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ok"] else 1

    if args.live:
        ok_creds, cred_msg = live_credentials_present()
        if not ok_creds:
            print(f"Live credential preflight: {cred_msg}", file=sys.stderr)
            if args.require_live_credentials:
                print(
                    "Blocked: needs Kalshi credentials "
                    "(set KALSHI_API_KEY / KALSHI_API_SECRET in .env).",
                    file=sys.stderr,
                )
                return 2
            print(
                "Continuing without credentials (public listing may still work); "
                "use --require-live-credentials to hard-fail.",
                file=sys.stderr,
            )

    from src.data.sources.kalshi import KalshiAdapter

    adapter = KalshiAdapter(demo=use_demo)

    interval = args.interval
    loop = interval > 0 and not args.once
    repeat = max(1, int(args.repeat))

    if not loop:
        try:
            last: Optional[Dict[str, Any]] = None
            for i in range(repeat):
                summary = _scan_once(adapter, args.output, args.variant)
                last = summary
                if repeat > 1:
                    print(
                        f"[{i + 1}/{repeat}] violation_count={summary.get('violation_count')} "
                        f"candidates={summary.get('events_partition_candidates')} "
                        f"markets={summary.get('markets_fetched')} "
                        f"api_mode={summary.get('api_mode')}",
                        file=sys.stderr,
                    )
                if i < repeat - 1:
                    time.sleep(3.0)
            assert last is not None
            print(json.dumps(last, indent=2, ensure_ascii=False))
            if repeat > 1:
                print(
                    json.dumps(
                        {
                            "batch_complete": True,
                            "repeat": repeat,
                            "variant": args.variant,
                            "api_mode": "live" if args.live else "demo",
                            "output": str(args.output),
                        },
                        indent=2,
                    ),
                    file=sys.stderr,
                )
        except OutputModeConflict as e:
            print(f"Output mode conflict: {e}", file=sys.stderr)
            return 2
        except Exception as e:
            print(f"Scan failed: {e}", file=sys.stderr)
            return 1
        return 0

    print(
        f"Looping every {interval}s ({'live' if args.live else 'demo'}); Ctrl+C to stop",
        file=sys.stderr,
    )
    while True:
        try:
            summary = _scan_once(adapter, args.output, args.variant)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
        except OutputModeConflict as e:
            print(f"Output mode conflict: {e}", file=sys.stderr)
            return 2
        except Exception as e:
            print(f"Scan failed: {e}", file=sys.stderr)
        time.sleep(interval)


if __name__ == "__main__":
    raise SystemExit(main())
