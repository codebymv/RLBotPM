#!/usr/bin/env python3
"""
Offline compare: H-KALS-001b demo PARKED G3 JSONL vs live Option B JSONL.

No network. No orders. Does not retune thresholds or promote any hypothesis.

Also reports Option B live G3-style 10-scan freeze readiness (bookkeeping only)
and refuses structural labels when demo/live JSONL mode purity fails.

  python research/scanners/compare_kals_001b_demo_live.py
  python research/scanners/compare_kals_001b_demo_live.py --json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_RLPM_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DEMO = (
    _RLPM_ROOT / "research" / "datasets" / "H-KALS-001b" / "scan_events.jsonl"
)
_DEFAULT_LIVE = (
    _RLPM_ROOT / "research" / "datasets" / "H-KALS-001b-live" / "scan_events.jsonl"
)
_SCANNER = _RLPM_ROOT / "research" / "scanners" / "kals_001_probability_sum_scan.py"
_G3_FREEZE_TARGET = 10


def _load_scanner():
    """Load Σp scanner module for shared purity audit (no network side effects)."""
    spec = importlib.util.spec_from_file_location("kals_001_scan_compare", _SCANNER)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load scanner at {_SCANNER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["kals_001_scan_compare"] = mod
    spec.loader.exec_module(mod)
    return mod


def audit_paths_purity(
    demo_path: Path,
    live_path: Path,
) -> Dict[str, Any]:
    """
    Offline purity for both streams. Demo must be demo-only; live must be live-only.
    """
    scan = _load_scanner()
    demo = scan.audit_jsonl_mode_purity(demo_path, expect_live=False)
    live = scan.audit_jsonl_mode_purity(live_path, expect_live=True)
    ok = bool(demo.get("ok")) and bool(live.get("ok"))
    errors: List[str] = []
    if not demo.get("ok"):
        errors.append("demo: " + "; ".join(demo.get("errors") or ["not ok"]))
    if not live.get("ok"):
        errors.append("live: " + "; ".join(live.get("errors") or ["not ok"]))
    return {
        "ok": ok,
        "demo": demo,
        "live": live,
        "errors": errors,
    }


def event_family(event_ticker: str) -> str:
    """Leading series token before the first hyphen (e.g. KXBTCY-27JAN0100 → KXBTCY)."""
    return (event_ticker or "").split("-", 1)[0]


def load_scan_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def iter_violations(rows: Sequence[Dict[str, Any]]) -> Iterable[Dict[str, Any]]:
    for row in rows:
        for v in row.get("violations") or []:
            yield v


def union_event_tickers(rows: Sequence[Dict[str, Any]]) -> Set[str]:
    return {str(v["event_ticker"]) for v in iter_violations(rows) if v.get("event_ticker")}


def union_families(rows: Sequence[Dict[str, Any]]) -> Set[str]:
    return {event_family(str(v["event_ticker"])) for v in iter_violations(rows) if v.get("event_ticker")}


def kind_counts(rows: Sequence[Dict[str, Any]]) -> Counter:
    return Counter(str(v.get("kind") or "?") for v in iter_violations(rows))


def family_kind_table(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Counter] = {}
    for v in iter_violations(rows):
        fam = event_family(str(v.get("event_ticker") or ""))
        if not fam:
            continue
        out.setdefault(fam, Counter())[str(v.get("kind") or "?")] += 1
    return {k: dict(v) for k, v in sorted(out.items(), key=lambda kv: (-sum(kv[1].values()), kv[0]))}


def scan_level_stats(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stats: List[Dict[str, Any]] = []
    for row in rows:
        cands = int(row.get("events_partition_candidates") or 0)
        viols = int(row.get("violation_count") or 0)
        viol_list = list(row.get("violations") or [])
        legs = [len(v.get("tickers") or []) for v in viol_list]
        sums = [float(v["sum_p"]) for v in viol_list if v.get("sum_p") is not None]
        ask_gaps = [
            float(v["toy_naive_long_ask_vs_par"])
            for v in viol_list
            if v.get("toy_naive_long_ask_vs_par") is not None
        ]
        stats.append(
            {
                "timestamp": row.get("timestamp"),
                "demo": row.get("demo"),
                "api_mode": row.get("api_mode"),
                "markets_fetched": row.get("markets_fetched"),
                "candidates": cands,
                "violations": viols,
                "violation_rate": round(viols / cands, 6) if cands else None,
                "mean_legs": round(sum(legs) / len(legs), 4) if legs else None,
                "mean_sum_p": round(sum(sums) / len(sums), 6) if sums else None,
                "ask_vs_par_negative": sum(1 for x in ask_gaps if x < 0),
                "ask_vs_par_nonneg": sum(1 for x in ask_gaps if x >= 0),
            }
        )
    return stats


def stable_live_events(live_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Events that appear as violations in every live scan (within-stream stickiness)."""
    if not live_rows:
        return {"n_scans": 0, "intersection": [], "union": []}
    sets = [union_event_tickers([r]) for r in live_rows]
    inter = set.intersection(*sets) if sets else set()
    uni = set.union(*sets) if sets else set()
    return {
        "n_scans": len(live_rows),
        "intersection": sorted(inter),
        "union": sorted(uni),
        "intersection_size": len(inter),
        "union_size": len(uni),
    }


def live_g3_freeze_status(
    live_rows: Sequence[Dict[str, Any]],
    *,
    purity_ok: bool,
    target: int = _G3_FREEZE_TARGET,
) -> Dict[str, Any]:
    """
    Parallel bookkeeping for Option B live vs demo's 10-scan G3 freeze.

    Pre-registered meaning (06 G3): after ``target`` successful scans, zero
    OVER/UNDER violations across the **first** ``target`` scans → existence
    FAIL; ≥1 → continue logging (not a tradeable PASS). Never invents capital
    PASS.

    Existence counts only the freeze window (``live_rows[:target]``). Later
    appends must not flip a zero-violation FAIL into VIOLATIONS_OBSERVED.
    """
    n = len(live_rows)
    target_n = int(target)
    # Pre-registered freeze window is the first N successful scans only.
    window = list(live_rows[:target_n])
    total_viol = sum(int(r.get("violation_count") or 0) for r in window)
    scans_with_viol = sum(1 for r in window if int(r.get("violation_count") or 0) > 0)
    remaining = max(0, target_n - n)
    freeze_ready = n >= target_n and purity_ok

    if not purity_ok:
        existence = "INCONCLUSIVE_DATA"
        note = (
            "Mode purity failed on live (or paired demo) JSONL; freeze bookkeeping "
            "is blocked until streams are pure."
        )
    elif n == 0:
        existence = "PENDING"
        note = "No live scans yet; append production --live rows first."
    elif n < target_n:
        existence = "PENDING"
        note = (
            f"{n}/{target_n} successful live scans; need {remaining} more before a "
            "G3-style freeze addendum. Not a capital gate."
        )
    elif total_viol == 0:
        existence = "FAIL"
        note = (
            f"First {target_n} live scans have zero OVER/UNDER violations — "
            "pre-registered G3 existence FAIL for this live window. Demo G3 stays "
            "PARKED; no retune; no capital."
        )
    else:
        # Match demo 07 language: zero-violation FAIL branch does not apply.
        existence = "VIOLATIONS_OBSERVED"
        note = (
            f"First {target_n} live scans frozen for bookkeeping: {total_viol} "
            f"violation rows across {scans_with_viol}/{target_n} scans"
            + (f" ({n} total appended)" if n > target_n else "")
            + ". Zero-violation FAIL branch does not apply. "
            "Not a Phase-4/5 PASS; do not promote."
        )

    return {
        "target_scans": target_n,
        "successful_live_scans": n,
        "freeze_window_scans": len(window),
        "scans_remaining_to_freeze": remaining,
        "total_violation_rows": total_viol,
        "scans_with_violations": scans_with_viol,
        "purity_ok": purity_ok,
        "g3_style_10_scan_freeze_ready": freeze_ready,
        "existence_gate": existence,
        "addendum_ready": freeze_ready and existence in ("FAIL", "VIOLATIONS_OBSERVED"),
        "capital_pass": False,
        "do_not_promote": True,
        "note": note,
    }


def replication_verdict(
    *,
    demo_scans: int,
    live_scans: int,
    demo_kinds: Counter,
    live_kinds: Counter,
    event_overlap: int,
    family_overlap: int,
    purity_ok: bool = True,
    freeze: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Honest, non-promotional assessment for Option B live replication.

    Promising = material OVER/UNDER violations appear on live under the same
    rule set B, with a similar UNDER-dominated mix — not identity of tickers
    and not a capital PASS. Impure provenance blocks structural labels.
    """
    live_total = sum(live_kinds.values())
    demo_total = sum(demo_kinds.values())
    live_has_violations = live_total > 0
    under_share_live = (live_kinds.get("UNDER", 0) / live_total) if live_total else None
    under_share_demo = (demo_kinds.get("UNDER", 0) / demo_total) if demo_total else None

    freeze = freeze or {}
    g3_style_ready = bool(freeze.get("g3_style_10_scan_freeze_ready"))
    existence = freeze.get("existence_gate")
    zero_violation_fail_branch_possible = (
        existence == "FAIL"
        or (live_total == 0 and live_scans >= _G3_FREEZE_TARGET and purity_ok)
    )

    if not purity_ok:
        label = "PROVENANCE_IMPURE"
        note = (
            "Demo/live JSONL failed mode purity audit — do not interpret structural "
            "replication until streams are unmixed."
        )
    elif live_scans == 0:
        label = "NO_LIVE_DATA"
        note = "No live scans to compare; append --live JSONL first."
    elif existence == "FAIL":
        # Freeze-window existence FAIL wins over whole-stream kind counts so
        # post-freeze appends cannot reopen STRUCTURAL_* / LIVE_EMPTY labels.
        label = "LIVE_G3_EXISTENCE_FAIL"
        note = (
            "Live G3-style freeze window has zero OVER/UNDER violations — "
            "pre-registered existence FAIL. Late appends do not reopen structural "
            "replication labels. Demo G3 stays PARKED; no capital."
        )
    elif not live_has_violations:
        label = "LIVE_EMPTY_VIOLATIONS"
        note = (
            "Live scans present but zero violations so far — opposite of demo G3 "
            "pattern; accumulate more batches before judging."
        )
    elif event_overlap == 0 and family_overlap == 0:
        label = "STRUCTURAL_REPLICATION_PROMISING"
        note = (
            "Live shows material sum-p violations every scan with UNDER-dominated mix, "
            "matching demo G3 phenomenology, but exact event/family sets do not "
            "overlap (different calendar / listing universe). Not a tradeable PASS."
        )
    elif event_overlap > 0 or family_overlap > 0:
        label = "STRUCTURAL_AND_PARTIAL_IDENTITY"
        note = (
            "Live replicates the violation phenomenon and shares some event or "
            "family identity with demo. Still not a capital PASS; toy ask fields "
            "must be checked before any economics claim."
        )
    else:
        label = "INCONCLUSIVE"
        note = "Insufficient contrast to classify."

    return {
        "label": label,
        "promising_for_more_live_batches": label
        in ("STRUCTURAL_REPLICATION_PROMISING", "STRUCTURAL_AND_PARTIAL_IDENTITY"),
        "g3_style_10_scan_freeze_ready": g3_style_ready,
        "zero_violation_fail_branch_possible": zero_violation_fail_branch_possible,
        "under_share_demo": under_share_demo,
        "under_share_live": under_share_live,
        "note": note,
        "do_not_promote": True,
        "capital_pass": False,
        "purity_ok": purity_ok,
    }


def compare_demo_live(
    demo_rows: Sequence[Dict[str, Any]],
    live_rows: Sequence[Dict[str, Any]],
    *,
    purity_ok: bool = True,
) -> Dict[str, Any]:
    demo_events = union_event_tickers(demo_rows)
    live_events = union_event_tickers(live_rows)
    demo_fams = union_families(demo_rows)
    live_fams = union_families(live_rows)
    demo_kinds = kind_counts(demo_rows)
    live_kinds = kind_counts(live_rows)
    event_overlap = demo_events & live_events
    family_overlap = demo_fams & live_fams
    freeze = live_g3_freeze_status(live_rows, purity_ok=purity_ok)

    report: Dict[str, Any] = {
        "demo": {
            "n_scans": len(demo_rows),
            "kind_counts": dict(demo_kinds),
            "unique_events": len(demo_events),
            "unique_families": sorted(demo_fams),
            "family_kind_table": family_kind_table(demo_rows),
            "scan_stats": scan_level_stats(demo_rows),
            "markets_fetched_range": _minmax_field(demo_rows, "markets_fetched"),
            "candidates_range": _minmax_field(demo_rows, "events_partition_candidates"),
            "violations_range": _minmax_field(demo_rows, "violation_count"),
        },
        "live": {
            "n_scans": len(live_rows),
            "kind_counts": dict(live_kinds),
            "unique_events": len(live_events),
            "unique_families": sorted(live_fams),
            "family_kind_table": family_kind_table(live_rows),
            "scan_stats": scan_level_stats(live_rows),
            "markets_fetched_range": _minmax_field(live_rows, "markets_fetched"),
            "candidates_range": _minmax_field(live_rows, "events_partition_candidates"),
            "violations_range": _minmax_field(live_rows, "violation_count"),
            "stable_events": stable_live_events(live_rows),
        },
        "overlap": {
            "exact_event_tickers": sorted(event_overlap),
            "exact_event_count": len(event_overlap),
            "families": sorted(family_overlap),
            "family_count": len(family_overlap),
            "demo_only_families": sorted(demo_fams - live_fams),
            "live_only_families": sorted(live_fams - demo_fams),
        },
        "live_g3_freeze": freeze,
        "verdict": replication_verdict(
            demo_scans=len(demo_rows),
            live_scans=len(live_rows),
            demo_kinds=demo_kinds,
            live_kinds=live_kinds,
            event_overlap=len(event_overlap),
            family_overlap=len(family_overlap),
            purity_ok=purity_ok,
            freeze=freeze,
        ),
    }
    return report


def _minmax_field(rows: Sequence[Dict[str, Any]], key: str) -> Optional[List[Any]]:
    vals = [r[key] for r in rows if r.get(key) is not None]
    if not vals:
        return None
    return [min(vals), max(vals)]


def format_report_text(report: Dict[str, Any]) -> str:
    d = report["demo"]
    l = report["live"]
    o = report["overlap"]
    v = report["verdict"]
    f = report.get("live_g3_freeze") or {}
    p = report.get("purity") or {}
    lines = [
        "H-KALS-001b demo PARKED G3 vs live Option B (offline)",
        "",
        f"Purity: ok={p.get('ok')}  "
        f"demo_lines={((p.get('demo') or {}).get('lines'))}  "
        f"live_lines={((p.get('live') or {}).get('lines'))}",
        f"Demo scans: {d['n_scans']}  kinds={d['kind_counts']}  "
        f"events={d['unique_events']}  markets_range={d['markets_fetched_range']}  "
        f"cands={d['candidates_range']}  viols={d['violations_range']}",
        f"Live scans: {l['n_scans']}  kinds={l['kind_counts']}  "
        f"events={l['unique_events']}  markets_range={l['markets_fetched_range']}  "
        f"cands={l['candidates_range']}  viols={l['violations_range']}",
        "",
        f"Exact event overlap: {o['exact_event_count']} {o['exact_event_tickers'][:12]}",
        f"Family overlap: {o['family_count']} {o['families']}",
        f"Demo-only families: {o['demo_only_families']}",
        f"Live-only families: {o['live_only_families']}",
        "",
        f"Live sticky events (all scans): "
        f"{l['stable_events'].get('intersection_size')}/"
        f"{l['stable_events'].get('union_size')}",
        "",
        f"Live G3 freeze: {f.get('successful_live_scans')}/{f.get('target_scans')}  "
        f"remaining={f.get('scans_remaining_to_freeze')}  "
        f"existence_gate={f.get('existence_gate')}  "
        f"addendum_ready={f.get('addendum_ready')}",
        f"  freeze note: {f.get('note')}",
        "",
        f"Verdict: {v['label']}",
        f"  promising_for_more_live_batches: {v['promising_for_more_live_batches']}",
        f"  g3_style_10_scan_freeze_ready: {v['g3_style_10_scan_freeze_ready']}",
        f"  capital_pass: {v['capital_pass']}  do_not_promote: {v['do_not_promote']}",
        f"  under_share_demo: {v['under_share_demo']}  under_share_live: {v['under_share_live']}",
        f"  note: {v['note']}",
    ]
    if p.get("errors"):
        lines.append(f"  purity_errors: {p['errors']}")
    return "\n".join(lines) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare H-KALS-001b demo vs live Σp violation sets (offline)"
    )
    parser.add_argument("--demo", type=Path, default=_DEFAULT_DEMO)
    parser.add_argument("--live", type=Path, default=_DEFAULT_LIVE)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full JSON report instead of text summary",
    )
    parser.add_argument(
        "--allow-impure",
        action="store_true",
        help="Compare even if demo/live mode purity fails (labels stay PROVENANCE_IMPURE)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    demo_rows = load_scan_rows(args.demo)
    live_rows = load_scan_rows(args.live)
    if not demo_rows:
        print(f"error: demo JSONL missing or empty: {args.demo}", file=sys.stderr)
        return 2
    if not live_rows:
        print(f"error: live JSONL missing or empty: {args.live}", file=sys.stderr)
        return 2

    purity = audit_paths_purity(args.demo, args.live)
    if not purity["ok"] and not args.allow_impure:
        print(
            "error: demo/live JSONL failed mode purity audit "
            "(use --allow-impure only for forensics):\n  "
            + "\n  ".join(purity.get("errors") or ["unknown"]),
            file=sys.stderr,
        )
        return 3

    report = compare_demo_live(demo_rows, live_rows, purity_ok=bool(purity["ok"]))
    report["paths"] = {"demo": str(args.demo), "live": str(args.live)}
    report["purity"] = purity

    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_report_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
