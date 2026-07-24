#!/usr/bin/env python3
"""Funding-only diagnostic for H-PERP-001 (insufficient calendar depth for Phase 3 gate)."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "datasets" / "H-PERP-001" / "btcusdt_swap_funding_okx.csv"
OUT = ROOT / "backtests" / "H-PERP-001_metrics.json"
V = 100.0  # USDT notional (design doc)


def main() -> None:
    rates: list[float] = []
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rates.append(float(row["fundingRate"]))
    pnls = [V * fr for fr in rates]
    s = (mean(pnls) / pstdev(pnls)) * math.sqrt(365 * 3) if len(pnls) > 5 and pstdev(pnls) > 1e-12 else 0.0
    out = {
        "hypothesis": "H-PERP-001",
        "n_intervals": len(rates),
        "note": "OKX public history depth ~3 months from this environment; Phase 3 12-mo gate NOT MET.",
        "sharpe_annualized_all_sample": round(float(s), 4),
        "cum_pnl_usdt": round(sum(pnls), 6),
        "verdict": "INCONCLUSIVE_DATA",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
