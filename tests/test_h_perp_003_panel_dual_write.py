"""Paper logger must append-only mirror snapshots into the offline panel CSV."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BOT = ROOT / "bot"
sys.path.insert(0, str(BOT))

from src.strategies.paper_trader import (  # noqa: E402
    H_PERP_003_PANEL_FIELDS,
    _append_h_perp_003_panel_row,
)


def test_panel_dual_write_appends_once(tmp_path, monkeypatch) -> None:
    panel = tmp_path / "btc_hedged_panel_okx.csv"
    monkeypatch.setattr(
        "src.strategies.paper_trader.H_PERP_003_PANEL_CSV",
        panel,
    )

    ev = {
        "fundingTime": 1_777_939_200_000,
        "fundingRate": -0.000048,
        "mark_candle_ts": 1_777_938_840_000,
        "mark_close": 79990.6,
        "spot_candle_ts": 1_777_938_840_000,
        "spot_close": 80031.2,
        "align_ok": 1,
        "mark_skew_ms": 0,
        "spot_skew_ms": 0,
    }
    _append_h_perp_003_panel_row(ev)
    _append_h_perp_003_panel_row(ev)  # idempotent

    with open(panel, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == H_PERP_003_PANEL_FIELDS
    assert len(rows) == 1
    assert int(rows[0]["fundingTime"]) == ev["fundingTime"]
    assert float(rows[0]["mark_close"]) == ev["mark_close"]
