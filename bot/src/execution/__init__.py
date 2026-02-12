"""Execution layer for live and paper trading.

Keep this module lightweight: avoid importing optional/heavy dependencies
(e.g. torch) at import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
	from .live_data_feed import LiveDataFeed
	from .paper_trader import PaperTradingEngine
	from .live_rl_trader import LiveRLPaperTrader


__all__ = ["LiveDataFeed", "PaperTradingEngine", "LiveRLPaperTrader"]


def __getattr__(name: str):
	if name == "LiveDataFeed":
		from .live_data_feed import LiveDataFeed as _LiveDataFeed

		return _LiveDataFeed
	if name == "PaperTradingEngine":
		from .paper_trader import PaperTradingEngine as _PaperTradingEngine

		return _PaperTradingEngine
	if name == "LiveRLPaperTrader":
		from .live_rl_trader import LiveRLPaperTrader as _LiveRLPaperTrader

		return _LiveRLPaperTrader
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
