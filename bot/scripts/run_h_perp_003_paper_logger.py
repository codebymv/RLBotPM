#!/usr/bin/env python3
"""Run the H-PERP-003 paper logger without starting any trading loop."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BOT_DIR))

from src.core.logger import get_logger  # noqa: E402
from src.strategies.paper_trader import append_h_perp_003_paper_snapshot  # noqa: E402

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=int, default=300, help="Seconds between polls.")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Write one snapshot if a new funding boundary is available, then exit.",
    )
    args = parser.parse_args(argv)

    scan_n = 0
    while True:
        try:
            append_h_perp_003_paper_snapshot(scan_n)
            logger.info("H-PERP-003 paper logger scan %s complete", scan_n)
        except Exception as exc:
            logger.warning("H-PERP-003 paper logger scan %s skipped: %s", scan_n, exc)
        if args.once:
            return 0
        scan_n += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
