"""Pre-flight checks for minimal live deployment."""

from __future__ import annotations

import os


def main() -> None:
    required = [
        "LIVE_TRADING_ENABLED",
        "EXCHANGE_API_KEY",
        "EXCHANGE_API_SECRET",
        "DATABASE_URL",
    ]
    missing = [key for key in required if not os.getenv(key)]

    if missing:
        print("Live deployment check failed. Missing env vars:")
        for key in missing:
            print(f"  - {key}")
        return

    if os.getenv("LIVE_TRADING_ENABLED", "false").lower() != "true":
        print("LIVE_TRADING_ENABLED is not true. Aborting live deployment.")
        return

    print("Pre-flight checks passed. Ready for minimal live deployment.")
    print("Recommendation: start with $100-500 and monitor daily.")


if __name__ == "__main__":
    main()
