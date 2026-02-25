"""Pre-flight checks for minimal live deployment."""

from __future__ import annotations

import os


def main() -> None:
    required = [
        "LIVE_TRADING_ENABLED",
        "KALSHI_API_KEY",
        "KALSHI_API_SECRET",
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

    allowed_sides = {
        side.strip().lower()
        for side in os.getenv("LIVE_ALLOWED_SIDES", "no").split(",")
        if side.strip()
    }
    allow_buy_yes = os.getenv("LIVE_ALLOW_BUY_YES", "false").lower() == "true"
    if "yes" in allowed_sides and not allow_buy_yes:
        print("LIVE_ALLOWED_SIDES includes yes, but LIVE_ALLOW_BUY_YES is not true. Aborting live deployment.")
        return

    has_webhook = bool(os.getenv("ALERT_WEBHOOK_URL"))
    has_smtp = all(
        os.getenv(key)
        for key in ["ALERT_EMAIL_TO", "ALERT_SMTP_HOST", "ALERT_SMTP_USER", "ALERT_SMTP_PASS"]
    )

    gleam_vars = ["GLEAM_TRIGGER_URL", "GLEAM_TRIGGER_KEY"]
    gleam_set = [key for key in gleam_vars if os.getenv(key)]
    has_gleam = len(gleam_set) == len(gleam_vars)
    if gleam_set and not has_gleam:
        missing_gl = [key for key in gleam_vars if key not in gleam_set]
        print(f"Warning: partial Gleam trigger config. Missing: {', '.join(missing_gl)}")

    if not has_webhook and not has_smtp and not has_gleam:
        print("Warning: no alert transport configured (set SMTP vars, ALERT_WEBHOOK_URL, or GLEAM_TRIGGER_* vars).")

    if has_gleam:
        print("Gleam voice alerts: configured.")

    print("Pre-flight checks passed. Ready for minimal live deployment.")
    print("Recommendation: start with $100-500 and monitor daily.")


if __name__ == "__main__":
    main()
