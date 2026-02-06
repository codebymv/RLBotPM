"""Alert system for critical events."""

from __future__ import annotations

from email.mime.text import MIMEText
from typing import Dict, List, Optional
import os
import smtplib

from ..core.logger import get_logger


logger = get_logger(__name__)


class AlertSystem:
    """
    Send alerts for critical events.

    Uses SMTP settings from environment if configured.
    """

    def __init__(self, email_to: List[str]):
        self.email_to = email_to
        self.smtp_host = os.getenv("ALERT_SMTP_HOST")
        self.smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))
        self.smtp_user = os.getenv("ALERT_SMTP_USER")
        self.smtp_pass = os.getenv("ALERT_SMTP_PASS")
        self.smtp_from = os.getenv("ALERT_SMTP_FROM", self.smtp_user or "alerts@rltrade")

    def send_alert(self, subject: str, message: str, severity: str = "info") -> None:
        logger.warning("Alert (%s): %s", severity, subject)
        if not self.smtp_host or not self.smtp_user or not self.smtp_pass:
            logger.info("SMTP not configured; alert message: %s", message)
            return

        msg = MIMEText(message)
        msg["Subject"] = f"[RLTrade:{severity.upper()}] {subject}"
        msg["From"] = self.smtp_from
        msg["To"] = ", ".join(self.email_to)

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_pass)
            server.sendmail(self.smtp_from, self.email_to, msg.as_string())

    def check_and_alert(self, metrics: Dict[str, float]) -> None:
        total_return = metrics.get("total_return_pct")
        win_rate = metrics.get("win_rate")
        num_trades = metrics.get("num_trades", 0)

        if total_return is not None and total_return < -0.10:
            self.send_alert(
                "Large Drawdown Alert",
                f"Portfolio down {total_return:.1%}",
                severity="warning",
            )

        if win_rate is not None and win_rate > 0.55 and num_trades > 100:
            self.send_alert(
                "Paper Trading Success",
                f"Win rate {win_rate:.1%} over {num_trades} trades",
                severity="success",
            )
