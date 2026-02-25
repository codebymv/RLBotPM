"""Alert system for critical events."""

from __future__ import annotations

from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Dict, List, Optional
import json
import os
import smtplib
import urllib.request

from ..core.logger import get_logger


logger = get_logger(__name__)


class AlertSystem:
    """
    Send alerts for critical events.

    Supports three transports (tried in order):
    1. Gleam trigger call  (GLEAM_TRIGGER_URL + GLEAM_TRIGGER_KEY)
    2. Generic webhook POST     (ALERT_WEBHOOK_* env vars)
    3. SMTP email               (ALERT_SMTP_* env vars)
    """

    def __init__(self, email_to: List[str]):
        self.email_to = email_to
        # SMTP
        self.smtp_host = os.getenv("ALERT_SMTP_HOST")
        self.smtp_port = int(os.getenv("ALERT_SMTP_PORT", "587"))
        self.smtp_user = os.getenv("ALERT_SMTP_USER")
        self.smtp_pass = os.getenv("ALERT_SMTP_PASS")
        self.smtp_from = os.getenv("ALERT_SMTP_FROM", self.smtp_user or "alerts@rltrade")
        # Generic webhook
        self.webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        self.webhook_bearer_token = os.getenv("ALERT_WEBHOOK_BEARER_TOKEN")
        self.webhook_timeout_seconds = float(os.getenv("ALERT_WEBHOOK_TIMEOUT_SECONDS", "5"))
        # Gleam event-alert trigger (2 env vars only)
        self.gleam_trigger_url = (os.getenv("GLEAM_TRIGGER_URL") or "").rstrip("/")
        self.gleam_trigger_key = os.getenv("GLEAM_TRIGGER_KEY", "")

    def _send_via_smtp(self, subject: str, message: str, severity: str) -> bool:
        if not self.email_to:
            return False
        if not self.smtp_host or not self.smtp_user or not self.smtp_pass:
            return False

        msg = MIMEText(message)
        msg["Subject"] = f"[RLTrade:{severity.upper()}] {subject}"
        msg["From"] = self.smtp_from
        msg["To"] = ", ".join(self.email_to)

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            server.login(self.smtp_user, self.smtp_pass)
            server.sendmail(self.smtp_from, self.email_to, msg.as_string())
        return True

    def _send_via_webhook(self, subject: str, message: str, severity: str) -> bool:
        if not self.webhook_url:
            return False

        payload = {
            "subject": subject,
            "message": message,
            "severity": severity,
            "source": "rlbot",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "email_to": self.email_to,
        }
        body = json.dumps(payload).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
        }
        if self.webhook_bearer_token:
            headers["Authorization"] = f"Bearer {self.webhook_bearer_token}"

        request = urllib.request.Request(
            self.webhook_url,
            data=body,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.webhook_timeout_seconds):
            pass
        return True

    # ------------------------------------------------------------------
    # Gleam trigger transport
    # ------------------------------------------------------------------
    def _send_via_gleam_trigger(self, subject: str, message: str, severity: str) -> bool:
        """Trigger an outbound voice alert call via the Gleam EVENT_ALERTS endpoint.

        POST GLEAM_TRIGGER_URL with Authorization: Bearer <GLEAM_TRIGGER_KEY>.
        The Gleam agent handles calling the configured alertRecipientPhone.
        """
        if not self.gleam_trigger_url or not self.gleam_trigger_key:
            return False

        payload = json.dumps({
            "event_type": subject,
            "severity": severity,
            "message": message,
            "source": "rlbot",
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.gleam_trigger_key}",
        }

        req = urllib.request.Request(
            self.gleam_trigger_url,
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp_body = json.loads(resp.read().decode("utf-8"))
            te_id = resp_body.get("data", {}).get("triggerEventId", "unknown")
            logger.info(
                "Gleam trigger alert sent (triggerEventId=%s, severity=%s): %s",
                te_id, severity, subject,
            )
        return True
        return True

    def send_alert(self, subject: str, message: str, severity: str = "info") -> None:
        logger.warning("Alert (%s): %s", severity, subject)
        gleam_ok = False
        webhook_ok = False
        smtp_ok = False

        try:
            gleam_ok = self._send_via_gleam_trigger(subject, message, severity)
        except Exception as exc:
            logger.error("Gleam trigger alert failed: %s", exc)

        try:
            webhook_ok = self._send_via_webhook(subject, message, severity)
        except Exception as exc:
            logger.error("Webhook alert failed: %s", exc)

        try:
            smtp_ok = self._send_via_smtp(subject, message, severity)
        except Exception as exc:
            logger.error("SMTP alert failed: %s", exc)

        if not gleam_ok and not webhook_ok and not smtp_ok:
            logger.info("No alert transport configured; alert message: %s", message)

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
