"""Bot heartbeat – keeps the dashboard alive indicator current.

Starts a daemon thread that POSTs to ``POST /api/bot/heartbeat`` every
``interval`` seconds.  Stopped automatically when the thread is daemon
(process exit) or explicitly via :meth:`BotHeartbeat.stop`.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

from ..core.logger import get_logger

logger = get_logger(__name__)

_API_BASE_URL = os.getenv("API_BASE_URL") or os.getenv("NEXT_PUBLIC_API_URL") or "http://localhost:8000"


class BotHeartbeat:
    """Periodic heartbeat sender.

    Args:
        interval: Seconds between heartbeat POSTs (default 60).
        metadata_fn: Optional callable that returns a dict of extra metadata
                     to include in each heartbeat payload (e.g. scan count,
                     open position count). Called just before each POST.
        api_base_url: Override the API base URL (default reads env).
    """

    def __init__(
        self,
        interval: int = 60,
        metadata_fn: Optional[Callable[[], Dict[str, Any]]] = None,
        api_base_url: Optional[str] = None,
        bot_id: str = "kalshi_paper",
    ) -> None:
        self.interval = interval
        self.metadata_fn = metadata_fn
        self.api_base_url = (api_base_url or _API_BASE_URL).rstrip("/")
        self.bot_id = bot_id
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        """Start the background heartbeat thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="BotHeartbeat", daemon=True)
        self._thread.start()
        logger.info("Bot heartbeat started (interval=%ds, bot_id=%s)", self.interval, self.bot_id)

    def stop(self) -> None:
        """Signal the heartbeat thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval + 5)
        logger.info("Bot heartbeat stopped")

    # ------------------------------------------------------------------
    def _run(self) -> None:
        # Send one immediately on startup so the dashboard lights up fast
        self._send()
        while not self._stop_event.wait(timeout=self.interval):
            self._send()

    def _send(self) -> None:
        payload: Dict[str, Any] = {
            "bot_id": self.bot_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if self.metadata_fn:
            try:
                payload["metadata"] = self.metadata_fn()
            except Exception as exc:
                logger.debug("heartbeat metadata_fn error: %s", exc)

        url = f"{self.api_base_url}/api/bot/heartbeat"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10):
                pass
            logger.debug("Heartbeat sent (bot_id=%s)", self.bot_id)
        except Exception as exc:
            # Heartbeat failures are non-fatal — the bot keeps running
            logger.debug("Heartbeat POST failed (will retry): %s", exc)
