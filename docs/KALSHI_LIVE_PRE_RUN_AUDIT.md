# Kalshi Live Pre-Run Audit

## Status
Implemented the highest-priority hardening items for the Kalshi live-trading path before the next production run.

## Findings And Disposition
### Fixed
- Resting orders now reserve exposure instead of being ignored by `deployed_capital` and `available_to_deploy`.
- Active-market limits now count both open positions and resting orders.
- Partial fills remain tracked without losing the still-resting remainder of the order.
- Transient `get_order()` failures no longer silently drop resting orders.
- Settlement reconciliation now skips phantom-removal logic when the exchange positions call fails instead of treating failure as zero positions.
- Unknown Kalshi order enum values now degrade safely with warnings instead of raising.
- The no-trade-before-resolution guard is now wired into live order placement.
- Live trade outcomes now feed the shared circuit breaker state, and live Kalshi execution disables the training-mode bypass.
- API replay now understands `order_resting`, `order_filled`, `order_closed`, and `phantom_removed`.
- Live combined metrics and bot status can now serve Kalshi live mode even when the RL/database side is unavailable.
- Dashboard bot-status and overview requests are now mode-aware for live Kalshi sessions.
- Live event push failures are now visible in bot logs instead of being silently swallowed.
- Startup now blocks by default when the Kalshi account already has open orders or positions that the bot cannot safely reconstruct.

### Remaining Known Limitations
- The fleet-level combined loss caps declared in `shared/config/fleet.yaml` are still configuration-only unless a supervisor enforces them.
- Live API replay tests that import `api.main` are skipped in environments without FastAPI installed.
- Restart recovery is intentionally conservative: the bot stops on pre-existing exchange state unless `KALSHI_ALLOW_UNRECONCILED_STARTUP=true` is set.

## Files Changed
- `bot/src/strategies/live_trader.py`
- `bot/src/execution/kalshi_client.py`
- `api/main.py`
- `dashboard/src/lib/api.ts`
- `dashboard/src/app/OverviewClient.tsx`
- `dashboard/src/app/page.tsx`
- `dashboard/src/app/bot-status/page.tsx`
- `dashboard/src/app/bot-status/BotStatusClient.tsx`
- `bot/tests/test_kalshi_live_audit.py`

## Go / No-Go Checklist
### Must Be True Before Live Start
1. `python bot/scripts/verify_kalshi_live_readiness.py` passes in the same shell/env that will run the bot.
2. `TRAINING_MODE=false` is set for the live bot process.
3. The Kalshi account has no pre-existing open orders or positions, unless you intentionally override with `KALSHI_ALLOW_UNRECONCILED_STARTUP=true`.
4. `API_BASE_URL` points to the API instance the dashboard reads from.
5. The dashboard in `mode=live` shows the bot as alive after startup heartbeat.
6. A dry run produces `session_start` and `scan_summary` events in both the local bot log and the API-backed dashboard.

### Recommended Smoke Test
1. Start a live run with the smallest intended sizing.
2. Confirm the exchange shows the expected order state.
3. Confirm the bot log shows either `order_resting`, `order_filled`, or `order_closed`.
4. Confirm the API/dashboard reflects the same state within one polling interval.
5. If a fill occurs, wait for settlement and verify the same contracts, cost, and P&L appear in the dashboard.

### No-Go Conditions
- The bot cannot reach `/api/live-trades` or `/api/bot/heartbeat`.
- The account contains leftover live orders/positions from a prior run and you have not manually reconciled them.
- The dashboard is only updating in paper mode or depends on a DB path you do not have available.
- The bot logs repeated Kalshi API errors or unknown payload shapes on startup.

## Suggested Operator Command Order
1. Run readiness check.
2. Run a dry run and inspect dashboard heartbeat.
3. Run one micro-size live session.
4. Validate exchange state against dashboard state.
5. Only then allow a longer unattended live run.
