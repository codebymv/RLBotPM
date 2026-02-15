# Dual Paper Runbook

This runbook starts both paper bots with conservative defaults:

- Crypto RL paper trader (live data, no real orders)
- Kalshi paper trader (demo mode, no real orders)

## Preconditions

- Activate and install bot dependencies:
  - `cd bot`
  - `.\venv\Scripts\activate`
  - `pip install -r requirements.txt`
- Confirm model exists:
  - `bot\models\final_run_165.zip`
- Confirm Kalshi connectivity:
  - `python main.py kalshi status`

## Quick readiness checks

- Crypto evaluation sanity:
  - `python main.py evaluate --model models/final_run_165.zip --episodes 20`
- Kalshi historical edge sanity:
  - `python main.py kalshi backtest-crypto --min-edge 0.01`

## Launch both paper bots

From `bot`:

- Dry run to print launch commands:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\run_dual_paper.ps1`
- Launch both:
  - `powershell -ExecutionPolicy Bypass -File .\scripts\run_dual_paper.ps1 -Launch`

Logs are written to:

- `bot\logs\paper_sessions\<timestamp>\crypto.stdout.log`
- `bot\logs\paper_sessions\<timestamp>\crypto.stderr.log`
- `bot\logs\paper_sessions\<timestamp>\kalshi.stdout.log`
- `bot\logs\paper_sessions\<timestamp>\kalshi.stderr.log`

## Conservative defaults in launcher

- Crypto:
  - model: `models/final_run_165.zip`
  - duration: `24h`
  - capital: `$1000`
  - interval: `1h`
- Kalshi:
  - demo mode enabled
  - interval: `300s`
  - bankroll: `$100`
  - edge range: `1%` to `20%`
  - max contracts/trade: `10`
  - max positions: `20`
  - max scans: `100`

## Recommended stop criteria

- Stop session if either:
  - Strategy behavior changes unexpectedly (large jump in invalid actions/errors)
  - Repeated API/auth failures
  - Drawdown exceeds your paper threshold for the test window
- Keep paper mode running at least 24-72h before any live capital decision.

## Manual commands (without launcher)

- Crypto:
  - `python main.py rl-paper-trade --model models/final_run_165.zip --duration 24 --capital 1000 --interval 1h --log-dir logs/paper_sessions/manual/crypto`
- Kalshi:
  - `python main.py kalshi paper-trade --demo --interval 300 --bankroll 100 --min-edge 0.01 --max-edge 0.20 --max-contracts 10 --max-positions 20 --max-scans 100`
