# RLBotPM project-level smoke tests

Repo-wide regression guards. Run with:

```bash
cd RLBotPM
python -m pytest tests/ -v
```

The bot's RL-specific tests live under [`bot/tests/`](../bot/tests/) and run
separately (they require the bot dependencies). These tests only depend on the
research stack and the standard library.

## Current tests

- `test_h_perp_003_smoke.py` — guards the D1 gate, the G6 placebo definition
  (must be `random_sign_flip`, never multiset-invariant), and the CSV
  append-only / `align_ok` invariants. Will need an update once D1 is met
  (track A5 of [architecture-audit-03](../research/architecture-audit-03.md)).
- `test_h_perp_003_tracking.py` — Phase 5 verifier: formula PASS/FAIL, Sharpe
  annualization parity with Phase 4 `sharpe_8h`, drift-gate FAIL when paper
  Sharpe is far from OOS, clear `fail_reasons` / `diagnosis`, honest
  non-blocking drift below `--drift-min-intervals`, and FAIL (not silent PASS)
  when Phase 4 metrics are missing or present-but-incomplete (null/unusable
  `sharpe_oos` / `profit_factor_oos`).
- `test_h_perp_003_panel_dual_write.py` — paper logger append-only panel mirror.
- `test_kals_001_live_sigma_p.py` — Option B Kalshi Σp: demo/live path separation,
  mix-refusal, rule-B UNDER/OVER fixtures, JSONL purity audit (no network).
- `test_kals_001b_demo_live_compare.py` — demo vs live compare labels, purity gate,
  live G3 10-scan freeze bookkeeping (no capital PASS; no network).
- `test_h_spot_002_signal.py` — H-SPOT-002 band signal / G6 method guards.
