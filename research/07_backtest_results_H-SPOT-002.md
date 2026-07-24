# 07 — Backtest results: H-SPOT-002

**Status: FAIL (Phase 4)** — data contract met (D1–D4); evidence gates **not** all true.
See [05_data_quality_H-SPOT-002.md](05_data_quality_H-SPOT-002.md).

Metrics below are the gate-eligible Phase 4 verdict under the frozen
[06_backtest_design_H-SPOT-002.md](06_backtest_design_H-SPOT-002.md).
**No parameter retune** after this run.

---

## Data window (current)

| Item | Value |
|------|--------|
| Aligned rows | 950 |
| `days_span` | 949.0 |
| First / last candle | See [PROVENANCE.md](datasets/H-SPOT-002/PROVENANCE.md) |
| OOS daily returns | 920 (post `W=30` warm-up; three equal blocks, all OOS) |
| Entries (position flips into non-flat) | 26 |
| D4 week density | OK (`d4_week_density_ok = true`) |
| Data acquisition | Dual Coinbase daily via [fetch_candles.py](datasets/H-SPOT-002/fetch_candles.py) |

---

## OOS protocol (reference)

After warm-up index `W=30`, split `[W, n)` into **three** equal-length contiguous blocks; **all three** are OOS. Net daily return uses lagged band position on BTC−ETH excess return minus two-leg taker fees, per [06 §2–§4](06_backtest_design_H-SPOT-002.md).

---

## Phase 4 metrics ([H-SPOT-002_metrics.json](backtests/H-SPOT-002_metrics.json))

Regenerate with `python RLBotPM/research/backtests/H-SPOT-002.py` after each new CSV. Pre-registered `--bootstrap-trials 500 --seed 42`.

| Field | Phase 4 run (2026-07-23 UTC, 949.0d) |
|-------|--------------------------------------|
| `n_rows` / `oos_days` | 950 / 920 |
| `n_entries` | 26 |
| `sharpe_oos` | **−1.3333** (**G1 FAIL**; gate ≥ 1.5) |
| `profit_factor_oos` | **0.773** (**G2 FAIL**; gate ≥ 1.4) |
| `cum_pnl_oos` | **−1.050262** |
| `cum_pnl_oos_2x_fee` | **−1.662262** (**G3 FAIL**; need > 0) |
| `seg_means` | `[0.00006, −0.003154, −0.000336]` — **1 / 3** positive (**G4 FAIL**) |
| `g5_concentration` | **True** (**G5 PASS**) |
| `g6_method` | `random_sign_flip` (frozen in 06 §5) |
| `g6_placebo_frac_beats` | **0.984** (**G6 FAIL**; need ≤ 0.05) |
| `data_contract_ok` | true |
| `verdict` | **FAIL** |

### Interpretation

On ~31 months of aligned Coinbase daily closes, the pre-registered BTC–ETH log-ratio z-band rule is **net negative** after retail two-leg fees. Sharpe is deeply below the 1.5 gate; profit factor is below 1; cumulative OOS and 2×-fee stress are both negative. Only one of three OOS segments has a positive mean. The sign-flip placebo beat-rate (~98%) is consistent with a weak/negative path (null often beats the realized cumulative sum). **G5 alone** passes; that does not overturn FAIL.

This is a clean mechanism falsification — **not** an invitation to widen `W`, soften `Z_ENTER`/`Z_EXIT`, or cut fees. H-SPOT-001 dual-SMA remains closed; do not fall back to retuning those windows either.

---

## Verdict

**FAIL** — `data_contract_ok = true` and at least one of `g1..g6` is false (here: G1, G2, G3, G4, G6).

Do **not** activate paper protocol or live capital for H-SPOT-002. Archive cleanly; update [NEXT_HYPOTHESIS.md](NEXT_HYPOTHESIS.md).

---

## Phase 4 runner (record)

1. Ran `python RLBotPM/research/backtests/H-SPOT-002.py` (deterministic, seed 42).
2. Read `H-SPOT-002_metrics.json`.
3. Table above matches that JSON (2026-07-23).
4. Verdict branch taken: **FAIL** ⇔ `data_contract_ok` true AND any gate false → archive; no paper.
