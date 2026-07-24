# 01 — Post-Mortem of the Lognormal-vs-Strike Strategy on Kalshi

> Status: Phase 1a deliverable of the Edge Research Reset plan.
> Author: Research cycle, 2026-04-19.
> Purpose: Document what was built, what was tested, what each test proved or failed to prove, with citations to concrete artifacts. Negative results are deliverables.

---

## 1. What was built

A Python live-trading system targeting Kalshi prediction markets, structured as:

- A market-data adapter (`bot/src/data/sources/kalshi.py`) and an execution client (`bot/src/execution/kalshi_client.py`) covering account/balance, positions, order placement, cancellation, settlements.
- An edge detector (`bot/src/strategies/kalshi_edges.py`) implementing several "edge types": `spot_vs_strike` (lognormal model), `crypto_spot_mispricing` (an earlier deep-OTM heuristic), `macro_data` (FRED-vs-Kalshi consensus model), `weather` (NOAA-vs-Kalshi).
- A live-trader loop (`bot/src/strategies/live_trader.py`) with admission filters, hybrid sleeves (fast vs macro), exit logic (profit-take, stop-loss, flatten-before-close), and a kill switch.
- Auxiliary tools: `bot/edge_audit.py`, `bot/research_scoreboard.py`, `bot/fundability_gate.py`, `bot/loss_narrative.py`, `bot/position_review.py`, `bot/unwind_macro.py`.

The infrastructure is solid — none of the failures below are caused by code bugs in the loop. They are caused by the strategy hypotheses the loop was asked to execute.

## 2. The two hypotheses we actually tested

The strategy reduced to a single underlying claim across two regimes:

> **The Kalshi-priced YES probability differs from a lognormal-diffusion-plus-static-vol estimate of the underlying asset's terminal probability, by enough to trade after fees.**

Two regimes of that claim were tested:

- **H2 (near-money model edge)** — lognormal-vs-strike on hourly and daily crypto buckets where the spot is within ~15% of the strike. Most of the 75 `spot_vs_strike` orders were in this regime.
- **H1 (far-OTM trivial edge)** — same model, but only on buckets where the spot is more than 15% away from the strike AND time-to-settlement is under 2 hours. The audit ([bot/edge_audit.py](../bot/edge_audit.py)) named this the "trivial edge" because in such conditions any reasonable model says "buy NO" — the model is not contributing.

## 3. Live trading scoreboard (artifacted)

From `bot/edge_audit.py` against `bot/logs/live_trades.jsonl` as of 2026-04-19:

| Edge type | Orders placed | Settled | W / L | Settled P&L | Avg reported edge | Unresolved |
|---|---|---|---|---|---|---|
| `crypto_spot_mispricing` (legacy H1-like detector) | 52 | 38 | 38 / 0 | **+$38.00** | 50.0% | 14 (27%) |
| `spot_vs_strike` (lognormal H2) | 75 | 8 | 0 / 8 | **-$1.65** | 16.7% | 67 (89%) |
| `macro_data` | 41 | 0 | — | **$0.00** | 57.0% | 41 (100%) |
| **Total** | **168** | **46** | **38 / 8** | **+$36.35 settled** | — | **122 (73%) unresolved** |

Three observations on this table that matter more than the headline P&L:

1. **The "wins" all came from one detector, on one regime, in one historical period.** The 38 wins are concentrated in `crypto_spot_mispricing` orders that were placed in the previous live run when BTC was at ~$70K and Kalshi had hourly buckets sitting at $81K+ strikes. That is the historical H1 condition. It produced a 100% paper-printed win rate.
2. **The current `spot_vs_strike` detector — which is the modern, more general lognormal model — is 0/8 in settlements with a -$1.65 P&L.** Every single near-money settlement we have is a loss. The 67 unresolved orders here include the long-dated macro-style holdings.
3. **73% of orders ever placed have not settled.** Most are macro buckets with months of holding period, or near-money crypto buckets that flatten-before-close exited at a loss before settlement. The bot has been *deploying capital it cannot recycle*.

## 4. What each hypothesis actually proved or failed to prove

### H1 — far-OTM trivial edge (>15% OTM, <2h)

**Status: FALSIFIED for the current regime.**

The audit cited 38 historical wins as evidence for H1. But those wins came from a transient market regime (BTC well below the strike ladder Kalshi was offering) and the live capital was sized small enough that we cannot statistically distinguish "edge" from "we got lucky for two weeks." More importantly, in the present regime:

- An H1-only mode was implemented in [bot/src/strategies/live_trader.py](../bot/src/strategies/live_trader.py) (constants `H1_CRYPTO_ASSETS`, `H1_MIN_SPOT_DISTANCE_PCT=0.15`, `H1_MIN_HOURS=0.5`, `H1_MAX_HOURS=2.0`).
- A candidate logger was added that records every NO-side crypto edge in the relevant window, regardless of whether it passed all gates. Output: [bot/logs/h1_candidates.jsonl](../bot/logs/h1_candidates.jsonl).

After two scans across 934 markets with 405 edges per scan, the breakdown of NO-side crypto candidates was:

| Gate | Count (of 98 candidates over multiple scans) |
|---|---|
| `pass_distance` (≥15% OTM) | 12 |
| `pass_window` (0.5–2h to settlement) | 29 |
| **`h1_passes_all` (both)** | **0** |

The 12 deep-OTM candidates that did exist were uniformly at 6+ hours to expiry, with `yes_ask` already at 1¢. Buying NO at 99¢ for a $1 payout is a 1.01% gross edge per trade. After Kalshi's 7% trading fee on profits (and a maker-taker spread of typically 1¢), the net edge is at or below zero.

**Conclusion**: the historical H1 wins were a one-time regime artifact. In the current market, the structural condition that made H1 work (deep-OTM short-dated crypto buckets with non-trivial mispricing) does not exist. Market makers price these correctly to within the minimum tick. There is no current-regime edge to exploit.

### H2 — near-money lognormal model edge

**Status: FALSIFIED with realized losses.**

8 of 8 near-money settlements lost money. -$1.65 net P&L on $38.27 cost basis (~4% loss rate) with a model claiming an average 16.7% edge. The audit's prediction that the lognormal model cannot distinguish 40% from 60% probability for short-dated near-money buckets was directly confirmed by these settlements. The fact that the model continued to *report* a 16.7% average edge while losing every single trade is the strongest possible falsification.

### Macro `macro_data` model

**Status: UNTESTED. Capital trapped.**

41 orders placed, 0 settlements. $35.16 of cost basis sitting in positions with months-long holding periods. The audit noted in writing before a single macro position settled that macro bets have "massive opportunity cost" and "no proven edge." The trapped capital validates the opportunity-cost concern without ever producing data on whether the model itself was correct. We also found accidental NO positions created by a bug in the unwind script that compounded the lockup.

**Conclusion**: even if the macro model has edge, the holding period makes it an inappropriate capital deployment for a $40 bankroll. We cannot test it in a useful timeframe and we cannot recycle the capital.

## 5. Cross-cutting failure modes

The two falsified hypotheses share root causes that are themselves valuable findings:

1. **The model edge is the spread.** When a model says "fair = 40c, market = 50c, edge = 10c," but the bid is 45c and the ask is 55c, the *executable* edge is 0c. The bot's reported edges were measured against mid-price; the executions paid bid/ask. After spread, most reported edges were imaginary. We added `MIN_EXECUTION_EDGE` and `MAX_SPREAD_CENTS` admission filters partway through, but by then capital was already deployed in the wrong places.
2. **No capacity check.** Many orders were placed in markets with `volume + open_interest < 5`. Those are not trades — those are us being the only counterparty in a stale market. We added `MIN_MARKET_ACTIVITY=5` later but it does not save earlier deployments.
3. **No pre-registration of evidence thresholds.** The strategy was tweaked many times in response to losing trades. Each tweak claimed to fix the prior failure mode. Without a documented "if the next 100 trades do X, we declare PASS/FAIL," there was no way to know when to stop tweaking.
4. **Capital was deployed before the hypothesis was tested offline.** Backtesting and walk-forward harnesses exist in the repo ([bot/src/strategies/kalshi_backtest.py](../bot/src/strategies/kalshi_backtest.py), [bot/src/strategies/walk_forward_crypto.py](../bot/src/strategies/walk_forward_crypto.py)) but they were not the gating mechanism for live deployment. The live bot was the test.
5. **No mechanism explanation for the edge.** "The model says fair is X and the market says Y" is not a mechanism. It is an observation. A real edge requires an explanation of *why* the market is willing to take the other side persistently. We never had one.

## 6. What is salvageable

- The Kalshi data adapter and execution client work correctly. They handle authentication, rate limits, order placement, cancellation, settlement detection. Reusable.
- The paper-trader scaffold ([bot/src/strategies/paper_trader.py](../bot/src/strategies/paper_trader.py)) and the live-trader loop are reusable for any new admission rule.
- The candidate-logging pattern (write every decision to JSONL whether or not we trade it) is the right pattern for evidence collection and should be the default for the next strategy.
- The walk-forward backtest harness exists and is reusable.

The lognormal model itself, the macro-FRED detector, and the weather detector all stay in the repo but **must not be re-enabled in live trading by this research cycle.** They have failed their evidence threshold.

## 7. Capital accounting at the start of this research cycle

Per the plan: "Pretend the $16 cash and $22 in stuck positions do not exist."

For completeness, the actual state on 2026-04-19 is roughly:
- Cash: $16.01
- Positions value (Kalshi mark-to-market): ~$22
- Realized cumulative P&L since funding: net loss after the H2 and macro deployments, partially offset by the historical H1 wins

These numbers do not enter any decision in this research cycle. If a hypothesis clears the fundability gate at the end of Phase 6, the live-deployment plan written at that point will treat starting capital as a fresh question.

## 8. Bottom line

We tested one core claim — that a lognormal-vs-strike model finds tradeable mispricings on Kalshi — across two regimes, and both regimes have been falsified by either realized losses or absence of current-regime setups. The infrastructure is good. The hypothesis was wrong. The next step is not "tweak the model"; it is "go back to the venue selection and the edge taxonomy from a clean slate."

Phase 1b ([02_venue_survey.md](02_venue_survey.md)) does that survey.
