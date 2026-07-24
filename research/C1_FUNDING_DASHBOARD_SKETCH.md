# C1 Sketch — Funding Monitor Dashboard

> **Status (2026-05-04):** SKETCH ONLY, deferred per
> [.cursor/plans/rlbotpm_comprehensive_execution_plan_*.plan.md](../../.cursor/plans/) §C1.
> Do not build until Track A (H-PERP-003) clears or fails cleanly.

## Why this exists

The H-PERP-003 daily-capture cron writes a continuously-growing CSV at
[research/datasets/H-PERP-003/btc_hedged_panel_okx.csv](../../research/datasets/H-PERP-003/btc_hedged_panel_okx.csv)
and a per-pull provenance log at
[research/datasets/H-PERP-003/pull_log.jsonl](../../research/datasets/H-PERP-003/pull_log.jsonl).
Operationally we need to know:

1. **Is the cron healthy?** (last successful pull, rows added, alignment)
2. **What does the funding curve currently look like?** (rolling mean,
   sign distribution, rate decile context)
3. **Are we approaching D1 depth?** (countdown to the 365d gate)

A standalone read-only chart accomplishes all three without depending on
any alpha existing — which is exactly the value proposition Track C is
supposed to support.

## Routing decision

Add a new top-level link to [dashboard/src/app/components/Nav.tsx](../../dashboard/src/app/components/Nav.tsx)
between `MARKET` and `EDGE`:

```diff
 const links = [
   { href: "/", label: "OVERVIEW", icon: "■" },
   { href: "/positions", label: "POSITIONS", icon: "▣" },
   { href: "/crypto", label: "MARKET", icon: "▲" },
+  { href: "/funding", label: "FUNDING", icon: "≈" },
   { href: "/edge-health", label: "EDGE", icon: "◆" },
   { href: "/bot-status", label: "STATUS", icon: "●" },
 ];
```

Page lives at `dashboard/src/app/funding/page.tsx` and consumes a single
JSON document from the API.

## API surface (one new route)

`GET /api/funding/h-perp-003`

Backed by [bot/scripts/funding_dashboard_snapshot.py](../../bot/scripts/funding_dashboard_snapshot.py),
which reads the CSV + JSONL and emits the document below. The API can
either re-run that script on demand (cheap — the CSV is small) or read a
pre-computed JSON written by the daily-capture cron.

Response shape:

```ts
type FundingSnapshotV1 = {
  schema: "funding-snapshot.v1";
  generated_at: string; // ISO8601
  hypothesis: "H-PERP-003";
  data_health: {
    rows: number;
    days_covered: number;
    d1_target_days: 365;
    d1_progress_pct: number;             // 100 * days_covered / 365
    align_ok_pct: number;                // 0..1
    last_pull_at: string | null;         // most recent pull_log entry
    last_pull_rows_added: number | null;
    last_pull_status: "ok" | "stale" | "error";
    consecutive_clean_days: number;      // for the A1 acceptance test
  };
  funding_series: Array<{
    fundingTime: number;                 // unix ms
    fundingRate: number;                 // raw period rate
    rolling_24h_mean: number | null;
  }>;
  decile_summary: {
    deciles: number[];                   // 11 cut points (0..10)
    current_decile: number;              // 0..9
  };
};
```

## Backend sketch — `bot/scripts/funding_dashboard_snapshot.py`

A tiny standalone script (no SQLAlchemy, no FastAPI dep) that builds the
snapshot from the CSV + JSONL. Safe to run inside the daily cron and
serve the resulting JSON statically; or expose under FastAPI when
[api/](../../api) gains a `/funding` route.

The implementation is intentionally tiny — under 150 LOC — so review
overhead is low and there is no duplication of `fetch_hedged_panel.py`.

## Frontend sketch — `dashboard/src/app/funding/page.tsx`

Server component that fetches the snapshot and renders three blocks:

1. **`<DataFreshness />`** banner reusing the existing component, feeding
   it `last_pull_at` and a 26h staleness threshold (one funding cycle +
   2h cron buffer).
2. **`<DepthProgressBar />`** new — `d1_progress_pct` linear bar with the
   numeric label "X / 365 days".
3. **`<FundingChart />`** new — area chart of `fundingRate`, with the
   rolling 24h mean overlay and a horizontal zero line. Reuses the
   styling/colors of [PnlChart.tsx](../../dashboard/src/app/components/PnlChart.tsx).

No new charting dependency required — the existing chart stack already
covers what is needed.

## Why this is a *sketch* and not built

The plan explicitly defers C1 until Track A (H-PERP-003) delivers a
clean verdict (PASS, FAIL, or INCONCLUSIVE-with-stop). Building the
dashboard before then risks two failure modes:

1. **Sunk-cost momentum:** if H-PERP-003 fails Phase 4, a fully built
   dashboard implies the project should "do something with it" rather
   than archive it cleanly.
2. **Stale infrastructure:** the JSON schema above is a guess at what
   matters most. The schema we actually want will become obvious once
   the gates are green.

Build only when one of:

- H-PERP-003 PASSes Phase 4 → dashboard becomes a paper-trading observer.
- H-PERP-003 INCONCLUSIVE/blocked → dashboard tracks the data debt across
  H-PERP-002 / H-SPOT-002 capture cycles too.
