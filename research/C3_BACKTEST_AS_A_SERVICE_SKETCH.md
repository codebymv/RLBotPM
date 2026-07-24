# C3 Sketch — Backtest-as-a-Service

> **Status (2026-05-04):** SKETCH ONLY, deferred per
> [.cursor/plans/rlbotpm_comprehensive_execution_plan_*.plan.md](../../.cursor/plans/) §C3.
> **Hard precondition:** **DO NOT BUILD** until H-PERP-003 actually passes
> Phase 4. Shipping a paid backtest service while the only candidate
> hypothesis has FAILed is selling a negative result.

## Why this exists (conditionally)

The H-PERP-003 backtest in
[research/backtests/H-PERP-003.py](backtests/H-PERP-003.py) packages
something most retail quants cannot replicate cleanly:

- A pre-registered design (`06`) frozen before evaluation.
- A data contract (`05`/D1–D4) that is structurally enforced by the
  script (the script returns `INCONCLUSIVE_DATA` when D1 is missed).
- A non-degenerate placebo (`G6` random sign-flip).
- A fee model that is fixed per design, not after seeing results.

If H-PERP-003 PASSes, this becomes a credible product: "upload your
hedged-carry CSV, get back a verdict against the same gates that gated
our PASS." The value is the *gate logic and discipline*, not the alpha.

## Hard precondition

This sketch is **null and void** unless one of:

1. H-PERP-003 PASSes Phase 4. Then BaaS becomes a "verify your own carry
   strategy against our PASSed gate set" product.
2. H-PERP-003 INCONCLUSIVE-with-clean-stop AND a follow-on hypothesis
   (H-PERP-002 rate-decile event study) PASSes. Then BaaS rebrands
   around whichever survived.

If neither happens, **archive this sketch** and never build it. Selling a
gate suite that the originating hypothesis failed against is a credibility
trap.

## Scope (intentional minimum)

- One HTTP endpoint: `POST /v1/backtest/h-perp-003`.
- One auth model: API key in header, no OAuth.
- One pricing tier (or free during beta).
- No dashboard, no user accounts UI — just a JSON API + a static landing
  page with the "what this does NOT do" disclaimers.

## API surface

`POST /v1/backtest/h-perp-003`

Request body:

```json
{
  "csv": "<base64-encoded user CSV, schema = same as btc_hedged_panel_okx.csv>",
  "options": {
    "bootstrap_trials": 5000,
    "seed": 1337
  }
}
```

Response body (mirrors
[research/backtests/H-PERP-003_metrics.json](backtests/H-PERP-003_metrics.json)
verbatim, with provenance fields added):

```json
{
  "schema": "h-perp-003-verdict.v1",
  "verdict": "PASS" | "FAIL" | "INCONCLUSIVE_DATA",
  "verdict_reason": "...",
  "data_contract": {
    "d1_days": 412.7,
    "d4_bad_weeks": 0
  },
  "gates": {
    "G1_oos_sharpe": 1.84,
    "G2_oos_pf": 1.62,
    "G3_fee_stress_cum_pnl": 12.4,
    "G4_segments_positive": "3/3",
    "G5_max_interval_share": 0.18,
    "G6_signflip_frac_beats": 0.012
  },
  "provenance": {
    "design_doc_sha": "<sha of 06_backtest_design_H-PERP-003.md at server build time>",
    "backtest_sha":   "<sha of backtests/H-PERP-003.py at server build time>",
    "service_version": "0.1.0"
  }
}
```

The `provenance.*sha` fields make it impossible to silently change the
gate set without bumping `service_version` — which is the same
pre-registration discipline as the source repo.

## Implementation sketch

`api/baas/main.py` (new submodule of the existing
[api/](../../api) FastAPI app). About 80 LOC because the heavy lifting
already exists in the script.

```python
# api/baas/main.py
from fastapi import APIRouter, HTTPException
from pathlib import Path
import base64, subprocess, tempfile, json

router = APIRouter(prefix="/v1/backtest")
SCRIPT = Path(__file__).resolve().parents[2] / "research" / "backtests" / "H-PERP-003.py"

@router.post("/h-perp-003")
def run_h_perp_003(payload: dict) -> dict:
    csv_b64 = payload.get("csv")
    if not csv_b64:
        raise HTTPException(400, "missing 'csv' field (base64 user CSV)")
    csv_bytes = base64.b64decode(csv_b64)
    opts = payload.get("options") or {}
    with tempfile.TemporaryDirectory() as td:
        csv_path = Path(td) / "panel.csv"
        csv_path.write_bytes(csv_bytes)
        out_path = Path(td) / "metrics.json"
        cmd = [
            "python", str(SCRIPT),
            "--csv", str(csv_path),
            "--out", str(out_path),
            "--bootstrap-trials", str(opts.get("bootstrap_trials", 5000)),
            "--seed", str(opts.get("seed", 1337)),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if not out_path.exists():
            raise HTTPException(500, f"backtest failed: {proc.stderr[-500:]}")
        result = json.loads(out_path.read_text())
    result["provenance"] = _provenance()  # fill in shas + version
    return result
```

The `H-PERP-003.py` script already accepts `--csv` / `--out` /
`--bootstrap-trials` / `--seed`, so no change is needed there for the
sketch. (Verify this is still true at build time; the script's argparse
section is the source of truth.)

## Operational sketch

- Run inside a small container (Python 3.13 + the script's pure-stdlib
  deps). 256MB RAM is plenty.
- Rate-limit at the gateway (e.g. 5 req/min per key) — the backtest
  itself is CPU-bound but small.
- Cap `--bootstrap-trials` at 100k server-side to bound runtime.
- Log every request hash + verdict to a metrics DB for capacity planning
  and abuse triage.

## Hard "no-go" checklist before launch

- [ ] H-PERP-003 has a written PASS verdict in
      [07_backtest_results_H-PERP-003.md](07_backtest_results_H-PERP-003.md).
- [ ] [10_fundability_review_H-PERP-003.md](10_fundability_review_H-PERP-003.md)
      has zero blocking NO.
- [ ] Landing page explicitly states "this is not investment advice and
      a PASS verdict is necessary but not sufficient for live capital."
- [ ] Service version is pinned in `provenance.service_version` so the
      next gate change forces a bump.
- [ ] Existing repo's pre-registration enforcement
      ([NEXT_HYPOTHESIS.md](NEXT_HYPOTHESIS.md)) is mirrored in the
      service repo.

## Why this is a *sketch* and not built

Same root cause as C1/C2: until A5 PASSes, nothing here is honest to
sell. Building the API, landing page, billing, etc., before a verdict
exists is sunk-cost momentum disguised as productivity.

The most expensive failure mode for this project is shipping
infrastructure for a strategy that does not work. Track A's job is to
gate that decision; Track C must defer to it.
