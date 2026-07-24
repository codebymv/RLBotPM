# C2 Sketch — Pre-Registration Template Repo

> **Status (2026-05-04):** SKETCH ONLY, deferred per
> [.cursor/plans/rlbotpm_comprehensive_execution_plan_*.plan.md](../../.cursor/plans/) §C2.
> Do not extract / publish until Track A clears or fails cleanly.

## Why this exists

The internal `06 → 05 → 07 → 08 → 10` workflow used in
[research/](.) is the single highest-leverage process artifact this
project produces. It is what kept H-SPOT-001 honest (`07` was a clean
FAIL) and what protected H-PERP-003 from G6 hindsight bias (see
[06_backtest_design_H-PERP-003.md](06_backtest_design_H-PERP-003.md) §5
amendment, [NEXT_HYPOTHESIS.md](NEXT_HYPOTHESIS.md) "Pre-registration
enforcement").

A standalone GitHub template repo distills this into something a
self-funded quant or a small research team can adopt without copying
crypto-specific code. It is also Track C's lowest-risk productization:
zero claim of alpha, just a workflow others can audit.

## Scope (intentional non-features)

This template **does not**:

- Provide a backtest engine. Users plug in their own.
- Pretend to be a profitability claim. The README explicitly says it is
  process scaffolding, not strategy.
- Wrap any specific exchange API. The data contract section is templated.

## Repo layout

```
research-preregistration-template/
├── README.md                      # 1-page guide + the "why" + the rule
├── LICENSE                        # MIT
├── .github/
│   └── workflows/
│       ├── pre-register-guard.yml # CI rejects PRs that bundle 06_*+07_*
│       └── timestamp.yml          # Stamps each 06_*/07_* commit
├── docs/
│   ├── workflow.md                # The 06→05→07→08→10 loop
│   ├── pre-registration.md        # The rule + canonical examples
│   └── audit-trail.md             # How to read your own history
└── research/
    ├── _templates/
    │   ├── 04_hypothesis_library.md
    │   ├── 05_data_quality_<ID>.md
    │   ├── 06_backtest_design_<ID>.md
    │   ├── 07_backtest_results_<ID>.md
    │   ├── 08_paper_protocol_<ID>.md
    │   ├── 10_fundability_review_<ID>.md
    │   └── architecture-audit-NN.md
    └── EXAMPLE_HYPOTHESIS/        # Toy hypothesis demonstrating the loop
        ├── 06_backtest_design_EXAMPLE-001.md
        ├── 05_data_quality_EXAMPLE-001.md
        ├── 07_backtest_results_EXAMPLE-001.md
        └── datasets/EXAMPLE-001/.gitkeep
```

## The single workflow rule (copied verbatim from this repo)

> Every change to a `06_backtest_design_<id>.md` file MUST be timestamped
> and land in the repo **before** the next run of the corresponding
> backtest. Reorder violation (results first → spec amended to fit) is
> the single most expensive research bug we can ship. Concretely:
>
> - Spec edits go in their own commit, separate from any `07_*` results commit.
> - The commit message includes the phrase `pre-register:` and lists the
>   gate(s) changed.
> - The commit lands strictly before the timestamp on the next `07_*`
>   write or metrics JSON refresh. CI / reviewer should reject a PR that
>   bundles them in one commit.

This is the canonical text from
[NEXT_HYPOTHESIS.md](NEXT_HYPOTHESIS.md) "Pre-registration enforcement".
The template repo's README opens with it.

## CI guard sketch — `.github/workflows/pre-register-guard.yml`

The guard fails PRs that touch a `06_backtest_design_*.md` AND the
matching `07_backtest_results_*.md` in the same commit. It runs on every
PR and is the structural enforcement of the rule above.

A minimal Python implementation lives at
[bot/scripts/check_preregistration.py](../../bot/scripts/check_preregistration.py)
in this repo and is reused verbatim by the template (a single 60-LOC
script, no extra deps beyond `git`).

## Templates worth highlighting

### `06_backtest_design_<ID>.md`

The most important template. It enforces the structure that makes the
gate set checkable:

- Data contract (D1–D4 named, no free text "we'll figure it out later").
- Instruments table.
- PnL formula in code-block math, frozen.
- Walk-forward partition.
- Evidence gates G1..GN with explicit thresholds.
- A "no fee/window/formula changes after this file is used to score
  results" sticky line at the top.

The H-PERP-003 `06` is the canonical worked example, including a
properly-pre-registered amendment (G6 random sign-flip).

### `architecture-audit-NN.md`

Reused from the audit series in this repo
([architecture-audit-00.md](architecture-audit-00.md) → -03). The
template emphasizes that audits are *findings* docs and operating-plan
docs live in their own file (`-03` here), not in a single sprawling
audit.

## Adoption funnel

The README documents three adoption paths from low to high commitment:

1. **Read-only:** copy the rule and the template into an existing repo.
2. **GitHub template:** click "Use this template" to fork, then start
   filling in `_templates/`.
3. **Hard adoption:** add the CI guard and the timestamp workflow to an
   existing private research repo.

## Why this is a *sketch* and not built

Same reasoning as [C1](C1_FUNDING_DASHBOARD_SKETCH.md): a public
template will get judged by whoever finds it. Better to publish *after*
this repo has run the workflow on a real PASS or a real clean FAIL,
because the README can then say "this workflow produced verdict X on
hypothesis Y" rather than "we hope this works".

Build only when:

- H-PERP-003 has a written PASS / FAIL / INCONCLUSIVE verdict, AND
- The audit-03 RL Track B has produced a clean run-174 candidate
  (PASS or honest FAIL on the new gates).

Both states give the template repo a real worked example to ship with.
