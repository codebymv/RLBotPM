#!/usr/bin/env python3
"""
Pre-registration guard — refuse a commit/PR that bundles a `06_*` design
edit with the matching `07_*` results.

Mode:
  - Default: inspect `git diff --name-only HEAD~1 HEAD` and fail if any
    pair (`06_backtest_design_<id>.md`, `07_backtest_results_<id>.md`)
    appears together for the same `<id>`.
  - With `--base <sha>`: inspect the diff between `<sha>` and `HEAD`
    instead — useful in CI where you want to compare against the merge
    base of `main`.

Exit codes:
  0  no violation
  1  violation found (one or more bundled <id>)
  2  invocation error (missing git, etc.)

This is the canonical implementation referenced by the C2 sketch
([research/C2_PREREGISTRATION_TEMPLATE_SKETCH.md](../../research/C2_PREREGISTRATION_TEMPLATE_SKETCH.md))
and by the cross-cutting rule in
[research/NEXT_HYPOTHESIS.md](../../research/NEXT_HYPOTHESIS.md)
"Pre-registration enforcement".
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable

DESIGN_RE = re.compile(r"^research/06_backtest_design_(?P<id>[A-Za-z0-9_-]+)\.md$")
RESULT_RE = re.compile(r"^research/07_backtest_results_(?P<id>[A-Za-z0-9_-]+)\.md$")


def _git_diff_names(base: str | None) -> list[str]:
    if base:
        cmd = ["git", "diff", "--name-only", base, "HEAD"]
    else:
        cmd = ["git", "diff", "--name-only", "HEAD~1", "HEAD"]
    try:
        out = subprocess.check_output(cmd, text=True)
    except FileNotFoundError:
        print("ERROR: git not found on PATH", file=sys.stderr)
        sys.exit(2)
    except subprocess.CalledProcessError as exc:
        print(f"ERROR: git diff failed: {exc}", file=sys.stderr)
        sys.exit(2)
    return [line.strip().replace("\\", "/") for line in out.splitlines() if line.strip()]


def _bundle_violations(paths: Iterable[str]) -> list[str]:
    designs: dict[str, str] = {}
    results: dict[str, str] = {}
    for p in paths:
        m = DESIGN_RE.match(p)
        if m:
            designs[m.group("id")] = p
            continue
        m = RESULT_RE.match(p)
        if m:
            results[m.group("id")] = p
    return sorted(set(designs) & set(results))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base",
        default=None,
        help="Compare HEAD against this base sha/branch (default: HEAD~1)",
    )
    parser.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help=(
            "Override git: pass an explicit list of touched paths. Useful "
            "for unit tests of this script."
        ),
    )
    args = parser.parse_args()

    paths = args.paths if args.paths is not None else _git_diff_names(args.base)
    violations = _bundle_violations(paths)
    if not violations:
        print("OK: no pre-registration violation in this diff.")
        return 0
    print(
        "FAIL: the following hypothesis ids have BOTH 06_* and 07_* changes "
        "in the same diff:",
        file=sys.stderr,
    )
    for hid in violations:
        print(f"  - {hid}", file=sys.stderr)
    print(
        "\nFix: split the diff into two commits — `pre-register: ...` for the\n"
        "06_* spec change, then a separate commit for the 07_* result.\n"
        "See research/NEXT_HYPOTHESIS.md 'Pre-registration enforcement'.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
