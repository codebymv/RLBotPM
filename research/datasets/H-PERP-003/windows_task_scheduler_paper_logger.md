# H-PERP-003 paper logger — Windows Task Scheduler setup

> Phase 5 paper observation needs a continuous 30-day / 90-snapshot window
> (see [`08_paper_protocol_H-PERP-003.md`](../../08_paper_protocol_H-PERP-003.md)).
> Backgrounding `run_h_perp_003_paper_logger.py` from a terminal is fragile —
> Cursor restarts, laptop sleeps, and reboots will silently kill the run and
> invalidate the wallclock count.
>
> The PowerShell installer at
> [`bot/scripts/install_h_perp_003_paper_task.ps1`](../../../bot/scripts/install_h_perp_003_paper_task.ps1)
> registers a real Windows Scheduled Task that survives all of those.

This is the paper-side companion to
[`windows_task_scheduler.md`](windows_task_scheduler.md), which covers the
once-a-day capture cron. They are independent and may both be installed.

## Prerequisites

- Python 3.11+ on PATH (`pythonw.exe` strongly preferred — the installer
  auto-detects it next to `python.exe`).
- `requests` installed.
- Repo cloned at a stable path, e.g.
  `C:\Users\roxas\OneDrive\Desktop\PROJECTS\RLBotPM`.
- Phase 4 verdict in [`07_backtest_results_H-PERP-003.md`](../../07_backtest_results_H-PERP-003.md)
  is **PASS** (already true 2026-05-05 UTC).

No elevation required. The task runs as the current interactive user with
`-RunLevel Limited`.

## Install

From the repo root:

```powershell
pwsh -File bot\scripts\install_h_perp_003_paper_task.ps1 -Action install
```

What this does:

| Knob | Value |
|------|-------|
| Task name | `RLBotPM-HPERP003-PaperLogger` |
| Triggers | `AtLogOn` (current user) — `AtStartup` requires elevation, see "Caveats" |
| Action | `cmd.exe /c "<pythonw> <runner> --interval 300 >> bot/logs/h_perp_003_paper_task.log 2>&1"` |
| Working directory | `bot/` |
| Time limit | none (long-running) |
| On failure | restart every 5 min, up to 999 times |
| Multiple instances | `IgnoreNew` (the runner is a singleton poll loop) |
| Power policy | `DontStopIfGoingOnBatteries`, `StartWhenAvailable` |

The installer also calls `Start-ScheduledTask` so the logger is running
within ~2 seconds of install.

## Verify

```powershell
pwsh -File bot\scripts\install_h_perp_003_paper_task.ps1 -Action verify
```

Expected output (excerpt):

```
Task        : RLBotPM-HPERP003-PaperLogger
State       : Running
LastRunTime : <recent UTC timestamp>
LastResult  : 0x00000000 (0)
NextRunTime : <triggered>
Runner log  : C:\...\RLBotPM\bot\logs\h_perp_003_paper_task.log
--- last 10 log lines ---
[2026-05-04 17:45:50] INFO     [__main__:33] H-PERP-003 paper logger scan 0 complete
...
```

Then confirm new funding boundaries are being appended:

```powershell
Get-Content "C:\Users\roxas\OneDrive\Desktop\PROJECTS\RLBotPM\bot\logs\paper_research_H-PERP-003.jsonl" -Tail 1
```

## Restart / stop

```powershell
pwsh -File bot\scripts\install_h_perp_003_paper_task.ps1 -Action start
pwsh -File bot\scripts\install_h_perp_003_paper_task.ps1 -Action stop
```

## Uninstall

```powershell
pwsh -File bot\scripts\install_h_perp_003_paper_task.ps1 -Action uninstall
```

## Caveats

- `AtStartup` triggers require elevation to register. The installer drops it
  and uses `AtLogOn` only so it can be run without UAC. Practical impact:
  after a reboot, the logger resumes the moment the user logs in (typically
  within a minute on a personal laptop). If you want true headless coverage
  across reboots without a logon, re-run the installer from an elevated
  PowerShell and re-add the `AtStartup` trigger by hand, or convert the task
  to run as `SYSTEM`.
- `LastResult = 0x00041301` is the documented Win32 status
  `SCHED_S_TASK_RUNNING`. It means "the task is currently running" — not an
  error. Failed runs surface as `0x00000001` or higher; treat any non-zero
  value other than `0x00041301` as a real failure to investigate.

## Operator notes

- The runner is **idempotent**: it only writes a new JSONL row when OKX
  exposes a new `fundingTime`. It is safe to run, restart, or double-trigger
  without producing duplicate rows.
- A laptop sleep across the 8-hour funding boundary is fine: the runner will
  catch up on the next poll. Phase 5's tracking-error gate (`08 §Tracking
  vs backtest`) verifies daily, not hourly.
- If the task shows `LastResult` other than `0`, read the runner log first:
  `Get-Content "<repo>\bot\logs\h_perp_003_paper_task.log" -Tail 50`. The
  most common cause is a transient OKX 5xx, which the runner already logs
  via `WARNING` and recovers from on the next poll.
- Acceptance criterion (`08 §Phase 5 minimum`): 30 calendar days OR
  90 funding snapshots in `paper_research_H-PERP-003.jsonl`, whichever
  comes first being the *tighter* of the two.
