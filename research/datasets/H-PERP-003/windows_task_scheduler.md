# H-PERP-003 daily capture — Windows Task Scheduler setup

> Alternative to `.github/workflows/h-perp-003-capture.yml` for operators who
> prefer to run the capture from the local Windows machine instead of GitHub
> Actions. Pick **one** scheduler — running both will produce duplicate
> commits / no-op runs but no data corruption (the script is idempotent).

## Prerequisites

- Python 3.11+ on PATH.
- `requests` installed (`pip install requests`).
- Repo cloned at a stable path, e.g. `C:\Users\roxas\OneDrive\Desktop\PROJECTS\RLBotPM`.

## One-shot install (PowerShell, run as your normal user)

```powershell
$repo = "C:\Users\roxas\OneDrive\Desktop\PROJECTS\RLBotPM"
$python = (Get-Command python).Source
$script = Join-Path $repo "research\datasets\H-PERP-003\daily_capture.py"
$logDir = Join-Path $repo "research\datasets\H-PERP-003\.task_logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$action = New-ScheduledTaskAction `
  -Execute $python `
  -Argument "`"$script`"" `
  -WorkingDirectory $repo

$trigger = New-ScheduledTaskTrigger -Daily -At 00:30

$settings = New-ScheduledTaskSettingsSet `
  -StartWhenAvailable `
  -DontStopIfGoingOnBatteries `
  -AllowStartIfOnBatteries `
  -ExecutionTimeLimit (New-TimeSpan -Minutes 15)

Register-ScheduledTask `
  -TaskName "RLBotPM-HPERP003-DailyCapture" `
  -Description "Append-only OKX hedged-panel pull for H-PERP-003 (Track A1)." `
  -Action $action `
  -Trigger $trigger `
  -Settings $settings `
  -RunLevel Limited
```

## Verify

```powershell
Get-ScheduledTask -TaskName "RLBotPM-HPERP003-DailyCapture" | Get-ScheduledTaskInfo
Start-ScheduledTask -TaskName "RLBotPM-HPERP003-DailyCapture"
```

Then read the latest line of the pull log:

```powershell
Get-Content `
  "C:\Users\roxas\OneDrive\Desktop\PROJECTS\RLBotPM\research\datasets\H-PERP-003\pull_log.jsonl" `
  -Tail 1
```

## Uninstall

```powershell
Unregister-ScheduledTask -TaskName "RLBotPM-HPERP003-DailyCapture" -Confirm:$false
```

## Notes

- The task runs the script even if the laptop was off at 00:30 UTC
  (`-StartWhenAvailable`) — next wake will trigger a missed run.
- The script is idempotent (`fundingTime`-keyed dedup), so an extra run is
  always safe.
- Acceptance criterion (architecture-audit-03 §A1): 7 consecutive days where
  `pull_log.jsonl` shows `ok: true` and `align_ok_pct >= 99`.
- This setup does **not** auto-commit to git. Operator either runs `git commit`
  manually after inspecting the diff, or relies on the GitHub Actions workflow
  for committing.
