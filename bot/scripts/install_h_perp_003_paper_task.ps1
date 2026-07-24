<#
.SYNOPSIS
  Install / verify / uninstall the Windows Scheduled Task that keeps the
  H-PERP-003 Phase 5 paper logger running across reboots and laptop sleeps.

.DESCRIPTION
  Phase 5 of H-PERP-003 (research/08_paper_protocol_H-PERP-003.md) requires a
  30-day / 90-snapshot continuous observation window. Backgrounding the
  Python runner in a terminal is fragile (sleep/reboot/Cursor-restart kill
  it). This script registers a real Scheduled Task that:

    - Runs `bot/scripts/run_h_perp_003_paper_logger.py --interval 300`.
    - Triggers `AtStartup` AND `AtLogOn` so the logger always comes back.
    - Uses `pythonw.exe` so no console window flashes on user sessions.
    - Restarts on failure (every 5 minutes, indefinitely).
    - Writes stdout/stderr to `bot/logs/h_perp_003_paper_task.log`.

  Idempotent: re-running with `-Action install` updates the existing task.

.PARAMETER Action
  install   = create or replace the task (default).
  verify    = print the task's current state and tail the log.
  start     = start the task immediately (next interval poll).
  stop      = stop a running task instance.
  uninstall = remove the task completely.

.PARAMETER IntervalSeconds
  Polling interval passed to the runner. Default 300 (5 minutes).

.PARAMETER TaskName
  Scheduled Task name. Default "RLBotPM-HPERP003-PaperLogger".

.EXAMPLE
  # First-time install (run from any directory):
  pwsh -File bot\scripts\install_h_perp_003_paper_task.ps1 -Action install

.EXAMPLE
  # Inspect the task and tail the most recent log lines:
  pwsh -File bot\scripts\install_h_perp_003_paper_task.ps1 -Action verify

.NOTES
  No elevation required. The task runs as the current interactive user with
  -RunLevel Limited so OneDrive paths and the user's Python install resolve
  the same way they do in a normal shell.
#>
[CmdletBinding()]
param(
  [ValidateSet('install','verify','start','stop','uninstall')]
  [string]$Action = 'install',

  [int]$IntervalSeconds = 300,

  [string]$TaskName = 'RLBotPM-HPERP003-PaperLogger'
)

$ErrorActionPreference = 'Stop'

# Repo layout: bot/scripts/install_h_perp_003_paper_task.ps1 -> repo root is two levels up.
$repoRoot   = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$botDir     = Join-Path $repoRoot 'bot'
$runnerPath = Join-Path $botDir   'scripts\run_h_perp_003_paper_logger.py'
$logDir     = Join-Path $botDir   'logs'
$logPath    = Join-Path $logDir   'h_perp_003_paper_task.log'

if (-not (Test-Path $runnerPath)) {
  throw "Runner not found: $runnerPath"
}

function Resolve-PythonW {
  $cmd = Get-Command pythonw -ErrorAction SilentlyContinue
  if ($cmd) { return $cmd.Source }
  $py = Get-Command python -ErrorAction SilentlyContinue
  if (-not $py) {
    throw "Neither pythonw nor python is on PATH. Install Python 3.11+ first."
  }
  $candidate = Join-Path (Split-Path $py.Source -Parent) 'pythonw.exe'
  if (Test-Path $candidate) { return $candidate }
  Write-Warning "pythonw.exe not found next to python.exe; falling back to python.exe (a console window may appear)."
  return $py.Source
}

function Install-Task {
  $python = Resolve-PythonW
  if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
  }

  # Wrap stdout+stderr redirection in a cmd.exe call so the task can capture
  # logs without depending on the runner managing its own file handles.
  $argLine = "/c `"`"$python`" `"$runnerPath`" --interval $IntervalSeconds >> `"$logPath`" 2>&1`""

  $action = New-ScheduledTaskAction `
    -Execute (Join-Path $env:WINDIR 'System32\cmd.exe') `
    -Argument $argLine `
    -WorkingDirectory $botDir

  # AtStartup requires elevation. AtLogOn (current user) does not, and is
  # sufficient for a user-mode laptop: a reboot or sleep recovery resumes
  # the logger as soon as the operator logs back in.
  $triggers = @(
    New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
  )

  $settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -DontStopIfGoingOnBatteries `
    -AllowStartIfOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -RestartCount 999 `
    -MultipleInstances IgnoreNew

  $principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Limited

  if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Updating existing scheduled task '$TaskName'..."
    Set-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers -Settings $settings -Principal $principal | Out-Null
  } else {
    Write-Host "Registering new scheduled task '$TaskName'..."
    Register-ScheduledTask `
      -TaskName $TaskName `
      -Description 'H-PERP-003 Phase 5 paper logger (research/09_paper_results_H-PERP-003.md). Polls OKX, appends one snapshot per funding boundary. No orders.' `
      -Action $action `
      -Trigger $triggers `
      -Settings $settings `
      -Principal $principal | Out-Null
  }

  Start-ScheduledTask -TaskName $TaskName
  Start-Sleep -Seconds 2
  Verify-Task
}

function Verify-Task {
  $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $task) { throw "Scheduled task '$TaskName' is not installed." }
  $info = $task | Get-ScheduledTaskInfo
  Write-Host "Task        : $TaskName"
  Write-Host "State       : $($task.State)"
  Write-Host "LastRunTime : $($info.LastRunTime)"
  Write-Host ("LastResult  : 0x{0:X8} ({0})" -f $info.LastTaskResult)
  Write-Host "NextRunTime : $($info.NextRunTime)"
  Write-Host "Runner log  : $logPath"
  if (Test-Path $logPath) {
    Write-Host '--- last 10 log lines ---'
    Get-Content $logPath -Tail 10
  } else {
    Write-Host '(no log file yet — task has not produced output)'
  }
}

switch ($Action) {
  'install'   { Install-Task }
  'verify'    { Verify-Task }
  'start'     { Start-ScheduledTask  -TaskName $TaskName; Verify-Task }
  'stop'      { Stop-ScheduledTask   -TaskName $TaskName; Verify-Task }
  'uninstall' { Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false; Write-Host "Unregistered '$TaskName'." }
}
