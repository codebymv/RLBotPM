param(
    [string]$ModelPath = "models/final_run_165.zip",
    [int]$CryptoDurationHours = 24,
    [double]$CryptoCapital = 1000.0,
    [int]$KalshiIntervalSeconds = 300,
    [double]$KalshiBankroll = 100.0,
    [double]$KalshiMinEdge = 0.01,
    [double]$KalshiMaxEdge = 0.20,
    [int]$KalshiMaxContracts = 10,
    [int]$KalshiMaxPositions = 20,
    [int]$KalshiMaxScans = 100,
    [switch]$Launch
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$botRoot = (Resolve-Path (Join-Path $scriptDir "..")).Path
$python = Join-Path $botRoot "venv\Scripts\python.exe"
$mainPy = Join-Path $botRoot "main.py"

if (!(Test-Path $python)) {
    throw "Python venv not found at $python"
}
if (!(Test-Path $mainPy)) {
    throw "main.py not found at $mainPy"
}

$modelFullPath = Join-Path $botRoot $ModelPath
if (!(Test-Path $modelFullPath)) {
    throw "Model not found at $modelFullPath"
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$sessionDir = Join-Path $botRoot "logs\paper_sessions\$timestamp"
New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null

$cryptoArgs = @(
    "main.py",
    "rl-paper-trade",
    "--model", $ModelPath,
    "--duration", $CryptoDurationHours.ToString(),
    "--capital", $CryptoCapital.ToString(),
    "--interval", "1h",
    "--log-dir", (Join-Path $sessionDir "crypto")
)

$kalshiArgs = @(
    "main.py",
    "kalshi",
    "paper-trade",
    "--demo",
    "--interval", $KalshiIntervalSeconds.ToString(),
    "--bankroll", $KalshiBankroll.ToString(),
    "--min-edge", $KalshiMinEdge.ToString(),
    "--max-edge", $KalshiMaxEdge.ToString(),
    "--max-contracts", $KalshiMaxContracts.ToString(),
    "--max-positions", $KalshiMaxPositions.ToString(),
    "--max-scans", $KalshiMaxScans.ToString()
)

Write-Host "Dual-paper session directory: $sessionDir"
Write-Host ""
Write-Host "Crypto command:"
Write-Host "$python $($cryptoArgs -join ' ')"
Write-Host ""
Write-Host "Kalshi command:"
Write-Host "$python $($kalshiArgs -join ' ')"
Write-Host ""

if (-not $Launch) {
    Write-Host "Dry run only. Re-run with -Launch to start both processes."
    exit 0
}

$cryptoOut = Join-Path $sessionDir "crypto.stdout.log"
$cryptoErr = Join-Path $sessionDir "crypto.stderr.log"
$kalshiOut = Join-Path $sessionDir "kalshi.stdout.log"
$kalshiErr = Join-Path $sessionDir "kalshi.stderr.log"

$cryptoProc = Start-Process `
    -FilePath $python `
    -ArgumentList $cryptoArgs `
    -WorkingDirectory $botRoot `
    -RedirectStandardOutput $cryptoOut `
    -RedirectStandardError $cryptoErr `
    -PassThru

$kalshiProc = Start-Process `
    -FilePath $python `
    -ArgumentList $kalshiArgs `
    -WorkingDirectory $botRoot `
    -RedirectStandardOutput $kalshiOut `
    -RedirectStandardError $kalshiErr `
    -PassThru

Write-Host "Launched crypto PID: $($cryptoProc.Id)"
Write-Host "Launched kalshi PID: $($kalshiProc.Id)"
Write-Host "Logs:"
Write-Host "  $cryptoOut"
Write-Host "  $cryptoErr"
Write-Host "  $kalshiOut"
Write-Host "  $kalshiErr"
