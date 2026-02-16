# RLTrade Paper Trading Startup Script
# This script starts all necessary components for paper trading

$ErrorActionPreference = "Stop"

Write-Host "🚀 RLTrade Paper Trading Startup" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the right directory
if (-not (Test-Path "bot\main.py")) {
    Write-Host "❌ Error: Must run from RLTrade root directory" -ForegroundColor Red
    exit 1
}

# Check environment
Write-Host "📋 Checking environment..." -ForegroundColor Yellow
if (-not (Test-Path ".env")) {
    Write-Host "❌ .env file not found!" -ForegroundColor Red
    Write-Host "   Please create .env with KALSHI_API_KEY and KALSHI_API_SECRET" -ForegroundColor Red
    exit 1
}

# Check API keys
$envContent = Get-Content .env -Raw
if ($envContent -notmatch "KALSHI_API_KEY" -or $envContent -notmatch "KALSHI_API_SECRET") {
    Write-Host "⚠️  Warning: KALSHI API keys not found in .env" -ForegroundColor Yellow
    Write-Host "   Paper trading may fail without API credentials" -ForegroundColor Yellow
    Start-Sleep -Seconds 3
}

Write-Host "✅ Environment OK" -ForegroundColor Green
Write-Host ""

# Start API Server
Write-Host "🔧 Starting API server..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\api'; python main.py" -WindowStyle Normal
Start-Sleep -Seconds 3
Write-Host "✅ API server started (http://localhost:8000)" -ForegroundColor Green
Write-Host ""

# Start Dashboard
Write-Host "🖥️  Starting dashboard..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$PWD\dashboard'; npm run dev" -WindowStyle Normal
Start-Sleep -Seconds 5
Write-Host "✅ Dashboard started (http://localhost:3000)" -ForegroundColor Green
Write-Host ""

# Ask user for paper trading settings
Write-Host "📊 Paper Trading Configuration" -ForegroundColor Cyan
Write-Host "==============================" -ForegroundColor Cyan
Write-Host ""

$bankroll = Read-Host "Starting capital (default: 100)"
if ([string]::IsNullOrWhiteSpace($bankroll)) { $bankroll = "100" }

$interval = Read-Host "Scan interval in seconds (default: 300)"
if ([string]::IsNullOrWhiteSpace($interval)) { $interval = "300" }

$mode = Read-Host "API mode - live or demo? (default: live)"
if ([string]::IsNullOrWhiteSpace($mode)) { $mode = "live" }

Write-Host ""
Write-Host "🎯 Starting paper trading with:" -ForegroundColor Green
Write-Host "   Capital: `$$bankroll" -ForegroundColor White
Write-Host "   Interval: $interval seconds" -ForegroundColor White
Write-Host "   Mode: $mode" -ForegroundColor White
Write-Host ""
Write-Host "⚠️  This will run continuously until stopped with Ctrl+C" -ForegroundColor Yellow
Write-Host ""
$confirm = Read-Host "Start paper trading? (y/n)"

if ($confirm -eq "y" -or $confirm -eq "Y" -or $confirm -eq "yes") {
    Write-Host ""
    Write-Host "🤖 Starting Kalshi paper trader..." -ForegroundColor Cyan
    Write-Host ""
    
    # Build command
    $modeFlag = if ($mode -eq "live") { "--live" } else { "--demo" }
    $command = "cd '$PWD\bot'; python main.py kalshi paper-trade --interval $interval --bankroll $bankroll $modeFlag"
    
    # Start paper trading
    Start-Process powershell -ArgumentList "-NoExit", "-Command", $command -WindowStyle Normal
    
    Write-Host ""
    Write-Host "✅ All systems running!" -ForegroundColor Green
    Write-Host ""
    Write-Host "📊 Monitoring URLs:" -ForegroundColor Cyan
    Write-Host "   Dashboard: http://localhost:3000" -ForegroundColor White
    Write-Host "   API Docs:  http://localhost:8000/docs" -ForegroundColor White
    Write-Host ""
    Write-Host "📁 Log files:" -ForegroundColor Cyan
    Write-Host "   Paper trades: bot\logs\paper_trades.jsonl" -ForegroundColor White
    Write-Host "   Bot logs:     bot\logs\" -ForegroundColor White
    Write-Host ""
    Write-Host "💡 Commands:" -ForegroundColor Cyan
    Write-Host "   Check status:  python bot\main.py kalshi paper-status" -ForegroundColor White
    Write-Host "   View logs:     Get-Content bot\logs\paper_trades.jsonl -Tail 20" -ForegroundColor White
    Write-Host ""
    Write-Host "⏸️  To stop: Close each PowerShell window or press Ctrl+C" -ForegroundColor Yellow
    Write-Host ""
    
} else {
    Write-Host ""
    Write-Host "❌ Paper trading cancelled" -ForegroundColor Red
    Write-Host ""
    Write-Host "⚠️  Note: API server and dashboard are still running" -ForegroundColor Yellow
    Write-Host "   Close their PowerShell windows to stop them" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "Press any key to exit..."
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
