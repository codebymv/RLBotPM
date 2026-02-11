#!/bin/bash
#
# Live Trading Bot - Production Runner
# Runs indefinitely with auto-restart on crash
#
# Usage:
#   ./bot/scripts/run_live_trader.sh              # Run in foreground
#   nohup ./bot/scripts/run_live_trader.sh &      # Run in background
#   ./bot/scripts/run_live_trader.sh 2>&1 | tee live_trading.log  # With logging
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_DIR="$SCRIPT_DIR/.."
cd "$BOT_DIR"

# Load environment from project root
if [ -f "$BOT_DIR/../.env" ]; then
    export $(grep -v '^#' "$BOT_DIR/../.env" | xargs)
fi

# Configuration
MODEL_PATH="models/best_model_run_118"
SYMBOL="BTC-USD"
INTERVAL="1h"
MODE="live"  # "paper" or "live"
MAX_POSITION_PCT="0.50"
DAILY_LOSS_LIMIT="0.10"
TICK_INTERVAL="300"  # 5 minutes between checks
RESTART_DELAY="60"   # Seconds to wait before restart on crash

echo "=============================================="
echo "  RLBot Live Trader - Production Mode"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Symbol: $SYMBOL"
echo "Mode: $MODE"
echo "Started: $(date)"
echo "=============================================="
echo ""

# Auto-restart loop
while true; do
    echo "[$(date)] Starting live trader..."
    
    python -c "
import sys
sys.path.insert(0, '.')

import os

# Load env manually (handles multiline secrets)
env_path = '$SCRIPT_DIR/.env'
if os.path.exists(env_path):
    with open(env_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value

from src.execution.live_trader import LiveTrader

trader = LiveTrader(
    model_path='$MODEL_PATH',
    symbol='$SYMBOL',
    interval='$INTERVAL',
    mode='$MODE',
    max_position_pct=$MAX_POSITION_PCT,
    daily_loss_limit_pct=$DAILY_LOSS_LIMIT,
)

# Run indefinitely (duration_hours=0)
trader.run(duration_hours=0, tick_interval_seconds=$TICK_INTERVAL)
" || true
    
    echo "[$(date)] Trader stopped. Restarting in $RESTART_DELAY seconds..."
    sleep $RESTART_DELAY
done
