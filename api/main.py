"""
FastAPI Backend for RLTrade Monitoring

This API provides endpoints for:
- Querying training runs and performance
- Real-time training metrics
- Trade history and analysis
- Risk management status
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List, Optional
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="RLTrade API",
    description="Monitoring and control API for RL Trading Bot",
    version="0.1.0"
)

# Database connection (for data source health)
DATABASE_URL = os.getenv("DATABASE_URL")
DB_ENGINE = create_engine(DATABASE_URL) if DATABASE_URL else None

# Configure CORS
origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - API health check"""
    return {
        "status": "online",
        "service": "RLTrade API",
        "version": "0.1.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not DB_ENGINE:
        return {
            "status": "degraded",
            "database": "not_configured",
            "timestamp": datetime.utcnow().isoformat(),
        }
    try:
        with DB_ENGINE.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
        status = "healthy"
    except Exception:
        db_status = "error"
        status = "degraded"

    return {
        "status": status,
        "database": db_status,
        "timestamp": datetime.utcnow().isoformat()
    }


# Training endpoints
@app.get("/api/training/runs")
async def get_training_runs(
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0)
):
    """
    Get list of training runs
    
    Args:
        limit: Maximum number of runs to return
        offset: Offset for pagination
    
    Returns:
        List of training runs with metadata
    """
    # TODO: Implement database query
    return {
        "runs": [
            {
                "id": 1,
                "started_at": datetime.utcnow().isoformat(),
                "status": "running",
                "total_episodes": 1000,
                "best_sharpe_ratio": 1.23
            }
        ],
        "total": 1,
        "limit": limit,
        "offset": offset
    }


@app.get("/api/training/runs/{run_id}")
async def get_training_run(run_id: int):
    """
    Get detailed information about a specific training run
    
    Args:
        run_id: Training run ID
    
    Returns:
        Training run details
    """
    # TODO: Implement database query
    return {
        "id": run_id,
        "started_at": datetime.utcnow().isoformat(),
        "status": "running",
        "total_episodes": 1000,
        "best_sharpe_ratio": 1.23,
        "config": {}
    }


@app.get("/api/training/runs/{run_id}/episodes")
async def get_episodes(
    run_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """
    Get episodes for a training run
    
    Args:
        run_id: Training run ID
        limit: Maximum episodes to return
        offset: Offset for pagination
    
    Returns:
        List of episodes with metrics
    """
    # TODO: Implement database query
    return {
        "episodes": [],
        "total": 0,
        "limit": limit,
        "offset": offset
    }


@app.get("/api/training/runs/{run_id}/metrics")
async def get_training_metrics(run_id: int):
    """
    Get aggregated metrics for a training run
    
    Args:
        run_id: Training run ID
    
    Returns:
        Aggregated training metrics
    """
    # TODO: Implement metrics calculation
    return {
        "run_id": run_id,
        "total_episodes": 1000,
        "mean_reward": 10.5,
        "best_reward": 25.3,
        "sharpe_ratio": 1.23,
        "win_rate": 0.58,
        "avg_drawdown": 0.12
    }


# Trade endpoints
@app.get("/api/trades")
async def get_trades(
    run_id: Optional[int] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """
    Get trade history
    
    Args:
        run_id: Filter by training run (optional)
        limit: Maximum trades to return
        offset: Offset for pagination
    
    Returns:
        List of trades
    """
    # TODO: Implement database query
    return {
        "trades": [],
        "total": 0,
        "limit": limit,
        "offset": offset
    }


@app.get("/api/trades/summary")
async def get_trades_summary(run_id: Optional[int] = None):
    """
    Get trade summary statistics
    
    Args:
        run_id: Filter by training run (optional)
    
    Returns:
        Trade summary stats
    """
    # TODO: Implement summary calculation
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "win_rate": 0.0,
        "avg_profit": 0.0,
        "avg_loss": 0.0,
        "profit_factor": 0.0
    }


# Model endpoints
@app.get("/api/models")
async def get_models(
    run_id: Optional[int] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """
    Get saved model checkpoints
    
    Args:
        run_id: Filter by training run (optional)
        limit: Maximum models to return
    
    Returns:
        List of model checkpoints
    """
    # TODO: Implement database query
    return {
        "models": [],
        "total": 0
    }


# Risk management endpoints
@app.get("/api/risk/status")
async def get_risk_status():
    """
    Get current risk management status
    
    Returns:
        Circuit breaker status and risk metrics
    """
    # TODO: Implement risk status check
    return {
        "status": "active",
        "circuit_breakers": {
            "daily_loss": {"status": "ok", "current": 0, "limit": 20},
            "weekly_loss": {"status": "ok", "current": 0, "limit": 50},
            "drawdown": {"status": "ok", "current": 0.05, "limit": 0.30}
        },
        "last_updated": datetime.utcnow().isoformat()
    }


# Markets endpoints
@app.get("/api/markets")
async def get_markets(
    limit: int = Query(50, ge=1, le=200),
    category: Optional[str] = None
):
    """
    Get crypto symbols in database
    
    Args:
        limit: Maximum markets to return
        category: Filter by category (optional)
    
    Returns:
        List of markets
    """
    # TODO: Implement database query
    return {
        "markets": [],
        "total": 0
    }


@app.get("/api/data-sources/health")
async def get_data_source_health():
    """
    Get health status for each data source based on latest candles in DB.
    """
    if not DB_ENGINE:
        raise HTTPException(status_code=503, detail="DATABASE_URL not configured")

    interval = os.getenv("DATA_INTERVAL", "1h")
    stale_seconds = _interval_to_seconds(interval) * 2
    now = datetime.utcnow()

    query = text(
        \"\"\"
        SELECT source, MAX(timestamp) AS last_ts
        FROM crypto_candles
        GROUP BY source
        \"\"\"
    )

    with DB_ENGINE.connect() as conn:
        rows = conn.execute(query).fetchall()

    results = []
    for source, last_ts in rows:
        if last_ts is None:
            results.append({
                "source": source,
                "ok": False,
                "reason": "no_data",
                "last_timestamp": None
            })
            continue

        age = (now - last_ts).total_seconds()
        ok = age <= stale_seconds
        results.append({
            "source": source,
            "ok": ok,
            "last_timestamp": last_ts.isoformat(),
            "age_seconds": age,
            "stale_threshold_seconds": stale_seconds
        })

    if not results:
        return {"sources": [], "status": "no_data"}

    overall_ok = all(r["ok"] for r in results if r["last_timestamp"])
    return {"sources": results, "status": "ok" if overall_ok else "stale"}


def _interval_to_seconds(interval: str) -> int:
    mapping = {
        "1m": 60,
        "5m": 300,
        "15m": 900,
        "30m": 1800,
        "1h": 3600,
        "4h": 14400,
        "1d": 86400,
    }
    return mapping.get(interval, 3600)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("API_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
