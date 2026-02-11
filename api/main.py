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
import json

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
    Get list of training runs from database.
    """
    if not DB_ENGINE:
        return {"runs": [], "total": 0, "limit": limit, "offset": offset}
    try:
        with DB_ENGINE.connect() as conn:
            count_row = conn.execute(
                text("SELECT COUNT(*) FROM training_runs")
            ).scalar()
            total = count_row or 0
            rows = conn.execute(
                text("""
                    SELECT id, started_at, ended_at, total_episodes,
                           best_sharpe_ratio, best_episode_reward, status, config_snapshot
                    FROM training_runs
                    ORDER BY id DESC
                    LIMIT :lim OFFSET :off
                """),
                {"lim": limit, "off": offset}
            ).fetchall()
        runs = [
            {
                "id": r[0],
                "started_at": r[1].isoformat() if r[1] else None,
                "ended_at": r[2].isoformat() if r[2] else None,
                "total_episodes": r[3] or 0,
                "best_sharpe_ratio": float(r[4]) if r[4] is not None else None,
                "best_episode_reward": float(r[5]) if r[5] is not None else None,
                "status": r[6],
                "config_snapshot": r[7],
            }
            for r in rows
        ]
        return {"runs": runs, "total": total, "limit": limit, "offset": offset}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/runs/{run_id}")
async def get_training_run(run_id: int):
    """Get detailed information about a specific training run."""
    if not DB_ENGINE:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        with DB_ENGINE.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT id, started_at, ended_at, total_episodes,
                           best_sharpe_ratio, best_episode_reward, status, config_snapshot
                    FROM training_runs WHERE id = :id
                """),
                {"id": run_id}
            ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Training run not found")
        return {
            "id": row[0],
            "started_at": row[1].isoformat() if row[1] else None,
            "ended_at": row[2].isoformat() if row[2] else None,
            "total_episodes": row[3] or 0,
            "best_sharpe_ratio": float(row[4]) if row[4] is not None else None,
            "best_episode_reward": float(row[5]) if row[5] is not None else None,
            "status": row[6],
            "config": row[7] or {},
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/runs/{run_id}/episodes")
async def get_episodes(
    run_id: int,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
):
    """Get episodes for a training run."""
    if not DB_ENGINE:
        return {"episodes": [], "total": 0, "limit": limit, "offset": offset}
    try:
        with DB_ENGINE.connect() as conn:
            total = conn.execute(
                text("SELECT COUNT(*) FROM episodes WHERE training_run_id = :id"),
                {"id": run_id}
            ).scalar() or 0
            rows = conn.execute(
                text("""
                    SELECT id, episode_num, started_at, ended_at, total_reward,
                           total_return_pct, sharpe_ratio, max_drawdown, num_trades,
                           num_winning_trades, final_capital
                    FROM episodes
                    WHERE training_run_id = :id
                    ORDER BY episode_num DESC
                    LIMIT :lim OFFSET :off
                """),
                {"id": run_id, "lim": limit, "off": offset}
            ).fetchall()
        episodes = [
            {
                "id": r[0],
                "episode_num": r[1],
                "started_at": r[2].isoformat() if r[2] else None,
                "ended_at": r[3].isoformat() if r[3] else None,
                "total_reward": float(r[4]) if r[4] is not None else None,
                "total_return_pct": float(r[5]) if r[5] is not None else None,
                "sharpe_ratio": float(r[6]) if r[6] is not None else None,
                "max_drawdown": float(r[7]) if r[7] is not None else None,
                "num_trades": r[8] or 0,
                "num_winning_trades": r[9] or 0,
                "final_capital": float(r[10]) if r[10] is not None else None,
            }
            for r in rows
        ]
        return {"episodes": episodes, "total": total, "limit": limit, "offset": offset}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/training/runs/{run_id}/metrics")
async def get_training_metrics(run_id: int):
    """Get aggregated metrics for a training run from episodes."""
    if not DB_ENGINE:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        with DB_ENGINE.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT
                        COUNT(*) AS total_episodes,
                        AVG(total_reward) AS mean_reward,
                        MAX(total_reward) AS best_reward,
                        AVG(sharpe_ratio) AS sharpe_ratio,
                        SUM(num_winning_trades)::float / NULLIF(SUM(num_trades), 0) AS win_rate,
                        AVG(max_drawdown) AS avg_drawdown
                    FROM episodes
                    WHERE training_run_id = :id
                """),
                {"id": run_id}
            ).fetchone()
        if not row or row[0] == 0:
            return {
                "run_id": run_id,
                "total_episodes": 0,
                "mean_reward": None,
                "best_reward": None,
                "sharpe_ratio": None,
                "win_rate": None,
                "avg_drawdown": None,
            }
        return {
            "run_id": run_id,
            "total_episodes": row[0],
            "mean_reward": float(row[1]) if row[1] is not None else None,
            "best_reward": float(row[2]) if row[2] is not None else None,
            "sharpe_ratio": float(row[3]) if row[3] is not None else None,
            "win_rate": float(row[4]) if row[4] is not None else None,
            "avg_drawdown": float(row[5]) if row[5] is not None else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Trade endpoints
@app.get("/api/trades")
async def get_trades(
    run_id: Optional[int] = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    """Get trade history from database."""
    if not DB_ENGINE:
        return {"trades": [], "total": 0, "limit": limit, "offset": offset}
    try:
        with DB_ENGINE.connect() as conn:
            if run_id is not None:
                total = conn.execute(
                    text("SELECT COUNT(*) FROM trades t JOIN episodes e ON t.episode_id = e.id WHERE e.training_run_id = :run_id"),
                    {"run_id": run_id}
                ).scalar() or 0
                rows = conn.execute(
                    text("""
                        SELECT t.id, t.episode_id, t.timestamp, t.market_id, t.action_name,
                               t.position_size, t.price, t.side, t.pnl, e.training_run_id
                        FROM trades t
                        JOIN episodes e ON t.episode_id = e.id
                        WHERE e.training_run_id = :run_id
                        ORDER BY t.timestamp DESC
                        LIMIT :lim OFFSET :off
                    """),
                    {"run_id": run_id, "lim": limit, "off": offset}
                ).fetchall()
            else:
                total = conn.execute(text("SELECT COUNT(*) FROM trades")).scalar() or 0
                rows = conn.execute(
                    text("""
                        SELECT t.id, t.episode_id, t.timestamp, t.market_id, t.action_name,
                               t.position_size, t.price, t.side, t.pnl, e.training_run_id
                        FROM trades t
                        JOIN episodes e ON t.episode_id = e.id
                        ORDER BY t.timestamp DESC
                        LIMIT :lim OFFSET :off
                    """),
                    {"lim": limit, "off": offset}
                ).fetchall()
        trades = [
            {
                "id": r[0],
                "episode_id": r[1],
                "timestamp": r[2].isoformat() if r[2] else None,
                "market_id": r[3],
                "action_name": r[4],
                "position_size": float(r[5]),
                "price": float(r[6]),
                "side": r[7],
                "pnl": float(r[8]) if r[8] is not None else None,
                "training_run_id": r[9],
            }
            for r in rows
        ]
        return {"trades": trades, "total": total, "limit": limit, "offset": offset}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/paper-trading/metrics")
async def get_paper_trading_metrics():
    """
    Return latest paper trading metrics.
    """
    metrics_path = os.getenv("PAPER_TRADING_METRICS_PATH", "./logs/paper_trading/metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception:
            pass

    return {
        "capital": 0.0,
        "total_return_pct": 0.0,
        "win_rate": 0.0,
        "num_trades": 0,
        "open_positions": 0,
        "recent_trades": [],
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/trades/summary")
async def get_trades_summary(run_id: Optional[int] = None):
    """Get trade summary statistics from database."""
    if not DB_ENGINE:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
        }
    try:
        if run_id is not None:
            sql = """
                SELECT COUNT(*),
                       SUM(CASE WHEN t.pnl > 0 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN t.pnl < 0 THEN 1 ELSE 0 END),
                       AVG(CASE WHEN t.pnl > 0 THEN t.pnl END),
                       AVG(CASE WHEN t.pnl < 0 THEN t.pnl END),
                       SUM(CASE WHEN t.pnl > 0 THEN t.pnl END),
                       SUM(CASE WHEN t.pnl < 0 THEN ABS(t.pnl) END)
                FROM trades t
                JOIN episodes e ON t.episode_id = e.id
                WHERE e.training_run_id = :run_id AND t.pnl IS NOT NULL
            """
            params = {"run_id": run_id}
        else:
            sql = """
                SELECT COUNT(*),
                       SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
                       SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END),
                       AVG(CASE WHEN pnl > 0 THEN pnl END),
                       AVG(CASE WHEN pnl < 0 THEN pnl END),
                       SUM(CASE WHEN pnl > 0 THEN pnl END),
                       SUM(CASE WHEN pnl < 0 THEN ABS(pnl) END)
                FROM trades WHERE pnl IS NOT NULL
            """
            params = {}
        with DB_ENGINE.connect() as conn:
            row = conn.execute(text(sql), params).fetchone()
        if not row or (row[0] or 0) == 0:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
            }
        total, winning, losing = int(row[0] or 0), int(row[1] or 0), int(row[2] or 0)
        avg_profit = float(row[3]) if row[3] is not None else 0.0
        avg_loss = float(row[4]) if row[4] is not None else 0.0
        gross_profit = float(row[5]) if row[5] is not None else 0.0
        gross_loss = float(row[6]) if row[6] is not None else 0.0
        pf = (gross_profit / gross_loss) if gross_loss and gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        return {
            "total_trades": total,
            "winning_trades": winning,
            "losing_trades": losing,
            "win_rate": (winning / total) if total else 0.0,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss,
            "profit_factor": pf,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Model endpoints
@app.get("/api/models")
async def get_models(
    run_id: Optional[int] = None,
    limit: int = Query(20, ge=1, le=100)
):
    """Get saved model checkpoints from database."""
    if not DB_ENGINE:
        return {"models": [], "total": 0}
    try:
        with DB_ENGINE.connect() as conn:
            if run_id is not None:
                total = conn.execute(
                    text("SELECT COUNT(*) FROM model_checkpoints WHERE training_run_id = :run_id"),
                    {"run_id": run_id}
                ).scalar() or 0
                rows = conn.execute(
                    text("""
                        SELECT id, training_run_id, episode_num, file_path, sharpe_ratio,
                               avg_reward, win_rate, is_best, created_at
                        FROM model_checkpoints
                        WHERE training_run_id = :run_id
                        ORDER BY episode_num DESC
                        LIMIT :lim
                    """),
                    {"run_id": run_id, "lim": limit}
                ).fetchall()
            else:
                total = conn.execute(text("SELECT COUNT(*) FROM model_checkpoints")).scalar() or 0
                rows = conn.execute(
                    text("""
                        SELECT id, training_run_id, episode_num, file_path, sharpe_ratio,
                               avg_reward, win_rate, is_best, created_at
                        FROM model_checkpoints
                        ORDER BY created_at DESC
                        LIMIT :lim
                    """),
                    {"lim": limit}
                ).fetchall()
        models = [
            {
                "id": r[0],
                "training_run_id": r[1],
                "episode_num": r[2],
                "file_path": r[3],
                "sharpe_ratio": float(r[4]) if r[4] is not None else None,
                "avg_reward": float(r[5]) if r[5] is not None else None,
                "win_rate": float(r[6]) if r[6] is not None else None,
                "is_best": bool(r[7]),
                "created_at": r[8].isoformat() if r[8] else None,
            }
            for r in rows
        ]
        return {"models": models, "total": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    """Get crypto symbols from database."""
    if not DB_ENGINE:
        return {"markets": [], "total": 0}
    try:
        with DB_ENGINE.connect() as conn:
            total = conn.execute(text("SELECT COUNT(*) FROM crypto_symbols")).scalar() or 0
            rows = conn.execute(
                text("SELECT id, source, symbol, base_asset, quote_asset, status FROM crypto_symbols LIMIT :lim"),
                {"lim": limit}
            ).fetchall()
        markets = [
            {"id": r[0], "source": r[1], "symbol": r[2], "base_asset": r[3], "quote_asset": r[4], "status": r[5]}
            for r in rows
        ]
        return {"markets": markets, "total": total}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Kalshi endpoints (for dashboard when pipeline is live)
@app.get("/api/kalshi/status")
async def get_kalshi_status():
    """Kalshi pipeline status: configured when env vars are set."""
    configured = bool(os.getenv("KALSHI_API_KEY") and os.getenv("KALSHI_API_SECRET"))
    return {
        "configured": configured,
        "message": "Kalshi connected" if configured else "Kalshi not configured (set KALSHI_API_KEY, KALSHI_API_SECRET)",
        "timestamp": datetime.utcnow().isoformat(),
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

    query = text("""
        SELECT source, MAX(timestamp) AS last_ts
        FROM crypto_candles
        GROUP BY source
    """)

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
