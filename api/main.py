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
origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip()]

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
    Return paper trading portfolio metrics from the kalshi_trades table.
    """
    if not DB_ENGINE:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        with DB_ENGINE.connect() as conn:
            # Overall stats
            row = conn.execute(text("""
                SELECT
                    COUNT(*) AS total,
                    SUM(CASE WHEN status='settled' AND pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    SUM(CASE WHEN status='settled' AND pnl <= 0 THEN 1 ELSE 0 END) AS losses,
                    COALESCE(SUM(CASE WHEN status='settled' THEN pnl END), 0) AS realized_pnl,
                    SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) AS open_count,
                    COALESCE(SUM(CASE WHEN status='open' THEN cost_dollars END), 0) AS open_cost
                FROM kalshi_trades
                WHERE mode = 'paper'
            """)).fetchone()

            total = int(row[0] or 0)
            wins = int(row[1] or 0)
            losses = int(row[2] or 0)
            realized_pnl = float(row[3] or 0)
            open_count = int(row[4] or 0)
            open_cost = float(row[5] or 0)
            settled = wins + losses
            win_rate = (wins / settled) if settled > 0 else 0.0

            # Side breakdown
            side_rows = conn.execute(text("""
                SELECT
                    side,
                    COUNT(*) AS total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    COALESCE(SUM(pnl), 0) AS pnl
                FROM kalshi_trades
                WHERE mode = 'paper' AND status = 'settled'
                GROUP BY side
            """)).fetchall()
            side_breakdown = {
                r[0]: {"total": int(r[1]), "wins": int(r[2]), "pnl": float(r[3])}
                for r in side_rows
            }

            # Recent trades (last 20)
            recent = conn.execute(text("""
                SELECT ticker, side, entry_price_cents, edge_value, contracts,
                       cost_dollars, status, outcome, pnl,
                       opened_at, settled_at, edge_type
                FROM kalshi_trades
                WHERE mode = 'paper'
                ORDER BY id DESC
                LIMIT 20
            """)).fetchall()

        recent_trades = [
            {
                "ticker": r[0],
                "side": r[1],
                "entry_price_cents": float(r[2]),
                "edge": float(r[3]) if r[3] else None,
                "contracts": int(r[4]),
                "cost": float(r[5]),
                "status": r[6],
                "outcome": r[7],
                "pnl": float(r[8]) if r[8] is not None else None,
                "opened_at": r[9].isoformat() if r[9] else None,
                "settled_at": r[10].isoformat() if r[10] else None,
                "edge_type": r[11],
            }
            for r in recent
        ]

        return {
            "total_trades": total,
            "wins": wins,
            "losses": losses,
            "win_rate": win_rate,
            "realized_pnl": realized_pnl,
            "open_positions": open_count,
            "open_cost": open_cost,
            "side_breakdown": side_breakdown,
            "recent_trades": recent_trades,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kalshi/trades")
async def get_kalshi_trades(
    mode: Optional[str] = Query(None, description="Filter by mode: paper or live"),
    status: Optional[str] = Query(None, description="Filter by status: open or settled"),
    side: Optional[str] = Query(None, description="Filter by side: yes or no"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Get Kalshi trades from the database."""
    if not DB_ENGINE:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        conditions = []
        params: dict = {"lim": limit, "off": offset}
        if mode:
            conditions.append("mode = :mode")
            params["mode"] = mode
        if status:
            conditions.append("status = :status")
            params["status"] = status
        if side:
            conditions.append("side = :side")
            params["side"] = side

        where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

        with DB_ENGINE.connect() as conn:
            total = conn.execute(
                text(f"SELECT COUNT(*) FROM kalshi_trades {where}"), params
            ).scalar() or 0

            rows = conn.execute(text(f"""
                SELECT id, ticker, event_ticker, series_ticker, side,
                       entry_price_cents, fair_price_cents, edge_value, edge_type,
                       contracts, cost_dollars, reasoning, status, outcome, pnl,
                       mode, session_id, opened_at, settled_at
                FROM kalshi_trades {where}
                ORDER BY id DESC
                LIMIT :lim OFFSET :off
            """), params).fetchall()

        trades = [
            {
                "id": r[0], "ticker": r[1], "event_ticker": r[2],
                "series_ticker": r[3], "side": r[4],
                "entry_price_cents": float(r[5]),
                "fair_price_cents": float(r[6]) if r[6] else None,
                "edge": float(r[7]) if r[7] else None,
                "edge_type": r[8], "contracts": int(r[9]),
                "cost": float(r[10]), "reasoning": r[11],
                "status": r[12], "outcome": r[13],
                "pnl": float(r[14]) if r[14] is not None else None,
                "mode": r[15], "session_id": r[16],
                "opened_at": r[17].isoformat() if r[17] else None,
                "settled_at": r[18].isoformat() if r[18] else None,
            }
            for r in rows
        ]
        return {"trades": trades, "total": total, "limit": limit, "offset": offset}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kalshi/positions")
async def get_kalshi_positions(
    mode: Optional[str] = Query("paper", description="paper or live"),
):
    """Get currently open Kalshi positions."""
    if not DB_ENGINE:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        with DB_ENGINE.connect() as conn:
            rows = conn.execute(text("""
                SELECT ticker, side, entry_price_cents, fair_price_cents,
                       edge_value, edge_type, contracts, cost_dollars,
                       reasoning, opened_at, series_ticker
                FROM kalshi_trades
                WHERE status = 'open' AND mode = :mode
                ORDER BY opened_at DESC
            """), {"mode": mode}).fetchall()

        positions = [
            {
                "ticker": r[0], "side": r[1],
                "entry_price_cents": float(r[2]),
                "fair_price_cents": float(r[3]) if r[3] else None,
                "edge": float(r[4]) if r[4] else None,
                "edge_type": r[5], "contracts": int(r[6]),
                "cost": float(r[7]), "reasoning": r[8],
                "opened_at": r[9].isoformat() if r[9] else None,
                "series_ticker": r[10],
            }
            for r in rows
        ]
        return {"positions": positions, "count": len(positions)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kalshi/pnl-series")
async def get_kalshi_pnl_series(
    mode: Optional[str] = Query("paper", description="paper or live"),
):
    """Get cumulative P&L time series for charting."""
    if not DB_ENGINE:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        with DB_ENGINE.connect() as conn:
            rows = conn.execute(text("""
                SELECT settled_at, pnl, side, ticker
                FROM kalshi_trades
                WHERE status = 'settled' AND mode = :mode AND settled_at IS NOT NULL
                ORDER BY settled_at ASC
            """), {"mode": mode}).fetchall()

        cumulative = 0.0
        series = []
        for r in rows:
            cumulative += float(r[1] or 0)
            series.append({
                "timestamp": r[0].isoformat() if r[0] else None,
                "pnl": float(r[1] or 0),
                "cumulative_pnl": cumulative,
                "side": r[2],
                "ticker": r[3],
            })
        return {"series": series, "total_pnl": cumulative}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kalshi/edge-health")
async def get_kalshi_edge_health(
    mode: Optional[str] = Query("paper", description="paper or live"),
):
    """Edge health metrics — rolling win rate and avg edge size."""
    if not DB_ENGINE:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        with DB_ENGINE.connect() as conn:
            # Overall settled stats by side
            rows = conn.execute(text("""
                SELECT
                    side,
                    COUNT(*) AS total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    AVG(edge_value) AS avg_edge,
                    AVG(pnl) AS avg_pnl,
                    SUM(pnl) AS total_pnl
                FROM kalshi_trades
                WHERE status = 'settled' AND mode = :mode
                GROUP BY side
            """), {"mode": mode}).fetchall()

            by_side = {}
            for r in rows:
                total = int(r[1])
                wins = int(r[2])
                by_side[r[0]] = {
                    "total": total,
                    "wins": wins,
                    "losses": total - wins,
                    "win_rate": wins / total if total > 0 else 0,
                    "avg_edge": float(r[3]) if r[3] else 0,
                    "avg_pnl": float(r[4]) if r[4] else 0,
                    "total_pnl": float(r[5]) if r[5] else 0,
                }

            # Edge type breakdown
            edge_rows = conn.execute(text("""
                SELECT
                    edge_type,
                    COUNT(*) AS total,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins,
                    AVG(edge_value) AS avg_edge,
                    SUM(pnl) AS total_pnl
                FROM kalshi_trades
                WHERE status = 'settled' AND mode = :mode
                GROUP BY edge_type
            """), {"mode": mode}).fetchall()

            by_edge_type = {}
            for r in edge_rows:
                total = int(r[1])
                wins = int(r[2])
                by_edge_type[r[0] or "unknown"] = {
                    "total": total,
                    "wins": wins,
                    "win_rate": wins / total if total > 0 else 0,
                    "avg_edge": float(r[3]) if r[3] else 0,
                    "total_pnl": float(r[4]) if r[4] else 0,
                }

        return {
            "by_side": by_side,
            "by_edge_type": by_edge_type,
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


# ──────────────────────────────────────────────────────────────────────
#  Crypto Market Data endpoints
# ──────────────────────────────────────────────────────────────────────

CRYPTO_ASSETS = ["BTC", "ETH", "SOL", "DOGE", "XRP"]
COINBASE_TICKER = "https://api.exchange.coinbase.com/products/{symbol}/ticker"
COINBASE_CANDLES = "https://api.exchange.coinbase.com/products/{symbol}/candles"

# Annualised vols calibrated from Kalshi settlement data
CALIBRATED_VOLS = {
    "BTC": 0.56, "ETH": 0.70, "SOL": 0.74, "DOGE": 0.65, "XRP": 0.71,
}

import httpx  # lightweight async HTTP – already transitively available


@app.get("/api/crypto/prices")
async def get_crypto_prices():
    """Live spot prices for all tracked crypto assets from Coinbase."""
    import time as _time
    results = {}
    async with httpx.AsyncClient(timeout=10) as client:
        for asset in CRYPTO_ASSETS:
            symbol = f"{asset}-USD"
            try:
                resp = await client.get(
                    COINBASE_TICKER.format(symbol=symbol)
                )
                if resp.status_code == 200:
                    data = resp.json()
                    results[asset] = {
                        "price": float(data["price"]),
                        "volume_24h": float(data.get("volume", 0)),
                        "bid": float(data.get("bid", 0)),
                        "ask": float(data.get("ask", 0)),
                        "symbol": symbol,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                else:
                    results[asset] = {"error": f"HTTP {resp.status_code}"}
            except Exception as e:
                results[asset] = {"error": str(e)}
    return {
        "prices": results,
        "volatilities": CALIBRATED_VOLS,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/api/crypto/candles/{asset}")
async def get_crypto_candles(
    asset: str,
    interval: str = Query("1h", regex="^(1m|5m|15m|1h|4h|1d)$"),
    limit: int = Query(48, ge=1, le=300),
):
    """Recent OHLCV candles for a crypto asset from Coinbase."""
    if asset.upper() not in CRYPTO_ASSETS:
        raise HTTPException(status_code=400, detail=f"Unsupported asset: {asset}")

    granularity_map = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
    symbol = f"{asset.upper()}-USD"
    params = {"granularity": granularity_map[interval]}

    async with httpx.AsyncClient(timeout=15) as client:
        resp = await client.get(
            COINBASE_CANDLES.format(symbol=symbol), params=params
        )
        if resp.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Coinbase error: {resp.status_code}")
        raw = resp.json()

    # Coinbase returns [time, low, high, open, close, volume] newest first
    candles = []
    for row in reversed(raw[:limit]):
        candles.append({
            "timestamp": datetime.utcfromtimestamp(row[0]).isoformat(),
            "open": float(row[3]),
            "high": float(row[2]),
            "low": float(row[1]),
            "close": float(row[4]),
            "volume": float(row[5]),
        })
    return {"asset": asset.upper(), "interval": interval, "candles": candles}


# ──────────────────────────────────────────────────────────────────────
#  Bot Operations endpoint
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/bot/status")
async def get_bot_status():
    """
    Aggregated bot status: latest session, scan stats, and configuration.
    """
    if not DB_ENGINE:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        with DB_ENGINE.connect() as conn:
            # Latest session
            sess = conn.execute(text("""
                SELECT session_id,
                       MIN(opened_at) AS started,
                       MAX(opened_at) AS last_trade,
                       COUNT(*) AS trades_opened,
                       SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) AS open_now,
                       SUM(CASE WHEN status='settled' AND pnl > 0 THEN 1 ELSE 0 END) AS wins,
                       SUM(CASE WHEN status='settled' AND pnl <= 0 THEN 1 ELSE 0 END) AS losses,
                       COALESCE(SUM(CASE WHEN status='settled' THEN pnl END), 0) AS realized_pnl
                FROM kalshi_trades
                WHERE mode = 'paper'
                GROUP BY session_id
                ORDER BY MIN(opened_at) DESC
                LIMIT 5
            """)).fetchall()

            sessions = []
            for r in sess:
                sessions.append({
                    "session_id": r[0],
                    "started_at": r[1].isoformat() if r[1] else None,
                    "last_trade_at": r[2].isoformat() if r[2] else None,
                    "trades_opened": int(r[3] or 0),
                    "open_now": int(r[4] or 0),
                    "wins": int(r[5] or 0),
                    "losses": int(r[6] or 0),
                    "realized_pnl": float(r[7] or 0),
                })

            # Overall counts
            totals = conn.execute(text("""
                SELECT COUNT(DISTINCT session_id) AS total_sessions,
                       COUNT(*) AS total_trades,
                       MIN(opened_at) AS first_trade,
                       MAX(opened_at) AS last_trade
                FROM kalshi_trades
                WHERE mode = 'paper'
            """)).fetchone()

        return {
            "sessions": sessions,
            "total_sessions": int(totals[0] or 0),
            "total_trades": int(totals[1] or 0),
            "first_trade_at": totals[2].isoformat() if totals[2] else None,
            "last_trade_at": totals[3].isoformat() if totals[3] else None,
            "strategy": {
                "name": "BUY_NO Lognormal Edge",
                "side_filter": "no",
                "min_edge": 0.02,
                "max_edge": 0.05,
                "min_price": 1,
                "max_price": 15,
                "assets": CRYPTO_ASSETS,
                "volatilities": CALIBRATED_VOLS,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ──────────────────────────────────────────────────────────────────────
#  Settled markets stats
# ──────────────────────────────────────────────────────────────────────

@app.get("/api/kalshi/market-stats")
async def get_kalshi_market_stats():
    """Summary of backfilled settled markets in the database."""
    if not DB_ENGINE:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        with DB_ENGINE.connect() as conn:
            row = conn.execute(text("""
                SELECT COUNT(*) AS total,
                       COUNT(DISTINCT event_ticker) AS events,
                       COUNT(DISTINCT series_ticker) AS series,
                       MIN(close_time) AS earliest,
                       MAX(close_time) AS latest
                FROM kalshi_settled_markets
            """)).fetchone()
        return {
            "total_markets": int(row[0] or 0),
            "total_events": int(row[1] or 0),
            "total_series": int(row[2] or 0),
            "earliest_close": row[3].isoformat() if row[3] else None,
            "latest_close": row[4].isoformat() if row[4] else None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
    port = int(os.getenv("PORT", os.getenv("API_PORT", 8000)))
    uvicorn.run(app, host="0.0.0.0", port=port)
