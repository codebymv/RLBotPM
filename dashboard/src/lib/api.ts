/**
 * Central API client used by both server components (initial SSR fetch)
 * and TanStack Query hooks (client-side polling).
 */

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export type StrategyMetrics = {
  total_trades: number;
  wins: number;
  losses: number;
  settled_trades: number;
  win_rate: number;
  realized_pnl: number;
  open_positions: number;
  open_cost: number;
  side_breakdown?: Record<string, { total: number; wins: number; pnl: number }>;
  recent_trades?: Array<Record<string, unknown>>;
};

export type CurrentSessionSnapshot = {
  session_id: string;
  started_at: string | null;
  open_positions: number;
  open_cost: number;
  settled_trades: number;
  wins: number;
  losses: number;
  win_rate: number | null;
  realized_pnl: number;
  last_scan: Record<string, unknown> | null;
};

export type CombinedMetricsResponse = {
  combined: StrategyMetrics;
  current_session: CurrentSessionSnapshot | null;
  by_strategy: {
    kalshi: StrategyMetrics;
    rl_crypto: StrategyMetrics;
  };
  timestamp: string;
};

export type PnlSeriesPoint = {
  timestamp: string;
  pnl: number;
  cumulative_pnl: number;
  side: string;
  ticker: string;
};

export type HeartbeatResponse = {
  bot_id: string;
  is_alive: boolean;
  stale: boolean;
  seconds_since_heartbeat: number | null;
  last_seen: string | null;
  metadata: Record<string, unknown>;
};

export type RiskStatusResponse = {
  status: string;
  stale: boolean;
  seconds_since_write: number | null;
  current_capital: number | null;
  peak_capital: number | null;
  circuit_breakers: {
    daily_loss: { status: string; current_usd: number; limit_usd: number; pct_used: number };
    weekly_loss: { status: string; current_usd: number; limit_usd: number; pct_used: number };
    drawdown: { status: string; current: number; limit: number; pct_used: number };
    consecutive_losses: { current: number; limit: number };
  };
  win_rate_last_20: number | null;
  api_error_count: number;
  recent_events: Array<{ timestamp: string; rule: string; description: string; severity: string }>;
  last_updated: string;
};

export type TradesSummaryResponse = {
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  avg_profit: number;
  avg_loss: number;
  profit_factor: number;
};

export type BotSession = {
  session_id: string;
  started_at: string | null;
  last_trade_at: string | null;
  trades_opened: number;
  open_now: number;
  wins: number;
  losses: number;
  realized_pnl: number;
};

export type BotStatusResponse = {
  sessions: BotSession[];
  total_sessions: number;
  total_trades: number;
  first_trade_at: string | null;
  last_trade_at: string | null;
  current_session: CurrentSessionSnapshot | null;
  strategy: {
    name: string;
    side_filter: "yes" | "no" | "both";
    allow_buy_yes: boolean;
    min_edge: number;
    max_edge: number;
    min_price: number;
    max_price: number;
    assets: string[];
    volatilities: Record<string, number>;
  };
  session_reconciliation?: {
    db_sessions: number;
    jsonl_session_starts: number;
    jsonl_session_ends: number;
    delta_db_minus_jsonl: number;
    jsonl_log_path: string;
  };
  timestamp: string;
};

// eslint-disable-next-line @typescript-eslint/no-explicit-any
async function get<T = any>(path: string): Promise<T | null> {
  try {
    const res = await fetch(`${BASE_URL}${path}`, { cache: "no-store" });
    if (!res.ok) return null;
    return res.json() as Promise<T>;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Health & system
// ---------------------------------------------------------------------------

export async function fetchHealth() {
  return get<{
    status: string;
    database: string;
    timestamp: string;
  }>("/health");
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

export async function fetchCombinedMetrics(mode = "paper") {
  return get<CombinedMetricsResponse>(`/api/metrics/combined?mode=${mode}`);
}

export async function fetchPaperTradingMetrics(mode?: string) {
  const query = mode ? `?mode=${mode}` : "";
  return get(`/api/paper-trading/metrics${query}`);
}

// ---------------------------------------------------------------------------
// Crypto prices
// ---------------------------------------------------------------------------

export async function fetchCryptoPrices() {
  return get(`/api/crypto/prices`);
}

// ---------------------------------------------------------------------------
// Bot status
// ---------------------------------------------------------------------------

export async function fetchBotStatus() {
  return get<BotStatusResponse>(`/api/bot/status`);
}

// ---------------------------------------------------------------------------
// Market stats
// ---------------------------------------------------------------------------

export async function fetchMarketStats() {
  return get(`/api/kalshi/market-stats`);
}

// ---------------------------------------------------------------------------
// Kalshi positions & trades
// ---------------------------------------------------------------------------

export async function fetchKalshiPositions(mode?: string) {
  const query = mode ? `?mode=${mode}` : "";
  return get(`/api/kalshi/positions${query}`);
}

export async function fetchEdgeHealth(mode = "paper") {
  return get(`/api/kalshi/edge-health?mode=${mode}`);
}

export async function fetchPnlSeries(mode = "paper") {
  return get(`/api/kalshi/pnl-series?mode=${mode}`);
}

// ---------------------------------------------------------------------------
// RL Crypto positions
// ---------------------------------------------------------------------------

export async function fetchRLPositions(mode = "paper") {
  return get(`/api/rl-crypto/positions?mode=${mode}`);
}

// ---------------------------------------------------------------------------
// Training runs
// ---------------------------------------------------------------------------

export async function fetchTrainingRuns(limit = 10) {
  const data = await get<{
    runs?: unknown[];
    total?: number;
    limit?: number;
    offset?: number;
  }>(`/api/training/runs?limit=${limit}`);
  return data?.runs ?? [];
}

// ---------------------------------------------------------------------------
// Heartbeat & risk
// ---------------------------------------------------------------------------

export async function fetchHeartbeat(botId = "fleet") {
  return get<HeartbeatResponse>(`/api/bot/heartbeat?bot_id=${botId}`);
}

export async function fetchRiskStatus() {
  return get<RiskStatusResponse>(`/api/risk/status`);
}

export async function fetchTradesSummary() {
  return get<TradesSummaryResponse>(`/api/trades/summary`);
}

export async function fetchPnlSeriesTyped(mode = "paper") {
  return get<{ series: PnlSeriesPoint[]; total_pnl: number }>(
    `/api/kalshi/pnl-series?mode=${mode}`
  );
}
