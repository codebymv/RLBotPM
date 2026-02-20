/**
 * Central API client used by both server components (initial SSR fetch)
 * and TanStack Query hooks (client-side polling).
 */

const BASE_URL =
  process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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
  return get(`/api/metrics/combined?mode=${mode}`);
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
  return get(`/api/bot/status`);
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
