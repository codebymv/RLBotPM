import Link from "next/link";

const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function getHealth() {
  try {
    const res = await fetch(`${baseUrl}/health`, { cache: "no-store" });
    if (!res.ok) return { status: "error" };
    return res.json();
  } catch {
    return { status: "unreachable" };
  }
}

async function getMetrics() {
  try {
    const res = await fetch(`${baseUrl}/api/paper-trading/metrics`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export default async function Page() {
  const [health, metrics] = await Promise.all([getHealth(), getMetrics()]);

  const pnlColor =
    metrics && metrics.realized_pnl > 0
      ? "text-green-400"
      : metrics && metrics.realized_pnl < 0
        ? "text-red-400"
        : "text-gray-300";

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-6 max-w-6xl mx-auto">
      <h1 className="text-3xl font-bold mb-1">RLTrade Dashboard</h1>
      <p className="text-gray-500 text-sm mb-8">
        Kalshi crypto edge trading bot &mdash; BUY_NO strategy
      </p>

      {/* Status bar */}
      <div className="flex gap-3 mb-8 text-sm">
        <span
          className={`px-3 py-1 rounded-full ${health.status === "healthy" ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"}`}
        >
          API: {health.status}
        </span>
        <span
          className={`px-3 py-1 rounded-full ${health.database === "connected" ? "bg-green-900 text-green-300" : "bg-yellow-900 text-yellow-300"}`}
        >
          DB: {health.database || "unknown"}
        </span>
      </div>

      {/* Key Metrics */}
      {metrics ? (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
            <Card label="Total Trades" value={metrics.total_trades} />
            <Card
              label="Win Rate"
              value={
                metrics.wins + metrics.losses > 0
                  ? `${(metrics.win_rate * 100).toFixed(1)}%`
                  : "—"
              }
              sub={`${metrics.wins}W / ${metrics.losses}L`}
            />
            <Card
              label="Realized P&L"
              value={`$${metrics.realized_pnl >= 0 ? "+" : ""}${metrics.realized_pnl.toFixed(2)}`}
              className={pnlColor}
            />
            <Card
              label="Open Positions"
              value={metrics.open_positions}
              sub={`$${metrics.open_cost.toFixed(2)} deployed`}
            />
          </div>

          {/* Side breakdown */}
          {metrics.side_breakdown &&
            Object.keys(metrics.side_breakdown).length > 0 && (
              <div className="mb-8">
                <h2 className="text-lg font-semibold mb-3">
                  Performance by Side
                </h2>
                <div className="grid grid-cols-2 gap-4">
                  {Object.entries(
                    metrics.side_breakdown as Record<
                      string,
                      { total: number; wins: number; pnl: number }
                    >,
                  ).map(([side, data]) => {
                    const wr =
                      data.total > 0
                        ? ((data.wins / data.total) * 100).toFixed(0)
                        : "0";
                    return (
                      <div
                        key={side}
                        className={`rounded-lg border p-4 ${side === "no" ? "border-green-800 bg-green-950/30" : "border-red-800 bg-red-950/30"}`}
                      >
                        <div className="text-sm text-gray-400 mb-1">
                          BUY_{side.toUpperCase()}
                        </div>
                        <div className="text-xl font-bold">
                          {wr}% win rate
                        </div>
                        <div className="text-sm text-gray-400">
                          {data.wins}W / {data.total - data.wins}L &middot; $
                          {data.pnl >= 0 ? "+" : ""}
                          {data.pnl.toFixed(2)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

          {/* Recent trades */}
          {metrics.recent_trades && metrics.recent_trades.length > 0 && (
            <div className="mb-8">
              <h2 className="text-lg font-semibold mb-3">Recent Trades</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-500 border-b border-gray-800">
                      <th className="text-left py-2 px-2">Ticker</th>
                      <th className="text-left py-2 px-2">Side</th>
                      <th className="text-right py-2 px-2">Price</th>
                      <th className="text-right py-2 px-2">Edge</th>
                      <th className="text-right py-2 px-2">Contracts</th>
                      <th className="text-right py-2 px-2">Cost</th>
                      <th className="text-left py-2 px-2">Status</th>
                      <th className="text-right py-2 px-2">P&amp;L</th>
                    </tr>
                  </thead>
                  <tbody>
                    {metrics.recent_trades.map(
                      (
                        t: {
                          ticker: string;
                          side: string;
                          entry_price_cents: number;
                          edge: number | null;
                          contracts: number;
                          cost: number;
                          status: string;
                          pnl: number | null;
                        },
                        i: number,
                      ) => (
                        <tr
                          key={`${t.ticker}-${i}`}
                          className="border-b border-gray-900 hover:bg-gray-900/50"
                        >
                          <td className="py-2 px-2 font-mono text-xs">
                            {t.ticker}
                          </td>
                          <td className="py-2 px-2">
                            <span
                              className={`px-2 py-0.5 rounded text-xs ${t.side === "no" ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"}`}
                            >
                              {t.side.toUpperCase()}
                            </span>
                          </td>
                          <td className="py-2 px-2 text-right">
                            {t.entry_price_cents}¢
                          </td>
                          <td className="py-2 px-2 text-right">
                            {t.edge ? `${(t.edge * 100).toFixed(1)}%` : "—"}
                          </td>
                          <td className="py-2 px-2 text-right">
                            {t.contracts}
                          </td>
                          <td className="py-2 px-2 text-right">
                            ${t.cost.toFixed(2)}
                          </td>
                          <td className="py-2 px-2">
                            <span
                              className={`text-xs ${t.status === "open" ? "text-yellow-400" : "text-gray-400"}`}
                            >
                              {t.status}
                            </span>
                          </td>
                          <td
                            className={`py-2 px-2 text-right font-medium ${t.pnl && t.pnl > 0 ? "text-green-400" : t.pnl && t.pnl < 0 ? "text-red-400" : ""}`}
                          >
                            {t.pnl !== null
                              ? `$${t.pnl >= 0 ? "+" : ""}${t.pnl.toFixed(2)}`
                              : "—"}
                          </td>
                        </tr>
                      ),
                    )}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </>
      ) : (
        <div className="text-gray-500 py-12 text-center">
          No trading data yet. Start the paper trader to see results here.
        </div>
      )}

      {/* Nav */}
      <div className="flex gap-4 mt-8 text-sm">
        <Link
          href="/positions"
          className="text-blue-400 hover:text-blue-300 underline"
        >
          Open Positions →
        </Link>
        <Link
          href="/edge-health"
          className="text-blue-400 hover:text-blue-300 underline"
        >
          Edge Health →
        </Link>
      </div>
    </main>
  );
}

function Card({
  label,
  value,
  sub,
  className = "",
}: {
  label: string;
  value: string | number;
  sub?: string;
  className?: string;
}) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <div className="text-xs text-gray-500 uppercase tracking-wide mb-1">
        {label}
      </div>
      <div className={`text-2xl font-bold ${className}`}>{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-1">{sub}</div>}
    </div>
  );
}
