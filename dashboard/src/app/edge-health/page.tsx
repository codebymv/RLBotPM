import Link from "next/link";

const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type SideStats = {
  total: number;
  wins: number;
  losses: number;
  win_rate: number;
  avg_edge: number;
  avg_pnl: number;
  total_pnl: number;
};

type EdgeTypeStats = {
  total: number;
  wins: number;
  win_rate: number;
  avg_edge: number;
  total_pnl: number;
};

async function getEdgeHealth() {
  try {
    const res = await fetch(`${baseUrl}/api/kalshi/edge-health?mode=paper`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function getPnlSeries() {
  try {
    const res = await fetch(`${baseUrl}/api/kalshi/pnl-series?mode=paper`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export default async function EdgeHealthPage() {
  const [health, pnl] = await Promise.all([getEdgeHealth(), getPnlSeries()]);

  const bySide: Record<string, SideStats> = health?.by_side || {};
  const byEdgeType: Record<string, EdgeTypeStats> = health?.by_edge_type || {};

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-6 max-w-7xl mx-auto">
      <h1 className="text-2xl font-bold mt-2 mb-1">Edge Health</h1>
      <p className="text-gray-500 text-sm mb-6">
        Is the statistical edge still working?
      </p>

      {/* By side */}
      {Object.keys(bySide).length > 0 ? (
        <div className="mb-8">
          <h2 className="text-lg font-semibold mb-3">By Side</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(bySide).map(([side, s]) => (
              <div
                key={side}
                className={`rounded-lg border p-5 ${side === "no" ? "border-green-800 bg-green-950/20" : "border-red-800 bg-red-950/20"}`}
              >
                <div className="text-sm text-gray-400 mb-2">
                  BUY_{side.toUpperCase()}
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div>
                    <div className="text-2xl font-bold">
                      {(s.win_rate * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-500">Win Rate</div>
                  </div>
                  <div>
                    <div className="text-2xl font-bold">
                      {s.wins}W / {s.losses}L
                    </div>
                    <div className="text-xs text-gray-500">Record</div>
                  </div>
                  <div>
                    <div
                      className={`text-2xl font-bold ${s.total_pnl >= 0 ? "text-green-400" : "text-red-400"}`}
                    >
                      ${s.total_pnl >= 0 ? "+" : ""}
                      {s.total_pnl.toFixed(2)}
                    </div>
                    <div className="text-xs text-gray-500">Total P&L</div>
                  </div>
                </div>
                <div className="mt-3 text-sm text-gray-400">
                  Avg edge: {(s.avg_edge * 100).toFixed(1)}% &middot; Avg P&L: $
                  {s.avg_pnl.toFixed(3)}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <div className="text-gray-500 py-8 text-center mb-8">
          No settled trades yet. Edge health data will appear after settlements.
        </div>
      )}

      {/* By edge type */}
      {Object.keys(byEdgeType).length > 0 && (
        <div className="mb-8">
          <h2 className="text-lg font-semibold mb-3">By Edge Type</h2>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 border-b border-gray-800">
                  <th className="text-left py-2 px-3">Edge Type</th>
                  <th className="text-right py-2 px-3">Trades</th>
                  <th className="text-right py-2 px-3">Win Rate</th>
                  <th className="text-right py-2 px-3">Avg Edge</th>
                  <th className="text-right py-2 px-3">Total P&L</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(byEdgeType).map(([etype, s]) => (
                  <tr
                    key={etype}
                    className="border-b border-gray-900 hover:bg-gray-900/50"
                  >
                    <td className="py-2 px-3 font-mono text-xs">{etype}</td>
                    <td className="py-2 px-3 text-right">{s.total}</td>
                    <td className="py-2 px-3 text-right">
                      {(s.win_rate * 100).toFixed(0)}%
                    </td>
                    <td className="py-2 px-3 text-right">
                      {(s.avg_edge * 100).toFixed(1)}%
                    </td>
                    <td
                      className={`py-2 px-3 text-right font-medium ${s.total_pnl >= 0 ? "text-green-400" : "text-red-400"}`}
                    >
                      ${s.total_pnl >= 0 ? "+" : ""}
                      {s.total_pnl.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* P&L series */}
      {pnl && pnl.series && pnl.series.length > 0 && (
        <div className="mb-8">
          <h2 className="text-lg font-semibold mb-3">Cumulative P&L</h2>
          <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <div className="flex items-end gap-1 h-32">
              {pnl.series.map(
                (
                  pt: { cumulative_pnl: number; ticker: string },
                  i: number,
                ) => {
                  const maxAbs = Math.max(
                    ...pnl.series.map(
                      (p: { cumulative_pnl: number }) =>
                        Math.abs(p.cumulative_pnl) || 1,
                    ),
                  );
                  const h = Math.max(
                    4,
                    Math.abs((pt.cumulative_pnl / maxAbs) * 100),
                  );
                  return (
                    <div
                      key={i}
                      className={`flex-1 rounded-t ${pt.cumulative_pnl >= 0 ? "bg-green-500" : "bg-red-500"}`}
                      style={{ height: `${h}%` }}
                      title={`$${pt.cumulative_pnl.toFixed(2)} — ${pt.ticker}`}
                    />
                  );
                },
              )}
            </div>
            <div className="text-right text-sm mt-2">
              <span
                className={
                  pnl.total_pnl >= 0 ? "text-green-400" : "text-red-400"
                }
              >
                Total: ${pnl.total_pnl >= 0 ? "+" : ""}
                {pnl.total_pnl.toFixed(2)}
              </span>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}
