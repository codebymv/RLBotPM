"use client";

import { useMode } from "../components/ModeToggle";
import { useBot } from "../components/BotSelector";
import { StatusPill } from "../components/StatusPill";
import { SectionHeader } from "../components/SectionHeader";
import { EmptyState } from "../components/EmptyState";
import { StrategyBadge } from "../components/StrategyBadge";

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

type Props = {
  health: any;
  pnl: any;
  combinedMetrics: any;
};

export default function EdgeHealthClient({ health, pnl, combinedMetrics }: Props) {
  const mode = useMode();
  const bot = useBot();
  const bySide: Record<string, SideStats> = health?.by_side || {};
  const byEdgeType: Record<string, EdgeTypeStats> = health?.by_edge_type || {};
  const rlMetrics = combinedMetrics?.by_strategy?.rl_crypto || {};
  const showKalshi = bot === "all" || bot === "kalshi";
  const showRL = bot === "all" || bot === "rl_crypto";

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-3 sm:p-4 max-w-6xl mx-auto grid-terminal">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between mb-6 pb-4 border-b border-gray-800/60">
        <div>
          <h1 className="text-4xl font-bold tracking-tight mb-2">
            Edge Health{bot !== "all" ? (bot === "rl_crypto" ? " (RL Crypto)" : " (Kalshi)") : ""}
          </h1>
          <p className="text-gray-500 text-base font-mono">
            {bot === "all"
              ? "Statistical edge and performance across strategies"
              : bot === "rl_crypto"
                ? "RL Crypto Bot performance summary"
                : "Statistical edge validation and performance diagnostics"}
          </p>
        </div>
        <div className="mt-4 sm:mt-0">
          <StatusPill mode={mode} />
        </div>
      </div>

      {/* RL Crypto Performance Summary */}
      {showRL && (
        <section className="mb-6">
          <SectionHeader
            title={bot === "all" ? "RL Crypto Bot — Performance" : "Performance Summary"}
            subtitle="Win rate and realized P&L from closed trades"
          />
          <div className="rounded-lg border border-purple-800/40 bg-purple-950/10 p-6">
            {bot === "all" && (
              <div className="flex items-center gap-2 mb-4">
                <StrategyBadge strategy="rl_crypto" />
                <span className="text-xs font-mono text-gray-500">RL Crypto Bot</span>
              </div>
            )}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="text-2xl font-mono font-bold tabular-nums">
                  {rlMetrics.total_trades ?? 0}
                </div>
                <div className="text-[10px] text-gray-600 uppercase tracking-widest font-mono">Total Trades</div>
              </div>
              <div>
                <div className="text-2xl font-mono font-bold tabular-nums">
                  {((rlMetrics.win_rate ?? 0) * 100).toFixed(0)}%
                </div>
                <div className="text-[10px] text-gray-600 uppercase tracking-widest font-mono">Win Rate</div>
              </div>
              <div>
                <div className="text-2xl font-mono font-bold tabular-nums">
                  {(rlMetrics.wins ?? 0)}W / {(rlMetrics.losses ?? 0)}L
                </div>
                <div className="text-[10px] text-gray-600 uppercase tracking-widest font-mono">Record</div>
              </div>
              <div>
                <div
                  className={`text-2xl font-mono font-bold tabular-nums ${
                    (rlMetrics.realized_pnl ?? 0) >= 0 ? "text-green-400" : "text-red-400"
                  }`}
                >
                  ${(rlMetrics.realized_pnl ?? 0) >= 0 ? "+" : ""}
                  {(rlMetrics.realized_pnl ?? 0).toFixed(2)}
                </div>
                <div className="text-[10px] text-gray-600 uppercase tracking-widest font-mono">Realized P&L</div>
              </div>
            </div>
            {(rlMetrics.total_trades ?? 0) === 0 && (
              <p className="text-sm text-gray-500 font-mono mt-4">No closed RL crypto trades yet.</p>
            )}
          </div>
        </section>
      )}

      {/* Kalshi — By Side */}
      {showKalshi && (Object.keys(bySide).length > 0 ? (
        <section className="mb-6">
          <SectionHeader
            title={bot === "all" ? "Kalshi — Performance by Side" : "Performance by Side"}
            subtitle="Win rate, edge realization, and P&L breakdown"
          />
          {bot === "all" && (
            <div className="flex items-center gap-2 mb-3">
              <StrategyBadge strategy="kalshi" />
              <span className="text-xs font-mono text-gray-500">Kalshi Market Bot</span>
            </div>
          )}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(bySide).map(([side, s]) => (
              <div
                key={side}
                className={`rounded-lg border p-6 ${
                  side === "no"
                    ? "border-green-800/60 bg-green-950/10"
                    : "border-red-800/60 bg-red-950/10"
                }`}
              >
                <div className="flex items-center justify-between mb-4">
                  <span className="text-xs font-mono font-bold uppercase tracking-widest text-gray-400">
                    BUY_{side.toUpperCase()}
                  </span>
                  <span
                    className={`px-2.5 py-1 rounded-md text-[9px] font-mono font-bold uppercase ${
                      side === "no"
                        ? "bg-green-900/60 text-green-300"
                        : "bg-red-900/60 text-red-300"
                    }`}
                  >
                    {s.total} TRADES
                  </span>
                </div>
                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div>
                    <div className="text-3xl font-mono font-bold tabular-nums">
                      {(s.win_rate * 100).toFixed(0)}%
                    </div>
                    <div className="text-[10px] text-gray-600 uppercase tracking-widest font-mono">
                      Win Rate
                    </div>
                  </div>
                  <div>
                    <div className="text-3xl font-mono font-bold tabular-nums">
                      {s.wins}W
                    </div>
                    <div className="text-[10px] text-gray-600 uppercase tracking-widest font-mono">
                      {s.losses}L
                    </div>
                  </div>
                  <div>
                    <div
                      className={`text-3xl font-mono font-bold tabular-nums ${
                        s.total_pnl >= 0 ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      ${s.total_pnl >= 0 ? "+" : ""}
                      {s.total_pnl.toFixed(2)}
                    </div>
                    <div className="text-[10px] text-gray-600 uppercase tracking-widest font-mono">
                      Total P&L
                    </div>
                  </div>
                </div>
                <div className="pt-3 border-t border-gray-800/40 text-xs font-mono text-gray-500">
                  Avg Edge: {(s.avg_edge * 100).toFixed(1)}% · Avg P&L: $
                  {s.avg_pnl.toFixed(3)}
                </div>
              </div>
            ))}
          </div>
        </section>
      ) : (
        bot === "all" ? null : (
          <EmptyState
            message={`No settled ${mode} Kalshi trades yet`}
            submessage="Edge health data will appear after settlements"
          />
        )
      ))}

      {/* Kalshi — By Edge Type */}
      {showKalshi && Object.keys(byEdgeType).length > 0 && (
        <section className="mb-6">
          <SectionHeader
            title="Performance by Edge Type"
            subtitle="Model variant effectiveness analysis"
          />
          <div className="hidden md:block overflow-x-auto rounded-lg border border-gray-800/60 bg-gray-900/20">
            <table className="w-full text-sm font-mono">
              <thead>
                <tr className="text-gray-500 border-b border-gray-800/60 text-[10px] uppercase tracking-widest">
                  <th className="text-left py-3 px-4 font-bold">Edge Type</th>
                  <th className="text-right py-3 px-4 font-bold">Trades</th>
                  <th className="text-right py-3 px-4 font-bold">Win Rate</th>
                  <th className="text-right py-3 px-4 font-bold">Avg Edge</th>
                  <th className="text-right py-3 px-4 font-bold">Total P&L</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(byEdgeType).map(([etype, s]) => (
                  <tr
                    key={etype}
                    className="border-b border-gray-900/40 hover:bg-gray-900/40 transition-colors"
                  >
                    <td className="py-3 px-4 text-xs font-bold">{etype}</td>
                    <td className="py-3 px-4 text-right tabular-nums">
                      {s.total}
                    </td>
                    <td className="py-3 px-4 text-right tabular-nums">
                      {(s.win_rate * 100).toFixed(0)}%
                    </td>
                    <td className="py-3 px-4 text-right tabular-nums text-cyan-400">
                      {(s.avg_edge * 100).toFixed(1)}%
                    </td>
                    <td
                      className={`py-3 px-4 text-right font-bold tabular-nums ${
                        s.total_pnl >= 0 ? "text-green-400" : "text-red-400"
                      }`}
                    >
                      ${s.total_pnl >= 0 ? "+" : ""}
                      {s.total_pnl.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Mobile Card View */}
          <div className="md:hidden space-y-3">
            {Object.entries(byEdgeType).map(([etype, s]) => (
              <div
                key={etype}
                className="rounded-lg border border-gray-800/60 bg-gray-900/20 p-4"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="text-sm font-bold font-mono">{etype}</div>
                  <div
                    className={`text-lg font-mono font-bold tabular-nums ${
                      s.total_pnl >= 0 ? "text-green-400" : "text-red-400"
                    }`}
                  >
                    ${s.total_pnl >= 0 ? "+" : ""}
                    {s.total_pnl.toFixed(2)}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3 text-sm font-mono">
                  <div>
                    <div className="text-[10px] text-gray-600 uppercase mb-0.5">
                      Trades
                    </div>
                    <div className="tabular-nums">{s.total}</div>
                  </div>
                  <div>
                    <div className="text-[10px] text-gray-600 uppercase mb-0.5">
                      Win Rate
                    </div>
                    <div className="tabular-nums">
                      {(s.win_rate * 100).toFixed(0)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-gray-600 uppercase mb-0.5">
                      Avg Edge
                    </div>
                    <div className="tabular-nums text-cyan-400">
                      {(s.avg_edge * 100).toFixed(1)}%
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] text-gray-600 uppercase mb-0.5">
                      {s.total} Total
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {/* Kalshi — P&L Series */}
      {showKalshi && pnl && pnl.series && pnl.series.length > 0 && (
        <section className="mb-6">
          <SectionHeader
            title="Cumulative P&L Series"
            subtitle={`Total: $${pnl.total_pnl >= 0 ? "+" : ""}${pnl.total_pnl.toFixed(2)} across ${pnl.series.length} settled trades`}
          />
          <div className="rounded-lg border border-gray-800/60 bg-gray-900/30 p-6">
            <div className="flex items-end gap-1 h-40">
              {pnl.series.map(
                (
                  pt: { cumulative_pnl: number; ticker: string },
                  i: number
                ) => {
                  const maxAbs = Math.max(
                    ...pnl.series.map(
                      (p: { cumulative_pnl: number }) =>
                        Math.abs(p.cumulative_pnl) || 1
                    )
                  );
                  const h = Math.max(
                    4,
                    Math.abs((pt.cumulative_pnl / maxAbs) * 100)
                  );
                  return (
                    <div
                      key={i}
                      className={`flex-1 rounded-t transition-all hover:opacity-80 ${
                        pt.cumulative_pnl >= 0 ? "bg-green-500" : "bg-red-500"
                      }`}
                      style={{ height: `${h}%` }}
                      title={`$${pt.cumulative_pnl.toFixed(2)} — ${pt.ticker}`}
                    />
                  );
                }
              )}
            </div>
            <div className="mt-6 flex items-center justify-between text-xs font-mono">
              <span className="text-gray-600">TRADE SEQUENCE</span>
              <span
                className={`font-bold ${
                  pnl.total_pnl >= 0 ? "text-green-400" : "text-red-400"
                }`}
              >
                CUMULATIVE: ${pnl.total_pnl >= 0 ? "+" : ""}
                {pnl.total_pnl.toFixed(2)}
              </span>
            </div>
          </div>
        </section>
      )}
    </main>
  );
}
