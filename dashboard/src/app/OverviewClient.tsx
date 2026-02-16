"use client";

import Image from "next/image";
import { useMode } from "./components/ModeToggle";
import { KpiCard } from "./components/KpiCard";
import { StatusPill } from "./components/StatusPill";
import { SectionHeader } from "./components/SectionHeader";
import { EmptyState } from "./components/EmptyState";
import { DataFreshness } from "./components/DataFreshness";

function fmt(n: number, decimals = 2) {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

type Props = {
  health: any;
  metrics: any;
  crypto: any;
  bot: any;
  mktStats: any;
};

export default function OverviewClient({
  health,
  metrics,
  crypto,
  bot,
  mktStats,
}: Props) {
  const mode = useMode();

  // Filter metrics by mode
  const modeData = metrics?.mode_breakdown?.[mode];
  const totalTrades = modeData?.total || 0;
  const wins = modeData?.wins || 0;
  const losses = modeData?.losses || 0;
  const pnl = modeData?.realized_pnl || 0;
  const openPositions = modeData?.open_positions || 0;
  const openCost = modeData?.open_cost || 0;
  const winRate = totalTrades > 0 ? (wins / totalTrades) * 100 : 0;

  // Filter recent trades by mode
  const recentTrades = metrics?.recent_trades?.filter(
    (t: any) => (t.mode || "paper") === mode
  ) || [];

  // Get side breakdown for current mode
  const sideBreakdown = metrics?.side_breakdown
    ? Object.entries(metrics.side_breakdown).map(([side, data]: [string, any]) => ({
        side,
        ...data,
      }))
    : [];

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-3 sm:p-4 max-w-6xl mx-auto grid-terminal">
      {/* Header with System Status */}
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between mb-6 pb-4 border-b border-gray-800/60">
        <div>
          <div className="flex items-center gap-3 mb-3">
            <Image
              src="/rltrade-icon.png"
              alt="RLTrade"
              width={56}
              height={56}
              className="w-14 h-14"
            />
            <Image
              src="/rltrade-text.png"
              alt="RLTrade"
              width={200}
              height={56}
              className="h-12 w-auto"
            />
          </div>
          <p className="text-gray-500 text-base font-mono tracking-wide">
            Reinforcement Learning Crypto Prediction Market Bot
          </p>
          <p className="text-gray-600 text-sm font-mono mt-1">
            Kalshi · Coinbase · PPO
          </p>
        </div>
        <div className="flex flex-col gap-2 mt-4 sm:mt-0">
          <div className="flex gap-2 items-center">
            <StatusPill mode={mode} />
            <DataFreshness lastUpdated={metrics?.last_updated} />
          </div>
          <div className="flex gap-2 text-xs">
            <SystemStatus
              label="API"
              ok={health.status === "healthy"}
              text={health.status}
            />
            <SystemStatus
              label="DB"
              ok={health.database === "connected"}
              text={health.database || "unknown"}
            />
          </div>
        </div>
      </div>

      {/* Trading Performance */}
      <section className="mb-6">
        <SectionHeader
          title="Trading Performance"
          subtitle={`Performance metrics for ${mode.toUpperCase()} trading mode`}
          actionHref="/positions"
          actionLabel="VIEW ALL POSITIONS →"
        />

        {metrics && modeData ? (
          <>
            {/* Primary KPIs */}
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 mb-5">
              <KpiCard
                label="Realized P&L"
                value={`$${pnl >= 0 ? "+" : ""}${fmt(pnl)}`}
                mode={mode}
                trend={pnl > 0 ? "up" : pnl < 0 ? "down" : "flat"}
              />
              <KpiCard
                label="Total Trades"
                value={totalTrades}
                sublabel={`${wins}W · ${losses}L`}
                mode="neutral"
              />
              <KpiCard
                label="Win Rate"
                value={totalTrades > 0 ? `${winRate.toFixed(1)}%` : "—"}
                sublabel={totalTrades > 0 ? `${wins} wins` : "No trades yet"}
                mode="neutral"
              />
              <KpiCard
                label="Open Positions"
                value={openPositions}
                sublabel={`$${fmt(openCost)} deployed`}
                mode="neutral"
              />
              <KpiCard
                label="Settled Markets"
                value={mktStats ? mktStats.total_markets.toLocaleString() : "—"}
                sublabel={mktStats ? `${mktStats.total_events} events` : "Kalshi data"}
                mode="neutral"
              />
            </div>

            {/* Side Breakdown */}
            {sideBreakdown.length > 0 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {sideBreakdown.map((data: any) => {
                  const sideWins = data.wins || 0;
                  const sideTotal = data.total || 0;
                  const sideLosses = sideTotal - sideWins;
                  const wr = sideTotal > 0 ? ((sideWins / sideTotal) * 100).toFixed(0) : "0";
                  return (
                    <div
                      key={data.side}
                      className={`rounded-lg border p-5 ${
                        data.side === "no"
                          ? "border-green-800/60 bg-green-950/10"
                          : "border-red-800/60 bg-red-950/10"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-3">
                        <span className="text-xs font-mono font-bold uppercase tracking-widest text-gray-400">
                          BUY_{data.side.toUpperCase()}
                        </span>
                        <span
                          className={`text-[10px] font-mono font-bold px-2.5 py-1 rounded-md ${
                            data.side === "no"
                              ? "bg-green-900/60 text-green-300"
                              : "bg-red-900/60 text-red-300"
                          }`}
                        >
                          {wr}% WIN
                        </span>
                      </div>
                      <div className="text-2xl font-mono font-bold mb-1">
                        {sideWins}W / {sideLosses}L
                      </div>
                      <div className="text-sm font-mono text-gray-400">
                        P&L: <span className={data.pnl >= 0 ? "text-green-400" : "text-red-400"}>
                          ${data.pnl >= 0 ? "+" : ""}{fmt(data.pnl)}
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </>
        ) : (
          <EmptyState
            message={`No ${mode} trading data available`}
            submessage="Start the paper trader or switch modes"
          />
        )}
      </section>

      {/* Live Crypto Prices */}
      <section className="mb-6">
        <SectionHeader
          title="Crypto Spot Prices"
          subtitle="Real-time market data from Coinbase"
          actionHref="/crypto"
          actionLabel="DETAILED VIEW →"
        />

        {crypto && crypto.prices ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {Object.entries(
              crypto.prices as Record<
                string,
                { price?: number; bid?: number; ask?: number; error?: string }
              >
            ).map(([asset, data]) => (
              <div
                key={asset}
                className="rounded-lg border border-gray-800/60 bg-gray-900/40 p-4 hover:bg-gray-900/60 transition-colors"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-bold text-sm font-mono">{asset}</span>
                  <span className="text-[9px] text-gray-600 font-mono">
                    σ{((crypto.volatilities?.[asset] || 0) * 100).toFixed(0)}%
                  </span>
                </div>
                {data.price ? (
                  <>
                    <div className="text-xl font-mono font-bold tabular-nums">
                      ${fmt(data.price, asset === "DOGE" || asset === "XRP" ? 4 : 2)}
                    </div>
                    {data.bid && data.ask ? (
                      <div className="text-[9px] text-gray-600 mt-1.5 font-mono">
                        {fmt(data.bid, 2)} / {fmt(data.ask, 2)}
                      </div>
                    ) : null}
                  </>
                ) : (
                  <div className="text-red-400 text-xs font-mono">
                    {data.error || "unavailable"}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <EmptyState message="Unable to fetch crypto prices" />
        )}
      </section>

      {/* Bot Operations */}
      <section className="mb-6">
        <SectionHeader
          title="Bot Configuration"
          subtitle="Strategy parameters and operational status"
          actionHref="/bot-status"
          actionLabel="FULL STATUS →"
        />

        {bot ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <KpiCard
              label="Total Sessions"
              value={bot.total_sessions}
              sublabel={
                bot.last_trade_at
                  ? `Last: ${new Date(bot.last_trade_at).toLocaleDateString()}`
                  : "No trades yet"
              }
              mode="neutral"
            />
            <KpiCard
              label="Strategy"
              value="BUY_NO"
              sublabel="Lognormal edge model"
              mode="neutral"
            />
            <KpiCard
              label="Edge Range"
              value={`${(bot.strategy.min_edge * 100).toFixed(0)}-${(bot.strategy.max_edge * 100).toFixed(0)}%`}
              sublabel={`Price ${bot.strategy.min_price}-${bot.strategy.max_price}¢`}
              mode="neutral"
            />
            <KpiCard
              label="Assets Tracked"
              value={bot.strategy.assets.length}
              sublabel={bot.strategy.assets.slice(0, 2).join(", ") + (bot.strategy.assets.length > 2 ? "..." : "")}
              mode="neutral"
            />
          </div>
        ) : (
          <EmptyState message="Bot status unavailable" />
        )}
      </section>

      {/* Recent Trades */}
      <section className="mb-6">
        <SectionHeader
          title={`Recent ${mode.charAt(0).toUpperCase() + mode.slice(1)} Trades`}
          subtitle={`Latest executions in ${mode} mode`}
          actionHref="/positions"
          actionLabel="ALL POSITIONS →"
        />

        {recentTrades.length > 0 ? (
          <>
            {/* Desktop Table View */}
            <div className="hidden md:block overflow-x-auto rounded-lg border border-gray-800/60 bg-gray-900/20">
              <table className="w-full text-sm font-mono">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-800/60 text-[10px] uppercase tracking-widest">
                    <th className="text-left py-3 px-4 font-bold">Ticker</th>
                    <th className="text-left py-3 px-4 font-bold">Side</th>
                    <th className="text-right py-3 px-4 font-bold">Price</th>
                    <th className="text-right py-3 px-4 font-bold">Edge</th>
                    <th className="text-right py-3 px-4 font-bold">Qty</th>
                    <th className="text-right py-3 px-4 font-bold">Cost</th>
                    <th className="text-left py-3 px-4 font-bold">Status</th>
                    <th className="text-right py-3 px-4 font-bold">P&L</th>
                  </tr>
                </thead>
                <tbody>
                  {recentTrades.slice(0, 10).map((t: any, i: number) => (
                    <tr
                      key={`${t.ticker}-${i}`}
                      className="border-b border-gray-900/40 hover:bg-gray-900/40 transition-colors"
                    >
                      <td className="py-3 px-4 text-xs font-bold">
                        {t.ticker}
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`px-2 py-0.5 rounded-md text-[9px] font-bold uppercase ${
                            t.side === "no"
                              ? "bg-green-900/60 text-green-300"
                              : "bg-red-900/60 text-red-300"
                          }`}
                        >
                          {t.side}
                        </span>
                      </td>
                      <td className="py-3 px-4 text-right tabular-nums">
                        {t.entry_price_cents}¢
                      </td>
                      <td className="py-3 px-4 text-right tabular-nums text-gray-400">
                        {t.edge ? `${(t.edge * 100).toFixed(1)}%` : "—"}
                      </td>
                      <td className="py-3 px-4 text-right tabular-nums">
                        {t.contracts}
                      </td>
                      <td className="py-3 px-4 text-right tabular-nums">
                        ${fmt(t.cost)}
                      </td>
                      <td className="py-3 px-4">
                        <span
                          className={`text-[9px] uppercase font-bold ${
                            t.status === "open" ? "text-amber-400" : "text-gray-500"
                          }`}
                        >
                          {t.status}
                        </span>
                      </td>
                      <td
                        className={`py-3 px-4 text-right font-bold tabular-nums ${
                          t.pnl && t.pnl > 0
                            ? "text-green-400"
                            : t.pnl && t.pnl < 0
                              ? "text-red-400"
                              : "text-gray-400"
                        }`}
                      >
                        {t.pnl !== null
                          ? `$${t.pnl >= 0 ? "+" : ""}${fmt(t.pnl)}`
                          : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Mobile Card View */}
            <div className="md:hidden space-y-3">
              {recentTrades.slice(0, 10).map((t: any, i: number) => (
                <div
                  key={`${t.ticker}-${i}`}
                  className="rounded-lg border border-gray-800/60 bg-gray-900/20 p-4"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="font-mono text-sm font-bold mb-1">
                        {t.ticker}
                      </div>
                      <span
                        className={`inline-block px-2 py-0.5 rounded-md text-[9px] font-bold uppercase ${
                          t.side === "no"
                            ? "bg-green-900/60 text-green-300"
                            : "bg-red-900/60 text-red-300"
                        }`}
                      >
                        {t.side}
                      </span>
                    </div>
                    <div
                      className={`text-lg font-mono font-bold tabular-nums ${
                        t.pnl && t.pnl > 0
                          ? "text-green-400"
                          : t.pnl && t.pnl < 0
                            ? "text-red-400"
                            : "text-gray-400"
                      }`}
                    >
                      {t.pnl !== null
                        ? `$${t.pnl >= 0 ? "+" : ""}${fmt(t.pnl)}`
                        : "—"}
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-3 text-sm font-mono">
                    <div>
                      <div className="text-[10px] text-gray-600 uppercase mb-0.5">
                        Price
                      </div>
                      <div className="tabular-nums">{t.entry_price_cents}¢</div>
                    </div>
                    <div>
                      <div className="text-[10px] text-gray-600 uppercase mb-0.5">
                        Edge
                      </div>
                      <div className="tabular-nums text-gray-400">
                        {t.edge ? `${(t.edge * 100).toFixed(1)}%` : "—"}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-gray-600 uppercase mb-0.5">
                        Qty
                      </div>
                      <div className="tabular-nums">{t.contracts}</div>
                    </div>
                    <div>
                      <div className="text-[10px] text-gray-600 uppercase mb-0.5">
                        Cost
                      </div>
                      <div className="tabular-nums">${fmt(t.cost)}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </>
        ) : (
          <EmptyState
            message={`No ${mode} trades recorded`}
            submessage="Trades will appear here once executed"
          />
        )}
      </section>
    </main>
  );
}

function SystemStatus({ label, ok, text }: { label: string; ok: boolean; text: string }) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-mono font-bold uppercase tracking-wider ${
        ok ? "bg-green-900/40 text-green-400" : "bg-red-900/40 text-red-400"
      }`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${ok ? "bg-green-400" : "bg-red-400"}`} />
      {label}: {text}
    </span>
  );
}
