"use client";

import Image from "next/image";
import { useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { useMode } from "./components/ModeToggle";
import { useBot } from "./components/BotSelector";
import { KpiCard } from "./components/KpiCard";
import { StatusPill } from "./components/StatusPill";
import { SectionHeader } from "./components/SectionHeader";
import { EmptyState } from "./components/EmptyState";
import { DataFreshness } from "./components/DataFreshness";
import { StrategyBadge } from "./components/StrategyBadge";
import { PnlChart } from "./components/PnlChart";
import { RiskStatusPanel } from "./components/RiskStatusPanel";
import {
  type BotStatusResponse,
  type CombinedMetricsResponse,
  fetchCombinedMetrics,
  fetchHealth,
  fetchCryptoPrices,
  fetchBotStatus,
  fetchMarketStats,
  fetchHeartbeat,
  fetchRiskStatus,
  fetchTradesSummary,
  fetchPnlSeriesTyped,
} from "../lib/api";
import { fmt } from "../lib/format";

type Props = {
  health: any;
  combinedMetrics: CombinedMetricsResponse | null;
  crypto: any;
  bot: BotStatusResponse | null;
  mktStats: any;
};

export default function OverviewClient({
  health: initialHealth,
  combinedMetrics: initialCombinedMetrics,
  crypto: initialCrypto,
  bot: initialBotStatus,
  mktStats: initialMktStats,
}: Props) {
  const mode = useMode();
  const bot = useBot();
  const searchParams = useSearchParams();
  const queryString = searchParams.toString();
  const link = (path: string) => (queryString ? `${path}?${queryString}` : path);

  const { data: combinedMetrics, dataUpdatedAt: metricsUpdatedAt } = useQuery({
    queryKey: ["combinedMetrics", mode],
    queryFn: () => fetchCombinedMetrics(mode),
    initialData: initialCombinedMetrics,
    refetchInterval: 15_000,
  });

  const { data: health } = useQuery({
    queryKey: ["health"],
    queryFn: fetchHealth,
    initialData: initialHealth,
    refetchInterval: 60_000,
  });

  const { data: crypto } = useQuery({
    queryKey: ["cryptoPrices"],
    queryFn: fetchCryptoPrices,
    initialData: initialCrypto,
    refetchInterval: 60_000,
  });

  const { data: botStatus } = useQuery({
    queryKey: ["botStatus"],
    queryFn: fetchBotStatus,
    initialData: initialBotStatus,
    refetchInterval: 60_000,
  });

  const { data: mktStats } = useQuery({
    queryKey: ["marketStats"],
    queryFn: fetchMarketStats,
    initialData: initialMktStats,
    refetchInterval: 60_000,
  });

  const { data: heartbeat } = useQuery({
    queryKey: ["heartbeat"],
    queryFn: () => fetchHeartbeat(),
    refetchInterval: 30_000,
  });

  const { data: riskStatus } = useQuery({
    queryKey: ["riskStatus"],
    queryFn: fetchRiskStatus,
    refetchInterval: 30_000,
  });

  const { data: tradesSummary } = useQuery({
    queryKey: ["tradesSummary"],
    queryFn: fetchTradesSummary,
    refetchInterval: 60_000,
  });

  const { data: pnlSeries } = useQuery({
    queryKey: ["pnlSeries", mode],
    queryFn: () => fetchPnlSeriesTyped(mode),
    refetchInterval: 30_000,
  });

  const kalshiData = combinedMetrics?.by_strategy?.kalshi ?? null;
  const rlData = combinedMetrics?.by_strategy?.rl_crypto ?? null;
  const combinedData = combinedMetrics?.combined ?? null;

  const strategyData: import("../lib/api").StrategyMetrics | null =
    bot === "all"
      ? combinedData
      : bot === "kalshi"
        ? kalshiData
        : rlData;

  const totalTrades = strategyData?.total_trades ?? 0;
  const wins = strategyData?.wins ?? 0;
  const losses = strategyData?.losses ?? 0;
  const settledTrades = strategyData?.settled_trades ?? wins + losses;
  const pnl = strategyData?.realized_pnl ?? 0;
  const openPositions = strategyData?.open_positions ?? 0;
  const openCost = strategyData?.open_cost ?? 0;
  const currentSession = combinedMetrics?.current_session ?? null;
  const sessionPnl = currentSession?.realized_pnl ?? null;
  const apiSettledWinRate = strategyData?.win_rate;
  const settledWinRate =
    typeof apiSettledWinRate === "number"
      ? apiSettledWinRate * 100
      : settledTrades > 0
        ? (wins / settledTrades) * 100
        : 0;

  const kalshiTrades = (kalshiData?.recent_trades || []).filter(
    (t: any) => (t.mode || "paper") === mode
  );
  const rlTrades = (rlData?.recent_trades || []).filter(
    (t: any) => (t.mode || "paper") === mode
  );
  const recentTrades =
    bot === "all"
      ? [...kalshiTrades, ...rlTrades]
          .sort((a: any, b: any) => {
            const dateA = a.opened_at || a.settled_at || "";
            const dateB = b.opened_at || b.settled_at || "";
            return dateB.localeCompare(dateA);
          })
          .slice(0, 20)
      : bot === "kalshi"
        ? kalshiTrades
        : rlTrades;

  const sideBreakdown =
    (bot === "all" || bot === "kalshi") && kalshiData?.side_breakdown
      ? Object.entries(kalshiData.side_breakdown!).map(([side, data]: [string, any]) => ({
          side,
          ...data,
        }))
      : [];
  const showSideBreakdown = (bot === "all" || bot === "kalshi") && sideBreakdown.length > 0;
  const showStrategyBreakdown = bot === "all";

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 px-4 sm:px-6 xl:px-10 py-6 max-w-[1600px] mx-auto grid-terminal">

      {/* ─── Header ─── */}
      <header className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 pb-6 mb-8 border-b border-gray-800/50">
        <div>
          <div className="flex items-center gap-3 mb-2">
            <Image src="/rltrade-icon.png" alt="RLTrade" width={48} height={48} className="w-12 h-12" />
            <Image src="/rltrade-text.png" alt="RLTrade" width={180} height={48} className="h-10 w-auto" />
          </div>
          <p className="text-gray-500 text-sm tracking-wide">
            Reinforcement Learning Crypto Prediction Market Bot
          </p>
        </div>
        <div className="flex flex-col items-start sm:items-end gap-2">
          <div className="flex gap-2 items-center flex-wrap">
            <StatusPill mode={mode} />
            <DataFreshness
              lastUpdated={
                metricsUpdatedAt
                  ? new Date(metricsUpdatedAt).toISOString()
                  : combinedMetrics?.timestamp
              }
            />
          </div>
          <div className="flex gap-2 text-xs flex-wrap">
            <SystemStatus label="API" ok={health?.status === "healthy"} text={health?.status ?? "unknown"} />
            <SystemStatus label="DB" ok={health?.database === "connected"} text={health?.database ?? "unknown"} />
            <BotLiveness heartbeat={heartbeat ?? null} />
          </div>
        </div>
      </header>

      {/* ─── 1. Portfolio Summary ─── */}
      <section aria-labelledby="portfolio-heading" className="mb-10">
        <SectionHeader
          id="portfolio-heading"
          title="Portfolio Overview"
          subtitle={
            bot === "all"
              ? `Combined ${mode} performance across all strategies`
              : `${bot === "rl_crypto" ? "RL Crypto" : "Kalshi"} · ${mode} mode`
          }
          actionHref={link("/positions")}
          actionLabel="Positions →"
        />

        {combinedMetrics && strategyData ? (
          <div className="space-y-6">

            {/* Hero P&L + Key Metrics */}
            <div className="grid grid-cols-1 lg:grid-cols-5 gap-4">
              <div className="lg:col-span-1 rounded-xl border border-gray-800/60 bg-gray-900/30 p-6 flex flex-col justify-center">
                <div className="text-xs uppercase tracking-widest text-gray-500 mb-2 font-medium">
                  Realized P&L
                </div>
                <div className={`text-4xl sm:text-5xl font-mono font-bold tabular-nums leading-tight ${pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                  ${pnl >= 0 ? "+" : ""}{fmt(pnl)}
                </div>
                {sessionPnl !== null && (
                  <div className="text-sm font-mono text-gray-500 mt-2">
                    Current session: ${sessionPnl >= 0 ? "+" : ""}{fmt(sessionPnl)}
                  </div>
                )}
              </div>

              <div className="lg:col-span-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
                <KpiCard
                  label="Win Rate"
                  value={settledTrades > 0 ? `${settledWinRate.toFixed(1)}%` : "—"}
                  sublabel={settledTrades > 0 ? `${wins}W / ${losses}L` : "No settled trades"}
                  mode="neutral"
                />
                <KpiCard
                  label="Profit Factor"
                  value={
                    tradesSummary && tradesSummary.total_trades > 0
                      ? tradesSummary.profit_factor === Infinity ? "∞" : fmt(tradesSummary.profit_factor)
                      : "—"
                  }
                  sublabel={
                    tradesSummary && tradesSummary.total_trades > 0
                      ? `+$${fmt(tradesSummary.avg_profit)} / -$${fmt(Math.abs(tradesSummary.avg_loss))}`
                      : "Avg win / loss"
                  }
                  mode="neutral"
                  trend={tradesSummary && tradesSummary.profit_factor > 1.3 ? "up" : tradesSummary && tradesSummary.profit_factor < 1.0 ? "down" : "flat"}
                />
                <KpiCard
                  label="Total Trades"
                  value={totalTrades}
                  sublabel={`${settledTrades} settled · ${openPositions} open`}
                  mode="neutral"
                />
                <KpiCard
                  label="Deployed"
                  value={`$${fmt(openCost)}`}
                  sublabel={`${openPositions} open position${openPositions !== 1 ? "s" : ""}`}
                  mode="neutral"
                />
              </div>
            </div>

            {/* Strategy Breakdown (when viewing "all") */}
            {showStrategyBreakdown && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <StrategyCard
                  strategy="kalshi"
                  label="Kalshi Prediction Markets"
                  pnl={kalshiData?.realized_pnl ?? 0}
                  settled={kalshiData?.settled_trades ?? (kalshiData?.wins ?? 0) + (kalshiData?.losses ?? 0)}
                  wins={kalshiData?.wins ?? 0}
                  losses={kalshiData?.losses ?? 0}
                  winRate={(kalshiData?.win_rate ?? 0) * 100}
                  openPositions={kalshiData?.open_positions ?? 0}
                />
                <StrategyCard
                  strategy="rl_crypto"
                  label="RL Crypto Spot Trading"
                  pnl={rlData?.realized_pnl ?? 0}
                  settled={rlData?.settled_trades ?? (rlData?.wins ?? 0) + (rlData?.losses ?? 0)}
                  wins={rlData?.wins ?? 0}
                  losses={rlData?.losses ?? 0}
                  winRate={(rlData?.win_rate ?? 0) * 100}
                  openPositions={rlData?.open_positions ?? 0}
                />
              </div>
            )}

            {/* Side Breakdown (Kalshi BUY_NO / BUY_YES) */}
            {showSideBreakdown && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                {sideBreakdown.map((data: any) => {
                  const sideWins = data.wins || 0;
                  const sideTotal = data.total || 0;
                  const sideLosses = sideTotal - sideWins;
                  const wr = sideTotal > 0 ? (sideWins / sideTotal) * 100 : 0;
                  const isNo = data.side === "no";
                  return (
                    <div
                      key={data.side}
                      className={`rounded-xl border p-5 ${
                        isNo
                          ? "border-green-800/50 bg-green-950/10"
                          : "border-red-800/50 bg-red-950/10"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-4">
                        <h4 className="text-sm font-bold uppercase tracking-wider text-gray-300">
                          BUY_{data.side.toUpperCase()}
                        </h4>
                        <span
                          className={`text-xs font-mono font-bold px-3 py-1 rounded-full ${
                            isNo ? "bg-green-900/60 text-green-300" : "bg-red-900/60 text-red-300"
                          }`}
                        >
                          {wr.toFixed(0)}% win
                        </span>
                      </div>
                      <div className="flex items-baseline justify-between">
                        <div>
                          <div className="text-3xl font-mono font-bold tabular-nums">
                            {sideWins}W <span className="text-gray-600">/</span> {sideLosses}L
                          </div>
                          <div className="text-sm font-mono text-gray-500 mt-1">
                            {sideTotal} settled trades
                          </div>
                        </div>
                        <div className="text-right">
                          <div className={`text-2xl font-mono font-bold tabular-nums ${data.pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
                            ${data.pnl >= 0 ? "+" : ""}{fmt(data.pnl)}
                          </div>
                          <div className="text-xs font-mono text-gray-600 mt-1">P&L</div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        ) : (
          <EmptyState
            message={`No ${mode} data for ${bot === "all" ? "any strategy" : bot === "rl_crypto" ? "RL Crypto" : "Kalshi"}`}
            submessage="Start the paper trader or switch modes"
          />
        )}
      </section>

      {/* ─── 2. Risk & Circuit Breakers ─── */}
      <section className="mb-10">
        <RiskStatusPanel risk={riskStatus ?? null} />
      </section>

      {/* ─── 3. Cumulative P&L Chart ─── */}
      {pnlSeries && pnlSeries.series && pnlSeries.series.length > 0 && (
        <section aria-labelledby="pnl-chart-heading" className="mb-10">
          <SectionHeader id="pnl-chart-heading" title="Cumulative P&L" subtitle="Settled trade performance over time" />
          <div className="rounded-xl border border-gray-800/60 bg-gray-900/20 p-4 sm:p-6">
            <PnlChart series={pnlSeries.series} height={280} />
          </div>
        </section>
      )}

      {/* ─── 4. Crypto Spot Prices ─── */}
      <section aria-labelledby="crypto-heading" className="mb-10">
        <SectionHeader
          id="crypto-heading"
          title="Crypto Spot Prices"
          subtitle="Real-time market data from Coinbase"
          actionHref={link("/crypto")}
          actionLabel="Detailed View →"
        />

        {crypto && crypto.prices ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 xl:grid-cols-6 gap-3">
            {Object.entries(
              crypto.prices as Record<
                string,
                { price?: number; bid?: number; ask?: number; error?: string; change_24h_pct?: number; volume_24h?: number }
              >
            ).map(([asset, data]) => (
              <div
                key={asset}
                className="rounded-xl border border-gray-800/50 bg-gray-900/30 p-4 hover:bg-gray-900/50 transition-colors"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-bold text-sm">{asset}</span>
                  <span className="text-[10px] text-gray-600 font-mono tabular-nums">
                    σ{((crypto.volatilities?.[asset] || 0) * 100).toFixed(0)}%
                  </span>
                </div>
                {data.price ? (
                  <>
                    <div className="text-xl font-mono font-bold tabular-nums">
                      ${fmt(data.price, asset === "DOGE" || asset === "XRP" ? 4 : 2)}
                    </div>
                    {data.change_24h_pct != null && (
                      <div
                        className={`text-xs font-mono font-bold mt-1.5 ${
                          data.change_24h_pct >= 0 ? "text-green-400" : "text-red-400"
                        }`}
                      >
                        {data.change_24h_pct >= 0 ? "↑" : "↓"}
                        {Math.abs(data.change_24h_pct * 100).toFixed(2)}%
                      </div>
                    )}
                    {data.bid && data.ask ? (
                      <div className="text-[10px] text-gray-600 mt-1.5 font-mono tabular-nums">
                        {fmt(data.bid, 2)} / {fmt(data.ask, 2)}
                      </div>
                    ) : null}
                  </>
                ) : (
                  <div className="text-red-400 text-xs font-mono">{data.error || "unavailable"}</div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <EmptyState message="Unable to fetch crypto prices" />
        )}
      </section>

      {/* ─── 5. Bot Configuration ─── */}
      <section aria-labelledby="config-heading" className="mb-10">
        <SectionHeader
          id="config-heading"
          title="Bot Configuration"
          subtitle="Strategy parameters and operational status"
          actionHref={link("/bot-status")}
          actionLabel="Full Status →"
        />

        {botStatus ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <KpiCard
              label="Total Sessions"
              value={botStatus.total_sessions}
              sublabel={
                botStatus.last_trade_at
                  ? `Last: ${new Date(botStatus.last_trade_at).toLocaleDateString()}`
                  : "No trades yet"
              }
              mode="neutral"
            />
            <KpiCard
              label="Strategy"
              value={
                botStatus.strategy.side_filter === "both"
                  ? "BUY_BOTH"
                  : `BUY_${botStatus.strategy.side_filter.toUpperCase()}`
              }
              sublabel={botStatus.strategy.allow_buy_yes ? "BUY_YES enabled" : "BUY_YES disabled"}
              mode="neutral"
            />
            <KpiCard
              label="Edge Range"
              value={`${(botStatus.strategy.min_edge * 100).toFixed(0)}–${(botStatus.strategy.max_edge * 100).toFixed(0)}%`}
              sublabel={`Price ${botStatus.strategy.min_price}–${botStatus.strategy.max_price}¢`}
              mode="neutral"
            />
            <KpiCard
              label="Assets Tracked"
              value={botStatus.strategy.assets.length}
              sublabel={botStatus.strategy.assets.slice(0, 3).join(", ") + (botStatus.strategy.assets.length > 3 ? "…" : "")}
              mode="neutral"
            />
          </div>
        ) : (
          <EmptyState message="Bot status unavailable" />
        )}

        {mktStats && (
          <div className="mt-3">
            <KpiCard
              label="Kalshi Dataset"
              value={mktStats.total_markets.toLocaleString() + " markets"}
              sublabel={`${mktStats.total_events} events · training coverage`}
              mode="neutral"
              className="max-w-sm"
            />
          </div>
        )}
      </section>

      {/* ─── 6. Recent Trades ─── */}
      <section aria-labelledby="trades-heading" className="mb-10">
        <SectionHeader
          id="trades-heading"
          title={`Recent Trades${bot !== "all" ? ` · ${bot === "rl_crypto" ? "RL Crypto" : "Kalshi"}` : ""}`}
          subtitle={`Latest ${mode} executions`}
          actionHref={link("/positions")}
          actionLabel="All Positions →"
        />

        {recentTrades.length > 0 ? (
          <>
            {/* Desktop */}
            <div className="hidden md:block rounded-xl border border-gray-800/50 bg-gray-900/20 overflow-hidden">
              <table className="w-full text-sm font-mono">
                <thead>
                  <tr className="text-gray-500 border-b border-gray-800/50 text-[10px] uppercase tracking-widest">
                    {bot === "all" && <th className="text-left py-3.5 px-5 font-bold">Bot</th>}
                    <th className="text-left py-3.5 px-5 font-bold">Ticker</th>
                    <th className="text-left py-3.5 px-5 font-bold">Side</th>
                    <th className="text-right py-3.5 px-5 font-bold">Price</th>
                    <th className="text-right py-3.5 px-5 font-bold">Edge</th>
                    <th className="text-right py-3.5 px-5 font-bold">Qty</th>
                    <th className="text-right py-3.5 px-5 font-bold">Cost</th>
                    <th className="text-left py-3.5 px-5 font-bold">Status</th>
                    <th className="text-right py-3.5 px-5 font-bold">P&amp;L</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-900/40">
                  {recentTrades.slice(0, 12).map((t: any, i: number) => (
                    <tr
                      key={`${t.ticker ?? t.symbol}-${t.strategy ?? bot}-${i}`}
                      className="hover:bg-gray-900/30 transition-colors"
                    >
                      {bot === "all" && (
                        <td className="py-3.5 px-5">
                          {t.strategy ? <StrategyBadge strategy={t.strategy} /> : "—"}
                        </td>
                      )}
                      <td className="py-3.5 px-5 text-xs font-bold">{t.ticker ?? t.symbol}</td>
                      <td className="py-3.5 px-5">
                        {t.side != null ? (
                          <span
                            className={`px-2 py-0.5 rounded-md text-[9px] font-bold uppercase ${
                              t.side === "no" ? "bg-green-900/50 text-green-300" : "bg-red-900/50 text-red-300"
                            }`}
                          >
                            {t.side}
                          </span>
                        ) : (
                          <span className="text-gray-600">—</span>
                        )}
                      </td>
                      <td className="py-3.5 px-5 text-right tabular-nums">
                        {t.entry_price_cents != null ? `${t.entry_price_cents}¢` : t.entry_price != null ? `$${fmt(t.entry_price)}` : "—"}
                      </td>
                      <td className="py-3.5 px-5 text-right tabular-nums text-gray-400">
                        {t.edge != null ? `${(t.edge * 100).toFixed(1)}%` : "—"}
                      </td>
                      <td className="py-3.5 px-5 text-right tabular-nums">{t.contracts ?? 1}</td>
                      <td className="py-3.5 px-5 text-right tabular-nums">${fmt(t.cost)}</td>
                      <td className="py-3.5 px-5">
                        <span className={`text-[10px] uppercase font-bold ${t.status === "open" ? "text-amber-400" : "text-gray-500"}`}>
                          {t.status ?? "—"}
                        </span>
                      </td>
                      <td className={`py-3.5 px-5 text-right font-bold tabular-nums ${
                        t.pnl != null && t.pnl > 0 ? "text-green-400" : t.pnl != null && t.pnl < 0 ? "text-red-400" : "text-gray-500"
                      }`}>
                        {t.pnl != null ? `$${t.pnl >= 0 ? "+" : ""}${fmt(t.pnl)}` : "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Mobile */}
            <div className="md:hidden space-y-3">
              {recentTrades.slice(0, 8).map((t: any, i: number) => (
                <div
                  key={`${t.ticker ?? t.symbol}-${t.strategy ?? bot}-m-${i}`}
                  className="rounded-xl border border-gray-800/50 bg-gray-900/20 p-4"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <div className="flex items-center gap-2 flex-wrap mb-1">
                        {bot === "all" && t.strategy && <StrategyBadge strategy={t.strategy} />}
                        <span className="font-mono text-sm font-bold">{t.ticker ?? t.symbol}</span>
                      </div>
                      {t.side != null && (
                        <span
                          className={`inline-block px-2 py-0.5 rounded-md text-[9px] font-bold uppercase ${
                            t.side === "no" ? "bg-green-900/50 text-green-300" : "bg-red-900/50 text-red-300"
                          }`}
                        >
                          {t.side}
                        </span>
                      )}
                    </div>
                    <div className={`text-lg font-mono font-bold tabular-nums ${
                      t.pnl != null && t.pnl > 0 ? "text-green-400" : t.pnl != null && t.pnl < 0 ? "text-red-400" : "text-gray-500"
                    }`}>
                      {t.pnl != null ? `$${t.pnl >= 0 ? "+" : ""}${fmt(t.pnl)}` : "—"}
                    </div>
                  </div>
                  <div className="grid grid-cols-3 gap-3 text-sm font-mono">
                    <div>
                      <div className="text-[10px] text-gray-600 uppercase mb-0.5">Price</div>
                      <div className="tabular-nums">
                        {t.entry_price_cents != null ? `${t.entry_price_cents}¢` : t.entry_price != null ? `$${fmt(t.entry_price)}` : "—"}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-gray-600 uppercase mb-0.5">Qty</div>
                      <div className="tabular-nums">{t.contracts ?? 1}</div>
                    </div>
                    <div>
                      <div className="text-[10px] text-gray-600 uppercase mb-0.5">Cost</div>
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

/* ─── Local Components ─── */

function StrategyCard({
  strategy,
  label,
  pnl,
  settled,
  wins,
  losses,
  winRate,
  openPositions,
}: {
  strategy: "kalshi" | "rl_crypto";
  label: string;
  pnl: number;
  settled: number;
  wins: number;
  losses: number;
  winRate: number;
  openPositions: number;
}) {
  const isKalshi = strategy === "kalshi";
  return (
    <div
      className={`rounded-xl border p-5 ${
        isKalshi ? "border-blue-800/40 bg-blue-950/10" : "border-purple-800/40 bg-purple-950/10"
      }`}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2.5">
          <StrategyBadge strategy={strategy} />
          <span className="text-sm font-medium text-gray-300">{label}</span>
        </div>
        {openPositions > 0 && (
          <span className="text-[10px] font-mono text-amber-400 bg-amber-900/30 px-2 py-0.5 rounded-full">
            {openPositions} open
          </span>
        )}
      </div>
      <div className={`text-3xl font-mono font-bold tabular-nums ${pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
        ${pnl >= 0 ? "+" : ""}{fmt(pnl)}
      </div>
      <div className="flex items-center gap-4 mt-3 text-sm font-mono text-gray-400">
        <span>{settled} settled</span>
        <span className="text-gray-700">·</span>
        <span>{wins}W / {losses}L</span>
        <span className="text-gray-700">·</span>
        <span>{winRate.toFixed(1)}%</span>
      </div>
    </div>
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

function BotLiveness({ heartbeat }: { heartbeat: import("../lib/api").HeartbeatResponse | null }) {
  if (!heartbeat) {
    return (
      <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-mono font-bold uppercase tracking-wider bg-gray-800/60 text-gray-500">
        <span className="w-1.5 h-1.5 rounded-full bg-gray-600" />
        Bot: unknown
      </span>
    );
  }

  const isAlive = heartbeat.is_alive;
  const secs = heartbeat.seconds_since_heartbeat;
  const agoStr =
    secs == null ? "" : secs < 120 ? `${Math.round(secs)}s ago` : `${Math.round(secs / 60)}m ago`;

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-mono font-bold uppercase tracking-wider ${
        isAlive ? "bg-green-900/40 text-green-400" : "bg-red-900/40 text-red-400"
      }`}
    >
      <span className={`w-1.5 h-1.5 rounded-full ${isAlive ? "bg-green-400 animate-pulse" : "bg-red-400"}`} />
      Bot: {isAlive ? "live" : `dead${agoStr ? ` · ${agoStr}` : ""}`}
    </span>
  );
}
