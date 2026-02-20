"use client";

import { useQuery } from "@tanstack/react-query";
import { useBot } from "../components/BotSelector";
import { SectionHeader } from "../components/SectionHeader";
import { EmptyState } from "../components/EmptyState";
import { KpiCard } from "../components/KpiCard";
import { StrategyBadge } from "../components/StrategyBadge";
import { DataFreshness } from "../components/DataFreshness";
import {
  fetchBotStatus,
  fetchMarketStats,
  fetchTrainingRuns,
} from "../../lib/api";

type Session = {
  session_id: string;
  started_at: string | null;
  last_trade_at: string | null;
  trades_opened: number;
  open_now: number;
  wins: number;
  losses: number;
  realized_pnl: number;
};

type BotData = {
  sessions: Session[];
  total_sessions: number;
  total_trades: number;
  first_trade_at: string | null;
  last_trade_at: string | null;
  strategy: {
    name: string;
    side_filter: string;
    min_edge: number;
    max_edge: number;
    min_price: number;
    max_price: number;
    assets: string[];
    volatilities: Record<string, number>;
  };
};

type TrainingRun = {
  id: number;
  started_at: string | null;
  ended_at: string | null;
  total_episodes: number;
  best_sharpe_ratio: number | null;
  best_episode_reward: number | null;
  status: string | null;
  config_snapshot: unknown;
};

function fmt(n: number, d = 2) {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });
}

function Stat({
  label,
  value,
  valueClass = "",
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div>
      <div className="text-[10px] text-gray-600 uppercase tracking-widest font-mono font-bold mb-1">
        {label}
      </div>
      <div className={`font-mono font-medium tabular-nums ${valueClass}`}>
        {value}
      </div>
    </div>
  );
}

type Props = {
  kalshiBot: BotData | null;
  mkt: any;
  trainingRuns: TrainingRun[];
};

export default function BotStatusClient({
  kalshiBot: initialKalshiBot,
  mkt: initialMkt,
  trainingRuns: initialTrainingRuns,
}: Props) {
  const bot = useBot();

  const { data: kalshiBot, dataUpdatedAt: botUpdatedAt } = useQuery({
    queryKey: ["botStatus"],
    queryFn: fetchBotStatus,
    initialData: initialKalshiBot,
    refetchInterval: 60_000,
  });

  const { data: mkt } = useQuery({
    queryKey: ["marketStats"],
    queryFn: fetchMarketStats,
    initialData: initialMkt,
    refetchInterval: 60_000,
  });

  const { data: trainingRunsData } = useQuery({
    queryKey: ["trainingRuns"],
    queryFn: () => fetchTrainingRuns(10),
    initialData: initialTrainingRuns,
    refetchInterval: 60_000,
  });
  const trainingRuns = Array.isArray(trainingRunsData)
    ? (trainingRunsData as TrainingRun[])
    : Array.isArray((trainingRunsData as { runs?: unknown[] } | null)?.runs)
      ? ((trainingRunsData as { runs: TrainingRun[] }).runs ?? [])
      : [];

  const showKalshi = bot === "all" || bot === "kalshi";
  const showRL = bot === "all" || bot === "rl_crypto";

  if (bot === "kalshi" && !kalshiBot) {
    return (
      <main className="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-6 max-w-7xl mx-auto grid-terminal">
        <h1 className="text-3xl font-bold tracking-tight mb-6">Bot Status</h1>
        <EmptyState message="Unable to load Kalshi bot status" />
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-3 sm:p-4 max-w-6xl mx-auto grid-terminal">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between mb-6 pb-4 border-b border-gray-800/60">
        <div>
          <h1 className="text-4xl font-bold tracking-tight mb-2">
            Bot Status{bot !== "all" ? (bot === "rl_crypto" ? " (RL Crypto)" : " (Kalshi)") : ""}
          </h1>
          <p className="text-gray-500 text-base font-mono">
            Strategy configuration, session history, and operational timeline
          </p>
        </div>
        <div className="mt-4 sm:mt-0">
          <DataFreshness
            lastUpdated={
              botUpdatedAt ? new Date(botUpdatedAt).toISOString() : undefined
            }
          />
        </div>
      </div>

      {/* RL Crypto — Training Runs */}
      {showRL && (
        <section className="mb-6">
          <SectionHeader
            title={bot === "all" ? "RL Crypto Bot — Training Runs" : "Training Runs"}
            subtitle="Latest PPO training runs (Run #, Sharpe, episodes)"
          />
          <div className="rounded-lg border border-purple-800/40 bg-purple-950/10 p-6">
            {bot === "all" && (
              <div className="flex items-center gap-2 mb-4">
                <StrategyBadge strategy="rl_crypto" />
                <span className="text-xs font-mono text-gray-500">RL Crypto Bot</span>
              </div>
            )}
            {trainingRuns.length === 0 ? (
              <EmptyState
                message="No training runs found"
                submessage="Run training to see run IDs and Sharpe ratios here"
              />
            ) : (
              <div className="space-y-3">
                {trainingRuns.map((run) => (
                  <div
                    key={run.id}
                    className="rounded-lg border border-gray-800/60 bg-gray-900/30 p-5 hover:bg-gray-900/40 transition-colors"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-sm font-bold">Run #{run.id}</span>
                        {run.status && (
                          <span className="text-[9px] font-mono font-bold px-2.5 py-1 rounded-md bg-gray-700/60 text-gray-300">
                            {run.status}
                          </span>
                        )}
                      </div>
                      {run.started_at && (
                        <span className="text-[10px] font-mono text-gray-600">
                          {new Date(run.started_at).toLocaleString("en-US", {
                            month: "short",
                            day: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                          })}
                        </span>
                      )}
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-sm font-mono">
                      <Stat
                        label="Best Sharpe"
                        value={run.best_sharpe_ratio != null ? fmt(run.best_sharpe_ratio) : "—"}
                        valueClass="text-purple-400"
                      />
                      <Stat
                        label="Best Episode Reward"
                        value={run.best_episode_reward != null ? fmt(run.best_episode_reward) : "—"}
                      />
                      <Stat label="Episodes" value={String(run.total_episodes)} />
                      <Stat
                        label="Ended"
                        value={
                          run.ended_at
                            ? new Date(run.ended_at).toLocaleDateString("en-US", {
                                month: "short",
                                day: "numeric",
                              })
                            : "—"
                        }
                      />
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      )}

      {/* Kalshi — Strategy Config */}
      {showKalshi && kalshiBot && (
        <>
          <section className="mb-6">
            <SectionHeader
              title={bot === "all" ? "Kalshi Market Bot — Strategy Configuration" : "Strategy Configuration"}
              subtitle="Active trading parameters and risk controls"
            />
            {bot === "all" && (
              <div className="flex items-center gap-2 mb-3">
                <StrategyBadge strategy="kalshi" />
                <span className="text-xs font-mono text-gray-500">Kalshi Market Bot</span>
              </div>
            )}
            <div className="rounded-lg border border-gray-800/60 bg-gray-900/30 p-6">
              <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 text-sm font-mono">
                <Stat label="Strategy" value={kalshiBot.strategy.name} />
                <Stat
                  label="Side Filter"
                  value={`BUY_${kalshiBot.strategy.side_filter.toUpperCase()}`}
                  valueClass="text-green-400"
                />
                <Stat
                  label="Edge Range"
                  value={`${(kalshiBot.strategy.min_edge * 100).toFixed(0)}–${(kalshiBot.strategy.max_edge * 100).toFixed(0)}%`}
                  valueClass="text-cyan-400"
                />
                <Stat
                  label="Price Range"
                  value={`${kalshiBot.strategy.min_price}–${kalshiBot.strategy.max_price}¢`}
                />
                <Stat label="Assets" value={kalshiBot.strategy.assets.join(", ")} />
                <Stat label="Total Sessions" value={String(kalshiBot.total_sessions)} />
              </div>

              <div className="mt-6 pt-6 border-t border-gray-800/40">
                <div className="text-[10px] text-gray-600 uppercase mb-3 font-mono font-bold tracking-widest">
                  Calibrated Annual Volatilities
                </div>
                <div className="flex flex-wrap gap-2">
                  {(Object.entries(kalshiBot.strategy.volatilities) as [string, number][]).map(([asset, vol]) => (
                    <span
                      key={asset}
                      className="px-3 py-1.5 rounded-md bg-gray-800/60 text-xs font-mono font-bold"
                    >
                      {asset}: {(vol * 100).toFixed(0)}%
                    </span>
                  ))}
                </div>
              </div>
            </div>
          </section>

          {/* Data Overview */}
          {mkt && (
            <section className="mb-6">
              <SectionHeader
                title="Market Data Snapshot"
                subtitle="Kalshi settled markets and historical coverage"
              />
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <KpiCard
                  label="Settled Markets"
                  value={mkt.total_markets.toLocaleString()}
                  mode="neutral"
                />
                <KpiCard
                  label="Events"
                  value={mkt.total_events.toLocaleString()}
                  mode="neutral"
                />
                <KpiCard
                  label="Series"
                  value={mkt.total_series.toLocaleString()}
                  mode="neutral"
                />
                <KpiCard
                  label="Date Range"
                  value={
                    mkt.earliest_close && mkt.latest_close
                      ? `${new Date(mkt.earliest_close).toLocaleDateString("en-US", { month: "short", day: "numeric" })} – ${new Date(mkt.latest_close).toLocaleDateString("en-US", { month: "short", day: "numeric" })}`
                      : "—"
                  }
                  mode="neutral"
                />
              </div>
            </section>
          )}

          {/* Session History */}
          <section className="mb-6">
            <SectionHeader
              title="Session History"
              subtitle={`${kalshiBot.sessions.length} recent trading sessions`}
            />
            {kalshiBot.sessions.length === 0 ? (
              <EmptyState message="No sessions recorded" />
            ) : (
              <div className="space-y-3">
                {(kalshiBot.sessions as Session[]).map((s) => {
                  const totalSettled = s.wins + s.losses;
                  const wr =
                    totalSettled > 0
                      ? ((s.wins / totalSettled) * 100).toFixed(0)
                      : "—";
                  return (
                    <div
                      key={s.session_id}
                      className="rounded-lg border border-gray-800/60 bg-gray-900/30 p-5 hover:bg-gray-900/40 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-4">
                        <div className="flex items-center gap-2">
                          <span className="font-mono text-sm font-bold">
                            {s.session_id}
                          </span>
                          {s.open_now > 0 && (
                            <span className="text-[9px] font-mono font-bold px-2.5 py-1 rounded-md bg-amber-900/60 text-amber-300">
                              {s.open_now} OPEN
                            </span>
                          )}
                        </div>
                        {s.started_at && (
                          <span className="text-[10px] font-mono text-gray-600">
                            {new Date(s.started_at).toLocaleString("en-US", {
                              month: "short",
                              day: "numeric",
                              hour: "2-digit",
                              minute: "2-digit",
                            })}
                          </span>
                        )}
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-5 gap-4 text-sm font-mono">
                        <Stat label="Trades" value={String(s.trades_opened)} />
                        <Stat label="Record" value={`${s.wins}W / ${s.losses}L`} />
                        <Stat
                          label="Win Rate"
                          value={wr === "—" ? "—" : `${wr}%`}
                        />
                        <Stat
                          label="Realized P&L"
                          value={`$${s.realized_pnl >= 0 ? "+" : ""}${fmt(s.realized_pnl)}`}
                          valueClass={
                            s.realized_pnl > 0
                              ? "text-green-400"
                              : s.realized_pnl < 0
                                ? "text-red-400"
                                : ""
                          }
                        />
                        <Stat label="Still Open" value={String(s.open_now)} />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </section>

          {/* Timeline */}
          <section className="mb-6">
            <SectionHeader title="Operational Timeline" subtitle="First and last trade timestamps" />
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <KpiCard
                label="First Trade"
                value={
                  kalshiBot.first_trade_at
                    ? new Date(kalshiBot.first_trade_at).toLocaleString("en-US", {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })
                    : "—"
                }
                mode="neutral"
              />
              <KpiCard
                label="Last Trade"
                value={
                  kalshiBot.last_trade_at
                    ? new Date(kalshiBot.last_trade_at).toLocaleString("en-US", {
                        month: "short",
                        day: "numeric",
                        year: "numeric",
                        hour: "2-digit",
                        minute: "2-digit",
                      })
                    : "—"
                }
                mode="neutral"
              />
            </div>
          </section>
        </>
      )}

      {showKalshi && !kalshiBot && (
        <EmptyState
          message="Kalshi bot status unavailable"
          submessage="Start the Kalshi paper trader to see strategy and sessions"
        />
      )}
    </main>
  );
}
