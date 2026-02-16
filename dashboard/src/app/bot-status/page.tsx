import { SectionHeader } from "../components/SectionHeader";
import { EmptyState } from "../components/EmptyState";
import { KpiCard } from "../components/KpiCard";

const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

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

async function getBotStatus(): Promise<BotData | null> {
  try {
    const res = await fetch(`${baseUrl}/api/bot/status`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function getMarketStats() {
  try {
    const res = await fetch(`${baseUrl}/api/kalshi/market-stats`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

function fmt(n: number, d = 2) {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });
}

export default async function BotStatusPage() {
  const [bot, mkt] = await Promise.all([getBotStatus(), getMarketStats()]);

  if (!bot) {
    return (
      <main className="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-6 max-w-7xl mx-auto grid-terminal">
        <h1 className="text-3xl font-bold tracking-tight mb-6">Bot Status</h1>
        <EmptyState message="Unable to load bot status" />
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-6 max-w-7xl mx-auto grid-terminal">
      {/* Header */}
      <div className="mb-8 pb-6 border-b border-gray-800/60">
        <h1 className="text-3xl font-bold tracking-tight mb-2">Bot Status</h1>
        <p className="text-gray-500 text-sm font-mono">
          Strategy configuration, session history, and operational timeline
        </p>
      </div>

      {/* Strategy Config */}
      <section className="mb-8">
        <SectionHeader
          title="Strategy Configuration"
          subtitle="Active trading parameters and risk controls"
        />
        <div className="rounded-lg border border-gray-800/60 bg-gray-900/30 p-6">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 text-sm font-mono">
            <Stat label="Strategy" value={bot.strategy.name} />
            <Stat
              label="Side Filter"
              value={`BUY_${bot.strategy.side_filter.toUpperCase()}`}
              valueClass="text-green-400"
            />
            <Stat
              label="Edge Range"
              value={`${(bot.strategy.min_edge * 100).toFixed(0)}–${(bot.strategy.max_edge * 100).toFixed(0)}%`}
              valueClass="text-cyan-400"
            />
            <Stat
              label="Price Range"
              value={`${bot.strategy.min_price}–${bot.strategy.max_price}¢`}
            />
            <Stat label="Assets" value={bot.strategy.assets.join(", ")} />
            <Stat label="Total Sessions" value={String(bot.total_sessions)} />
          </div>

          {/* Volatilities */}
          <div className="mt-6 pt-6 border-t border-gray-800/40">
            <div className="text-[10px] text-gray-600 uppercase mb-3 font-mono font-bold tracking-widest">
              Calibrated Annual Volatilities
            </div>
            <div className="flex flex-wrap gap-2">
              {Object.entries(bot.strategy.volatilities).map(([asset, vol]) => (
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
        <section className="mb-8">
          <SectionHeader
            title="Market Data Snapshot"
            subtitle="Kalshi settled markets and historical coverage"
          />
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
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
      <section className="mb-8">
        <SectionHeader
          title="Session History"
          subtitle={`${bot.sessions.length} recent trading sessions`}
        />
        {bot.sessions.length === 0 ? (
          <EmptyState message="No sessions recorded" />
        ) : (
          <div className="space-y-3">
            {bot.sessions.map((s) => {
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
      <section className="mb-8">
        <SectionHeader title="Operational Timeline" subtitle="First and last trade timestamps" />
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <KpiCard
            label="First Trade"
            value={
              bot.first_trade_at
                ? new Date(bot.first_trade_at).toLocaleString("en-US", {
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
              bot.last_trade_at
                ? new Date(bot.last_trade_at).toLocaleString("en-US", {
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
    </main>
  );
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
