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
      <main className="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-6 max-w-7xl mx-auto">
        <h1 className="text-2xl font-bold mt-2 mb-4">Bot Status</h1>
        <div className="text-gray-500 py-12 text-center text-sm rounded-lg border border-gray-800/50 bg-gray-900/30">
          Unable to load bot status.
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-6 max-w-7xl mx-auto">
      <h1 className="text-2xl font-bold mt-2 mb-1">Bot Status</h1>
      <p className="text-gray-500 text-sm mb-6">
        Session history, strategy configuration, and data overview
      </p>

      {/* Strategy Config */}
      <section className="mb-8">
        <h2 className="text-lg font-semibold mb-3">Strategy Configuration</h2>
        <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-5">
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-4 text-sm">
            <Stat label="Strategy" value={bot.strategy.name} />
            <Stat
              label="Side Filter"
              value={`BUY_${bot.strategy.side_filter.toUpperCase()}`}
            />
            <Stat
              label="Edge Range"
              value={`${(bot.strategy.min_edge * 100).toFixed(0)}–${(bot.strategy.max_edge * 100).toFixed(0)}%`}
            />
            <Stat
              label="Price Range"
              value={`${bot.strategy.min_price}–${bot.strategy.max_price}¢`}
            />
            <Stat
              label="Assets"
              value={bot.strategy.assets.join(", ")}
            />
            <Stat
              label="Total Sessions"
              value={String(bot.total_sessions)}
            />
          </div>

          {/* Volatilities */}
          <div className="mt-4 pt-4 border-t border-gray-800">
            <div className="text-[10px] text-gray-500 uppercase mb-2">
              Calibrated Annual Volatilities
            </div>
            <div className="flex flex-wrap gap-2">
              {Object.entries(bot.strategy.volatilities).map(([asset, vol]) => (
                <span
                  key={asset}
                  className="px-2 py-1 rounded bg-gray-800 text-xs font-mono"
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
          <h2 className="text-lg font-semibold mb-3">Data Overview</h2>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <Card
              label="Settled Markets"
              value={mkt.total_markets.toLocaleString()}
            />
            <Card label="Events" value={mkt.total_events.toLocaleString()} />
            <Card label="Series" value={mkt.total_series.toLocaleString()} />
            <Card
              label="Date Range"
              value={
                mkt.earliest_close && mkt.latest_close
                  ? `${new Date(mkt.earliest_close).toLocaleDateString("en-US", { month: "short", day: "numeric" })} – ${new Date(mkt.latest_close).toLocaleDateString("en-US", { month: "short", day: "numeric" })}`
                  : "—"
              }
            />
          </div>
        </section>
      )}

      {/* Session History */}
      <section className="mb-8">
        <h2 className="text-lg font-semibold mb-3">Recent Sessions</h2>
        {bot.sessions.length === 0 ? (
          <div className="text-gray-500 py-8 text-center text-sm rounded-lg border border-gray-800/50 bg-gray-900/30">
            No sessions recorded.
          </div>
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
                  className="rounded-lg border border-gray-800 bg-gray-900/60 p-4"
                >
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="font-mono text-sm text-gray-300">
                        {s.session_id}
                      </span>
                      {s.open_now > 0 && (
                        <span className="text-[10px] px-2 py-0.5 rounded-full bg-yellow-900/60 text-yellow-300">
                          {s.open_now} open
                        </span>
                      )}
                    </div>
                    {s.started_at && (
                      <span className="text-xs text-gray-500">
                        {new Date(s.started_at).toLocaleString("en-US", {
                          month: "short",
                          day: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                        })}
                      </span>
                    )}
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-5 gap-3 text-sm">
                    <Stat label="Trades" value={String(s.trades_opened)} />
                    <Stat label="Record" value={`${s.wins}W / ${s.losses}L`} />
                    <Stat label="Win Rate" value={wr === "—" ? "—" : `${wr}%`} />
                    <Stat
                      label="Realized P&L"
                      value={`$${s.realized_pnl >= 0 ? "+" : ""}${fmt(s.realized_pnl)}`}
                      className={
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
        <h2 className="text-lg font-semibold mb-3">Timeline</h2>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <Card
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
          />
          <Card
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
          />
        </div>
      </section>
    </main>
  );
}

function Stat({
  label,
  value,
  className = "",
}: {
  label: string;
  value: string;
  className?: string;
}) {
  return (
    <div>
      <div className="text-[10px] text-gray-500 uppercase">{label}</div>
      <div className={`font-mono ${className}`}>{value}</div>
    </div>
  );
}

function Card({
  label,
  value,
}: {
  label: string;
  value: string | number;
}) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
      <div className="text-[10px] text-gray-500 uppercase tracking-wide mb-1">
        {label}
      </div>
      <div className="text-xl font-bold">{value}</div>
    </div>
  );
}
