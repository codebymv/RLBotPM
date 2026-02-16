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

async function getCryptoPrices() {
  try {
    const res = await fetch(`${baseUrl}/api/crypto/prices`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function getBotStatus() {
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

function fmt(n: number, decimals = 2) {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

export default async function Page() {
  const [health, metrics, crypto, bot, mktStats] = await Promise.all([
    getHealth(),
    getMetrics(),
    getCryptoPrices(),
    getBotStatus(),
    getMarketStats(),
  ]);

  const pnlColor =
    metrics && metrics.realized_pnl > 0
      ? "text-green-400"
      : metrics && metrics.realized_pnl < 0
        ? "text-red-400"
        : "text-gray-300";

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-6 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold mb-1">RLTrade Dashboard</h1>
          <p className="text-gray-500 text-sm">
            Crypto prediction market trading bot &mdash; Kalshi &middot; Coinbase
          </p>
        </div>
        <div className="flex gap-2 mt-3 sm:mt-0 text-xs">
          <StatusBadge
            label="API"
            ok={health.status === "healthy"}
            text={health.status}
          />
          <StatusBadge
            label="DB"
            ok={health.database === "connected"}
            text={health.database || "unknown"}
          />
        </div>
      </div>

      {/* ── Trading Performance ── */}
      <Section title="Trading Performance">
        {metrics ? (
          <>
            {/* Mode breakdown */}
            {metrics.mode_breakdown && Object.keys(metrics.mode_breakdown).length > 0 && (
              <div className="flex gap-2 mb-4">
                {Object.entries(
                  metrics.mode_breakdown as Record<string, { total: number; wins: number; losses: number; realized_pnl: number; open_positions: number; open_cost: number }>
                ).map(([m, d]) => (
                  <div key={m} className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs ${
                    m === 'live' ? 'border-blue-700/60 bg-blue-950/20 text-blue-300' : 'border-gray-700/60 bg-gray-900/40 text-gray-400'
                  }`}>
                    <span className={`w-2 h-2 rounded-full ${m === 'live' ? 'bg-blue-400' : 'bg-gray-500'}`} />
                    <span className="font-medium uppercase">{m}</span>
                    <span>{d.total} trades</span>
                    <span className="text-gray-600">|</span>
                    <span>{d.wins}W/{d.losses}L</span>
                    <span className="text-gray-600">|</span>
                    <span className={d.realized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                      ${d.realized_pnl >= 0 ? '+' : ''}{fmt(d.realized_pnl)}
                    </span>
                  </div>
                ))}
              </div>
            )}
            <div className="grid grid-cols-2 lg:grid-cols-5 gap-3 mb-4">
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
                value={`$${metrics.realized_pnl >= 0 ? "" : ""}${fmt(metrics.realized_pnl)}`}
                className={pnlColor}
              />
              <Card
                label="Open Positions"
                value={metrics.open_positions}
                sub={`$${fmt(metrics.open_cost)} deployed`}
              />
              <Card
                label="Settled Markets"
                value={mktStats ? mktStats.total_markets.toLocaleString() : "—"}
                sub={mktStats ? `${mktStats.total_events} events` : ""}
              />
            </div>

            {/* Side breakdown */}
            {metrics.side_breakdown &&
              Object.keys(metrics.side_breakdown).length > 0 && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mb-2">
                  {Object.entries(
                    metrics.side_breakdown as Record<
                      string,
                      { total: number; wins: number; pnl: number }
                    >,
                  ).map(([side, data]) => {
                    const losses = data.total - data.wins;
                    const wr =
                      data.total > 0
                        ? ((data.wins / data.total) * 100).toFixed(0)
                        : "0";
                    return (
                      <div
                        key={side}
                        className={`rounded-lg border p-4 ${side === "no" ? "border-green-800/60 bg-green-950/20" : "border-red-800/60 bg-red-950/20"}`}
                      >
                        <div className="flex items-center justify-between mb-1">
                          <span className="text-sm text-gray-400">
                            BUY_{side.toUpperCase()}
                          </span>
                          <span
                            className={`text-xs px-2 py-0.5 rounded ${side === "no" ? "bg-green-900/60 text-green-300" : "bg-red-900/60 text-red-300"}`}
                          >
                            {wr}% win rate
                          </span>
                        </div>
                        <div className="text-lg font-bold">
                          {data.wins}W / {losses}L
                        </div>
                        <div className="text-sm text-gray-400">
                          P&L: ${data.pnl >= 0 ? "+" : ""}{fmt(data.pnl)}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
          </>
        ) : (
          <Empty text="No trading data yet. Start the paper trader." />
        )}
      </Section>

      {/* ── Live Crypto Prices ── */}
      <Section
        title="Crypto Spot Prices"
        action={{ href: "/crypto", label: "Details →" }}
      >
        {crypto && crypto.prices ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
            {Object.entries(
              crypto.prices as Record<
                string,
                { price?: number; bid?: number; ask?: number; error?: string }
              >,
            ).map(([asset, data]) => (
              <div
                key={asset}
                className="rounded-lg border border-gray-800 bg-gray-900/60 p-4"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-bold text-sm">{asset}</span>
                  <span className="text-[10px] text-gray-500">
                    σ={((crypto.volatilities?.[asset] || 0) * 100).toFixed(0)}%
                  </span>
                </div>
                {data.price ? (
                  <>
                    <div className="text-xl font-mono font-bold">
                      ${fmt(data.price, asset === "DOGE" || asset === "XRP" ? 4 : 2)}
                    </div>
                    {data.bid && data.ask ? (
                      <div className="text-[10px] text-gray-500 mt-1">
                        bid ${fmt(data.bid, 2)} / ask ${fmt(data.ask, 2)}
                      </div>
                    ) : null}
                  </>
                ) : (
                  <div className="text-red-400 text-xs">
                    {data.error || "unavailable"}
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <Empty text="Unable to fetch crypto prices." />
        )}
      </Section>

      {/* ── Bot Operations ── */}
      <Section
        title="Bot Operations"
        action={{ href: "/bot-status", label: "Details →" }}
      >
        {bot ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
            <Card
              label="Total Sessions"
              value={bot.total_sessions}
              sub={
                bot.last_trade_at
                  ? `Last trade ${new Date(bot.last_trade_at).toLocaleDateString()}`
                  : ""
              }
            />
            <Card label="Strategy" value="BUY_NO" sub="Lognormal edge" />
            <Card
              label="Edge Range"
              value={`${(bot.strategy.min_edge * 100).toFixed(0)}-${(bot.strategy.max_edge * 100).toFixed(0)}%`}
              sub={`Price ${bot.strategy.min_price}-${bot.strategy.max_price}¢`}
            />
            <Card
              label="Assets Tracked"
              value={bot.strategy.assets.length}
              sub={bot.strategy.assets.join(", ")}
            />
          </div>
        ) : (
          <Empty text="Bot status unavailable." />
        )}
      </Section>

      {/* ── Recent Trades ── */}
      <Section
        title="Recent Trades"
        action={{ href: "/positions", label: "All Positions →" }}
      >
        {metrics?.recent_trades && metrics.recent_trades.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-gray-500 border-b border-gray-800 text-xs">
                  <th className="text-left py-2 px-2">Ticker</th>
                  <th className="text-left py-2 px-2">Mode</th>
                  <th className="text-left py-2 px-2">Side</th>
                  <th className="text-right py-2 px-2">Price</th>
                  <th className="text-right py-2 px-2">Edge</th>
                  <th className="text-right py-2 px-2">Qty</th>
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
                      mode: string;
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
                      className="border-b border-gray-900/60 hover:bg-gray-900/40"
                    >
                      <td className="py-1.5 px-2 font-mono text-xs">
                        {t.ticker}
                      </td>
                      <td className="py-1.5 px-2">
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                          t.mode === 'live' ? 'bg-blue-900/60 text-blue-300' : 'bg-gray-800 text-gray-400'
                        }`}>
                          {t.mode?.toUpperCase() || 'PAPER'}
                        </span>
                      </td>
                      <td className="py-1.5 px-2">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${t.side === "no" ? "bg-green-900/60 text-green-300" : "bg-red-900/60 text-red-300"}`}
                        >
                          {t.side.toUpperCase()}
                        </span>
                      </td>
                      <td className="py-1.5 px-2 text-right font-mono">
                        {t.entry_price_cents}¢
                      </td>
                      <td className="py-1.5 px-2 text-right">
                        {t.edge ? `${(t.edge * 100).toFixed(1)}%` : "—"}
                      </td>
                      <td className="py-1.5 px-2 text-right">
                        {t.contracts}
                      </td>
                      <td className="py-1.5 px-2 text-right font-mono">
                        ${fmt(t.cost)}
                      </td>
                      <td className="py-1.5 px-2">
                        <span
                          className={`text-xs ${t.status === "open" ? "text-yellow-400" : "text-gray-400"}`}
                        >
                          {t.status}
                        </span>
                      </td>
                      <td
                        className={`py-1.5 px-2 text-right font-mono font-medium ${t.pnl && t.pnl > 0 ? "text-green-400" : t.pnl && t.pnl < 0 ? "text-red-400" : ""}`}
                      >
                        {t.pnl !== null
                          ? `$${t.pnl >= 0 ? "+" : ""}${fmt(t.pnl)}`
                          : "—"}
                      </td>
                    </tr>
                  ),
                )}
              </tbody>
            </table>
          </div>
        ) : (
          <Empty text="No trades recorded yet." />
        )}
      </Section>
    </main>
  );
}

/* ── Reusable components ── */

function Section({
  title,
  action,
  children,
}: {
  title: string;
  action?: { href: string; label: string };
  children: React.ReactNode;
}) {
  return (
    <section className="mb-8">
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-lg font-semibold">{title}</h2>
        {action && (
          <Link
            href={action.href}
            className="text-blue-400 hover:text-blue-300 text-xs"
          >
            {action.label}
          </Link>
        )}
      </div>
      {children}
    </section>
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
    <div className="rounded-lg border border-gray-800 bg-gray-900/60 p-4">
      <div className="text-[10px] text-gray-500 uppercase tracking-wide mb-1">
        {label}
      </div>
      <div className={`text-2xl font-bold ${className}`}>{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-1">{sub}</div>}
    </div>
  );
}

function StatusBadge({
  label,
  ok,
  text,
}: {
  label: string;
  ok: boolean;
  text: string;
}) {
  return (
    <span
      className={`px-2.5 py-1 rounded-full text-xs ${ok ? "bg-green-900/60 text-green-300" : "bg-red-900/60 text-red-300"}`}
    >
      {label}: {text}
    </span>
  );
}

function Empty({ text }: { text: string }) {
  return (
    <div className="text-gray-500 py-8 text-center text-sm rounded-lg border border-gray-800/50 bg-gray-900/30">
      {text}
    </div>
  );
}
