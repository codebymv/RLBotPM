import { SectionHeader } from "../components/SectionHeader";
import { DataFreshness } from "../components/DataFreshness";

const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type PriceData = {
  price?: number;
  volume_24h?: number;
  bid?: number;
  ask?: number;
  symbol?: string;
  error?: string;
};

type Candle = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
};

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

async function getCandles(asset: string) {
  try {
    const res = await fetch(
      `${baseUrl}/api/crypto/candles/${asset}?interval=1h&limit=24`,
      { cache: "no-store" }
    );
    if (!res.ok) return [];
    const data = await res.json();
    return data.candles || [];
  } catch {
    return [];
  }
}

function fmt(n: number, decimals = 2) {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}

const ASSETS = ["BTC", "ETH", "SOL", "DOGE", "XRP"];

export default async function CryptoPage() {
  const crypto = await getCryptoPrices();
  const prices: Record<string, PriceData> = crypto?.prices || {};
  const vols: Record<string, number> = crypto?.volatilities || {};

  // Fetch 24h candles for each asset
  const candleResults = await Promise.all(ASSETS.map((a) => getCandles(a)));
  const candles: Record<string, Candle[]> = {};
  ASSETS.forEach((a, i) => {
    candles[a] = candleResults[i];
  });

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-3 sm:p-4 max-w-6xl mx-auto grid-terminal">
      {/* Header */}
      <div className="mb-6 pb-4 border-b border-gray-800/60">
        <div className="flex items-end justify-between">
          <div>
            <h1 className="text-4xl font-bold tracking-tight mb-2">
              Market Data
            </h1>
            <p className="text-gray-500 text-base font-mono">
              Live spot prices from Coinbase · calibrated volatilities from
              Kalshi
            </p>
          </div>
          <DataFreshness lastUpdated={crypto?.last_updated} />
        </div>
      </div>

      {/* Context Ribbon */}
      <div className="mb-5 rounded-lg border border-blue-900/40 bg-blue-950/10 p-4">
        <div className="flex items-center gap-2 mb-1">
          <span className="text-blue-400 text-xs">ⓘ</span>
          <span className="text-xs font-mono font-bold uppercase tracking-widest text-blue-400">
            Context
          </span>
        </div>
        <p className="text-xs text-gray-400 font-mono leading-relaxed">
          Market data is for context only. Trading decisions are based on Kalshi
          prediction markets, not spot crypto positions. Volatilities are
          calibrated from historical settlement data.
        </p>
      </div>

      {/* Asset Cards */}
      <div className="space-y-4">
        {ASSETS.map((asset) => {
          const p = prices[asset];
          const vol = vols[asset];
          const c = candles[asset] || [];
          const decimals = asset === "DOGE" || asset === "XRP" ? 4 : 2;

          // 24h range from candles
          const highs = c.map((x) => x.high);
          const lows = c.map((x) => x.low);
          const high24 = highs.length ? Math.max(...highs) : null;
          const low24 = lows.length ? Math.min(...lows) : null;
          const totalVol = c.reduce((s, x) => s + x.volume, 0);

          // Sparkline: mini bar chart of close prices
          const closes = c.map((x) => x.close);
          const minClose = closes.length ? Math.min(...closes) : 0;
          const maxClose = closes.length ? Math.max(...closes) : 1;
          const range = maxClose - minClose || 1;

          // Change
          const change24 =
            closes.length >= 2
              ? ((closes[closes.length - 1] - closes[0]) / closes[0]) * 100
              : null;

          return (
            <div
              key={asset}
              className="rounded-lg border border-gray-800/60 bg-gray-900/30 p-6 hover:bg-gray-900/40 transition-colors"
            >
              {/* Header row */}
              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between mb-5 gap-3">
                <div className="flex items-center gap-3">
                  <span className="text-2xl font-bold font-mono tracking-tight">
                    {asset}
                  </span>
                  <span className="text-gray-600 text-xs font-mono">
                    {asset}-USD
                  </span>
                  {vol !== undefined && (
                    <span className="text-[9px] font-mono font-bold px-2.5 py-1 rounded-md bg-gray-800/60 text-gray-400 uppercase tracking-wider">
                      σ {(vol * 100).toFixed(0)}% ANNUAL
                    </span>
                  )}
                </div>
                {p?.price ? (
                  <div className="flex items-baseline gap-4">
                    <span className="text-3xl font-mono font-bold tabular-nums">
                      ${fmt(p.price, decimals)}
                    </span>
                    {change24 !== null && (
                      <span
                        className={`text-sm font-mono font-bold ${
                          change24 >= 0 ? "text-green-400" : "text-red-400"
                        }`}
                      >
                        {change24 >= 0 ? "+" : ""}
                        {change24.toFixed(2)}%
                      </span>
                    )}
                  </div>
                ) : (
                  <span className="text-red-400 text-sm font-mono">
                    {p?.error || "unavailable"}
                  </span>
                )}
              </div>

              {/* Stats row */}
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-4 mb-5">
                <Stat
                  label="Bid"
                  value={p?.bid ? `$${fmt(p.bid, decimals)}` : "—"}
                />
                <Stat
                  label="Ask"
                  value={p?.ask ? `$${fmt(p.ask, decimals)}` : "—"}
                />
                <Stat
                  label="24h High"
                  value={high24 !== null ? `$${fmt(high24, decimals)}` : "—"}
                />
                <Stat
                  label="24h Low"
                  value={low24 !== null ? `$${fmt(low24, decimals)}` : "—"}
                />
                <Stat
                  label="24h Volume"
                  value={totalVol > 0 ? fmt(totalVol, 1) : "—"}
                />
              </div>

              {/* Sparkline */}
              {closes.length > 1 && (
                <div className="pt-4 border-t border-gray-800/40">
                  <div className="text-[9px] text-gray-600 uppercase font-mono font-bold tracking-widest mb-2">
                    24H Close Price Chart
                  </div>
                  <div className="h-20 flex items-end gap-0.5">
                    {closes.map((v, i) => {
                      const h = Math.max(4, ((v - minClose) / range) * 100);
                      const isUp = i === 0 ? v >= closes[0] : v >= closes[i - 1];
                      return (
                        <div
                          key={i}
                          className={`flex-1 rounded-t transition-all hover:opacity-80 ${
                            isUp ? "bg-green-500/70" : "bg-red-500/70"
                          }`}
                          style={{ height: `${h}%` }}
                          title={`${c[i]?.timestamp?.slice(11, 16)} — $${fmt(v, decimals)}`}
                        />
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </main>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[10px] text-gray-600 uppercase tracking-widest font-mono font-bold mb-1">
        {label}
      </div>
      <div className="font-mono font-medium tabular-nums">{value}</div>
    </div>
  );
}
