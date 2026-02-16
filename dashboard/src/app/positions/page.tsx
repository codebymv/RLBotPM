import Link from "next/link";

const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Position = {
  ticker: string;
  side: string;
  entry_price_cents: number;
  fair_price_cents: number | null;
  edge: number | null;
  edge_type: string;
  contracts: number;
  cost: number;
  reasoning: string;
  opened_at: string | null;
  series_ticker: string;
};

type PriceData = { price?: number; error?: string };

// Map series ticker → asset symbol for spot price lookup
function tickerToAsset(ticker: string): string | null {
  const t = ticker.toUpperCase();
  if (t.startsWith("KXBTC") || t.startsWith("KXBTCD")) return "BTC";
  if (t.startsWith("KXETH") || t.startsWith("KXETHD")) return "ETH";
  if (t.startsWith("KXSOL") || t.startsWith("KXSOLD")) return "SOL";
  if (t.startsWith("KXDOGE") || t.startsWith("KXDOGED")) return "DOGE";
  if (t.startsWith("KXXRP") || t.startsWith("KXXRPD")) return "XRP";
  return null;
}

function fmt(n: number, d = 2) {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });
}

async function getPositions() {
  try {
    const res = await fetch(`${baseUrl}/api/kalshi/positions?mode=paper`, {
      cache: "no-store",
    });
    if (!res.ok) return { positions: [], count: 0 };
    return res.json();
  } catch {
    return { positions: [], count: 0 };
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

export default async function PositionsPage() {
  const [data, crypto] = await Promise.all([getPositions(), getCryptoPrices()]);
  const positions: Position[] = data.positions || [];
  const prices: Record<string, PriceData> = crypto?.prices || {};
  const totalCost = positions.reduce((s, p) => s + p.cost, 0);

  // Compute expected max profit for BUY_NO positions
  const totalExpectedProfit = positions.reduce((s, p) => {
    if (p.side === "no") {
      return s + ((100 - p.entry_price_cents) / 100) * p.contracts;
    }
    return s + (p.entry_price_cents / 100) * p.contracts;
  }, 0);

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-4 sm:p-6 max-w-7xl mx-auto">
      <h1 className="text-2xl font-bold mt-2 mb-1">Open Positions</h1>
      <p className="text-gray-500 text-sm mb-6">
        {positions.length} positions &middot; ${fmt(totalCost)} deployed
        &middot; max profit ${fmt(totalExpectedProfit)} if all settle in our
        favor
      </p>

      {positions.length === 0 ? (
        <div className="text-gray-500 py-12 text-center text-sm rounded-lg border border-gray-800/50 bg-gray-900/30">
          No open positions.
        </div>
      ) : (
        <div className="space-y-3">
          {positions.map((p, idx) => {
            const asset = tickerToAsset(p.ticker);
            const spot = asset ? prices[asset]?.price : null;
            const decimals =
              asset === "DOGE" || asset === "XRP" ? 4 : 2;

            // Expected profit if position wins
            const maxProfit =
              p.side === "no"
                ? ((100 - p.entry_price_cents) / 100) * p.contracts
                : (p.entry_price_cents / 100) * p.contracts;

            return (
              <div
                key={`${p.ticker}-${idx}`}
                className="rounded-lg border border-gray-800 bg-gray-900/60 p-4"
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm font-medium">
                      {p.ticker}
                    </span>
                    {asset && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-800 text-gray-400">
                        {asset}
                      </span>
                    )}
                  </div>
                  <span
                    className={`px-2 py-0.5 rounded text-xs font-medium ${p.side === "no" ? "bg-green-900/60 text-green-300" : "bg-red-900/60 text-red-300"}`}
                  >
                    BUY_{p.side.toUpperCase()}
                  </span>
                </div>

                <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3 text-sm">
                  <Stat label="Entry Price" value={`${p.entry_price_cents}¢`} />
                  <Stat
                    label="Fair Value"
                    value={
                      p.fair_price_cents
                        ? `${p.fair_price_cents.toFixed(0)}¢`
                        : "—"
                    }
                  />
                  <Stat
                    label="Edge"
                    value={p.edge ? `${(p.edge * 100).toFixed(1)}%` : "—"}
                  />
                  <Stat label="Contracts" value={String(p.contracts)} />
                  <Stat label="Cost" value={`$${fmt(p.cost)}`} />
                  <Stat
                    label="Max Profit"
                    value={`$${fmt(maxProfit)}`}
                    className="text-green-400"
                  />
                  {spot ? (
                    <Stat
                      label={`${asset} Spot`}
                      value={`$${fmt(spot, decimals)}`}
                      className="text-blue-400"
                    />
                  ) : (
                    <Stat label="Spot" value="—" />
                  )}
                </div>

                {p.reasoning && (
                  <div className="mt-2 text-xs text-gray-500 leading-relaxed">
                    {p.reasoning}
                  </div>
                )}

                {p.opened_at && (
                  <div className="mt-1 text-[10px] text-gray-600">
                    Opened{" "}
                    {new Date(p.opened_at).toLocaleString("en-US", {
                      month: "short",
                      day: "numeric",
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
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
