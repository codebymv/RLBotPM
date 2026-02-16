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

export default async function PositionsPage() {
  const data = await getPositions();
  const positions: Position[] = data.positions || [];
  const totalCost = positions.reduce((s, p) => s + p.cost, 0);

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-6 max-w-6xl mx-auto">
      <Link href="/" className="text-blue-400 text-sm hover:underline">
        ← Dashboard
      </Link>
      <h1 className="text-2xl font-bold mt-4 mb-1">Open Positions</h1>
      <p className="text-gray-500 text-sm mb-6">
        {positions.length} positions &middot; ${totalCost.toFixed(2)} deployed
      </p>

      {positions.length === 0 ? (
        <div className="text-gray-500 py-12 text-center">
          No open positions.
        </div>
      ) : (
        <div className="space-y-3">
          {positions.map((p) => (
            <div
              key={p.ticker}
              className="rounded-lg border border-gray-800 bg-gray-900 p-4"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-mono text-sm">{p.ticker}</span>
                <span
                  className={`px-2 py-0.5 rounded text-xs ${p.side === "no" ? "bg-green-900 text-green-300" : "bg-red-900 text-red-300"}`}
                >
                  BUY_{p.side.toUpperCase()}
                </span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3 text-sm">
                <div>
                  <span className="text-gray-500">Price</span>
                  <div>{p.entry_price_cents}¢</div>
                </div>
                <div>
                  <span className="text-gray-500">Fair</span>
                  <div>
                    {p.fair_price_cents
                      ? `${p.fair_price_cents.toFixed(0)}¢`
                      : "—"}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">Edge</span>
                  <div>
                    {p.edge ? `${(p.edge * 100).toFixed(1)}%` : "—"}
                  </div>
                </div>
                <div>
                  <span className="text-gray-500">Contracts</span>
                  <div>{p.contracts}</div>
                </div>
                <div>
                  <span className="text-gray-500">Cost</span>
                  <div>${p.cost.toFixed(2)}</div>
                </div>
              </div>
              {p.reasoning && (
                <div className="mt-2 text-xs text-gray-500 truncate">
                  {p.reasoning}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </main>
  );
}
