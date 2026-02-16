"use client";

import { useMode } from "../components/ModeToggle";
import { StatusPill } from "../components/StatusPill";
import { SectionHeader } from "../components/SectionHeader";
import { EmptyState } from "../components/EmptyState";

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
  mode: string;
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

type Props = {
  data: { positions: Position[]; count: number };
  crypto: any;
};

export default function PositionsClient({ data, crypto }: Props) {
  const mode = useMode();
  const positions: Position[] = data.positions || [];
  const prices: Record<string, PriceData> = crypto?.prices || {};

  // Filter positions by mode
  const filteredPositions = positions.filter(
    (p) => (p.mode || "paper") === mode
  );

  const totalCost = filteredPositions.reduce((s, p) => s + p.cost, 0);
  const totalExpectedProfit = filteredPositions.reduce((s, p) => {
    if (p.side === "no") {
      return s + ((100 - p.entry_price_cents) / 100) * p.contracts;
    }
    return s + (p.entry_price_cents / 100) * p.contracts;
  }, 0);

  // Group by asset
  const byAsset = filteredPositions.reduce(
    (acc, p) => {
      const asset = tickerToAsset(p.ticker) || "OTHER";
      if (!acc[asset]) acc[asset] = [];
      acc[asset].push(p);
      return acc;
    },
    {} as Record<string, Position[]>
  );

  const assetGroups = Object.entries(byAsset).sort(([a], [b]) =>
    a.localeCompare(b)
  );

  return (
    <main className="min-h-screen bg-gray-950 text-gray-100 p-3 sm:p-4 max-w-6xl mx-auto grid-terminal">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between mb-5 pb-4 border-b border-gray-800/60">
        <div>
          <h1 className="text-4xl font-bold tracking-tight mb-2">
            Open Positions
          </h1>
          <p className="text-gray-500 text-base font-mono">
            {filteredPositions.length} position{filteredPositions.length !== 1 ? "s" : ""} ·{" "}
            <span className="text-gray-300 font-bold">${fmt(totalCost)}</span>{" "}
            deployed · max profit{" "}
            <span className="text-green-400 font-bold">
              ${fmt(totalExpectedProfit)}
            </span>
          </p>
        </div>
        <div className="mt-4 sm:mt-0">
          <StatusPill mode={mode} />
        </div>
      </div>

      {/* Positions */}
      {filteredPositions.length === 0 ? (
        <EmptyState
          message={`No open ${mode} positions`}
          submessage="Positions will appear here once opened"
        />
      ) : (
        <div className="space-y-5">
          {assetGroups.map(([asset, groupPositions]) => {
            const groupCost = groupPositions.reduce((s, p) => s + p.cost, 0);
            const groupProfit = groupPositions.reduce((s, p) => {
              if (p.side === "no") {
                return s + ((100 - p.entry_price_cents) / 100) * p.contracts;
              }
              return s + (p.entry_price_cents / 100) * p.contracts;
            }, 0);

            return (
              <section key={asset}>
                <SectionHeader
                  title={asset}
                  subtitle={`${groupPositions.length} position${groupPositions.length !== 1 ? "s" : ""} · $${fmt(groupCost)} deployed · max profit $${fmt(groupProfit)}`}
                  className="mb-3"
                />

                <div className="space-y-3">
                  {groupPositions.map((p, idx) => {
                    const spot = prices[asset]?.price || null;
                    const decimals = asset === "DOGE" || asset === "XRP" ? 4 : 2;

                    const maxProfit =
                      p.side === "no"
                        ? ((100 - p.entry_price_cents) / 100) * p.contracts
                        : (p.entry_price_cents / 100) * p.contracts;

                    return (
                      <div
                        key={`${p.ticker}-${idx}`}
                        className={`rounded-lg border p-5 transition-all hover:border-gray-700/80 ${
                          mode === "live"
                            ? "border-cyan-800/40 bg-cyan-950/10"
                            : "border-amber-800/40 bg-amber-950/5"
                        }`}
                      >
                        <div className="flex items-start justify-between mb-4">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              <span className="font-mono text-sm font-bold">
                                {p.ticker}
                              </span>
                              <span
                                className={`px-2 py-0.5 rounded-md text-[9px] font-mono font-bold uppercase ${
                                  p.side === "no"
                                    ? "bg-green-900/60 text-green-300"
                                    : "bg-red-900/60 text-red-300"
                                }`}
                              >
                                BUY_{p.side}
                              </span>
                            </div>
                            {p.opened_at && (
                              <div className="text-[10px] text-gray-600 font-mono">
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
                          <div className="text-right">
                            <div className="text-xs text-gray-500 uppercase font-mono mb-0.5">
                              Max Profit
                            </div>
                            <div className="text-lg font-mono font-bold text-green-400 tabular-nums">
                              ${fmt(maxProfit)}
                            </div>
                          </div>
                        </div>

                        <div className="grid grid-cols-3 sm:grid-cols-4 lg:grid-cols-6 gap-4 mb-3">
                          <Stat label="Entry" value={`${p.entry_price_cents}¢`} />
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
                            valueClass="text-cyan-400"
                          />
                          <Stat label="Contracts" value={String(p.contracts)} />
                          <Stat label="Cost" value={`$${fmt(p.cost)}`} />
                          {spot ? (
                            <Stat
                              label={`${asset} Spot`}
                              value={`$${fmt(spot, decimals)}`}
                              valueClass="text-blue-400"
                            />
                          ) : (
                            <Stat label="Spot" value="—" />
                          )}
                        </div>

                        {p.reasoning && (
                          <div className="text-xs text-gray-500 leading-relaxed pt-3 border-t border-gray-800/40 font-mono">
                            {p.reasoning}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </section>
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
  valueClass = "",
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div>
      <div className="text-[10px] text-gray-600 uppercase tracking-widest mb-1 font-mono font-bold">
        {label}
      </div>
      <div className={`font-mono font-medium tabular-nums ${valueClass}`}>
        {value}
      </div>
    </div>
  );
}
