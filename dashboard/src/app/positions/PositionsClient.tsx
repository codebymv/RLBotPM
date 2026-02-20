"use client";

import { useQuery } from "@tanstack/react-query";
import { useMode } from "../components/ModeToggle";
import { useBot } from "../components/BotSelector";
import { StatusPill } from "../components/StatusPill";
import { EmptyState } from "../components/EmptyState";
import { StrategyBadge } from "../components/StrategyBadge";
import { DataFreshness } from "../components/DataFreshness";
import {
  fetchKalshiPositions,
  fetchRLPositions,
  fetchCryptoPrices,
} from "../../lib/api";

type KalshiPosition = {
  strategy: "kalshi";
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

type RLPosition = {
  strategy: "rl_crypto";
  id: number;
  symbol: string;
  action: string;
  entry_price: number;
  position_size: number;
  model_path: string;
  regime: string | null;
  hold_steps: number;
  mode: string;
  session_id: string | null;
  opened_at: string | null;
};

type UnifiedPosition = KalshiPosition | RLPosition;

function isKalshi(p: UnifiedPosition): p is KalshiPosition {
  return p.strategy === "kalshi";
}

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

function getAsset(p: UnifiedPosition): string {
  if (p.strategy === "kalshi") return tickerToAsset(p.ticker) || "OTHER";
  return p.symbol.split("-")[0] || "OTHER";
}

function getCost(p: UnifiedPosition): number {
  if (p.strategy === "kalshi") return p.cost;
  return p.entry_price * p.position_size;
}

function fmt(n: number, d = 2) {
  return n.toLocaleString("en-US", {
    minimumFractionDigits: d,
    maximumFractionDigits: d,
  });
}

type Props = {
  kalshiData: { positions: Omit<KalshiPosition, "strategy">[]; count: number };
  rlData: { positions: Omit<RLPosition, "strategy">[]; count: number };
  crypto: any;
};

export default function PositionsClient({
  kalshiData: initialKalshiData,
  rlData: initialRlData,
  crypto: initialCrypto,
}: Props) {
  const mode = useMode();
  const bot = useBot();

  const { data: kalshiData, dataUpdatedAt: positionsUpdatedAt } = useQuery({
    queryKey: ["kalshiPositions"],
    queryFn: () => fetchKalshiPositions(),
    initialData: initialKalshiData,
    refetchInterval: 15_000,
  });

  const { data: rlData } = useQuery({
    queryKey: ["rlPositions", mode],
    queryFn: () => fetchRLPositions(mode),
    initialData: initialRlData,
    refetchInterval: 15_000,
  });

  const { data: crypto } = useQuery({
    queryKey: ["cryptoPrices"],
    queryFn: fetchCryptoPrices,
    initialData: initialCrypto,
    refetchInterval: 60_000,
  });

  const prices: Record<string, PriceData> = (crypto as any)?.prices || {};

  const kalshiPositions: KalshiPosition[] = (kalshiData.positions || [])
    .filter((p) => (p.mode || "paper") === mode)
    .map((p) => ({ ...p, strategy: "kalshi" as const }));
  const rlPositions: RLPosition[] = (rlData.positions || []).map((p) => ({
    ...p,
    strategy: "rl_crypto" as const,
  }));

  const allMerged: UnifiedPosition[] =
    bot === "all"
      ? [...kalshiPositions, ...rlPositions]
      : bot === "kalshi"
        ? kalshiPositions
        : rlPositions;

  const totalCost = allMerged.reduce((s, p) => s + getCost(p), 0);
  const totalExpectedProfit = allMerged.reduce((s, p) => {
    if (!isKalshi(p)) return s;
    if (p.side === "no") return s + ((100 - p.entry_price_cents) / 100) * p.contracts;
    return s + (p.entry_price_cents / 100) * p.contracts;
  }, 0);
  const hasKalshiProfit = allMerged.some(isKalshi);

  // Group by asset
  const byAsset = allMerged.reduce(
    (acc, p) => {
      const asset = getAsset(p);
      if (!acc[asset]) acc[asset] = [];
      acc[asset].push(p);
      return acc;
    },
    {} as Record<string, UnifiedPosition[]>
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
            Open Positions{bot !== "all" ? ` (${bot === "rl_crypto" ? "RL" : "Kalshi"})` : ""}
          </h1>
          <div className="text-gray-500 text-base font-mono space-y-1">
            <div>
              {allMerged.length} position{allMerged.length !== 1 ? "s" : ""} ·{" "}
              <span className="text-gray-300 font-bold">${fmt(totalCost)}</span>{" "}
              deployed
            </div>
            {hasKalshiProfit && (
              <div className="md:inline md:ml-1">
                <span className="hidden md:inline">· </span>
                max profit{" "}
                <span className="text-green-400 font-bold">
                  ${fmt(totalExpectedProfit)}
                </span>
              </div>
            )}
          </div>
        </div>
        <div className="mt-4 sm:mt-0 flex flex-col items-end gap-2">
          <StatusPill mode={mode} />
          <DataFreshness
            lastUpdated={
              positionsUpdatedAt ? new Date(positionsUpdatedAt).toISOString() : undefined
            }
          />
        </div>
      </div>

      {/* Positions */}
      {allMerged.length === 0 ? (
        <EmptyState
          message={`No open ${mode} positions${bot !== "all" ? ` for ${bot === "rl_crypto" ? "RL Crypto Bot" : "Kalshi Bot"}` : ""}`}
          submessage="Positions will appear here once opened"
        />
      ) : (
        <div className="space-y-5">
          {assetGroups.map(([asset, groupPositions]) => {
            const groupCost = groupPositions.reduce((s, p) => s + getCost(p), 0);
            const groupProfit = groupPositions.reduce((s, p) => {
              if (!isKalshi(p)) return s;
              if (p.side === "no") return s + ((100 - p.entry_price_cents) / 100) * p.contracts;
              return s + (p.entry_price_cents / 100) * p.contracts;
            }, 0);
            const groupHasKalshi = groupPositions.some(isKalshi);

            return (
              <section key={asset}>
                <div className="mb-3">
                  <h2 className="text-xl font-bold tracking-tight">{asset}</h2>
                  <div className="text-sm text-gray-500 mt-0.5 space-y-0.5">
                    <div>
                      {groupPositions.length} position{groupPositions.length !== 1 ? "s" : ""} · ${fmt(groupCost)} deployed
                    </div>
                    {groupHasKalshi && (
                      <div className="md:inline md:ml-1">
                        <span className="hidden md:inline">· </span>
                        max profit ${fmt(groupProfit)}
                      </div>
                    )}
                  </div>
                </div>

                <div className="space-y-3">
                  {groupPositions.map((p, idx) => {
                    if (p.strategy === "rl_crypto") {
                      const spot = prices[asset]?.price ?? null;
                      const decimals = asset === "DOGE" || asset === "XRP" ? 4 : 2;
                      const cost = getCost(p);
                      return (
                        <div
                          key={`rl-${p.id}-${idx}`}
                          className={`rounded-lg border p-5 transition-all hover:border-gray-700/80 ${
                            mode === "live"
                              ? "border-cyan-800/40 bg-cyan-950/10"
                              : "border-purple-800/40 bg-purple-950/10"
                          }`}
                        >
                          <div className="flex items-start justify-between mb-4">
                            <div className="flex-1">
                              <div className="flex items-center gap-2 mb-1">
                                {bot === "all" && <StrategyBadge strategy="rl_crypto" />}
                                <span className="font-mono text-sm font-bold">
                                  {p.symbol}
                                </span>
                                <span className="px-2 py-0.5 rounded-md text-[9px] font-mono font-bold uppercase bg-purple-900/60 text-purple-300">
                                  {p.action.toUpperCase()}
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
                                Cost
                              </div>
                              <div className="text-lg font-mono font-bold tabular-nums">
                                ${fmt(cost)}
                              </div>
                            </div>
                          </div>
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-3">
                            <Stat label="Entry" value={`$${fmt(p.entry_price)}`} />
                            <Stat label="Size" value={fmt(p.position_size)} />
                            <Stat label="Cost" value={`$${fmt(cost)}`} />
                            {p.regime && (
                              <Stat label="Regime" value={p.regime} valueClass="text-purple-400" />
                            )}
                            {spot != null && (
                              <Stat
                                label={`${asset} Spot`}
                                value={`$${fmt(spot, decimals)}`}
                                valueClass="text-blue-400"
                              />
                            )}
                          </div>
                        </div>
                      );
                    }

                    const pK = p as KalshiPosition;
                    const spot = prices[asset]?.price || null;
                    const decimals = asset === "DOGE" || asset === "XRP" ? 4 : 2;
                    const maxProfit =
                      pK.side === "no"
                        ? ((100 - pK.entry_price_cents) / 100) * pK.contracts
                        : (pK.entry_price_cents / 100) * pK.contracts;

                    return (
                      <div
                        key={`${pK.ticker}-${idx}`}
                        className={`rounded-lg border p-5 transition-all hover:border-gray-700/80 ${
                          mode === "live"
                            ? "border-cyan-800/40 bg-cyan-950/10"
                            : "border-amber-800/40 bg-amber-950/5"
                        }`}
                      >
                        <div className="flex items-start justify-between mb-4">
                          <div className="flex-1">
                            <div className="flex items-center gap-2 mb-1">
                              {bot === "all" && <StrategyBadge strategy="kalshi" />}
                              <span className="font-mono text-sm font-bold">
                                {pK.ticker}
                              </span>
                              <span
                                className={`px-2 py-0.5 rounded-md text-[9px] font-mono font-bold uppercase ${
                                  pK.side === "no"
                                    ? "bg-green-900/60 text-green-300"
                                    : "bg-red-900/60 text-red-300"
                                }`}
                              >
                                BUY_{pK.side}
                              </span>
                            </div>
                            {pK.opened_at && (
                              <div className="text-[10px] text-gray-600 font-mono">
                                Opened{" "}
                                {new Date(pK.opened_at).toLocaleString("en-US", {
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

                        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-3">
                          <Stat label="Entry" value={`${pK.entry_price_cents}¢`} />
                          <Stat
                            label="Fair Value"
                            value={
                              pK.fair_price_cents
                                ? `${pK.fair_price_cents.toFixed(0)}¢`
                                : "—"
                            }
                          />
                          <Stat
                            label="Edge"
                            value={pK.edge ? `${(pK.edge * 100).toFixed(1)}%` : "—"}
                            valueClass="text-cyan-400"
                          />
                          <Stat label="Contracts" value={String(pK.contracts)} />
                          <Stat label="Cost" value={`$${fmt(pK.cost)}`} />
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

                        {pK.reasoning && (
                          <div className="text-xs text-gray-500 leading-relaxed pt-3 border-t border-gray-800/40 font-mono">
                            {pK.reasoning}
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
