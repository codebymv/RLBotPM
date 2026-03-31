"use client";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import { type BotSession } from "../../lib/api";

type Props = {
  sessions: BotSession[];
  height?: number;
};

function fmtDate(iso: string | null) {
  if (!iso) return "—";
  try {
    return new Date(iso).toLocaleDateString("en-US", {
      month: "short",
      day: "numeric",
    });
  } catch {
    return iso;
  }
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const s = payload[0].payload as BotSession & { label: string };
  const pnl = s.realized_pnl;
  const settled = s.wins + s.losses;
  const wr = settled > 0 ? ((s.wins / settled) * 100).toFixed(0) : "—";
  return (
    <div className="rounded-md border border-gray-700 bg-gray-900 p-3 text-xs font-mono shadow-lg min-w-[140px]">
      <div className="text-gray-400 mb-1">{s.label}</div>
      <div className={`font-bold ${pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
        {pnl >= 0 ? "+" : ""}${Math.abs(pnl).toFixed(2)}
      </div>
      <div className="text-gray-400 mt-1">
        {s.wins}W / {s.losses}L · {wr}% WR
      </div>
      <div className="text-gray-500 mt-0.5">{s.trades_opened} trades</div>
    </div>
  );
}

export function SessionHistoryChart({ sessions, height = 180 }: Props) {
  if (!sessions || sessions.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border border-gray-800/60 bg-gray-900/20 text-gray-600 text-sm font-mono"
        style={{ height }}
      >
        No session history yet
      </div>
    );
  }

  // Oldest → newest left to right
  const data = [...sessions].reverse().map((s, i) => ({
    ...s,
    label: fmtDate(s.started_at) || `Session ${i + 1}`,
  }));

  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart
          data={data}
          margin={{ top: 4, right: 8, bottom: 0, left: 8 }}
          barCategoryGap="30%"
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#1f2937"
            vertical={false}
          />
          <XAxis
            dataKey="label"
            tick={{ fill: "#6b7280", fontSize: 10, fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
          />
          <YAxis
            tickFormatter={(v: number) => `$${v.toFixed(0)}`}
            tick={{ fill: "#6b7280", fontSize: 10, fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
            width={52}
          />
          <ReferenceLine y={0} stroke="#374151" strokeDasharray="4 2" />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
          <Bar dataKey="realized_pnl" radius={[3, 3, 0, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.realized_pnl >= 0 ? "#4ade80" : "#f87171"}
                fillOpacity={0.8}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
