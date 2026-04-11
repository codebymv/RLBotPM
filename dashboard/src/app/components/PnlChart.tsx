"use client";

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { type PnlSeriesPoint } from "../../lib/api";

type Props = {
  series: PnlSeriesPoint[];
  height?: number;
};

function fmtDate(iso: string) {
  try {
    const d = new Date(iso);
    return d.toLocaleDateString("en-US", { month: "short", day: "numeric" });
  } catch {
    return iso;
  }
}

function fmtDollar(n: number) {
  const sign = n >= 0 ? "+" : "";
  return `${sign}$${Math.abs(n).toFixed(2)}`;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function CustomTooltip({ active, payload }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0].payload as PnlSeriesPoint;
  return (
    <div className="rounded-md border border-gray-700 bg-gray-900 p-3 text-xs font-mono shadow-lg">
      <div className="text-gray-400 mb-1">{fmtDate(d.timestamp)}</div>
      <div className="font-bold text-gray-100">{d.ticker}</div>
      <div className={`mt-1 ${d.pnl >= 0 ? "text-green-400" : "text-red-400"}`}>
        Trade: {fmtDollar(d.pnl)}
      </div>
      <div className="text-gray-300 mt-0.5">
        Cumulative: {fmtDollar(d.cumulative_pnl)}
      </div>
    </div>
  );
}

export function PnlChart({ series, height = 220 }: Props) {
  if (!series || series.length === 0) {
    return (
      <div
        className="flex items-center justify-center rounded-lg border border-gray-800/60 bg-gray-900/20 text-gray-600 text-sm font-mono"
        style={{ height }}
      >
        No settled trades yet — chart will appear once trades resolve
      </div>
    );
  }

  const isPositive =
    series.length > 0 &&
    series[series.length - 1].cumulative_pnl >= 0;

  const lineColor = isPositive ? "#4ade80" : "#f87171";
  const gradientId = "pnl-gradient";

  return (
    <div style={{ height }} className="w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={series}
          margin={{ top: 4, right: 8, bottom: 0, left: 8 }}
        >
          <defs>
            <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={lineColor} stopOpacity={0.2} />
              <stop offset="95%" stopColor={lineColor} stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#1f2937"
            vertical={false}
          />
          <XAxis
            dataKey="timestamp"
            tickFormatter={fmtDate}
            tick={{ fill: "#6b7280", fontSize: 10, fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            tickFormatter={(v: number) => `$${v.toFixed(0)}`}
            tick={{ fill: "#6b7280", fontSize: 10, fontFamily: "monospace" }}
            tickLine={false}
            axisLine={false}
            width={52}
          />
          <ReferenceLine y={0} stroke="#374151" strokeDasharray="4 2" />
          <Tooltip content={<CustomTooltip />} />
          <Area
            type="monotone"
            dataKey="cumulative_pnl"
            stroke={lineColor}
            strokeWidth={2}
            fill={`url(#${gradientId})`}
            dot={false}
            activeDot={{ r: 4, fill: lineColor, stroke: "#111827", strokeWidth: 2 }}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
