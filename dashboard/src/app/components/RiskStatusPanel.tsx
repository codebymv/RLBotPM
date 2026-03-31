"use client";

import { type RiskStatusResponse } from "../../lib/api";

type Props = {
  risk: RiskStatusResponse | null;
};

function StatusDot({ ok, warn }: { ok: boolean; warn?: boolean }) {
  const color = !ok ? "bg-red-500" : warn ? "bg-amber-400" : "bg-green-500";
  const pulse = !ok ? "animate-pulse" : "";
  return (
    <span className={`inline-block w-2 h-2 rounded-full ${color} ${pulse} flex-shrink-0`} />
  );
}

function BreacherBar({
  label,
  pctUsed,
  current,
  limit,
  unit = "$",
  isTriggered,
}: {
  label: string;
  pctUsed: number;
  current: number;
  limit: number;
  unit?: string;
  isTriggered: boolean;
}) {
  const clamped = Math.min(pctUsed * 100, 100);
  const barColor = isTriggered
    ? "bg-red-500"
    : clamped > 70
    ? "bg-amber-500"
    : "bg-green-500/70";

  const fmtVal = unit === "$" ? `$${Math.abs(current).toFixed(2)}` : `${(current * 100).toFixed(1)}%`;
  const fmtLimit = unit === "$" ? `$${limit}` : `${(limit * 100).toFixed(0)}%`;

  return (
    <div className="mb-3">
      <div className="flex justify-between text-[10px] font-mono text-gray-500 mb-1">
        <span className="uppercase tracking-wider">{label}</span>
        <span>
          <span className={isTriggered ? "text-red-400 font-bold" : "text-gray-400"}>
            {fmtVal}
          </span>
          <span className="text-gray-600"> / {fmtLimit}</span>
        </span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-gray-800/80">
        <div
          className={`h-1.5 rounded-full transition-all ${barColor}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
}

export function RiskStatusPanel({ risk }: Props) {
  if (!risk) {
    return (
      <div className="rounded-lg border border-gray-800/60 bg-gray-900/20 p-4">
        <div className="text-[10px] uppercase tracking-widest text-gray-500 mb-2">
          Risk / Circuit Breakers
        </div>
        <div className="text-xs text-gray-600 font-mono">
          No risk data — bot not running or state file missing
        </div>
      </div>
    );
  }

  const cbStatus = risk.status;
  const isTriggered = cbStatus === "paused" || cbStatus === "triggered";
  const isStale = risk.stale;

  const statusColor = isTriggered
    ? "text-red-400 border-red-800/60 bg-red-950/10"
    : isStale
    ? "text-amber-400 border-amber-800/40 bg-amber-950/10"
    : "text-green-400 border-green-800/40 bg-green-950/10";

  const cb = risk.circuit_breakers;

  return (
    <div className={`rounded-lg border p-4 ${statusColor}`}>
      <div className="flex items-center justify-between mb-3">
        <div className="text-[10px] uppercase tracking-widest text-gray-500">
          Risk / Circuit Breakers
        </div>
        <div className="flex items-center gap-1.5">
          <StatusDot ok={!isTriggered && !isStale} warn={isStale} />
          <span className="text-[10px] font-mono font-bold uppercase">
            {isStale ? "STALE" : cbStatus.toUpperCase()}
          </span>
          {isStale && risk.seconds_since_write != null && (
            <span className="text-[9px] text-gray-600 font-mono">
              ({Math.round(risk.seconds_since_write / 60)}m ago)
            </span>
          )}
        </div>
      </div>

      {cb && (
        <>
          <BreacherBar
            label="Daily Loss"
            pctUsed={cb.daily_loss?.pct_used ?? 0}
            current={cb.daily_loss?.current_usd ?? 0}
            limit={cb.daily_loss?.limit_usd ?? 20}
            unit="$"
            isTriggered={cb.daily_loss?.status === "triggered"}
          />
          <BreacherBar
            label="Weekly Loss"
            pctUsed={cb.weekly_loss?.pct_used ?? 0}
            current={cb.weekly_loss?.current_usd ?? 0}
            limit={cb.weekly_loss?.limit_usd ?? 50}
            unit="$"
            isTriggered={cb.weekly_loss?.status === "triggered"}
          />
          <BreacherBar
            label="Drawdown"
            pctUsed={cb.drawdown?.pct_used ?? 0}
            current={cb.drawdown?.current ?? 0}
            limit={cb.drawdown?.limit ?? 0.3}
            unit="%"
            isTriggered={cb.drawdown?.status === "triggered"}
          />
          <div className="flex items-center justify-between text-[10px] font-mono text-gray-500 mt-2">
            <span className="uppercase tracking-wider">Consec. Losses</span>
            <span>
              <span className={cb.consecutive_losses.current >= cb.consecutive_losses.limit ? "text-red-400 font-bold" : "text-gray-400"}>
                {cb.consecutive_losses?.current ?? 0}
              </span>
              <span className="text-gray-600"> / {cb.consecutive_losses?.limit ?? 5}</span>
            </span>
          </div>
        </>
      )}

      {risk.recent_events && risk.recent_events.length > 0 && (
        <div className="mt-3 pt-3 border-t border-gray-800/40">
          <div className="text-[9px] uppercase tracking-wider text-gray-600 mb-1.5">Recent Events</div>
          {risk.recent_events.slice(-3).map((ev, i) => (
            <div key={i} className="text-[9px] font-mono text-amber-400/80 truncate mb-0.5">
              {ev.rule}: {ev.description}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
