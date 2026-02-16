type KpiCardProps = {
  label: string;
  value: string | number;
  sublabel?: string;
  mode?: "paper" | "live" | "neutral";
  trend?: "up" | "down" | "flat";
  className?: string;
};

export function KpiCard({
  label,
  value,
  sublabel,
  mode = "neutral",
  trend,
  className = "",
}: KpiCardProps) {
  const modeStyles = {
    paper: "border-amber-800/40 bg-amber-950/10",
    live: "border-cyan-700/60 bg-cyan-950/20 shadow-cyan-900/10 shadow-sm",
    neutral: "border-gray-800/60 bg-gray-900/40",
  };

  const trendIcon = trend === "up" ? "↑" : trend === "down" ? "↓" : null;
  const trendLabel = trend === "up" ? "trending up" : trend === "down" ? "trending down" : "";

  return (
    <div
      role="article"
      aria-label={`${label}: ${value}${sublabel ? `, ${sublabel}` : ""}${trendLabel ? `, ${trendLabel}` : ""}`}
      className={`rounded-lg border p-4 transition-colors ${modeStyles[mode]} ${className}`}
    >
      <div className="text-[11px] uppercase tracking-widest text-gray-500 mb-1.5 font-medium">
        {label}
      </div>
      <div className="flex items-baseline gap-2">
        <div className="text-3xl font-mono font-bold tabular-nums">{value}</div>
        {trendIcon && (
          <span className="text-base text-gray-400" aria-label={trendLabel}>
            {trendIcon}
          </span>
        )}
      </div>
      {sublabel && (
        <div className="text-sm text-gray-500 mt-1 font-mono">{sublabel}</div>
      )}
    </div>
  );
}
