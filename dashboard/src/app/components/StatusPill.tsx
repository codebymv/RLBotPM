type StatusPillProps = {
  mode: "paper" | "live";
  className?: string;
};

export function StatusPill({ mode, className = "" }: StatusPillProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-bold uppercase tracking-wider ${
        mode === "live"
          ? "bg-cyan-500/20 text-cyan-300 border border-cyan-700/60"
          : "bg-amber-500/20 text-amber-400 border border-amber-800/40"
      } ${className}`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${mode === "live" ? "bg-cyan-400 animate-pulse" : "bg-amber-500"}`}
      />
      <span className="font-mono">{mode}</span>
    </span>
  );
}
