type StrategyBadgeProps = {
  strategy: "rl_crypto" | "kalshi";
  className?: string;
};

export function StrategyBadge({ strategy, className = "" }: StrategyBadgeProps) {
  const styles = {
    rl_crypto: "bg-purple-500/20 text-purple-300 border-purple-700/60",
    kalshi: "bg-blue-500/20 text-blue-300 border-blue-700/60",
  };

  const labels = {
    rl_crypto: "RL",
    kalshi: "K",
  };

  return (
    <span
      className={`inline-flex items-center px-2 py-0.5 rounded-md text-[9px] font-mono font-bold border ${styles[strategy]} ${className}`}
    >
      {labels[strategy]}
    </span>
  );
}
