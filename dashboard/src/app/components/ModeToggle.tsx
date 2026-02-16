"use client";

import { useRouter, useSearchParams, usePathname } from "next/navigation";

export type TradingMode = "paper" | "live";

export function ModeToggle() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const currentMode = (searchParams.get("mode") || "paper") as TradingMode;

  const setMode = (mode: TradingMode) => {
    const params = new URLSearchParams(searchParams);
    params.set("mode", mode);
    router.push(`${pathname}?${params.toString()}`);
  };

  return (
    <div
      role="group"
      aria-label="Trading mode selector"
      className="inline-flex rounded-lg border border-gray-700/60 bg-gray-900/40 p-0.5 backdrop-blur-sm"
    >
      <button
        onClick={() => setMode("paper")}
        aria-pressed={currentMode === "paper"}
        aria-label="Switch to paper trading mode"
        className={`min-h-[44px] sm:min-h-0 px-4 py-1.5 rounded-md text-xs font-medium tracking-wide transition-all focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:ring-offset-2 focus:ring-offset-gray-950 ${
          currentMode === "paper"
            ? "bg-amber-500/20 text-amber-300 shadow-sm"
            : "text-gray-500 hover:text-gray-300"
        }`}
      >
        <span className="font-mono">PAPER</span>
      </button>
      <button
        onClick={() => setMode("live")}
        aria-pressed={currentMode === "live"}
        aria-label="Switch to live trading mode"
        className={`min-h-[44px] sm:min-h-0 px-4 py-1.5 rounded-md text-xs font-medium tracking-wide transition-all focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:ring-offset-2 focus:ring-offset-gray-950 ${
          currentMode === "live"
            ? "bg-cyan-500/20 text-cyan-300 shadow-sm"
            : "text-gray-500 hover:text-gray-300"
        }`}
      >
        <span className="font-mono">LIVE</span>
      </button>
    </div>
  );
}

export function useMode(): TradingMode {
  const searchParams = useSearchParams();
  return (searchParams.get("mode") || "paper") as TradingMode;
}
