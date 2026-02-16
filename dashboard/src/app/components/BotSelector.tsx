"use client";

import { useRouter, useSearchParams, usePathname } from "next/navigation";

export type TradingBot = "all" | "rl_crypto" | "kalshi";

export function BotSelector() {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const currentBot = (searchParams.get("bot") || "all") as TradingBot;

  const setBot = (bot: TradingBot) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("bot", bot);
    router.push(`${pathname}?${params.toString()}`);
  };

  return (
    <div
      role="group"
      aria-label="Bot selector"
      className="inline-flex rounded-lg border border-gray-700/60 bg-gray-900/40 p-0.5 backdrop-blur-sm"
    >
      <button
        onClick={() => setBot("all")}
        aria-pressed={currentBot === "all"}
        aria-label="View all strategies"
        className={`min-h-[44px] sm:min-h-0 px-3 py-1.5 rounded-md text-xs font-medium tracking-wide transition-all focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:ring-offset-2 focus:ring-offset-gray-950 ${
          currentBot === "all"
            ? "bg-cyan-500/20 text-cyan-300 shadow-sm"
            : "text-gray-500 hover:text-gray-300"
        }`}
      >
        <span className="font-mono">ALL</span>
      </button>
      <button
        onClick={() => setBot("rl_crypto")}
        aria-pressed={currentBot === "rl_crypto"}
        aria-label="View RL Crypto Bot only"
        className={`min-h-[44px] sm:min-h-0 px-3 py-1.5 rounded-md text-xs font-medium tracking-wide transition-all focus:outline-none focus:ring-2 focus:ring-purple-500/50 focus:ring-offset-2 focus:ring-offset-gray-950 ${
          currentBot === "rl_crypto"
            ? "bg-purple-500/20 text-purple-300 shadow-sm"
            : "text-gray-500 hover:text-gray-300"
        }`}
      >
        <span className="font-mono">RL</span>
      </button>
      <button
        onClick={() => setBot("kalshi")}
        aria-pressed={currentBot === "kalshi"}
        aria-label="View Kalshi Market Bot only"
        className={`min-h-[44px] sm:min-h-0 px-3 py-1.5 rounded-md text-xs font-medium tracking-wide transition-all focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:ring-offset-2 focus:ring-offset-gray-950 ${
          currentBot === "kalshi"
            ? "bg-blue-500/20 text-blue-300 shadow-sm"
            : "text-gray-500 hover:text-gray-300"
        }`}
      >
        <span className="font-mono">KALSHI</span>
      </button>
    </div>
  );
}

export function useBot(): TradingBot {
  const searchParams = useSearchParams();
  return (searchParams.get("bot") || "all") as TradingBot;
}
