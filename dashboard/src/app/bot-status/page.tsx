import { Suspense } from "react";
import { SectionHeader } from "../components/SectionHeader";
import { EmptyState } from "../components/EmptyState";
import BotStatusClient from "./BotStatusClient";

const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Session = {
  session_id: string;
  started_at: string | null;
  last_trade_at: string | null;
  trades_opened: number;
  open_now: number;
  wins: number;
  losses: number;
  realized_pnl: number;
};

type BotData = {
  sessions: Session[];
  total_sessions: number;
  total_trades: number;
  first_trade_at: string | null;
  last_trade_at: string | null;
  strategy: {
    name: string;
    side_filter: string;
    min_edge: number;
    max_edge: number;
    min_price: number;
    max_price: number;
    assets: string[];
    volatilities: Record<string, number>;
  };
};

type TrainingRun = {
  id: number;
  started_at: string | null;
  ended_at: string | null;
  total_episodes: number;
  best_sharpe_ratio: number | null;
  best_episode_reward: number | null;
  status: string | null;
  config_snapshot: unknown;
};

async function getBotStatus(): Promise<BotData | null> {
  try {
    const res = await fetch(`${baseUrl}/api/bot/status`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function getMarketStats() {
  try {
    const res = await fetch(`${baseUrl}/api/kalshi/market-stats`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function getTrainingRuns(): Promise<{ runs: TrainingRun[]; total: number }> {
  try {
    const res = await fetch(`${baseUrl}/api/training/runs?limit=10`, {
      cache: "no-store",
    });
    if (!res.ok) return { runs: [], total: 0 };
    const data = await res.json();
    return { runs: data.runs || [], total: data.total || 0 };
  } catch {
    return { runs: [], total: 0 };
  }
}

export default async function BotStatusPage() {
  const [kalshiBot, mkt, trainingRuns] = await Promise.all([
    getBotStatus(),
    getMarketStats(),
    getTrainingRuns(),
  ]);

  return (
    <Suspense fallback={<div className="p-6">Loading bot status...</div>}>
      <BotStatusClient
        kalshiBot={kalshiBot}
        mkt={mkt}
        trainingRuns={trainingRuns.runs}
      />
    </Suspense>
  );
}
