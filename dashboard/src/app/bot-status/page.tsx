import { Suspense } from "react";
import { SectionHeader } from "../components/SectionHeader";
import { EmptyState } from "../components/EmptyState";
import BotStatusClient from "./BotStatusClient";
import type { BotStatusResponse } from "../../lib/api";

const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type BotData = BotStatusResponse;

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
