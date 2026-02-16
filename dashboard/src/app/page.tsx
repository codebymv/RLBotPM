import { Suspense } from "react";
import OverviewClient from "./OverviewClient";

const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function getHealth() {
  try {
    const res = await fetch(`${baseUrl}/health`, { cache: "no-store" });
    if (!res.ok) return { status: "error" };
    return res.json();
  } catch {
    return { status: "unreachable" };
  }
}

async function getMetrics() {
  try {
    const res = await fetch(`${baseUrl}/api/paper-trading/metrics`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function getCryptoPrices() {
  try {
    const res = await fetch(`${baseUrl}/api/crypto/prices`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function getBotStatus() {
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

async function getCombinedMetrics(mode: string) {
  try {
    const res = await fetch(`${baseUrl}/api/metrics/combined?mode=${mode}`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

type PageProps = {
  searchParams?: { mode?: string; bot?: string };
};

export default async function Page({ searchParams }: PageProps) {
  const mode = searchParams?.mode || "paper";

  const [health, combinedMetrics, crypto, bot, mktStats] = await Promise.all([
    getHealth(),
    getCombinedMetrics(mode),
    getCryptoPrices(),
    getBotStatus(),
    getMarketStats(),
  ]);

  return (
    <Suspense fallback={<div className="p-6">Loading...</div>}>
      <OverviewClient
        health={health}
        combinedMetrics={combinedMetrics}
        crypto={crypto}
        bot={bot}
        mktStats={mktStats}
      />
    </Suspense>
  );
}
