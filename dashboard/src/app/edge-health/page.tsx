import { Suspense } from "react";
import EdgeHealthClient from "./EdgeHealthClient";

const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function getEdgeHealth(mode: string) {
  try {
    const res = await fetch(`${baseUrl}/api/kalshi/edge-health?mode=${mode}`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

async function getPnlSeries(mode: string) {
  try {
    const res = await fetch(`${baseUrl}/api/kalshi/pnl-series?mode=${mode}`, {
      cache: "no-store",
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}

export default async function EdgeHealthPage({
  searchParams,
}: {
  searchParams: { mode?: string };
}) {
  const mode = searchParams.mode || "paper";
  const [health, pnl] = await Promise.all([
    getEdgeHealth(mode),
    getPnlSeries(mode),
  ]);

  return (
    <Suspense fallback={<div className="p-6">Loading edge health...</div>}>
      <EdgeHealthClient health={health} pnl={pnl} />
    </Suspense>
  );
}
