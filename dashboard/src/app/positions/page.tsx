import { Suspense } from "react";
import PositionsClient from "./PositionsClient";

const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function getPositions() {
  try {
    const res = await fetch(`${baseUrl}/api/kalshi/positions`, {
      cache: "no-store",
    });
    if (!res.ok) return { positions: [], count: 0 };
    return res.json();
  } catch {
    return { positions: [], count: 0 };
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

export default async function PositionsPage() {
  const [data, crypto] = await Promise.all([getPositions(), getCryptoPrices()]);

  return (
    <Suspense fallback={<div className="p-6">Loading positions...</div>}>
      <PositionsClient data={data} crypto={crypto} />
    </Suspense>
  );
}
