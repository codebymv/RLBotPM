'use client';

import { useEffect, useState } from 'react';

type Trade = {
  symbol?: string;
  pnl?: number;
  timestamp?: string;
};

type PaperTradingMetrics = {
  capital: number;
  total_return_pct: number;
  win_rate: number;
  num_trades: number;
  open_positions: number;
  recent_trades?: Trade[];
};

export default function PaperTradingPage() {
  const [metrics, setMetrics] = useState<PaperTradingMetrics | null>(null);

  useEffect(() => {
    const interval = setInterval(async () => {
      const response = await fetch('/api/paper-trading/metrics');
      if (!response.ok) {
        return;
      }
      const data = await response.json();
      setMetrics(data);
    }, 5000);

    return () => clearInterval(interval);
  }, []);

  if (!metrics) {
    return <div className="p-6">Loading paper trading metrics...</div>;
  }

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Paper Trading Monitor</h1>

      <div className="grid grid-cols-4 gap-4 mb-6">
        <div className="rounded border p-4">
          <div className="text-sm text-gray-500">Capital</div>
          <div className="text-xl font-semibold">${metrics.capital.toFixed(2)}</div>
        </div>
        <div className="rounded border p-4">
          <div className="text-sm text-gray-500">Total Return</div>
          <div className="text-xl font-semibold">{(metrics.total_return_pct * 100).toFixed(2)}%</div>
        </div>
        <div className="rounded border p-4">
          <div className="text-sm text-gray-500">Win Rate</div>
          <div className="text-xl font-semibold">{(metrics.win_rate * 100).toFixed(2)}%</div>
        </div>
        <div className="rounded border p-4">
          <div className="text-sm text-gray-500">Trades</div>
          <div className="text-xl font-semibold">{metrics.num_trades}</div>
        </div>
      </div>

      <div className="rounded border p-4">
        <div className="text-lg font-semibold mb-3">Recent Trades</div>
        {metrics.recent_trades && metrics.recent_trades.length > 0 ? (
          <ul className="space-y-2">
            {metrics.recent_trades.map((trade, idx) => (
              <li key={`${trade.timestamp ?? idx}`} className="text-sm">
                {trade.symbol ?? 'N/A'} — PnL: {trade.pnl?.toFixed(2) ?? '0.00'}
              </li>
            ))}
          </ul>
        ) : (
          <div className="text-sm text-gray-500">No recent trades.</div>
        )}
      </div>
    </div>
  );
}
