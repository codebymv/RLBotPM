# RLTrade Dashboard

Next.js dashboard for monitoring the RL trading bot.

## Features

- Real-time training metrics
- Episode performance visualization
- Trade history and analysis
- Risk management status
- Model checkpoint management

## Setup

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)

## Configuration

Set the API URL in `.env.local`:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Development

This is a Next.js 14 application using:
- App Router
- TailwindCSS for styling
- TanStack Query for data fetching
- Recharts for visualizations

## Deployment

Deploy to Railway as a separate service from the `/dashboard` subdirectory.

## How To Read Overview KPIs

- **Total Trades (All)**: all records for the selected bot/mode (open + settled/closed).
- **Settled Win Rate**: `wins / (wins + losses)` for settled/closed outcomes only.
- **Open Positions**: currently open positions that have not settled/closed yet.
- **Kalshi Settled Markets**: size of the backfilled Kalshi dataset, not your bot's settled trade count.

Example:

- If the dashboard shows `53W / 6L`, settled trades are `59`.
- Settled win rate is `53 / 59 = 89.8%`.
- Total Trades (All) can still be higher than `59` because it includes open positions.
