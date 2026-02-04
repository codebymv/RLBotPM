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
