const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function getDataSourceHealth() {
  const res = await fetch(`${baseUrl}/api/data-sources/health`, {
    cache: "no-store",
  });
  if (!res.ok) {
    return { status: "error", sources: [] };
  }
  return res.json();
}

async function getKalshiStatus() {
  try {
    const res = await fetch(`${baseUrl}/api/kalshi/status`, { cache: "no-store" });
    if (!res.ok) return { configured: false, message: "API error" };
    return res.json();
  } catch {
    return { configured: false, message: "Unavailable" };
  }
}

export default async function Page() {
  const [health, kalshiStatus] = await Promise.all([
    getDataSourceHealth(),
    getKalshiStatus(),
  ]);

  return (
    <main style={{ padding: "24px" }}>
      <h1>RLTrade Dashboard</h1>

      {kalshiStatus && (
        <section style={{ marginBottom: "24px" }}>
          <h2 style={{ fontSize: "1.1rem", marginBottom: "8px" }}>Kalshi</h2>
          <p style={{ color: kalshiStatus.configured ? "green" : "gray" }}>
            {kalshiStatus.message}
          </p>
        </section>
      )}

      <p>Data Sources Health</p>
      <pre
        style={{
          padding: "12px",
          background: "#111",
          color: "#eee",
          borderRadius: "8px",
          overflowX: "auto",
        }}
      >
        {JSON.stringify(health, null, 2)}
      </pre>
    </main>
  );
}
