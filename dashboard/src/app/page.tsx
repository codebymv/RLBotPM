async function getDataSourceHealth() {
  const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
  const res = await fetch(`${baseUrl}/api/data-sources/health`, {
    cache: "no-store",
  });
  if (!res.ok) {
    return { status: "error", sources: [] };
  }
  return res.json();
}

export default async function Page() {
  const health = await getDataSourceHealth();

  return (
    <main style={{ padding: "24px" }}>
      <h1>RLTrade Dashboard</h1>
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
