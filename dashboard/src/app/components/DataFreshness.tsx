type DataFreshnessProps = {
  lastUpdated?: string | null;
  className?: string;
};

export function DataFreshness({
  lastUpdated,
  className = "",
}: DataFreshnessProps) {
  if (!lastUpdated) return null;

  const date = new Date(lastUpdated);
  const now = new Date();
  const ageSeconds = (now.getTime() - date.getTime()) / 1000;

  const isStale = ageSeconds > 300; // 5 minutes

  const relativeTime =
    ageSeconds < 60
      ? `${Math.floor(ageSeconds)}s ago`
      : ageSeconds < 3600
        ? `${Math.floor(ageSeconds / 60)}m ago`
        : `${Math.floor(ageSeconds / 3600)}h ago`;

  return (
    <div
      className={`inline-flex items-center gap-1.5 text-[10px] font-mono ${
        isStale ? "text-amber-500" : "text-gray-600"
      } ${className}`}
    >
      <span className={`w-1 h-1 rounded-full ${isStale ? "bg-amber-500" : "bg-gray-600"}`} />
      <span>{relativeTime}</span>
    </div>
  );
}
