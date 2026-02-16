type EmptyStateProps = {
  message: string;
  submessage?: string;
  className?: string;
};

export function EmptyState({
  message,
  submessage,
  className = "",
}: EmptyStateProps) {
  return (
    <div
      role="status"
      aria-live="polite"
      aria-label={`${message}${submessage ? `. ${submessage}` : ""}`}
      className={`flex flex-col items-center justify-center py-16 px-4 rounded-lg border border-gray-800/50 bg-gray-900/20 ${className}`}
    >
      <div className="text-gray-600 text-4xl mb-3 font-mono" aria-hidden="true">—</div>
      <p className="text-sm text-gray-400 font-medium">{message}</p>
      {submessage && (
        <p className="text-xs text-gray-600 mt-1">{submessage}</p>
      )}
    </div>
  );
}
