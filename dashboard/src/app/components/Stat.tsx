export function Stat({
  label,
  value,
  valueClass = "",
}: {
  label: string;
  value: string;
  valueClass?: string;
}) {
  return (
    <div>
      <div className="text-[10px] text-gray-600 uppercase tracking-widest mb-1 font-mono font-bold">
        {label}
      </div>
      <div className={`font-mono font-medium tabular-nums ${valueClass}`}>
        {value}
      </div>
    </div>
  );
}
