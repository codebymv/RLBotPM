import Link from "next/link";

type SectionHeaderProps = {
  title: string;
  subtitle?: string;
  actionHref?: string;
  actionLabel?: string;
  className?: string;
};

export function SectionHeader({
  title,
  subtitle,
  actionHref,
  actionLabel,
  className = "",
}: SectionHeaderProps) {
  return (
    <div className={`flex items-end justify-between mb-3 ${className}`}>
      <div>
        <h2 className="text-xl font-bold tracking-tight">{title}</h2>
        {subtitle && (
          <p className="text-sm text-gray-500 mt-0.5">{subtitle}</p>
        )}
      </div>
      {actionHref && actionLabel && (
        <Link
          href={actionHref}
          className="text-sm text-gray-400 hover:text-gray-200 transition-colors font-mono tracking-wide"
        >
          {actionLabel}
        </Link>
      )}
    </div>
  );
}
