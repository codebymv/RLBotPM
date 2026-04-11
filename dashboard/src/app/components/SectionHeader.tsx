import Link from "next/link";

type SectionHeaderProps = {
  id?: string;
  title: string;
  subtitle?: string;
  actionHref?: string;
  actionLabel?: string;
  className?: string;
};

export function SectionHeader({
  id,
  title,
  subtitle,
  actionHref,
  actionLabel,
  className = "",
}: SectionHeaderProps) {
  return (
    <div className={`flex items-end justify-between mb-4 ${className}`}>
      <div>
        <h2 id={id} className="text-xl font-bold tracking-tight">{title}</h2>
        {subtitle && (
          <p className="text-sm text-gray-500 mt-0.5">{subtitle}</p>
        )}
      </div>
      {actionHref && actionLabel && (
        <Link
          href={actionHref}
          className="text-sm text-gray-400 hover:text-gray-200 transition-colors tracking-wide"
        >
          {actionLabel}
        </Link>
      )}
    </div>
  );
}
