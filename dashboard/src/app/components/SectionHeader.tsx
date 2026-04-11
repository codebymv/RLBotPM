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
    <div className={`flex items-end justify-between mb-5 ${className}`}>
      <div>
        <h2 id={id} className="text-xl font-bold tracking-tight">{title}</h2>
        {subtitle && (
          <p className="text-sm text-gray-500 mt-0.5">{subtitle}</p>
        )}
      </div>
      {actionHref && actionLabel && (
        <Link
          href={actionHref}
          className="text-sm font-medium text-gray-400 hover:text-cyan-400 transition-colors tracking-wide shrink-0 ml-4"
        >
          {actionLabel}
        </Link>
      )}
    </div>
  );
}
