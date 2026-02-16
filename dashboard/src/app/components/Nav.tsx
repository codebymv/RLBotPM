"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname } from "next/navigation";
import { ModeToggle } from "./ModeToggle";

const links = [
  { href: "/", label: "OVERVIEW", icon: "■" },
  { href: "/positions", label: "POSITIONS", icon: "▣" },
  { href: "/crypto", label: "MARKET", icon: "▲" },
  { href: "/edge-health", label: "EDGE", icon: "◆" },
  { href: "/bot-status", label: "STATUS", icon: "●" },
];

export default function Nav() {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Main navigation"
      className="border-b border-gray-800/60 bg-gray-950/95 backdrop-blur-md sticky top-0 z-50 shadow-lg shadow-black/20"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center justify-between h-16 gap-4">
        {/* Logo */}
        <Link
          href="/"
          aria-label="RLTrade home"
          className="flex items-center gap-2 shrink-0 hover:opacity-80 transition-opacity focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:ring-offset-2 focus:ring-offset-gray-950 rounded-sm"
        >
          <Image
            src="/rltrade-icon.png"
            alt="RLTrade"
            width={32}
            height={32}
            className="w-8 h-8"
          />
          <Image
            src="/rltrade-text.png"
            alt="RLTrade"
            width={120}
            height={32}
            className="h-7 w-auto hidden sm:block"
          />
        </Link>

        {/* Navigation Links */}
        <div className="flex gap-1 overflow-x-auto flex-1" role="navigation">
          {links.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                aria-current={active ? "page" : undefined}
                className={`flex items-center gap-1.5 px-3 py-2 rounded-md text-[11px] font-mono font-medium whitespace-nowrap transition-all focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:ring-offset-2 focus:ring-offset-gray-950 min-h-[44px] sm:min-h-0 ${
                  active
                    ? "bg-cyan-500/20 text-cyan-300 shadow-sm"
                    : "text-gray-500 hover:text-gray-200 hover:bg-gray-800/50"
                }`}
              >
                <span className="text-xs opacity-60" aria-hidden="true">{l.icon}</span>
                <span>{l.label}</span>
              </Link>
            );
          })}
        </div>

        {/* Mode Toggle */}
        <div className="shrink-0">
          <ModeToggle />
        </div>
      </div>
    </nav>
  );
}
