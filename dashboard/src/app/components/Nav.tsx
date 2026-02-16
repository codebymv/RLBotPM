"use client";

import { useState } from "react";
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
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <nav
      aria-label="Main navigation"
      className="border-b border-gray-800/60 bg-gray-950/95 backdrop-blur-md sticky top-0 z-50 shadow-lg shadow-black/20"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-16 gap-4">
          {/* Logo */}
          <Link
            href="/"
            aria-label="RLTrade home"
            className="flex items-center shrink-0 hover:opacity-80 transition-opacity focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:ring-offset-2 focus:ring-offset-gray-950 rounded-sm"
          >
            <Image
              src="/rltrade-icon.png"
              alt="RLTrade"
              width={32}
              height={32}
              className="w-8 h-8"
            />
          </Link>

          {/* Desktop Navigation Links */}
          <div className="hidden md:flex gap-1 flex-1" role="navigation">
            {links.map((l) => {
              const active = pathname === l.href;
              return (
                <Link
                  key={l.href}
                  href={l.href}
                  aria-current={active ? "page" : undefined}
                  className={`flex items-center gap-1.5 px-3 py-2 rounded-md text-[11px] font-mono font-medium whitespace-nowrap transition-all focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:ring-offset-2 focus:ring-offset-gray-950 ${
                    active
                      ? "bg-cyan-500/20 text-cyan-300 shadow-sm"
                      : "text-gray-500 hover:text-gray-200 hover:bg-gray-800/50"
                  }`}
                >
                  <span className="text-xs opacity-60" aria-hidden="true">
                    {l.icon}
                  </span>
                  <span>{l.label}</span>
                </Link>
              );
            })}
          </div>

          {/* Mode Toggle - Desktop */}
          <div className="hidden md:block shrink-0">
            <ModeToggle />
          </div>

          {/* Hamburger Menu Button - Mobile */}
          <button
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-label="Toggle menu"
            aria-expanded={mobileMenuOpen}
            className="md:hidden p-2 rounded-md text-gray-400 hover:text-gray-200 hover:bg-gray-800/50 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 focus:ring-offset-2 focus:ring-offset-gray-950"
          >
            <svg
              className="w-6 h-6"
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth="2"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              {mobileMenuOpen ? (
                <path d="M6 18L18 6M6 6l12 12" />
              ) : (
                <path d="M4 6h16M4 12h16M4 18h16" />
              )}
            </svg>
          </button>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <>
            {/* Backdrop */}
            <div
              className="md:hidden fixed inset-0 bg-black/60 backdrop-blur-sm z-40"
              onClick={() => setMobileMenuOpen(false)}
              aria-hidden="true"
            />
            
            {/* Menu */}
            <div className="md:hidden relative z-50 py-4 space-y-2 border-t border-gray-800/60">
            {links.map((l) => {
              const active = pathname === l.href;
              return (
                <Link
                  key={l.href}
                  href={l.href}
                  onClick={() => setMobileMenuOpen(false)}
                  aria-current={active ? "page" : undefined}
                  className={`flex items-center gap-2 px-4 py-3 rounded-md text-sm font-mono font-medium transition-all ${
                    active
                      ? "bg-cyan-500/20 text-cyan-300 shadow-sm"
                      : "text-gray-500 hover:text-gray-200 hover:bg-gray-800/50"
                  }`}
                >
                  <span className="text-base opacity-60" aria-hidden="true">
                    {l.icon}
                  </span>
                  <span>{l.label}</span>
                </Link>
              );
            })}
            <div className="pt-3 border-t border-gray-800/60">
              <ModeToggle />
            </div>
          </div>
          </>
        )}
      </div>
    </nav>
  );
}
