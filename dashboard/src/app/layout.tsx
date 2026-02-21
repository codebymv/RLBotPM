import type { Metadata } from "next";
import "./globals.css";
import Nav from "./components/Nav";
import { Providers } from "./providers";

export const metadata: Metadata = {
  title: "RLTrade Dashboard",
  description: "Crypto prediction market trading bot dashboard",
  icons: {
    icon: "/rltrade-icon.png",
    shortcut: "/rltrade-icon.png",
    apple: "/rltrade-icon.png",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-gray-950 text-gray-100 antialiased">
        <Providers>
          <Nav />
          {children}
        </Providers>
      </body>
    </html>
  );
}
