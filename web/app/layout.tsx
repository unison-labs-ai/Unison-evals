import type { Metadata } from "next";
import type { ReactNode } from "react";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "unison-evals",
  description: "Public benchmark harness for Unison and comparable systems.",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <header className="border-b" style={{ borderColor: "var(--border)" }}>
          <div className="mx-auto max-w-5xl px-4 py-3 flex items-center justify-between">
            <Link href="/" className="font-mono text-sm tracking-tight">
              unison-evals
            </Link>
            <nav className="flex gap-4 text-sm">
              <Link href="/runs/new" className="hover:opacity-70">
                New run
              </Link>
              <Link href="/" className="hover:opacity-70">
                History
              </Link>
              <Link href="/?tab=leaderboard" className="hover:opacity-70">
                Leaderboard
              </Link>
              <Link href="/runs/compare" className="hover:opacity-70">
                Compare
              </Link>
              <a
                href="https://github.com/Unison-Workspace/Unison-evals"
                target="_blank"
                rel="noreferrer"
                className="hover:opacity-70"
              >
                GitHub
              </a>
            </nav>
          </div>
        </header>
        <main className="mx-auto max-w-5xl px-4 py-8">{children}</main>
      </body>
    </html>
  );
}
