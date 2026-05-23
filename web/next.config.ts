import type { NextConfig } from "next";

const config: NextConfig = {
  reactStrictMode: true,
  // Proxy /api/* to the FastAPI server during dev so the browser can hit
  // it without CORS gymnastics.
  async rewrites() {
    const target = process.env.UNISON_EVALS_API ?? "http://localhost:8001";
    return [{ source: "/api/:path*", destination: `${target}/api/:path*` }];
  },
};

export default config;
