import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  // Each route exports as a folder (out/chat/index.html), which is what the
  // backend's StaticFiles(html=True) serves for /chat/; out/chat.html it would not.
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
