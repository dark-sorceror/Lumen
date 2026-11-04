import type { Metadata } from "next";
import "./globals.css";

// The chat's own root layout. The landing at / has another, so each page
// loads only its own stylesheet.

export const metadata: Metadata = {
  title: "Lumen",
  description: "Minimal chat frontend for Lumen.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
