import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";

// The landing's own root layout; the chat at /chat/ has another. Sora draws
// the wordmark and the headings. It ships with the page, so nothing is
// fetched at build or run time.
const sora = localFont({
  src: "./fonts/Sora-Variable.woff2",
  weight: "100 800",
  variable: "--font-sora",
  display: "swap",
});

export const metadata: Metadata = {
  title: "illumen",
  description: "See what your AI sees: how full its context is, what it forgot, and what each answer drew on.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={sora.variable}>
      <body>{children}</body>
    </html>
  );
}
