import type { Metadata } from "next";
import type { ReactNode } from "react";
import { Nav } from "@/components/Nav";
import { ThemeSync } from "@/components/common";
import { TooltipProvider } from "@/components/ui/tooltip";
import { IdentityProvider } from "@/lib/identity";
import "./globals.css";

export const metadata: Metadata = {
  title: "EAER 演示",
  description: "EAER 的简易 Web UI：提问、会话、上传、审核、归档、证据与强制验证清单",
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="zh-CN" suppressHydrationWarning>
      <body className="min-h-screen bg-muted/30 antialiased">
        <ThemeSync />
        <TooltipProvider>
          <IdentityProvider>
            <Nav />
            <main className="mx-auto max-w-6xl px-4 py-6">{children}</main>
          </IdentityProvider>
        </TooltipProvider>
      </body>
    </html>
  );
}
