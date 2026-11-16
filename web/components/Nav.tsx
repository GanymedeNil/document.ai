"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useIdentity } from "@/lib/identity";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { ToneBadge } from "./common";

const LINKS: [string, string][] = [
  ["/", "提问"],
  ["/sessions", "会话"],
  ["/upload", "上传"],
  ["/review", "审核"],
  ["/documents", "文档"],
  ["/evidence", "证据"],
  ["/config", "强制验证"],
  ["/identities", "身份"],
];

export function Nav() {
  const path = usePathname();
  const { ready, identities, active, setActive } = useIdentity();
  return (
    <header className="sticky top-0 z-20 border-b bg-background/95 backdrop-blur">
      <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-2 px-4 py-2">
        <Link href="/" className="mr-2 font-heading text-base font-bold">EAER</Link>
        <nav className="flex flex-wrap gap-1">
          {LINKS.map(([href, label]) => {
            const on = href === "/" ? path === "/" : path.startsWith(href);
            return (
              <Button key={href} asChild size="sm" variant={on ? "secondary" : "ghost"}>
                <Link href={href}>{label}</Link>
              </Button>
            );
          })}
        </nav>
        <div className="ml-auto flex items-center gap-2 text-sm">
          {ready && identities.length === 0 && (
            <Link href="/identities"><ToneBadge tone="warn">未设置身份</ToneBadge></Link>
          )}
          {identities.length > 0 && (
            <>
              <span className="text-muted-foreground">当前身份</span>
              <Select value={active?.id ?? ""} onValueChange={setActive}>
                <SelectTrigger size="sm" className="min-w-56" aria-label="切换身份">
                  <SelectValue placeholder="选择身份" />
                </SelectTrigger>
                <SelectContent align="end">
                  {identities.map((i) => (
                    <SelectItem key={i.id} value={i.id}>
                      {i.label}{i.roles?.length ? `（${i.roles.join("、")}）` : ""}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </>
          )}
        </div>
      </div>
    </header>
  );
}
