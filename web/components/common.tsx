"use client";

// 各页面共用的小部件，全部基于 shadcn/ui 组件。

import { useEffect, type ReactNode } from "react";
import { ChevronRightIcon } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";

export type Tone = "good" | "warn" | "bad" | "info" | "neutral";

/** 业务状态的颜色（shadcn 的 Badge、Alert 只有中性与 destructive 两种色调）。用工具类而不是自定义类，
 *  这样 cn（tailwind-merge）能盖掉组件自带的 bg-card、bg-secondary。 */
export const TONE: Record<Tone, string> = {
  good: "border-transparent bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  warn: "border-transparent bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  bad: "border-transparent bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
  info: "border-transparent bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300",
  neutral: "",
};

/** 带业务色调的 Badge：good 已验证 / 已完成，warn 未验证 / 进行中，bad 拒答 / 失败。 */
export function ToneBadge({ tone = "neutral", className, children, title }: {
  tone?: Tone; className?: string; children: ReactNode; title?: string;
}) {
  return (
    <Badge variant={tone === "neutral" ? "outline" : "secondary"} title={title}
      className={cn(TONE[tone], className)}>
      {children}
    </Badge>
  );
}

export function PageTitle({ title, description, children }: { title: ReactNode; description?: ReactNode; children?: ReactNode }) {
  return (
    <div className="mb-4 flex flex-wrap items-end justify-between gap-3">
      <div>
        <h1 className="font-heading text-xl font-semibold">{title}</h1>
        {description && <p className="mt-1 text-sm text-muted-foreground">{description}</p>}
      </div>
      {children && <div className="flex flex-wrap items-center gap-2">{children}</div>}
    </div>
  );
}

export function Field({ label, htmlFor, hint, className, children }: {
  label: ReactNode; htmlFor?: string; hint?: ReactNode; className?: string; children: ReactNode;
}) {
  return (
    <div className={cn("grid gap-1.5", className)}>
      <Label htmlFor={htmlFor} className="text-muted-foreground">{label}</Label>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  );
}

export function Mono({ children, className, title }: { children: ReactNode; className?: string; title?: string }) {
  return <span className={cn("font-mono text-xs", className)} title={title}>{children}</span>;
}

export function Code({ children }: { children: ReactNode }) {
  return <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">{children}</code>;
}

/** 可展开的 JSON（入库提示、更正等）。 */
export function JsonCollapsible({ label = "查看", value }: { label?: string; value: unknown }) {
  return (
    <Collapsible>
      <CollapsibleTrigger className="group flex items-center gap-1 text-xs text-primary hover:underline">
        <ChevronRightIcon className="size-3 transition-transform group-data-[state=open]:rotate-90" />
        {label}
      </CollapsibleTrigger>
      <CollapsibleContent>
        <pre className="mt-1 max-w-md overflow-x-auto rounded-md bg-muted p-2 text-xs whitespace-pre-wrap break-all">
          {JSON.stringify(value, null, 2)}
        </pre>
      </CollapsibleContent>
    </Collapsible>
  );
}

/** 跟随系统的深色模式：shadcn 的主题变量挂在 html.dark 上。 */
export function ThemeSync() {
  useEffect(() => {
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const apply = () => document.documentElement.classList.toggle("dark", mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);
  return null;
}
