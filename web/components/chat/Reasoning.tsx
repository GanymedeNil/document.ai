"use client";

// Reasoning 样式（shadcn AI Elements 的 Reasoning，基于 Trace）：
// - 模型自身的思考（生成服务开启思考时，接口按步骤返回 reasoning）：每个步骤的每次思考各一行（文档：一条消息里有多段思考时
//   各自成行、放在同一个 Trace 里），点开看思考文字（按 Markdown 显示）；
// - 最后一行是总的状态：思考时“思考中”并展开，结束后变成“已思考 · 用时 N 秒”并收起，里面是 EAER 的执行过程
//   （经过的模式、后台执行的步骤、时间点怎么定的、核对结果）。用户自己展开或收起过，以用户的为准。

import { useEffect, useRef, useState } from "react";
import { BrainIcon, CircleAlertIcon, CircleCheckIcon, CircleSlashIcon } from "lucide-react";
import type { ReasoningStep } from "@/lib/types";
import { Markdown } from "@/components/Markdown";
import {
  Trace, TraceCollapsible, TraceCollapsibleContent, TraceCollapsibleTrigger, TraceDescription, TraceIcon, TraceLine,
  TraceTitle,
} from "@/components/ui/trace";

export type ReasoningState = "thinking" | "done" | "cancelled" | "error";

const TITLE: Record<ReasoningState, string> = { thinking: "思考中", done: "已思考", cancelled: "已取消", error: "执行失败" };

export const STEP_LABEL: Record<ReasoningStep["step"], string> = {
  prepare: "时间点与路由",
  first: "检索与生成",
  escalate: "升级判断（Deep）",
  verify: "验证",
  assemble: "组装回答",
};

function seconds(ms: number): string {
  return `${Math.max(1, Math.round(ms / 1000))} 秒`;
}

/** 一段可展开的行：follow 为 true 时跟着它开合（思考时展开、结束时收起），用户点过就以用户的为准。 */
function Collapsible({ follow, line, children, contentClassName }: {
  follow: boolean; line: React.ReactNode; children: React.ReactNode; contentClassName?: string;
}) {
  const [open, setOpen] = useState(follow);
  const userChose = useRef(false);
  useEffect(() => {
    if (!userChose.current) setOpen(follow);
  }, [follow]);
  return (
    <TraceCollapsible open={open} onOpenChange={(o) => { userChose.current = true; setOpen(o); }}>
      <TraceCollapsibleTrigger>{line}</TraceCollapsibleTrigger>
      <TraceCollapsibleContent className={contentClassName}>{children}</TraceCollapsibleContent>
    </TraceCollapsible>
  );
}

export function Reasoning({ state, lines, startedAt, durationMs, thoughts = [] }: {
  state: ReasoningState;
  /** EAER 的执行过程，放在最后一行里 */
  lines: string[];
  /** 思考开始的时刻（毫秒），思考中时用来显示已用时间 */
  startedAt?: number | null;
  /** 思考用时；不知道时不显示 */
  durationMs?: number | null;
  /** 模型自身的思考，按步骤 */
  thoughts?: ReasoningStep[];
}) {
  const thinking = state === "thinking";
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!thinking || !startedAt) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [thinking, startedAt]);

  const icon = thinking ? <BrainIcon /> : state === "done" ? <CircleCheckIcon /> : state === "cancelled" ? <CircleSlashIcon /> : <CircleAlertIcon />;
  const description = thinking
    ? (startedAt ? seconds(now - startedAt) : null)
    : durationMs ? `用时 ${seconds(durationMs)}` : null;
  const summary = (
    <TraceLine status={thinking ? "running" : state === "error" ? "error" : undefined}>
      <TraceIcon>{icon}</TraceIcon>
      <TraceTitle>{TITLE[state]}</TraceTitle>
      {description && <TraceDescription>{description}</TraceDescription>}
    </TraceLine>
  );

  return (
    <Trace>
      {thoughts.map((t, i) => (
        <Collapsible key={i} follow={thinking} contentClassName="max-h-96 overflow-y-auto text-xs"
          line={
            <TraceLine>
              <TraceIcon><BrainIcon /></TraceIcon>
              <TraceTitle>模型思考 · {STEP_LABEL[t.step] ?? t.step}</TraceTitle>
              <TraceDescription>{t.text.length} 字</TraceDescription>
            </TraceLine>
          }>
          <Markdown>{t.text}</Markdown>
        </Collapsible>
      ))}
      {lines.length === 0 ? summary : (
        <Collapsible follow={thinking} line={summary} contentClassName="text-xs whitespace-pre-wrap">
          {lines.join("\n")}
        </Collapsible>
      )}
    </Trace>
  );
}
