"use client";

// 对话中的一问一答：用户的问题（右侧气泡）与 EAER 的回答（思考过程、回答、结论与引用）。

import { Fragment } from "react";
import { ArrowRightIcon, BookOpenIcon, SquareIcon } from "lucide-react";
import type { ApiResult, StreamEvent } from "@/lib/api";
import { useIdentity } from "@/lib/identity";
import { useEvidence } from "@/lib/useEvidence";
import { formatLocator, formatTime, GRADE_SHORT, MODE_LABEL, TIME_SOURCE_LABEL } from "@/lib/format";
import type { Citation, Claim, Mode, Progress, ReasoningStep, SessionItem, TimeBasis } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { ApiError } from "@/components/ApiError";
import { ToneBadge } from "@/components/common";
import { Reasoning, type ReasoningState } from "./Reasoning";
import type { SourceView } from "./SourceDrawer";

export function UserBubble({ text, at }: { text: string; at?: string }) {
  return (
    <div className="flex flex-col items-end gap-1">
      <div className="max-w-[85%] rounded-2xl rounded-br-md bg-primary px-4 py-2 text-sm whitespace-pre-wrap text-primary-foreground">
        {text}
      </div>
      {at && <span className="text-[11px] text-muted-foreground">{formatTime(at)}</span>}
    </div>
  );
}

function Avatar() {
  return (
    <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-foreground text-[11px] font-bold text-background">E</div>
  );
}

export function ModePath({ path }: { path: Mode[] }) {
  if (!path?.length) return null;
  return (
    <span className="inline-flex items-center gap-1">
      {path.map((m, i) => (
        <Fragment key={i}>
          {i > 0 && <ArrowRightIcon className="size-3 text-muted-foreground" />}
          <Badge variant={i === path.length - 1 ? "default" : "outline"} className="h-4 px-1.5 text-[10px]">{MODE_LABEL[m]}</Badge>
        </Fragment>
      ))}
    </span>
  );
}

// ---------- 思考过程的文字 ----------

function timeLines(tb: TimeBasis | Record<string, never> | null | undefined): string[] {
  if (!tb || !("source" in tb)) return [];
  const parts = [`时间点：${TIME_SOURCE_LABEL[tb.source] ?? tb.source}`];
  if (tb.as_of_valid) parts.push(`按 ${tb.as_of_valid} 实际生效的版本`);
  if (tb.as_of_system) parts.push(`回溯到截至 ${tb.as_of_system} 系统所知的记录`);
  const out = [parts.join("，")];
  if (tb.unresolved?.length) out.push(`未能换算成日期的表述（保留原文）：${tb.unresolved.join("、")}`);
  return out;
}

export function progressLines(p: Progress | undefined): string[] {
  if (!p) return ["已转入后台执行（超过同步时限），等待进度…"];
  const out = ["已转入后台执行（超过同步时限）"];
  if (p.mode_path.length) out.push(`模式：${p.mode_path.map((m) => MODE_LABEL[m]).join(" → ")}`);
  out.push(`已完成：${p.done.length ? p.done.join(" → ") : "（尚无）"}`);
  if (p.next) out.push(`下一步：${p.next}`);
  out.push(`检索轮数：${p.retrieval_rounds}`);
  if (p.waiting) out.push(`等待重试：${p.waiting}${p.retry_after ? `（最早 ${formatTime(p.retry_after)}）` : ""}`);
  return out;
}

export function streamLines(events: StreamEvent[]): string[] {
  const out: string[] = [];
  for (const e of events) {
    if (e.event === "mode") out.push(`进入 ${MODE_LABEL[(e.data as { mode: Mode }).mode]} 模式`);
    if (e.event === "notice") out.push(`说明：${(e.data as { text: string }).text}`);
  }
  return out;
}

function doneLines(it: SessionItem, notices: string[] | undefined): string[] {
  const out: string[] = [];
  if (it.mode_path.length) out.push(`经过的模式：${it.mode_path.map((m) => MODE_LABEL[m]).join(" → ")}${it.background ? "（曾转入后台）" : ""}`);
  out.push(...timeLines(it.time_basis));
  for (const n of notices ?? []) out.push(`说明：${n}`);
  if (it.abstained) out.push("没有足够的证据，拒答（证据不足、无权或已归档时不编造）");
  else if (it.claims.length) {
    const ok = it.claims.filter((c) => c.status === "supported").length;
    const docs = new Set(it.claims.flatMap((c) => c.citations.map((x) => x.version_id))).size;
    const evs = new Set(it.claims.flatMap((c) => c.citations.map((x) => x.evidence_id))).size;
    out.push(`核对：${ok} 条结论已验证，${it.claims.length - ok} 条未验证；引用 ${docs} 个版本的 ${evs} 处证据`);
  }
  if (!it.reasoning?.length) out.push("模型没有给出思考内容（生成服务未开启思考时没有）");
  return out;
}

// ---------- 结论与引用 ----------

interface Ref { n: number; c: Citation }

function numberRefs(claims: Claim[]): { refs: Ref[]; num: Map<string, number> } {
  const num = new Map<string, number>();
  const refs: Ref[] = [];
  for (const cl of claims) {
    for (const c of cl.citations) {
      if (!num.has(c.evidence_id)) {
        num.set(c.evidence_id, refs.length + 1);
        refs.push({ n: refs.length + 1, c });
      }
    }
  }
  return { refs, num };
}

const fullText = (c: Citation): SourceView => ({ kind: "version", id: c.version_id, focus: c.evidence_id });

function RefRow({ r, onOpen }: { r: Ref; onOpen: (v: SourceView) => void }) {
  const { active } = useIdentity();
  const ev = useEvidence(active?.token ?? null, r.c.evidence_id);
  const e = ev?.ok ? ev.data : null;
  return (
    <li id={`ref-${r.c.evidence_id}`} className="grid gap-0.5 rounded-md border bg-muted/30 px-2.5 py-1.5 text-xs">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
        <span className="font-mono text-muted-foreground">[{r.n}]</span>
        <span className="font-medium">{e ? `《${e.document.title}》` : "文档"}第 {r.c.version_no ?? "?"} 版</span>
        <span className="text-muted-foreground">{formatLocator(r.c.locator)}</span>
        <ToneBadge tone={r.c.valid_grade === "explicit" ? "good" : r.c.valid_grade ? "warn" : "neutral"} className="h-4 text-[10px]">
          有效期来源：{r.c.valid_grade ? GRADE_SHORT[r.c.valid_grade] : "无（记录类）"}
        </ToneBadge>
        {e?.status === "searchable" && (
          <span className="text-muted-foreground">文档日期 {e.version.doc_date ?? "无"} · 出处 {e.version.publisher ?? "无"}</span>
        )}
        <span className="ml-auto flex gap-2">
          <button type="button" className="cursor-pointer text-primary hover:underline"
            onClick={() => onOpen({ kind: "evidence", id: r.c.evidence_id })}>证据</button>
          {e?.status !== "archived" && (
            <button type="button" className="inline-flex cursor-pointer items-center gap-0.5 text-primary hover:underline"
              onClick={() => onOpen(fullText(r.c))}>
              <BookOpenIcon className="size-3" />全文
            </button>
          )}
        </span>
      </div>
      {ev === null && <span className="flex items-center gap-1 text-muted-foreground"><Spinner className="size-3" />读取证据…</span>}
      {e?.status === "searchable" && (
        <blockquote className="border-l-2 pl-2 text-muted-foreground whitespace-pre-wrap">
          {e.content.length > 200 ? e.content.slice(0, 200) + "…" : e.content}
        </blockquote>
      )}
      {e?.status === "archived" && <span className="text-amber-700 dark:text-amber-300">{e.notice}（不返回原文）</span>}
      {ev && !ev.ok && (
        <span className="text-destructive">证据接口 {ev.status}{ev.status === 404 ? "：不存在或无权查看" : ev.error ? `：${ev.error.message}` : ""}</span>
      )}
    </li>
  );
}

function Claims({ claims, onOpen }: { claims: Claim[]; onOpen: (v: SourceView) => void }) {
  const { refs, num } = numberRefs(claims);
  return (
    <div className="grid gap-2">
      <ul className="grid gap-1.5">
        {claims.map((c, i) => (
          <li key={i} className="flex flex-wrap items-start gap-1.5 text-sm">
            <ToneBadge tone={c.status === "supported" ? "good" : "warn"} className="mt-0.5">{c.status === "supported" ? "已验证" : "未验证"}</ToneBadge>
            {c.core && <Badge variant="outline" className="mt-0.5">核心</Badge>}
            <span className="min-w-0 flex-1">
              {c.text}
              {c.citations.map((x) => (
                <button key={x.evidence_id} type="button" title="在全文中查看这一处" onClick={() => onOpen(fullText(x))}
                  className="ml-1 inline-flex h-4 min-w-4 cursor-pointer items-center justify-center rounded bg-muted px-1 align-text-top font-mono text-[10px] text-muted-foreground hover:bg-primary hover:text-primary-foreground">
                  {num.get(x.evidence_id)}
                </button>
              ))}
            </span>
          </li>
        ))}
      </ul>
      {refs.length > 0 && (
        <div className="grid gap-1">
          <div className="text-xs font-medium text-muted-foreground">引用（{refs.length}）</div>
          <ol className="grid gap-1">{refs.map((r) => <RefRow key={r.c.evidence_id} r={r} onOpen={onOpen} />)}</ol>
        </div>
      )}
    </div>
  );
}

// ---------- 回答 ----------

export interface TurnExtra {
  notices?: string[];
  durationMs?: number;
  error?: string;
}

/** 会话里已有记录的一次回答（已完成、后台进行中、已取消、失败）。 */
export function AssistantTurn({ it, extra, onCancel, cancelling, onOpen }: {
  it: SessionItem; extra?: TurnExtra; onCancel?: () => void; cancelling?: boolean;
  /** 在抽屉里打开证据或全文 */
  onOpen: (v: SourceView) => void;
}) {
  const state: ReasoningState = it.status === "running" ? "thinking" : it.status === "completed" ? "done"
    : it.status === "cancelled" ? "cancelled" : "error";
  const lines = it.status === "running" ? progressLines(it.progress)
    : it.status === "completed" ? doneLines(it, extra?.notices)
    : it.status === "failed" ? [extra?.error ?? "执行失败（如模型服务调用最终失败），没有回答"] : ["查询已取消，没有回答"];
  return (
    <div className="flex gap-3">
      <Avatar />
      <div className="grid min-w-0 flex-1 gap-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <Reasoning state={state} lines={lines} startedAt={it.status === "running" ? Date.parse(it.asked_at) : null}
            durationMs={extra?.durationMs} thoughts={it.reasoning ?? []} />
          {it.status === "running" && onCancel && (
            <Button size="xs" variant="outline" onClick={onCancel} disabled={cancelling}>
              {cancelling ? <Spinner /> : <SquareIcon />}取消
            </Button>
          )}
        </div>
        {it.status === "completed" && (
          <>
            <div className="text-sm leading-relaxed whitespace-pre-wrap">{it.answer || <span className="text-muted-foreground">（无）</span>}</div>
            {it.abstained && <div><ToneBadge tone="bad">拒答</ToneBadge></div>}
            {it.claims.length > 0 && <Claims claims={it.claims} onOpen={onOpen} />}
            <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
              <ModePath path={it.mode_path} />
              <span>query {it.query_id.slice(0, 8)}</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

/** 正在等接口返回的一次提问：思考中；流式时边收边显示回答文字。 */
export function PendingTurn({ startedAt, events }: { startedAt: number; events: StreamEvent[] }) {
  const delta = events.filter((e) => e.event === "delta").map((e) => (e.data as { text: string }).text).join("");
  const lines = ["检索、生成与核对中（同步时限默认 30 秒，超过转入后台）", ...streamLines(events)];
  const thoughts = events.filter((e) => e.event === "reasoning").map((e) => e.data as ReasoningStep);
  return (
    <div className="flex gap-3">
      <Avatar />
      <div className="grid min-w-0 flex-1 gap-2.5">
        <Reasoning state="thinking" lines={lines} startedAt={startedAt} thoughts={thoughts} />
        {delta && <div className="text-sm leading-relaxed whitespace-pre-wrap">{delta}</div>}
      </div>
    </div>
  );
}

/** 没有记进会话的失败（无权、频率限制、服务不可用等）。 */
export function ErrorTurn({ result }: { result: ApiResult<unknown> }) {
  return (
    <div className="flex gap-3">
      <Avatar />
      <div className="min-w-0 flex-1"><ApiError result={result} title="提问失败" /></div>
    </div>
  );
}
