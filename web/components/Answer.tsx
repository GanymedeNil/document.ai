"use client";

// 回答的展示：时间依据、说明、回答正文、每条结论的状态与引用的证据。
// 用于提问页的即时结果，也用于会话历史的每一条。

import Link from "next/link";
import { Fragment } from "react";
import { ArrowRightIcon } from "lucide-react";
import { useIdentity } from "@/lib/identity";
import { useEvidence } from "@/lib/useEvidence";
import { DOC_TYPE_LABEL, formatLocator, GRADE_SHORT, MODE_LABEL, TIME_SOURCE_LABEL } from "@/lib/format";
import type { Citation, Claim, Mode, TimeBasis } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Separator } from "@/components/ui/separator";
import { Spinner } from "@/components/ui/spinner";
import { Code, Mono, ToneBadge } from "./common";

export function ModePath({ path, mode }: { path: Mode[]; mode?: Mode | null }) {
  if (!path?.length) return <span className="text-muted-foreground">—</span>;
  return (
    <span className="inline-flex items-center gap-1">
      {path.map((m, i) => (
        <Fragment key={i}>
          {i > 0 && <ArrowRightIcon className="size-3 text-muted-foreground" />}
          <Badge variant={mode === m && i === path.length - 1 ? "default" : "outline"}>{MODE_LABEL[m]}</Badge>
        </Fragment>
      ))}
    </span>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="grid gap-1.5">
      <h4 className="text-xs font-medium text-muted-foreground">{title}</h4>
      {children}
    </div>
  );
}

export function TimeBasisView({ tb }: { tb: TimeBasis | Record<string, never> | null | undefined }) {
  if (!tb || !("source" in tb)) return <span className="text-sm text-muted-foreground">—</span>;
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-0.5 text-sm">
      <dt className="text-muted-foreground">依据来源</dt>
      <dd>{TIME_SOURCE_LABEL[tb.source] ?? tb.source}</dd>
      <dt className="text-muted-foreground">实际生效日</dt>
      <dd>{tb.as_of_valid ?? <span className="text-muted-foreground">未指定</span>}</dd>
      <dt className="text-muted-foreground">系统所知截至</dt>
      <dd>{tb.as_of_system ?? <span className="text-muted-foreground">未指定（不回溯）</span>}</dd>
      {tb.unresolved && tb.unresolved.length > 0 && (
        <>
          <dt className="text-muted-foreground">未能换算</dt>
          <dd>{tb.unresolved.map((u) => <Code key={u}>{u}</Code>)}（保留原文，不给日期）</dd>
        </>
      )}
    </dl>
  );
}

function CitationRow({ c }: { c: Citation }) {
  const { active } = useIdentity();
  const ev = useEvidence(active?.token ?? null, c.evidence_id);
  const e = ev?.ok ? ev.data : null;
  return (
    <li className="grid gap-1 rounded-md border bg-muted/30 p-2.5 text-sm">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <Link href={`/evidence/${c.evidence_id}`} className="font-mono text-xs text-primary hover:underline">{c.evidence_id}</Link>
        {e ? <span className="font-medium">《{e.document.title}》</span> : <Mono className="text-muted-foreground">文档 {c.document_id.slice(0, 8)}</Mono>}
        <span>第 {c.version_no ?? "?"} 版</span>
        <span className="text-muted-foreground">定位：{formatLocator(c.locator)}</span>
        <ToneBadge tone={c.valid_grade === "explicit" ? "good" : c.valid_grade ? "warn" : "neutral"}>
          有效期来源：{c.valid_grade ? GRADE_SHORT[c.valid_grade] : "无（记录类）"}
        </ToneBadge>
      </div>
      {ev === null && <span className="flex items-center gap-1 text-xs text-muted-foreground"><Spinner className="size-3" />读取证据…</span>}
      {e?.status === "searchable" && (
        <>
          <div className="flex flex-wrap gap-x-3 text-xs text-muted-foreground">
            <span>文档日期：{e.version.doc_date ?? "无"}</span>
            <span>出处：{e.version.publisher ?? "无"}</span>
            {e.document.doc_type && <span>{DOC_TYPE_LABEL[e.document.doc_type]}</span>}
            {e.validity && <span>有效期 {e.validity.valid_from ?? "?"} ~ {e.validity.valid_to ?? "至今"}</span>}
          </div>
          <blockquote className="border-l-2 pl-2 text-xs text-muted-foreground whitespace-pre-wrap">
            {e.content.length > 160 ? e.content.slice(0, 160) + "…" : e.content}
          </blockquote>
        </>
      )}
      {e?.status === "archived" && <div><ToneBadge tone="warn">{e.notice}（不返回原文）</ToneBadge></div>}
      {ev && !ev.ok && (
        <div>
          <ToneBadge tone="bad">
            证据接口 {ev.status}{ev.status === 404 ? "：不存在或无权查看" : ev.error ? `：${ev.error.message}` : ""}
          </ToneBadge>
        </div>
      )}
    </li>
  );
}

export function ClaimList({ claims }: { claims: Claim[] }) {
  if (!claims.length) {
    return <p className="text-sm text-muted-foreground">没有结论（拒答时不给结论；被判为不支持的结论不返回）。</p>;
  }
  return (
    <ol className="grid gap-3">
      {claims.map((c, i) => (
        <li key={i} className="grid gap-2">
          <div className="flex flex-wrap items-start gap-2 text-sm">
            <span className="text-muted-foreground">{i + 1}.</span>
            <ToneBadge tone={c.status === "supported" ? "good" : "warn"}>{c.status === "supported" ? "已验证" : "未验证"}</ToneBadge>
            <Badge variant="outline">{c.core ? "核心结论" : "补充"}</Badge>
            <span className="flex-1">{c.text}</span>
          </div>
          {c.citations.length > 0 ? (
            <ul className="ml-5 grid gap-1.5">
              {c.citations.map((ct) => <CitationRow key={ct.evidence_id + ct.version_id} c={ct} />)}
            </ul>
          ) : (
            <p className="ml-5 text-xs text-muted-foreground">没有引用</p>
          )}
        </li>
      ))}
    </ol>
  );
}

export function AnswerBody(props: {
  answer: string | null;
  abstained: boolean | null;
  claims: Claim[];
  notices?: string[];
  timeBasis: TimeBasis | Record<string, never> | null;
  mode: Mode | null;
  modePath: Mode[];
}) {
  const { answer, abstained, claims, notices, timeBasis, mode, modePath } = props;
  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span className="text-muted-foreground">模式</span>
        <ModePath path={modePath} mode={mode} />
        {abstained && <ToneBadge tone="bad">拒答（证据不足、无权或已归档时不编造）</ToneBadge>}
      </div>
      <div className="grid gap-4 md:grid-cols-2">
        <Section title="时间依据"><TimeBasisView tb={timeBasis} /></Section>
        {notices && notices.length > 0 && (
          <Section title="说明">
            <ul className="list-disc pl-5 text-sm">{notices.map((n, i) => <li key={i}>{n}</li>)}</ul>
          </Section>
        )}
      </div>
      <Section title="回答">
        <div className="rounded-md bg-muted px-3 py-2 text-sm whitespace-pre-wrap">
          {answer || <span className="text-muted-foreground">（无）</span>}
        </div>
      </Section>
      <Separator />
      <Section title={`结论与证据（${claims.length}）`}>
        <ClaimList claims={claims} />
      </Section>
    </div>
  );
}
