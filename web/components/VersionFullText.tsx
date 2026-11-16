"use client";

// 版本全文（GET /v1/versions/{id}）：该版本的全部证据按页、正文段落、表格、图表排列，focus 为证据 id 时高亮并滚到那一处；
// 可下载原件（GET /v1/versions/{id}/original）。可见规则与证据接口相同；待审核的版本只对本租户的审核人开放。
// 独立页面（/versions/{id}）与对话里的抽屉共用。

import { Fragment, useEffect, useState } from "react";
import { DownloadIcon, FileTextIcon } from "lucide-react";
import { call, downloadOriginal, type ApiResult } from "@/lib/api";
import { useIdentity } from "@/lib/identity";
import { DOC_TYPE_LABEL, formatLocator, formatTime, GRADE_SHORT } from "@/lib/format";
import type { VersionEvidence, VersionView } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Spinner } from "@/components/ui/spinner";
import { ApiError } from "./ApiError";
import { Mono, ToneBadge } from "./common";
import { cn } from "@/lib/utils";

type Block =
  | { kind: "page"; page: number }
  | { kind: "heading"; text: string; depth: number }
  | { kind: "text"; ev: VersionEvidence }
  | { kind: "table"; table: string; rows: VersionEvidence[] }
  | { kind: "figure"; ev: VersionEvidence };

/** 把按顺序排好的证据排成可读的块：换页时加分页线，章节路径变化时加小标题，同一张表的单元格并成一块。 */
function toBlocks(list: VersionEvidence[]): Block[] {
  const out: Block[] = [];
  let page: unknown = undefined;
  let section = "";
  for (const ev of list) {
    const loc = ev.locator as Record<string, unknown>;
    if (loc.page !== undefined && loc.page !== page) {
      page = loc.page;
      out.push({ kind: "page", page: Number(loc.page) });
    }
    if (ev.modality === "text") {
      const sec = Array.isArray(loc.section) ? (loc.section as string[]) : [];
      const key = sec.join(" / ");
      if (key && key !== section) out.push({ kind: "heading", text: sec[sec.length - 1], depth: sec.length });
      section = key;
      out.push({ kind: "text", ev });
    } else if (ev.modality === "table") {
      const t = String(loc.table ?? "?");
      const last = out[out.length - 1];
      if (last?.kind === "table" && last.table === t) last.rows.push(ev);
      else out.push({ kind: "table", table: t, rows: [ev] });
    } else {
      out.push({ kind: "figure", ev });
    }
  }
  return out;
}

function Marked({ ev, focus, showIds, children, className }: {
  ev: VersionEvidence; focus: string | null; showIds: boolean; children: React.ReactNode; className?: string;
}) {
  const on = ev.evidence_id === focus;
  return (
    <div data-evidence={ev.evidence_id} title={`${ev.evidence_id} · ${formatLocator(ev.locator)}`}
      className={cn("scroll-mt-24 rounded-md px-2 py-1 transition-colors",
        on && "bg-amber-100 ring-2 ring-amber-400 dark:bg-amber-950 dark:ring-amber-600", className)}>
      {children}
      {(showIds || on) && (
        <div className="mt-0.5 text-[11px] text-muted-foreground">
          <Mono>{ev.evidence_id}</Mono> · {formatLocator(ev.locator)}{ev.parse_stage === "deep" ? " · 深度解析" : ""}
          {on && <Badge variant="outline" className="ml-2 h-4 text-[10px]">被引用的这一处</Badge>}
        </div>
      )}
    </div>
  );
}

export function VersionFullText({ versionId, focus = null, showTitle = true }: {
  versionId: string; focus?: string | null; showTitle?: boolean;
}) {
  const { active, ready } = useIdentity();
  const token = active?.token ?? null;
  const [r, setR] = useState<ApiResult<VersionView> | null>(null);
  const [showIds, setShowIds] = useState(false);
  const [dl, setDl] = useState<ApiResult<null> | null>(null);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    if (!ready) return;
    let alive = true;
    setR(null);
    setDl(null);
    call<VersionView>(token, "GET", `/v1/versions/${encodeURIComponent(versionId)}`).then((x) => { if (alive) setR(x); });
    return () => { alive = false; };
  }, [ready, token, versionId]);

  // 数据到了以后滚到被引用的那一处（在页面或抽屉里都按 data-evidence 找）
  useEffect(() => {
    if (!focus || !r?.ok) return;
    const t = setTimeout(() => document.querySelector(`[data-evidence="${focus}"]`)?.scrollIntoView({ block: "center", behavior: "smooth" }), 150);
    return () => clearTimeout(t);
  }, [focus, r]);

  if (!r) return <p className="flex items-center gap-2 text-sm text-muted-foreground"><Spinner />读取全文…</p>;
  if (!r.ok) return <ApiError result={r} title="读取全文失败" />;
  const v = r.data!;
  const title = `《${v.document.title}》第 ${v.version.version_no} 版`;
  if (v.status === "archived") {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><ToneBadge tone="warn">已归档</ToneBadge>{title}</CardTitle>
          <CardDescription>{v.notice}：已归档的资料只给提示，不返回原文与原件。</CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const blocks = toBlocks(v.evidence);
  const download = async () => {
    setDownloading(true);
    setDl(await downloadOriginal(token, v.version.id, v.original.filename));
    setDownloading(false);
  };

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          {showTitle && <h1 className="font-heading text-xl font-semibold">{title}</h1>}
          <p className="mt-1 text-sm text-muted-foreground">
            {v.access === "review"
              ? "待审核的版本：只有本租户的审核人能看到全文。表格与图表在审核通过后的深度解析里才抽取，审核时可下载原件对照。"
              : "该版本的全文（解析出的全部证据）。"}
          </p>
        </div>
        <Button variant="outline" size="sm" onClick={download} disabled={downloading}>
          {downloading ? <Spinner /> : <DownloadIcon />}下载原件
        </Button>
      </div>
      <ApiError result={dl} title="下载原件失败" />

      <Card size="sm">
        <CardContent>
          <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm sm:grid-cols-[max-content_1fr_max-content_1fr]">
            <dt className="text-muted-foreground">状态</dt>
            <dd>
              {v.status === "searchable" ? <ToneBadge tone="good">可搜索</ToneBadge> : <ToneBadge tone="warn">待审核</ToneBadge>}
              {v.access === "review" && <ToneBadge tone="info" className="ml-1">审核人查看</ToneBadge>}
            </dd>
            <dt className="text-muted-foreground">文档类型</dt>
            <dd>{v.document.doc_type ? DOC_TYPE_LABEL[v.document.doc_type] : <span className="text-muted-foreground">未确认</span>}</dd>
            <dt className="text-muted-foreground">上传时间</dt><dd>{formatTime(v.version.uploaded_at)}</dd>
            <dt className="text-muted-foreground">审核通过</dt><dd>{v.version.approved_at ? formatTime(v.version.approved_at) : <span className="text-muted-foreground">未通过</span>}</dd>
            <dt className="text-muted-foreground">文档日期</dt><dd>{v.version.doc_date ?? <span className="text-muted-foreground">无</span>}</dd>
            <dt className="text-muted-foreground">出处</dt><dd>{v.version.publisher ?? <span className="text-muted-foreground">无</span>}</dd>
            <dt className="text-muted-foreground">有效期</dt>
            <dd>
              {v.validity
                ? <>{v.validity.valid_from ?? "—"}（{GRADE_SHORT[v.validity.valid_from_grade]}）~ {v.validity.valid_to ?? "至今"}{v.validity.valid_to && `（${GRADE_SHORT[v.validity.valid_to_grade]}）`}</>
                : <span className="text-muted-foreground">{v.status === "pending_review" ? "待审核确认" : "无"}</span>}
            </dd>
            <dt className="text-muted-foreground">原件</dt>
            <dd className="flex items-center gap-1"><FileTextIcon className="size-3.5" />{v.original.filename}<span className="text-xs text-muted-foreground">（{v.original.content_type}）</span></dd>
          </dl>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>全文（{v.evidence.length} 条证据）</CardTitle>
          <CardDescription className="flex flex-wrap items-center gap-3">
            {focus && <span>高亮的是被引用的那一处。</span>}
            <span className="flex items-center gap-2">
              <Checkbox id={`show-ids-${versionId}`} checked={showIds} onCheckedChange={(x) => setShowIds(x === true)} />
              <Label htmlFor={`show-ids-${versionId}`} className="font-normal">显示证据 id 与定位</Label>
            </span>
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-1 text-sm leading-7">
          {blocks.length === 0 && <p className="text-muted-foreground">没有证据（解析尚未完成，或文档里没有可解析的内容）。</p>}
          {blocks.map((b, i) => (
            <Fragment key={i}>
              {b.kind === "page" && (
                <div className="my-2 flex items-center gap-2 text-xs text-muted-foreground"><span className="h-px flex-1 bg-border" />第 {b.page} 页<span className="h-px flex-1 bg-border" /></div>
              )}
              {b.kind === "heading" && (
                <h3 className={cn("mt-3 font-heading font-semibold", b.depth <= 1 ? "text-base" : "text-sm")}>{b.text}</h3>
              )}
              {b.kind === "text" && (
                <Marked ev={b.ev} focus={focus} showIds={showIds}><p className="whitespace-pre-wrap">{b.ev.content}</p></Marked>
              )}
              {b.kind === "table" && (
                <div className="my-2 rounded-lg border p-2">
                  <div className="mb-1 text-xs font-medium text-muted-foreground">表 {b.table}（{b.rows.length} 个单元格）</div>
                  {b.rows.map((ev) => (
                    <Marked key={ev.evidence_id} ev={ev} focus={focus} showIds={showIds} className="text-xs leading-6">{ev.content}</Marked>
                  ))}
                </div>
              )}
              {b.kind === "figure" && (
                <div className="my-2 rounded-lg border border-dashed p-2">
                  <Marked ev={b.ev} focus={focus} showIds={showIds}>
                    <div className="text-xs font-medium text-muted-foreground">图 {String((b.ev.locator as Record<string, unknown>).figure ?? "")}（看图转写）</div>
                    <p className="whitespace-pre-wrap">{b.ev.content}</p>
                  </Marked>
                </div>
              )}
            </Fragment>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}
