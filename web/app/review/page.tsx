"use client";

// 审核（需要 reviewer 角色）：GET /v1/review/tasks 列出任务（待审核、已通过、已删除），
// POST /v1/review/decisions 逐份或批量通过、删除（驳回），通过时可补录更正。每份单独成事务，逐份返回结果。

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Fragment, Suspense, useCallback, useEffect, useState } from "react";
import { BookOpenIcon, CheckIcon, PencilIcon, PlusIcon, RefreshCwIcon, Trash2Icon, XIcon } from "lucide-react";
import { call, type ApiResult } from "@/lib/api";
import { useIdentity } from "@/lib/identity";
import { DOC_TYPE_LABEL, EXTRACTION_LABEL, formatTime, GRADE_SHORT, TASK_STATUS_LABEL } from "@/lib/format";
import type {
  Corrections, Decision, DecisionResult, DocType, RelationInput, ReviewTask, ReviewTaskPage, ValidGrade,
} from "@/lib/types";
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter,
  AlertDialogHeader, AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ApiError } from "@/components/ApiError";
import { Code, Field, JsonCollapsible, Mono, PageTitle, ToneBadge, TONE } from "@/components/common";

type Status = "open" | "approved" | "deleted";
type DateMode = "keep" | "set" | "clear";

interface DateDraft { mode: DateMode; value: string; grade: ValidGrade }
interface RelDraft { relation: NonNullable<RelationInput["relation"]>; to_document_id: string; to_version_no: string }
interface Draft {
  doc_type: "" | DocType;
  from: DateDraft;
  to: DateDraft;
  relMode: "keep" | "set";
  relations: RelDraft[];
}

const emptyDraft = (): Draft => ({
  doc_type: "",
  from: { mode: "keep", value: "", grade: "explicit" },
  to: { mode: "keep", value: "", grade: "explicit" },
  relMode: "keep",
  relations: [],
});

/** 只带出需要更正的字段；什么都不改时返回 undefined。 */
function toCorrections(d: Draft | undefined): Corrections | undefined {
  if (!d) return undefined;
  const c: Corrections = {};
  if (d.doc_type) c.doc_type = d.doc_type;
  if (d.from.mode === "set" && d.from.value) c.valid_from = { value: d.from.value, grade: d.from.grade };
  if (d.from.mode === "clear") c.valid_from = null;
  if (d.to.mode === "set" && d.to.value) c.valid_to = { value: d.to.value, grade: d.to.grade };
  if (d.to.mode === "clear") c.valid_to = null;
  if (d.relMode === "set") {
    c.relations = d.relations
      .filter((r) => r.to_document_id.trim())
      .map((r) => ({ relation: r.relation, to_document_id: r.to_document_id.trim(), to_version_no: Number(r.to_version_no) || 1 }));
  }
  return Object.keys(c).length ? c : undefined;
}

function str(v: unknown): string | null {
  return typeof v === "string" && v ? v : null;
}

function DraftValidity({ d }: { d: Record<string, unknown> | null }) {
  if (!d) return <span className="text-muted-foreground">—</span>;
  const from = str(d.valid_from);
  const to = str(d.valid_to);
  const fg = str(d.valid_from_grade) as ValidGrade | null;
  const tg = str(d.valid_to_grade) as ValidGrade | null;
  const unresolved = str(d.unresolved_text);
  return (
    <div className="grid gap-0.5 text-xs">
      <span>生效：{from ?? "—"}{fg && <span className="text-muted-foreground">（{GRADE_SHORT[fg] ?? fg}）</span>}</span>
      <span>作废：{to ?? "—"}{tg && <span className="text-muted-foreground">（{GRADE_SHORT[tg] ?? tg}）</span>}</span>
      {unresolved && <span className="text-amber-700 dark:text-amber-300">未能换算：{unresolved}</span>}
    </div>
  );
}

function DateEditor({ id, label, v, onChange }: { id: string; label: string; v: DateDraft; onChange: (v: DateDraft) => void }) {
  return (
    <Field label={label} htmlFor={id}>
      <div className="flex flex-wrap gap-2">
        <NativeSelect id={id} value={v.mode} onChange={(e) => onChange({ ...v, mode: e.target.value as DateMode })}>
          <NativeSelectOption value="keep">不改</NativeSelectOption>
          <NativeSelectOption value="set">补录为</NativeSelectOption>
          <NativeSelectOption value="clear">清除（等级记为 unknown）</NativeSelectOption>
        </NativeSelect>
        {v.mode === "set" && (
          <>
            <Input type="date" className="w-40" aria-label={`${label}日期`} value={v.value} onChange={(e) => onChange({ ...v, value: e.target.value })} />
            <NativeSelect aria-label={`${label}等级`} value={v.grade} onChange={(e) => onChange({ ...v, grade: e.target.value as ValidGrade })}>
              <NativeSelectOption value="explicit">explicit（明确）</NativeSelectOption>
              <NativeSelectOption value="inferred">inferred（推断）</NativeSelectOption>
              <NativeSelectOption value="unknown">unknown（未知）</NativeSelectOption>
            </NativeSelect>
          </>
        )}
      </div>
    </Field>
  );
}

function CorrectionEditor({ taskId, d, onChange }: { taskId: string; d: Draft; onChange: (d: Draft) => void }) {
  const setRel = (i: number, r: Partial<RelDraft>) =>
    onChange({ ...d, relations: d.relations.map((x, j) => (j === i ? { ...x, ...r } : x)) });
  const c = toCorrections(d);
  return (
    <div className="grid gap-4 rounded-lg border bg-muted/40 p-4">
      <div className="grid gap-4 md:grid-cols-2">
        <Field label="文档类型" htmlFor={`${taskId}-doctype`}>
          <NativeSelect id={`${taskId}-doctype`} value={d.doc_type} onChange={(e) => onChange({ ...d, doc_type: e.target.value as "" | DocType })}>
            <NativeSelectOption value="">不改</NativeSelectOption>
            <NativeSelectOption value="normative">规范类（normative）</NativeSelectOption>
            <NativeSelectOption value="record">记录类（record）</NativeSelectOption>
          </NativeSelect>
        </Field>
        <Field label="与已发布版本的关系" htmlFor={`${taskId}-rel`}>
          <NativeSelect id={`${taskId}-rel`} value={d.relMode} onChange={(e) => onChange({ ...d, relMode: e.target.value as "keep" | "set" })}>
            <NativeSelectOption value="keep">不改（用识别出的关系）</NativeSelectOption>
            <NativeSelectOption value="set">替换为下面的列表（可为空）</NativeSelectOption>
          </NativeSelect>
        </Field>
        <DateEditor id={`${taskId}-from`} label="生效日 valid_from" v={d.from} onChange={(from) => onChange({ ...d, from })} />
        <DateEditor id={`${taskId}-to`} label="作废日 valid_to（不再有效的第一天）" v={d.to} onChange={(to) => onChange({ ...d, to })} />
      </div>
      {d.relMode === "set" && (
        <div className="grid gap-2">
          {d.relations.map((r, i) => (
            <div className="flex flex-wrap gap-2" key={i}>
              <NativeSelect aria-label="关系" value={r.relation} onChange={(e) => setRel(i, { relation: e.target.value as RelDraft["relation"] })}>
                <NativeSelectOption value="supersedes">supersedes（替代）</NativeSelectOption>
                <NativeSelectOption value="conflicts">conflicts（冲突）</NativeSelectOption>
                <NativeSelectOption value="references">references（引用）</NativeSelectOption>
              </NativeSelect>
              <Input placeholder="目标文档 id" className="min-w-72 flex-1 font-mono" value={r.to_document_id}
                onChange={(e) => setRel(i, { to_document_id: e.target.value })} />
              <Input placeholder="版本号" type="number" min={1} className="w-24" value={r.to_version_no}
                onChange={(e) => setRel(i, { to_version_no: e.target.value })} />
              <Button type="button" size="icon" variant="ghost" aria-label="移除关系"
                onClick={() => onChange({ ...d, relations: d.relations.filter((_, j) => j !== i) })}>
                <XIcon />
              </Button>
            </div>
          ))}
          <div>
            <Button type="button" size="sm" variant="outline" onClick={() =>
              onChange({ ...d, relations: [...d.relations, { relation: "supersedes", to_document_id: "", to_version_no: "1" }] })}>
              <PlusIcon />添加关系
            </Button>
          </div>
        </div>
      )}
      <p className="text-xs text-muted-foreground">
        将提交的更正：<Code>{JSON.stringify(c ?? null)}</Code>（补录的时间记为 explicit 并记录补录人）
      </p>
    </div>
  );
}

function ReviewPage() {
  const params = useSearchParams();
  const router = useRouter();
  const { active, ready } = useIdentity();
  const token = active?.token ?? null;
  const status = (params.get("status") as Status) || "open";
  const batchFilter = params.get("batch_id") ?? "";
  const [batchInput, setBatchInput] = useState(batchFilter);
  const [limit, setLimit] = useState(50);
  const [cursors, setCursors] = useState<(string | null)[]>([null]);
  const [page, setPage] = useState<ApiResult<ReviewTaskPage> | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [drafts, setDrafts] = useState<Record<string, Draft>>({});
  const [editing, setEditing] = useState<Set<string>>(new Set());
  const [submitting, setSubmitting] = useState(false);
  const [decideRes, setDecideRes] = useState<ApiResult<{ results: DecisionResult[] }> | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<Decision[] | null>(null);
  const [nonce, setNonce] = useState(0);

  const cursor = cursors[cursors.length - 1];

  const setQuery = (next: { status?: Status; batch_id?: string }) => {
    const q = new URLSearchParams();
    const s = next.status ?? status;
    const b = next.batch_id ?? batchFilter;
    if (s !== "open") q.set("status", s);
    if (b) q.set("batch_id", b);
    router.replace(`/review${q.size ? `?${q}` : ""}`, { scroll: false });
    setCursors([null]);
    setSelected(new Set());
  };

  const load = useCallback(async () => {
    const q = new URLSearchParams({ status, limit: String(limit) });
    if (batchFilter) q.set("batch_id", batchFilter);
    if (cursor) q.set("cursor", cursor);
    const r = await call<ReviewTaskPage>(token, "GET", `/v1/review/tasks?${q}`);
    setPage(r);
    return r;
  }, [token, status, batchFilter, cursor, limit]);

  // 待审核的任务还在分类与抽取（queued、running）时，每 3 秒刷新一次
  useEffect(() => {
    if (!ready) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      const r = await load();
      if (!alive) return;
      const pending = r.ok && r.data!.items.some((t) => t.extraction_status === "queued" || t.extraction_status === "running");
      if (pending) timer = setTimeout(tick, 3000);
    };
    tick();
    return () => { alive = false; if (timer) clearTimeout(timer); };
  }, [ready, load, nonce]);

  const items = page?.ok ? page.data!.items : [];
  const resultOf = new Map((decideRes?.ok ? decideRes.data!.results : []).map((r) => [r.task_id, r]));

  const submitDecisions = async (decisions: Decision[]) => {
    setSubmitting(true);
    const r = await call<{ results: DecisionResult[] }>(token, "POST", "/v1/review/decisions", { json: { decisions } });
    setSubmitting(false);
    setDecideRes(r);
    if (r.ok) {
      const done = new Set(r.data!.results.filter((x) => x.ok).map((x) => x.task_id));
      setSelected((s) => new Set([...s].filter((id) => !done.has(id))));
      setEditing((s) => new Set([...s].filter((id) => !done.has(id))));
      setNonce((n) => n + 1);
    }
  };

  // 删除不可撤销，先确认
  const decide = (decisions: Decision[]) => {
    if (decisions.some((d) => d.decision === "delete")) setConfirmDelete(decisions);
    else submitDecisions(decisions);
  };

  const decisionFor = (t: ReviewTask, decision: "approve" | "delete"): Decision => {
    const c = decision === "approve" ? toCorrections(drafts[t.task_id]) : undefined;
    return c ? { task_id: t.task_id, decision, corrections: c } : { task_id: t.task_id, decision };
  };

  const toggle = (id: string, on: boolean) => setSelected((s) => {
    const n = new Set(s);
    if (on) n.add(id); else n.delete(id);
    return n;
  });
  const toggleEdit = (id: string) => {
    setEditing((s) => { const n = new Set(s); if (n.has(id)) n.delete(id); else n.add(id); return n; });
    setDrafts((d) => d[id] ? d : { ...d, [id]: emptyDraft() });
  };
  const selectedTasks = items.filter((t) => selected.has(t.task_id));
  const cols = status === "open" ? 8 : status === "approved" ? 8 : 4;

  return (
    <div className="grid gap-4">
      <PageTitle title="审核" description="每份单独成事务，一份失败不影响其他份；不做按规则自动通过。需要 reviewer 角色。">
        <Button variant="outline" size="sm" onClick={() => setNonce((n) => n + 1)}><RefreshCwIcon />刷新</Button>
      </PageTitle>

      <div className="flex flex-wrap items-center gap-3">
        <Tabs value={status} onValueChange={(v) => setQuery({ status: v as Status })}>
          <TabsList>
            {(["open", "approved", "deleted"] as Status[]).map((s) => (
              <TabsTrigger key={s} value={s}>{TASK_STATUS_LABEL[s]}</TabsTrigger>
            ))}
          </TabsList>
        </Tabs>
        <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); setQuery({ batch_id: batchInput.trim() }); }}>
          <Input placeholder="按审核批次 batch_id 筛选" className="w-80 font-mono" value={batchInput} onChange={(e) => setBatchInput(e.target.value)} />
          <Button type="submit" variant="secondary">筛选</Button>
          {batchFilter && <Button type="button" variant="ghost" onClick={() => { setBatchInput(""); setQuery({ batch_id: "" }); }}>清除</Button>}
        </form>
        <div className="ml-auto flex items-center gap-2 text-sm text-muted-foreground">
          每页
          <NativeSelect size="sm" aria-label="每页条数" value={limit} onChange={(e) => { setLimit(Number(e.target.value)); setCursors([null]); }}>
            {[10, 20, 50, 100, 200].map((n) => <NativeSelectOption key={n} value={n}>{n}</NativeSelectOption>)}
          </NativeSelect>
        </div>
      </div>

      {status === "open" && (
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span>已选 {selectedTasks.length} 份</span>
          <Button size="sm" disabled={!selectedTasks.length || submitting}
            onClick={() => decide(selectedTasks.map((t) => decisionFor(t, "approve")))}>
            <CheckIcon />批量通过
          </Button>
          <Button size="sm" variant="destructive" disabled={!selectedTasks.length || submitting}
            onClick={() => decide(selectedTasks.map((t) => decisionFor(t, "delete")))}>
            <Trash2Icon />批量删除（驳回）
          </Button>
          <span className="text-xs text-muted-foreground">展开“更正”填写的内容随通过一起提交。</span>
        </div>
      )}

      <ApiError result={page} title="读取审核任务失败" />
      <ApiError result={decideRes} title="提交审核决定失败" />
      {decideRes?.ok && (
        <Alert className={decideRes.data!.results.every((r) => r.ok) ? TONE.good : TONE.bad}>
          <AlertTitle>
            提交了 {decideRes.data!.results.length} 份：成功 {decideRes.data!.results.filter((r) => r.ok).length}，
            失败 {decideRes.data!.results.filter((r) => !r.ok).length}
          </AlertTitle>
          {decideRes.data!.results.some((r) => !r.ok) && (
            <AlertDescription className="text-current/80">
              {decideRes.data!.results.filter((r) => !r.ok).map((r) => (
                <p key={r.task_id}>任务 <Mono>{r.task_id.slice(0, 8)}</Mono>：{r.error}</p>
              ))}
            </AlertDescription>
          )}
        </Alert>
      )}

      {page?.ok && (
        <Card size="sm">
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  {status === "open" && (
                    <TableHead className="w-8">
                      <Checkbox aria-label="全选" checked={items.length > 0 && selectedTasks.length === items.length}
                        onCheckedChange={(v) => setSelected(v === true ? new Set(items.map((t) => t.task_id)) : new Set())} />
                    </TableHead>
                  )}
                  <TableHead>文档</TableHead>
                  <TableHead>版本</TableHead>
                  <TableHead>上传时间</TableHead>
                  {status !== "deleted" && <TableHead>类型 / 抽取</TableHead>}
                  {status === "open" && <TableHead>待确认有效期</TableHead>}
                  {status !== "deleted" && <TableHead>入库提示</TableHead>}
                  {status !== "open" && <TableHead>{status === "deleted" ? "删除人 / 时间" : "审核人 / 时间"}</TableHead>}
                  {status === "approved" && <TableHead>更正</TableHead>}
                  {status === "open" && <TableHead className="text-right">操作</TableHead>}
                </TableRow>
              </TableHeader>
              <TableBody>
                {items.length === 0 && (
                  <TableRow><TableCell colSpan={cols} className="py-6 text-center text-muted-foreground">没有{TASK_STATUS_LABEL[status]}的任务。</TableCell></TableRow>
                )}
                {items.map((t) => {
                  const res = resultOf.get(t.task_id);
                  const done = t.extraction_status === "done";
                  const corr = toCorrections(drafts[t.task_id]);
                  return (
                    <Fragment key={t.task_id}>
                      <TableRow data-state={selected.has(t.task_id) ? "selected" : undefined}>
                        {status === "open" && (
                          <TableCell className="align-top">
                            <Checkbox aria-label={`选择 ${t.title ?? t.task_id}`} checked={selected.has(t.task_id)}
                              onCheckedChange={(v) => toggle(t.task_id, v === true)} />
                          </TableCell>
                        )}
                        <TableCell className="align-top whitespace-normal">
                          <div className="font-medium">
                            {t.title ?? <span className="text-muted-foreground">（已删除，不保留内容）</span>}
                            {t.version_id && (
                              <Link href={`/versions/${t.version_id}`} className="ml-2 inline-flex items-center gap-0.5 text-xs font-normal text-primary hover:underline">
                                <BookOpenIcon className="size-3" />全文
                              </Link>
                            )}
                          </div>
                          <div className="text-xs text-muted-foreground">
                            任务 <Mono>{t.task_id.slice(0, 8)}</Mono>
                            {t.document_id && <> · 文档 <Mono>{t.document_id}</Mono></>}
                          </div>
                          {t.batch_id && <div className="text-xs text-muted-foreground">批次 <Mono>{t.batch_id.slice(0, 8)}</Mono></div>}
                          {t.import_batch_id && <div className="text-xs text-muted-foreground">导入批次 {t.import_batch_id}</div>}
                          {res && !res.ok && <div className="text-xs text-destructive">{res.error}</div>}
                        </TableCell>
                        <TableCell className="align-top">{t.version_no ?? "—"}</TableCell>
                        <TableCell className="align-top text-xs">{formatTime(t.uploaded_at)}</TableCell>
                        {status !== "deleted" && (
                          <TableCell className="align-top">
                            <div className="grid justify-items-start gap-1 text-xs">
                              {t.doc_type ? DOC_TYPE_LABEL[t.doc_type] : <span className="text-muted-foreground">未确认</span>}
                              <ToneBadge tone={done ? "good" : t.extraction_status === "failed" ? "bad" : "warn"}>
                                {t.extraction_status ? EXTRACTION_LABEL[t.extraction_status] : "—"}
                              </ToneBadge>
                            </div>
                          </TableCell>
                        )}
                        {status === "open" && <TableCell className="align-top"><DraftValidity d={t.draft_validity} /></TableCell>}
                        {status !== "deleted" && (
                          <TableCell className="align-top">
                            {t.suggestions && Object.keys(t.suggestions).length ? <JsonCollapsible value={t.suggestions} /> : <span className="text-muted-foreground">—</span>}
                          </TableCell>
                        )}
                        {status !== "open" && (
                          <TableCell className="align-top text-xs">{t.reviewer ?? "—"}<div className="text-muted-foreground">{formatTime(t.decided_at)}</div></TableCell>
                        )}
                        {status === "approved" && (
                          <TableCell className="align-top">{t.corrections ? <JsonCollapsible value={t.corrections} /> : <span className="text-muted-foreground">—</span>}</TableCell>
                        )}
                        {status === "open" && (
                          <TableCell className="align-top">
                            <div className="flex justify-end gap-1.5">
                              <Button size="xs" disabled={submitting} onClick={() => decide([decisionFor(t, "approve")])}
                                title={done ? undefined : "分类与抽取完成后才能通过"}>
                                <CheckIcon />通过{corr ? "（带更正）" : ""}
                              </Button>
                              <Button size="xs" variant={editing.has(t.task_id) ? "secondary" : "outline"} onClick={() => toggleEdit(t.task_id)}>
                                <PencilIcon />{editing.has(t.task_id) ? "收起更正" : "更正"}
                              </Button>
                              <Button size="xs" variant="destructive" disabled={submitting} onClick={() => decide([decisionFor(t, "delete")])}>
                                <Trash2Icon />删除（驳回）
                              </Button>
                            </div>
                          </TableCell>
                        )}
                      </TableRow>
                      {editing.has(t.task_id) && drafts[t.task_id] && (
                        <TableRow className="hover:bg-transparent">
                          <TableCell colSpan={cols} className="whitespace-normal">
                            <CorrectionEditor taskId={t.task_id} d={drafts[t.task_id]} onChange={(d) => setDrafts((x) => ({ ...x, [t.task_id]: d }))} />
                          </TableCell>
                        </TableRow>
                      )}
                    </Fragment>
                  );
                })}
              </TableBody>
            </Table>
            <div className="mt-3 flex items-center gap-2">
              <Button size="sm" variant="outline" disabled={cursors.length <= 1} onClick={() => setCursors((c) => c.slice(0, -1))}>上一页</Button>
              <span className="text-xs text-muted-foreground">第 {cursors.length} 页</span>
              <Button size="sm" variant="outline" disabled={!page.data!.next_cursor} onClick={() => setCursors((c) => [...c, page.data!.next_cursor])}>下一页</Button>
              {status === "approved" && (
                <Button size="sm" variant="link" asChild className="ml-auto"><Link href="/documents">去归档或取消归档 →</Link></Button>
              )}
            </div>
          </CardContent>
        </Card>
      )}

      <AlertDialog open={confirmDelete !== null} onOpenChange={(o) => { if (!o) setConfirmDelete(null); }}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>删除 {confirmDelete?.length ?? 0} 份待审核版本？</AlertDialogTitle>
            <AlertDialogDescription>
              删除后不可恢复。审核记录保留删除人与时间，不保留内容；可在“已删除”里查看。
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>取消</AlertDialogCancel>
            <AlertDialogAction variant="destructive" onClick={() => { const d = confirmDelete; setConfirmDelete(null); if (d) submitDecisions(d); }}>
              删除
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

export default function Page() {
  return (
    <Suspense fallback={<p className="text-sm text-muted-foreground">加载中…</p>}>
      <ReviewPage />
    </Suspense>
  );
}
