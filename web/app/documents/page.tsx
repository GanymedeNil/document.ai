"use client";

// 文档的归档与取消归档（POST /v1/documents/{id}/archive、/unarchive，需要 editor 角色）。
// 接口没有“列出文档”的操作：这里合并两处来源——已通过的审核任务（需要 reviewer 角色才能读）与本浏览器上传过的文档；
// 也可以直接输入文档 id。归档后的提问一律查不到（回溯也不例外）；取消归档恢复为可搜索，不重新审核。

import { useCallback, useEffect, useState } from "react";
import { ArchiveIcon, ArchiveRestoreIcon, InfoIcon, RefreshCwIcon } from "lucide-react";
import { call, type ApiResult } from "@/lib/api";
import { useIdentity } from "@/lib/identity";
import { listUploads } from "@/lib/storage";
import { clearEvidenceCache } from "@/lib/useEvidence";
import type { ArchiveResult, ReviewTaskPage } from "@/lib/types";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ApiError } from "@/components/ApiError";
import { Mono, PageTitle, ToneBadge } from "@/components/common";

interface DocRow {
  document_id: string;
  title: string;
  versions: number[];
  from: Set<string>;
}

export default function DocumentsPage() {
  const { active, ready, identities } = useIdentity();
  const token = active?.token ?? null;
  const [rows, setRows] = useState<DocRow[]>([]);
  const [listRes, setListRes] = useState<ApiResult<ReviewTaskPage> | null>(null);
  const [actions, setActions] = useState<Record<string, ApiResult<ArchiveResult>>>({});
  const [busy, setBusy] = useState<string | null>(null);
  const [manual, setManual] = useState("");

  const load = useCallback(async () => {
    const map = new Map<string, DocRow>();
    const add = (id: string, title: string, version: number | null, from: string) => {
      const r = map.get(id) ?? { document_id: id, title, versions: [], from: new Set<string>() };
      if (version && !r.versions.includes(version)) r.versions.push(version);
      if (title && (!r.title || r.title.startsWith("（"))) r.title = title;
      r.from.add(from);
      map.set(id, r);
    };
    // 已通过的审核任务（最多读 5 页）
    let cursor: string | null = null;
    let first: ApiResult<ReviewTaskPage> | null = null;
    for (let i = 0; i < 5; i++) {
      const q = new URLSearchParams({ status: "approved", limit: "200" });
      if (cursor) q.set("cursor", cursor);
      const r = await call<ReviewTaskPage>(token, "GET", `/v1/review/tasks?${q}`);
      first ??= r;
      if (!r.ok) break;
      for (const t of r.data!.items) if (t.document_id) add(t.document_id, t.title ?? "", t.version_no, "审核任务");
      cursor = r.data!.next_cursor;
      if (!cursor) break;
    }
    setListRes(first);
    for (const u of listUploads()) add(u.document_id, u.title ?? `（${u.filename}）`, null, "本浏览器上传");
    setRows([...map.values()]);
  }, [token]);

  useEffect(() => { if (ready) load(); }, [ready, load]);
  // 换身份后清掉上一个身份的操作结果，免得看错
  useEffect(() => setActions({}), [token]);

  const act = async (id: string, what: "archive" | "unarchive") => {
    setBusy(id + what);
    const r = await call<ArchiveResult>(token, "POST", `/v1/documents/${encodeURIComponent(id)}/${what}`);
    setBusy(null);
    setActions((a) => ({ ...a, [id]: r }));
    clearEvidenceCache();
  };

  const reviewerHint = identities.find((i) => i.roles?.includes("reviewer"));

  const buttons = (id: string, size: "xs" | "sm") => (
    <div className="flex gap-1.5">
      <Button size={size} variant="outline" disabled={!id || !!busy} onClick={() => act(id, "archive")}>
        {busy === id + "archive" ? <Spinner /> : <ArchiveIcon />}归档
      </Button>
      <Button size={size} variant="outline" disabled={!id || !!busy} onClick={() => act(id, "unarchive")}>
        {busy === id + "unarchive" ? <Spinner /> : <ArchiveRestoreIcon />}取消归档
      </Button>
    </div>
  );

  return (
    <div className="grid gap-4">
      <PageTitle title="文档：归档与取消归档" description="归档与取消归档需要 editor 角色。">
        <Button variant="outline" size="sm" onClick={load}><RefreshCwIcon />刷新</Button>
      </PageTitle>
      <Alert>
        <InfoIcon />
        <AlertTitle>归档以文档为单位</AlertTitle>
        <AlertDescription>
          归档全部可搜索版本，之后的提问一律查不到（回溯查询也不例外）；历史回答保留引用记录，点开引用只给归档提示。
          取消归档恢复为可搜索，不重新审核，审核通过时间不变。
        </AlertDescription>
      </Alert>

      <Card size="sm">
        <CardContent className="grid gap-2">
          <div className="flex flex-wrap gap-2">
            <Input value={manual} onChange={(e) => setManual(e.target.value)} className="min-w-80 flex-1 font-mono" placeholder="按文档 id（UUID）操作" aria-label="文档 id" />
            {buttons(manual.trim(), "sm")}
          </div>
          {manual.trim() && actions[manual.trim()] && <ActionResult r={actions[manual.trim()]} />}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>文档列表</CardTitle>
          <CardDescription>
            {listRes && !listRes.ok
              ? <>当前身份读不到审核任务（HTTP {listRes.status}），只列出本浏览器上传过的文档。{reviewerHint && <>切换成有 reviewer 角色的身份（如 {reviewerHint.label}）可看到全部已通过的文档。</>}</>
              : "已通过的审核任务与本浏览器上传过的文档。"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {rows.length === 0 ? <p className="text-sm text-muted-foreground">没有已知的文档。</p> : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>标题 / 文档 id</TableHead>
                  <TableHead>已知版本</TableHead>
                  <TableHead>来源</TableHead>
                  <TableHead>操作</TableHead>
                  <TableHead className="w-56 min-w-56">结果</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {rows.map((d) => (
                  <TableRow key={d.document_id}>
                    <TableCell className="whitespace-normal">
                      <div className="font-medium">{d.title || <span className="text-muted-foreground">—</span>}</div>
                      <Mono className="text-muted-foreground">{d.document_id}</Mono>
                    </TableCell>
                    <TableCell className="whitespace-normal">{d.versions.length ? d.versions.sort((a, b) => a - b).map((v) => `第 ${v} 版`).join("、") : <span className="text-muted-foreground">—</span>}</TableCell>
                    <TableCell className="text-xs whitespace-normal">{[...d.from].join("、")}</TableCell>
                    <TableCell>{buttons(d.document_id, "xs")}</TableCell>
                    <TableCell className="w-56 min-w-56 whitespace-normal">{actions[d.document_id] && <ActionResult r={actions[d.document_id]} compact />}</TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function ActionResult({ r, compact }: { r: ApiResult<ArchiveResult>; compact?: boolean }) {
  if (!r.ok) {
    if (compact) {
      return <ToneBadge tone="bad" className="h-auto whitespace-normal" title={r.error?.message}>HTTP {r.status} {r.error?.code}：{r.error?.message}</ToneBadge>;
    }
    return <ApiError result={r} title="操作失败" />;
  }
  const d = r.data!;
  const text = `${d.status === "archived" ? "已归档" : "已恢复为可搜索"} ${d.versions.length} 个版本`;
  return compact
    ? <ToneBadge tone="good" className="h-auto whitespace-normal">{text}</ToneBadge>
    : <p className="text-sm">{text}：<Mono>{d.versions.join("、")}</Mono></p>;
}
