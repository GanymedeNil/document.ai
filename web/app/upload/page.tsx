"use client";

// 上传文档（POST /v1/documents，需要 editor 角色）：进入待审核。内容完全相同的版本已存在时返回 200、duplicate 为 true。

import Link from "next/link";
import { useEffect, useState } from "react";
import { CircleCheckIcon, UploadIcon } from "lucide-react";
import { call, type ApiResult } from "@/lib/api";
import { useIdentity } from "@/lib/identity";
import { listUploads, rememberUpload, type UploadRecord } from "@/lib/storage";
import { formatTime, uuid } from "@/lib/format";
import type { UploadResult } from "@/lib/types";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ApiError } from "@/components/ApiError";
import { Code, Field, Mono, PageTitle, ToneBadge, TONE } from "@/components/common";

export default function UploadPage() {
  const { active } = useIdentity();
  const [file, setFile] = useState<File | null>(null);
  const [principals, setPrincipals] = useState("R_STAFF");
  const [title, setTitle] = useState("");
  const [sourceRef, setSourceRef] = useState("");
  const [importBatch, setImportBatch] = useState("");
  const [batchId, setBatchId] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<ApiResult<UploadResult> | null>(null);
  const [history, setHistory] = useState<UploadRecord[]>([]);
  const [inputKey, setInputKey] = useState(0);

  useEffect(() => setHistory(listUploads()), []);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;
    const form = new FormData();
    form.append("file", file, file.name);
    form.append("principals", principals);
    if (title.trim()) form.append("title", title.trim());
    if (sourceRef.trim()) form.append("source_ref", sourceRef.trim());
    if (importBatch.trim()) form.append("import_batch_id", importBatch.trim());
    if (batchId.trim()) form.append("batch_id", batchId.trim());
    setBusy(true);
    const r = await call<UploadResult>(active?.token ?? null, "POST", "/v1/documents", { form });
    setBusy(false);
    setResult(r);
    if (r.ok) {
      rememberUpload({
        document_id: r.data!.document_id, version_id: r.data!.version_id, filename: file.name,
        title: title.trim() || null, principals, duplicate: r.data!.duplicate,
        uploaded_by: active?.label ?? "", at: new Date().toISOString(),
      });
      setHistory(listUploads());
      setFile(null);
      setInputKey((k) => k + 1);
    }
  };

  return (
    <div className="grid gap-4">
      <PageTitle title="上传文档" description="上传后进入待审核，后台解析、分类与抽取完成后由审核人通过。需要 editor 角色。" />
      <Card>
        <form onSubmit={submit} className="contents">
          <CardContent className="grid gap-4 sm:grid-cols-2">
            <Field label="文件" htmlFor="file" hint="HTML、Word；装了 Docling 时另有 PDF、PPTX、XLSX、图片">
              <Input key={inputKey} id="file" type="file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            </Field>
            <Field label="可访问的角色 principals（必填）" htmlFor="principals" hint="分号或逗号分隔，如 R_STAFF;R_FIN">
              <Input id="principals" value={principals} onChange={(e) => setPrincipals(e.target.value)} />
            </Field>
            <Field label="标题" htmlFor="title" hint="不填用解析出的标题">
              <Input id="title" value={title} onChange={(e) => setTitle(e.target.value)} />
            </Field>
            <Field label="来源标识 source_ref" htmlFor="source_ref" hint="同一来源的后续上传成为该文档的新版本；不填取文件名">
              <Input id="source_ref" value={sourceRef} onChange={(e) => setSourceRef(e.target.value)} />
            </Field>
            <Field label="批量导入批次 import_batch_id" htmlFor="import_batch_id" hint="同一批次的版本时间来源等级为 unknown">
              <Input id="import_batch_id" value={importBatch} onChange={(e) => setImportBatch(e.target.value)} />
            </Field>
            <Field label="审核批次 batch_id（UUID）" htmlFor="batch_id" hint="审核任务可按它筛选">
              <div className="flex gap-2">
                <Input id="batch_id" value={batchId} onChange={(e) => setBatchId(e.target.value)} className="font-mono" />
                <Button type="button" variant="outline" onClick={() => setBatchId(uuid())}>生成</Button>
              </div>
            </Field>
          </CardContent>
          <CardFooter className="justify-between gap-2">
            <span className="text-xs text-muted-foreground">当前身份：{active?.label ?? "未选"}</span>
            <Button type="submit" disabled={busy || !file || !principals.trim()}>
              {busy ? <Spinner /> : <UploadIcon />}上传
            </Button>
          </CardFooter>
        </form>
      </Card>

      <ApiError result={result} title="上传失败" />
      {result?.ok && (
        <Alert className={TONE.good}>
          <CircleCheckIcon />
          <AlertTitle>
            {result.data!.duplicate ? "内容相同的版本已存在，未重复入库（HTTP 200）" : "已接收，等待后台解析与审核（HTTP 201）"}
          </AlertTitle>
          <AlertDescription className="text-current/80">
            <p>文档 <Code>{result.data!.document_id}</Code> 版本 <Code>{result.data!.version_id}</Code> 状态 {result.data!.status}</p>
            <p>
              下一步：换成 reviewer 身份到
              <Link href={batchId ? `/review?batch_id=${batchId}` : "/review"} className="mx-1 underline">审核</Link>
              页通过它（分类与抽取完成后才能通过）。
            </p>
          </AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>本浏览器上传过的文档</CardTitle>
          <CardDescription>接口没有列出文档的操作，这里只是本浏览器的记录。</CardDescription>
        </CardHeader>
        <CardContent>
          {history.length === 0 ? <p className="text-sm text-muted-foreground">还没有。</p> : (
            <Table>
              <TableHeader>
                <TableRow><TableHead>文件</TableHead><TableHead>标题</TableHead><TableHead>principals</TableHead><TableHead>文档 id</TableHead><TableHead>版本 id</TableHead><TableHead>上传人</TableHead><TableHead>时间</TableHead></TableRow>
              </TableHeader>
              <TableBody>
                {history.map((u) => (
                  <TableRow key={u.version_id}>
                    <TableCell>{u.filename} {u.duplicate && <ToneBadge>重复</ToneBadge>}</TableCell>
                    <TableCell>{u.title ?? <span className="text-muted-foreground">—</span>}</TableCell>
                    <TableCell>{u.principals}</TableCell>
                    <TableCell><Mono title={u.document_id}>{u.document_id.slice(0, 8)}</Mono></TableCell>
                    <TableCell><Mono title={u.version_id}>{u.version_id.slice(0, 8)}</Mono></TableCell>
                    <TableCell>{u.uploaded_by}</TableCell>
                    <TableCell className="text-xs text-muted-foreground">{formatTime(u.at)}</TableCell>
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
