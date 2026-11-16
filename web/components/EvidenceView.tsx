"use client";

// 证据详情（GET /v1/evidence/{id}）：可搜索且有权时给原文、定位、有效期、文档日期与出处；
// 已归档且有权时只给归档提示；其余（不存在、无权、待审核、已删除）一律 404，响应体除 request_id 外相同。

import { useEffect, useState } from "react";
import { UsersIcon } from "lucide-react";
import type { ApiResult } from "@/lib/api";
import { useIdentity, type Identity } from "@/lib/identity";
import { fetchEvidence } from "@/lib/useEvidence";
import { DOC_TYPE_LABEL, formatLocator, formatTime, GRADE_LABEL } from "@/lib/format";
import type { Evidence } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Spinner } from "@/components/ui/spinner";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ApiError } from "./ApiError";
import { Code, Mono, ToneBadge } from "./common";

function Row({ k, children }: { k: string; children: React.ReactNode }) {
  return (
    <>
      <dt className="text-muted-foreground">{k}</dt>
      <dd>{children}</dd>
    </>
  );
}

export function EvidenceDetail({ r }: { r: ApiResult<Evidence> }) {
  if (!r.ok) return <ApiError result={r} title="读取证据失败" />;
  const e = r.data!;
  if (e.status === "archived") {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><ToneBadge tone="warn">已归档</ToneBadge>{e.notice}</CardTitle>
          <CardDescription>已归档的资料只给提示，不返回原文。</CardDescription>
        </CardHeader>
        <CardContent>
          <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1 text-sm">
            <Row k="文档">《{e.document.title}》 <Mono className="text-muted-foreground">{e.document.id}</Mono></Row>
            <Row k="版本">第 {e.version.version_no} 版 <Mono className="text-muted-foreground">{e.version.id}</Mono></Row>
          </dl>
        </CardContent>
      </Card>
    );
  }
  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex flex-wrap items-center gap-2">
          <ToneBadge tone="good">可搜索</ToneBadge>
          <Badge variant="outline">{{ text: "文字", table: "表格", figure: "图表" }[e.modality]}</Badge>
          <Badge variant="outline">{e.parse_stage === "deep" ? "深度解析" : "快速解析"}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4">
        <blockquote className="rounded-md border-l-4 bg-muted px-3 py-2 text-sm whitespace-pre-wrap">{e.content}</blockquote>
        <dl className="grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1.5 text-sm">
          <Row k="定位">{formatLocator(e.locator)} <Code>{JSON.stringify(e.locator)}</Code></Row>
          <Row k="文档">
            《{e.document.title}》{e.document.doc_type ? `（${DOC_TYPE_LABEL[e.document.doc_type]}）` : "（类型未定）"}{" "}
            <Mono className="text-muted-foreground">{e.document.id}</Mono>
          </Row>
          <Row k="版本">
            第 {e.version.version_no} 版，审核通过于 {formatTime(e.version.approved_at)} <Mono className="text-muted-foreground">{e.version.id}</Mono>
          </Row>
          <Row k="文档日期">{e.version.doc_date ?? <span className="text-muted-foreground">无</span>}</Row>
          <Row k="出处">{e.version.publisher ?? <span className="text-muted-foreground">无（数据源没有给出）</span>}</Row>
          <Row k="有效期">
            {e.validity ? (
              <>
                生效 {e.validity.valid_from ?? "—"}（{GRADE_LABEL[e.validity.valid_from_grade]}）；
                作废 {e.validity.valid_to ?? "—"}（{GRADE_LABEL[e.validity.valid_to_grade]}）
                <div className="text-xs text-muted-foreground">作废日是不再有效的第一天。</div>
              </>
            ) : <span className="text-muted-foreground">无（记录类文档没有有效期）</span>}
          </Row>
        </dl>
      </CardContent>
    </Card>
  );
}

/** 用每个已设置的身份各取一次同一条证据，对比状态码：无权与不存在都是 404。 */
export function EvidenceCompare({ id }: { id: string }) {
  const { identities } = useIdentity();
  const [rows, setRows] = useState<{ who: Identity; r: ApiResult<Evidence> }[] | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => setRows(null), [id]);

  const run = async () => {
    setBusy(true);
    const out = await Promise.all(identities.map(async (who) => ({ who, r: await fetchEvidence(who.token, id, true) })));
    setRows(out);
    setBusy(false);
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>按身份对比</CardTitle>
        <CardDescription>
          另可试一个不存在的 id（如 <Code>ev_0000000000000000</Code>），响应体除 request_id 外与无权时相同。
        </CardDescription>
      </CardHeader>
      <CardContent className="grid gap-3">
        <div>
          <Button variant="outline" onClick={run} disabled={busy || !identities.length}>
            {busy ? <Spinner /> : <UsersIcon />}用全部 {identities.length} 个身份各取一次
          </Button>
        </div>
        {rows && (
          <Table>
            <TableHeader>
              <TableRow><TableHead>身份</TableHead><TableHead>文档权限角色</TableHead><TableHead>HTTP</TableHead><TableHead>结果</TableHead></TableRow>
            </TableHeader>
            <TableBody>
              {rows.map(({ who, r }) => (
                <TableRow key={who.id}>
                  <TableCell>{who.label}</TableCell>
                  <TableCell className="text-xs">{who.principals?.length ? who.principals.join("、") : <span className="text-muted-foreground">无</span>}</TableCell>
                  <TableCell><ToneBadge tone={r.ok ? "good" : r.status === 404 ? "warn" : "bad"}>{r.status}</ToneBadge></TableCell>
                  <TableCell className="text-xs whitespace-normal">
                    {r.ok ? (r.data!.status === "archived" ? "已归档（只给提示）" : "可见：返回原文")
                      : <><Code>{r.error?.code}</Code> {r.error?.message}</>}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </CardContent>
    </Card>
  );
}
