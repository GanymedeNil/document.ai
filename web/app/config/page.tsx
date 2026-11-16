"use client";

// 强制验证清单（GET /v1/config/forced-verify 任何已认证身份可读；POST 追加需要 admin 角色）。
// 只能增加，接口不提供删除或修改；命中任一规则的提问必须走 Verified，增加后下一次提问立即生效。

import { useCallback, useEffect, useState } from "react";
import { CircleCheckIcon, PlusIcon, ShieldCheckIcon } from "lucide-react";
import { call, type ApiResult } from "@/lib/api";
import { useIdentity } from "@/lib/identity";
import { formatTime } from "@/lib/format";
import type { ForcedVerifyList, ForcedVerifyRule } from "@/lib/types";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select";
import { Spinner } from "@/components/ui/spinner";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ApiError } from "@/components/ApiError";
import { Code, Field, Mono, PageTitle, TONE } from "@/components/common";

type Kind = ForcedVerifyRule["kind"];

const KIND_LABEL: Record<Kind, string> = {
  keyword: "问题含该词",
  document: "答案引用该文档",
  doc_type: "答案引用该类文档",
  all: "全部提问",
};

export default function ConfigPage() {
  const { active, ready } = useIdentity();
  const token = active?.token ?? null;
  const [list, setList] = useState<ApiResult<ForcedVerifyList> | null>(null);
  const [kind, setKind] = useState<Kind>("keyword");
  const [value, setValue] = useState("");
  const [desc, setDesc] = useState("");
  const [busy, setBusy] = useState(false);
  const [added, setAdded] = useState<ApiResult<ForcedVerifyRule> | null>(null);

  const load = useCallback(async () => setList(await call<ForcedVerifyList>(token, "GET", "/v1/config/forced-verify")), [token]);
  useEffect(() => { if (ready) load(); }, [ready, load]);
  useEffect(() => setAdded(null), [token]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    const body = { kind, value: kind === "all" ? null : value.trim(), description: desc.trim() || null };
    const r = await call<ForcedVerifyRule>(token, "POST", "/v1/config/forced-verify", { json: body });
    setBusy(false);
    setAdded(r);
    if (r.ok) { setValue(kind === "doc_type" ? "normative" : ""); setDesc(""); load(); }
  };

  return (
    <div className="grid gap-4">
      <PageTitle title="强制验证清单" description="命中任一规则的提问必须走 Verified。内置规则不能关闭；租户只能追加，接口不提供删除或修改。" />
      <ApiError result={list} title="读取清单失败" />
      {list?.ok && (
        <div className="grid gap-4 lg:grid-cols-[1fr_2fr]">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2"><ShieldCheckIcon className="size-4" />内置规则（{list.data!.builtin.length}）</CardTitle>
            </CardHeader>
            <CardContent>
              <ul className="grid gap-2 text-sm">
                {list.data!.builtin.map((b) => <li key={b.id}><Code>{b.id}</Code> {b.description}</li>)}
              </ul>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>租户追加的规则（{list.data!.rules.length}）</CardTitle>
              <CardDescription>按追加顺序。</CardDescription>
            </CardHeader>
            <CardContent>
              {list.data!.rules.length === 0 ? <p className="text-sm text-muted-foreground">还没有。</p> : (
                <Table>
                  <TableHeader>
                    <TableRow><TableHead>id</TableHead><TableHead>种类</TableHead><TableHead>取值</TableHead><TableHead>说明</TableHead><TableHead>追加人</TableHead><TableHead>时间</TableHead></TableRow>
                  </TableHeader>
                  <TableBody>
                    {list.data!.rules.map((r) => (
                      <TableRow key={r.id}>
                        <TableCell><Mono>{r.id}</Mono></TableCell>
                        <TableCell>{KIND_LABEL[r.kind]}</TableCell>
                        <TableCell className="whitespace-normal">{r.value ?? <span className="text-muted-foreground">—</span>}</TableCell>
                        <TableCell className="whitespace-normal">{r.description ?? <span className="text-muted-foreground">—</span>}</TableCell>
                        <TableCell>{r.created_by}</TableCell>
                        <TableCell className="text-xs text-muted-foreground">{formatTime(r.created_at)}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      <Card>
        <form onSubmit={submit} className="contents">
          <CardHeader>
            <CardTitle>追加规则</CardTitle>
            <CardDescription>需要 admin 角色。同种类、同取值的规则已存在时原样返回。</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-4 md:grid-cols-3">
            <Field label="种类" htmlFor="kind">
              <NativeSelect id="kind" className="w-full" value={kind} onChange={(e) => { setKind(e.target.value as Kind); setValue(e.target.value === "doc_type" ? "normative" : ""); }}>
                {(Object.keys(KIND_LABEL) as Kind[]).map((k) => <NativeSelectOption key={k} value={k}>{KIND_LABEL[k]}（{k}）</NativeSelectOption>)}
              </NativeSelect>
            </Field>
            <Field label="取值" htmlFor="value">
              {kind === "doc_type" ? (
                <NativeSelect id="value" className="w-full" value={value} onChange={(e) => setValue(e.target.value)}>
                  <NativeSelectOption value="normative">规范类（normative）</NativeSelectOption>
                  <NativeSelectOption value="record">记录类（record）</NativeSelectOption>
                </NativeSelect>
              ) : (
                <Input id="value" value={kind === "all" ? "" : value} disabled={kind === "all"} onChange={(e) => setValue(e.target.value)} maxLength={100}
                  placeholder={kind === "keyword" ? "如：安全生产" : kind === "document" ? "文档 id（UUID）" : "全部提问，不需要取值"}
                  className={kind === "document" ? "font-mono" : ""} />
              )}
            </Field>
            <Field label="说明（可选，200 字以内）" htmlFor="desc">
              <Input id="desc" value={desc} onChange={(e) => setDesc(e.target.value)} maxLength={200} />
            </Field>
          </CardContent>
          <CardFooter className="justify-between">
            <span className="text-xs text-muted-foreground">当前身份：{active?.label ?? "未选"}</span>
            <Button type="submit" disabled={busy || (kind !== "all" && !value.trim())}>{busy ? <Spinner /> : <PlusIcon />}追加</Button>
          </CardFooter>
        </form>
      </Card>
      <ApiError result={added} title="追加失败" />
      {added?.ok && (
        <Alert className={TONE.good}>
          <CircleCheckIcon />
          <AlertTitle>{added.status === 201 ? "已追加（HTTP 201）" : "同种类、同取值的规则已存在，原样返回（HTTP 200）"}</AlertTitle>
          <AlertDescription className="text-current/80"><Mono>{added.data!.id}</Mono></AlertDescription>
        </Alert>
      )}
    </div>
  );
}
