"use client";

// 身份管理：导入令牌表（与接口进程的 EAER_TOKENS_FILE 同一格式）或逐个添加令牌，随时切换，便于演示权限差异。
// 令牌只保存在本浏览器（localStorage），只用于本地演示；生产环境由 SSO 提供身份。

import { useState } from "react";
import { ImportIcon, InfoIcon, PlusIcon, ShieldQuestionIcon, Trash2Icon } from "lucide-react";
import { call, type ApiResult } from "@/lib/api";
import { parseTokensFile, useIdentity } from "@/lib/identity";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Textarea } from "@/components/ui/textarea";
import { Code, Field, Mono, PageTitle, ToneBadge } from "@/components/common";

const EXAMPLE = `{
  "令牌-editor": {"user": "editor1", "tenant": "t1", "roles": ["editor"]},
  "令牌-reviewer": {"user": "reviewer1", "tenant": "t1", "roles": ["reviewer"]},
  "令牌-user": {"user": "user1", "tenant": "t1", "roles": [], "principals": ["R_STAFF"]}
}`;

export default function IdentitiesPage() {
  const { identities, active, setActive, add, remove, importTokens } = useIdentity();
  const [json, setJson] = useState("");
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const [label, setLabel] = useState("");
  const [token, setToken] = useState("");
  const [show, setShow] = useState(false);
  const [checks, setChecks] = useState<Record<string, ApiResult<unknown>>>({});

  const doImport = () => {
    try {
      parseTokensFile(json); // 先校验，给出可读的错误
      const n = importTokens(json);
      setMsg({ ok: true, text: `已导入 ${n} 个身份` });
      setJson("");
    } catch (e) {
      setMsg({ ok: false, text: `导入失败：${(e as Error).message}` });
    }
  };

  const checkAll = async () => {
    const out: Record<string, ApiResult<unknown>> = {};
    await Promise.all(identities.map(async (i) => { out[i.id] = await call(i.token, "GET", "/v1/config/forced-verify"); }));
    setChecks(out);
  };

  return (
    <div className="grid gap-4">
      <PageTitle title="身份" description="令牌只存在本浏览器，只用于本地演示；客户现场由 SSO 提供身份。" />
      <Alert>
        <InfoIcon />
        <AlertTitle>接口用 <Code>Authorization: Bearer 令牌</Code> 识别身份</AlertTitle>
        <AlertDescription>
          参考实现的令牌表就是接口进程的 <Code>EAER_TOKENS_FILE</Code>：<Code>roles</Code> 是接口角色（editor 上传与归档、reviewer 审核、
          admin 追加强制验证规则），<Code>principals</Code> 是文档权限角色（决定能看到哪些文档）。
        </AlertDescription>
      </Alert>

      <Card>
        <CardHeader>
          <CardTitle>已设置的身份（{identities.length}）</CardTitle>
          <CardDescription>选中的是当前身份，也可以在右上角切换。</CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3">
          {identities.length === 0 ? <p className="text-sm text-muted-foreground">还没有。导入令牌表或在下面添加。</p> : (
            <>
              <div className="flex flex-wrap items-center gap-4">
                <Button size="sm" variant="outline" onClick={checkAll}><ShieldQuestionIcon />检查全部令牌</Button>
                <div className="flex items-center gap-2">
                  <Checkbox id="show-token" checked={show} onCheckedChange={(v) => setShow(v === true)} />
                  <Label htmlFor="show-token" className="font-normal">显示令牌</Label>
                </div>
              </div>
              <RadioGroup value={active?.id ?? ""} onValueChange={setActive} aria-label="当前身份">
                <Table>
                  <TableHeader>
                    <TableRow><TableHead className="w-8" /><TableHead>名称</TableHead><TableHead>租户</TableHead><TableHead>接口角色</TableHead><TableHead>文档权限角色</TableHead><TableHead>令牌</TableHead><TableHead>检查</TableHead><TableHead /></TableRow>
                  </TableHeader>
                  <TableBody>
                    {identities.map((i) => (
                      <TableRow key={i.id} data-state={active?.id === i.id ? "selected" : undefined}>
                        <TableCell><RadioGroupItem value={i.id} id={`id-${i.id}`} aria-label={`设 ${i.label} 为当前身份`} /></TableCell>
                        <TableCell><Label htmlFor={`id-${i.id}`}>{i.label}</Label></TableCell>
                        <TableCell>{i.tenant ?? <span className="text-muted-foreground">?</span>}</TableCell>
                        <TableCell>{i.roles?.length ? i.roles.join("、") : <span className="text-muted-foreground">无</span>}</TableCell>
                        <TableCell>{i.principals?.length ? i.principals.join("、") : <span className="text-muted-foreground">无（看不到证据）</span>}</TableCell>
                        <TableCell><Mono>{show ? i.token : `${i.token.slice(0, 4)}…`}</Mono></TableCell>
                        <TableCell>
                          {checks[i.id] && (checks[i.id].ok
                            ? <ToneBadge tone="good">有效</ToneBadge>
                            : <ToneBadge tone="bad">HTTP {checks[i.id].status} {checks[i.id].error?.code}</ToneBadge>)}
                        </TableCell>
                        <TableCell className="text-right">
                          <Button size="icon-xs" variant="ghost" aria-label={`移除 ${i.label}`} onClick={() => remove(i.id)}><Trash2Icon /></Button>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </RadioGroup>
            </>
          )}
        </CardContent>
      </Card>

      <div className="grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>导入令牌表</CardTitle>
            <CardDescription>粘贴接口进程用的令牌表 JSON；同一令牌再次导入会更新，不重复。</CardDescription>
          </CardHeader>
          <CardContent className="grid gap-3">
            <Textarea value={json} onChange={(e) => setJson(e.target.value)} placeholder={EXAMPLE} className="min-h-36 font-mono text-xs" aria-label="令牌表 JSON" />
            {msg && <p className={msg.ok ? "text-sm text-emerald-700 dark:text-emerald-300" : "text-sm text-destructive"}>{msg.text}</p>}
          </CardContent>
          <CardFooter className="justify-end">
            <Button onClick={doImport} disabled={!json.trim()}><ImportIcon />导入</Button>
          </CardFooter>
        </Card>
        <Card>
          <form className="contents" onSubmit={(e) => {
            e.preventDefault();
            add({ label: label.trim() || `令牌 ${token.slice(0, 4)}…`, token: token.trim() });
            setLabel(""); setToken("");
          }}>
            <CardHeader>
              <CardTitle>添加单个令牌</CardTitle>
              <CardDescription>不知道角色时也可以添加，接口会按令牌判断。</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3">
              <Field label="名称" htmlFor="new-label"><Input id="new-label" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="如：财务部员工" /></Field>
              <Field label="令牌" htmlFor="new-token"><Input id="new-token" value={token} onChange={(e) => setToken(e.target.value)} className="font-mono" /></Field>
            </CardContent>
            <CardFooter className="justify-end">
              <Button type="submit" variant="secondary" disabled={!token.trim()}><PlusIcon />添加</Button>
            </CardFooter>
          </form>
        </Card>
      </div>
    </div>
  );
}
