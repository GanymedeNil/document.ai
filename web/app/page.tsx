"use client";

// 提问（POST /v1/query）：选模式（自动或 Fast / Deep / Verified，只能提高）、时间点、是否流式；
// 超过同步时限转入后台时，在本页显示进度并可以取消。

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useEffect, useRef, useState } from "react";
import { SendIcon } from "lucide-react";
import { call, streamQuery, type ApiResult, type StreamEvent } from "@/lib/api";
import { identityKey, useIdentity } from "@/lib/identity";
import { rememberSession } from "@/lib/storage";
import { uuid } from "@/lib/format";
import type { Mode, QueryRequest, QueryResponse } from "@/lib/types";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { ApiError } from "@/components/ApiError";
import { AnswerBody } from "@/components/Answer";
import { Field, Mono, PageTitle, ToneBadge } from "@/components/common";
import { SessionView } from "@/components/SessionView";

const SAMPLES = [
  "一线城市的住宿报销上限是多少？",
  "2026 年 3 月 15 日，一线城市的住宿报销上限是多少？",
  "住宿报销上限从多少改成了多少？",
];

function AskPage() {
  const params = useSearchParams();
  const router = useRouter();
  const { active, ready } = useIdentity();
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<"" | Mode>("");
  const [asOfValid, setAsOfValid] = useState("");
  const [asOfSystem, setAsOfSystem] = useState("");
  const [stream, setStream] = useState(false);
  const [sessionId, setSessionId] = useState<string>(""); // 空串表示开始新会话
  const [busy, setBusy] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState<ApiResult<QueryResponse> | null>(null);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [lastSession, setLastSession] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    const s = params.get("session");
    if (s) setSessionId(s);
  }, [params]);

  useEffect(() => {
    if (!busy) return;
    const t0 = Date.now();
    const t = setInterval(() => setElapsed(Math.round((Date.now() - t0) / 1000)), 500);
    return () => clearInterval(t);
  }, [busy]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;
    // 新会话的 id 由客户端生成（接口允许），这样即使请求失败或转入后台，也知道去哪里看
    const sid = sessionId || uuid();
    const body: QueryRequest = {
      question: question.trim(),
      session_id: sid,
      as_of_valid: asOfValid || null,
      as_of_system: asOfSystem || null,
      mode_hint: mode || null,
      stream,
    };
    setBusy(true);
    setElapsed(0);
    setResult(null);
    setEvents([]);
    setLastSession(sid);
    abort.current = new AbortController();
    try {
      const r = stream
        ? await streamQuery(active?.token ?? null, body, (ev) => setEvents((xs) => [...xs, ev]), abort.current.signal)
        : await call<QueryResponse>(active?.token ?? null, "POST", "/v1/query", { json: body, signal: abort.current.signal });
      setResult(r);
      if (r.ok || r.status === 502) {
        // 记入本浏览器的会话列表（接口没有列出会话的操作）；502 时提问也记入了 query_log
        rememberSession(identityKey(active), sid, body.question);
        setSessionId(sid);
        router.replace(`/?session=${sid}`, { scroll: false });
      }
    } catch (err) {
      if ((err as Error).name !== "AbortError") throw err;
    } finally {
      setBusy(false);
    }
  };

  const data = result?.ok ? result.data! : null;
  const deltaText = events.filter((e) => e.event === "delta").map((e) => (e.data as { text: string }).text).join("");
  const continuing = sessionId || lastSession;

  return (
    <div className="grid gap-4">
      <PageTitle title="提问" description="回答中的每条结论附引用：文档、版本、定位与有效期来源；拿不准的标为“未验证”。" />
      {ready && !active && (
        <Alert>
          <AlertTitle>还没有身份</AlertTitle>
          <AlertDescription>先到 <Link href="/identities" className="text-primary underline">身份</Link> 页导入令牌表。</AlertDescription>
        </Alert>
      )}
      <Card>
        <form onSubmit={submit} className="contents">
          <CardContent className="grid gap-4">
            <Textarea id="question" value={question} onChange={(e) => setQuestion(e.target.value)} maxLength={2000}
              placeholder="输入问题，如：一线城市的住宿报销上限是多少？" aria-label="问题" className="min-h-20" />
            <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
              示例
              {SAMPLES.map((s) => (
                <Button type="button" size="xs" variant="outline" key={s} onClick={() => setQuestion(s)}>{s}</Button>
              ))}
            </div>
            <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Field label="模式（mode_hint，只能提高）" htmlFor="mode">
                <NativeSelect id="mode" className="w-full" value={mode} onChange={(e) => setMode(e.target.value as "" | Mode)}>
                  <NativeSelectOption value="">自动（由 Router 决定）</NativeSelectOption>
                  <NativeSelectOption value="fast">Fast</NativeSelectOption>
                  <NativeSelectOption value="deep">Deep</NativeSelectOption>
                  <NativeSelectOption value="verified">Verified</NativeSelectOption>
                </NativeSelect>
              </Field>
              <Field label="某天实际生效（as_of_valid）" htmlFor="as_of_valid">
                <Input id="as_of_valid" type="date" value={asOfValid} onChange={(e) => setAsOfValid(e.target.value)} />
              </Field>
              <Field label="截至某天系统所知（as_of_system，回溯）" htmlFor="as_of_system">
                <Input id="as_of_system" type="date" value={asOfSystem} onChange={(e) => setAsOfSystem(e.target.value)} />
              </Field>
              <Field label="会话" htmlFor="session">
                <NativeSelect id="session" className="w-full" value={sessionId ? "continue" : "new"} onChange={(e) => {
                  if (e.target.value === "new") { setSessionId(""); router.replace("/", { scroll: false }); }
                  else if (lastSession) setSessionId(lastSession);
                }}>
                  <NativeSelectOption value="new">开始新会话</NativeSelectOption>
                  {continuing && <NativeSelectOption value="continue">接着会话 {continuing.slice(0, 8)}</NativeSelectOption>}
                </NativeSelect>
              </Field>
            </div>
          </CardContent>
          <CardFooter className="flex flex-wrap items-center gap-3">
            <div className="flex items-center gap-2">
              <Checkbox id="stream" checked={stream} onCheckedChange={(v) => setStream(v === true)} />
              <Label htmlFor="stream" className="font-normal">流式返回（SSE：mode、notice、delta、claim、done 事件）</Label>
            </div>
            <span className="text-xs text-muted-foreground">时间点都不填时按问题里的时间点，没有则按当前生效。</span>
            <div className="ml-auto flex items-center gap-2">
              {busy && (
                <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Spinner />等待回答 {elapsed} 秒（同步时限默认 30 秒，超过转入后台）
                </span>
              )}
              {busy && <Button type="button" variant="outline" onClick={() => abort.current?.abort()}>停止等待</Button>}
              <Button type="submit" disabled={busy || !question.trim() || !active}><SendIcon />提问</Button>
            </div>
          </CardFooter>
        </form>
      </Card>

      {stream && events.length > 0 && (
        <Card size="sm">
          <CardHeader><CardTitle>流式事件（{events.length}）</CardTitle></CardHeader>
          <CardContent className="grid gap-2">
            <div className="max-h-52 overflow-y-auto font-mono text-xs">
              {events.map((e, i) => (
                <div key={i}><span className="font-semibold">{e.event}</span> {e.event === "done" ? "（完整响应，见下）" : JSON.stringify(e.data)}</div>
              ))}
            </div>
            {deltaText && !data && <div className="rounded-md bg-muted px-3 py-2 text-sm">{deltaText}</div>}
          </CardContent>
        </Card>
      )}

      <ApiError result={result} title="提问失败" />

      {data && data.status === "completed" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2"><ToneBadge tone="good">已完成</ToneBadge>回答</CardTitle>
            <CardDescription>
              会话 <Link href={`/sessions/${data.session_id}`} className="text-primary hover:underline"><Mono>{data.session_id}</Mono></Link>
            </CardDescription>
          </CardHeader>
          <CardContent>
            <AnswerBody answer={data.answer} abstained={data.abstained} claims={data.claims} notices={data.notices}
              timeBasis={data.time_basis} mode={data.mode} modePath={data.mode_path} />
          </CardContent>
        </Card>
      )}

      {data && data.status !== "completed" && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ToneBadge tone={data.status === "running" ? "warn" : "bad"}>{data.status === "running" ? "已转入后台" : "已取消"}</ToneBadge>
            </CardTitle>
            <CardDescription>
              {data.status === "running"
                ? "超过同步时限，查询在当前步骤结束后交给后台任务进程；下面每 2 秒刷新一次会话进度。"
                : "查询在执行中被取消。"}
            </CardDescription>
          </CardHeader>
          <CardContent><SessionView sessionId={data.session_id} /></CardContent>
        </Card>
      )}
    </div>
  );
}

export default function Page() {
  return (
    <Suspense fallback={<p className="text-sm text-muted-foreground">加载中…</p>}>
      <AskPage />
    </Suspense>
  );
}
