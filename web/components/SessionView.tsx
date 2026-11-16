"use client";

// 会话历史：每次提问的状态、回答与引用记录；有进行中的查询时每 2 秒刷新进度，可以取消。

import Link from "next/link";
import { useEffect, useState } from "react";
import { RefreshCwIcon, SquareIcon } from "lucide-react";
import { call, type ApiResult } from "@/lib/api";
import { useIdentity } from "@/lib/identity";
import { formatTime, MODE_LABEL, QUERY_STATUS_LABEL } from "@/lib/format";
import type { CancelResult, Progress, Session } from "@/lib/types";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Spinner } from "@/components/ui/spinner";
import { ApiError } from "./ApiError";
import { AnswerBody, ModePath } from "./Answer";
import { Code, Mono, ToneBadge, type Tone, TONE } from "./common";

const STATUS_TONE: Record<string, Tone> = { completed: "good", running: "warn", cancelled: "bad", failed: "bad" };

function ProgressView({ p }: { p: Progress }) {
  return (
    <Alert className={TONE.warn}>
      <Spinner />
      <AlertTitle>
        后台执行中{p.next && <> · 下一步：{p.next}</>} · 检索轮数 {p.retrieval_rounds}
        {p.mode_path.length > 0 && <> · 模式 {p.mode_path.map((m) => MODE_LABEL[m]).join(" → ")}</>}
      </AlertTitle>
      <AlertDescription className="text-current/80">
        <p>已完成：{p.done.length ? p.done.join(" → ") : "（尚无）"}</p>
        {p.waiting && <p>等待重试：{p.waiting}{p.retry_after && <>（最早 {formatTime(p.retry_after)}）</>}</p>}
      </AlertDescription>
    </Alert>
  );
}

export function SessionView({ sessionId, pollMs = 2000 }: { sessionId: string; pollMs?: number }) {
  const { active, ready } = useIdentity();
  const token = active?.token ?? null;
  const [res, setRes] = useState<ApiResult<Session> | null>(null);
  const [cancelRes, setCancelRes] = useState<ApiResult<CancelResult> | null>(null);
  const [cancelling, setCancelling] = useState(false);
  // 改变它即重新开始读取与轮询（刷新、取消之后）
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (!ready) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | null = null;
    const tick = async () => {
      const r = await call<Session>(token, "GET", `/v1/sessions/${encodeURIComponent(sessionId)}`);
      if (!alive) return;
      setRes(r);
      if (r.ok && r.data!.items.some((i) => i.status === "running")) timer = setTimeout(tick, pollMs);
    };
    tick();
    return () => {
      alive = false;
      if (timer) clearTimeout(timer);
    };
  }, [ready, token, sessionId, pollMs, nonce]);

  const reload = () => setNonce((n) => n + 1);

  const cancel = async () => {
    setCancelling(true);
    const r = await call<CancelResult>(token, "POST", `/v1/sessions/${encodeURIComponent(sessionId)}/cancel`);
    setCancelRes(r);
    setCancelling(false);
    reload();
  };

  if (!res) return <p className="flex items-center gap-2 text-sm text-muted-foreground"><Spinner />读取会话…</p>;
  if (!res.ok) return <ApiError result={res} title="读取会话失败" />;
  const s = res.data!;
  const running = s.items.some((i) => i.status === "running");

  return (
    <div className="grid gap-4">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <span>会话 <Code>{s.session_id}</Code>，共 {s.items.length} 次提问</span>
        <Button size="sm" variant="outline" onClick={reload}><RefreshCwIcon />刷新</Button>
        {running && (
          <Button size="sm" variant="destructive" onClick={cancel} disabled={cancelling}>
            {cancelling ? <Spinner /> : <SquareIcon />}取消进行中的查询
          </Button>
        )}
        <Button size="sm" variant="link" asChild>
          <Link href={`/?session=${s.session_id}`}>接着这个会话提问 →</Link>
        </Button>
      </div>
      <ApiError result={cancelRes} title="取消失败" />
      {cancelRes?.ok && (
        <Alert className={TONE.good}>
          <AlertTitle>已提交取消</AlertTitle>
          <AlertDescription className="text-current/80">
            {cancelRes.data!.items.map((i) => (
              <p key={i.query_id}>
                <Mono>{i.query_id.slice(0, 8)}</Mono> {i.status === "cancelled" ? "已取消（排队中）" : "当前步骤结束后停止"}
              </p>
            ))}
          </AlertDescription>
        </Alert>
      )}
      {s.items.map((it, idx) => (
        <Card key={it.query_id} size="sm">
          <CardHeader>
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <span className="font-medium">#{idx + 1}</span>
              <ToneBadge tone={STATUS_TONE[it.status]}>{QUERY_STATUS_LABEL[it.status] ?? it.status}</ToneBadge>
              {it.background && <ToneBadge tone="info">曾转入后台</ToneBadge>}
              <span className="text-xs text-muted-foreground">{formatTime(it.asked_at)}</span>
              <Mono className="text-muted-foreground">query {it.query_id.slice(0, 8)}</Mono>
              <ModePath path={it.mode_path} mode={it.mode} />
            </div>
            <CardTitle className="pt-1">问：{it.question}</CardTitle>
          </CardHeader>
          <CardContent>
            {it.status === "running" && it.progress && <ProgressView p={it.progress} />}
            {it.status === "completed" && (
              <AnswerBody answer={it.answer} abstained={it.abstained} claims={it.claims} timeBasis={it.time_basis}
                mode={it.mode} modePath={it.mode_path} />
            )}
            {it.status === "cancelled" && <p className="text-sm text-muted-foreground">已取消，没有回答。</p>}
            {it.status === "failed" && <p className="text-sm text-destructive">执行失败（如模型服务调用最终失败），没有回答。</p>}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
