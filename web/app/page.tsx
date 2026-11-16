"use client";

// 对话（POST /v1/query、GET /v1/sessions/{id}、POST /v1/sessions/{id}/cancel）：左边是当前身份在本浏览器里的会话，
// 右边是一问一答。回答前有“思考过程”（Reasoning）：经过的模式、后台执行的步骤、时间点怎么定的、核对结果。
// 超过同步时限转入后台的提问在对话里显示进度并可以取消。会话 id 由客户端生成（接口允许），失败或转入后台也知道去哪里看。

import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useRef, useState } from "react";
import { ArrowUpIcon, CalendarIcon, MessageSquarePlusIcon, SquareIcon, XIcon } from "lucide-react";
import { call, streamQuery, type ApiResult, type StreamEvent } from "@/lib/api";
import { identityKey, useIdentity } from "@/lib/identity";
import { forgetSession, listSessions, rememberSession, type SessionRecord } from "@/lib/storage";
import { formatTime, uuid } from "@/lib/format";
import type { AnswerPolicy, CancelResult, Mode, QueryRequest, QueryResponse, Session, SessionItem } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { NativeSelect, NativeSelectOption } from "@/components/ui/native-select";
import { Spinner } from "@/components/ui/spinner";
import { Textarea } from "@/components/ui/textarea";
import { ApiError } from "@/components/ApiError";
import { ToneBadge } from "@/components/common";
import { AssistantTurn, ErrorTurn, PendingTurn, UserBubble, type TurnExtra } from "@/components/chat/Turns";
import { SourceDrawer, type SourceView } from "@/components/chat/SourceDrawer";
import { cn } from "@/lib/utils";

const SAMPLES = [
  "一线城市的住宿报销上限是多少？",
  "2026 年 3 月 15 日，一线城市的住宿报销上限是多少？",
  "住宿报销上限从多少改成了多少？",
  "我在北京住一周最高能报销多少？",
];

/** 等接口返回的一次提问；sessionId 是它所属的会话：切到别的会话时不在那里显示 */
interface Pending { sessionId: string; question: string; startedAt: number; events: StreamEvent[] }
interface LocalError { question: string; result: ApiResult<unknown>; afterCount: number }

function ChatPage() {
  const params = useSearchParams();
  const router = useRouter();
  const { active, ready } = useIdentity();
  const token = active?.token ?? null;
  const who = identityKey(active);
  const urlSession = params.get("session");

  const [sessionId, setSessionId] = useState<string | null>(null);
  const sessionRef = useRef<string | null>(null);
  const [items, setItems] = useState<SessionItem[]>([]);
  const itemsRef = useRef<SessionItem[]>([]);
  const [loadErr, setLoadErr] = useState<ApiResult<Session> | null>(null);
  const [loading, setLoading] = useState(false);
  const [pending, setPending] = useState<Pending | null>(null);
  const [errors, setErrors] = useState<LocalError[]>([]);
  const [extra, setExtra] = useState<Record<string, TurnExtra>>({});
  const [sessions, setSessions] = useState<SessionRecord[]>([]);
  const [cancelling, setCancelling] = useState(false);
  const [cancelErr, setCancelErr] = useState<ApiResult<CancelResult> | null>(null);
  const [openId, setOpenId] = useState("");
  const [drawer, setDrawer] = useState<SourceView[]>([]);   // 抽屉里的浏览记录，空表示关闭
  const openSource = (v: SourceView) => setDrawer([v]);

  const [text, setText] = useState("");
  const [mode, setMode] = useState<"" | Mode>("");
  // 回答方式：空为租户默认；宽松可以按单价算合计、按常识归类，严格只认证据原文
  const [policy, setPolicy] = useState<"" | "lenient" | "strict">("");
  const [tenantStrict, setTenantStrict] = useState<boolean | null>(null);
  const [asOfValid, setAsOfValid] = useState("");
  const [asOfSystem, setAsOfSystem] = useState("");
  const [showTime, setShowTime] = useState(false);
  const [stream, setStream] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const prevWho = useRef<string | undefined>(undefined);

  const reset = (sid: string | null) => {
    sessionRef.current = sid;
    setSessionId(sid);
    itemsRef.current = [];
    setItems([]);
    setErrors([]);
    setLoadErr(null);
    setCancelErr(null);
  };

  const loadSession = useCallback(async (sid: string) => {
    const r = await call<Session>(token, "GET", `/v1/sessions/${encodeURIComponent(sid)}`);
    if (sessionRef.current !== sid) return r;              // 已切到别的会话
    if (r.ok) {
      // 在后台跑完的提问：用时按提问时刻到发现完成时估算
      const prev = new Map(itemsRef.current.map((i) => [i.query_id, i.status]));
      const finished = r.data!.items.filter((i) => prev.get(i.query_id) === "running" && i.status !== "running");
      if (finished.length) {
        setExtra((x) => {
          const y = { ...x };
          for (const i of finished) y[i.query_id] = { ...y[i.query_id], durationMs: y[i.query_id]?.durationMs ?? Date.now() - Date.parse(i.asked_at) };
          return y;
        });
      }
      itemsRef.current = r.data!.items;
      setItems(r.data!.items);
      setLoadErr(null);
    } else {
      setLoadErr(r);
    }
    return r;
  }, [token]);

  // 换身份：会话只属于发起它的用户，回到新对话
  useEffect(() => {
    if (!ready) return;
    if (prevWho.current !== undefined && prevWho.current !== who) {
      reset(null);
      router.replace("/", { scroll: false });
    }
    prevWho.current = who;
    setSessions(listSessions(who));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, who]);

  // 租户默认的回答方式，显示在“回答方式”的默认项上
  useEffect(() => {
    if (!ready || !token) { setTenantStrict(null); return; }
    call<AnswerPolicy>(token, "GET", "/v1/config/answer-policy").then((r) => setTenantStrict(r.ok ? r.data!.strict : null));
  }, [ready, token]);

  // 地址栏里的会话（点左边的会话、刷新页面、从别处链接过来）
  useEffect(() => {
    if (!ready || urlSession === sessionRef.current) return;
    reset(urlSession);
    if (urlSession) {
      setLoading(true);
      loadSession(urlSession).finally(() => setLoading(false));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, urlSession]);

  // 有后台进行中的提问时每 2 秒刷新进度
  useEffect(() => {
    if (!sessionId || !items.some((i) => i.status === "running")) return;
    const t = setTimeout(() => loadSession(sessionId), 2000);
    return () => clearTimeout(t);
  }, [items, sessionId, loadSession]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [items.length, errors.length, pending?.question, pending?.events.length]);

  const send = async (question?: string) => {
    const q = (question ?? text).trim();
    if (!q || pending || !active) return;
    const sid = sessionRef.current ?? uuid();
    if (!sessionRef.current) {
      reset(sid);
      router.replace(`/?session=${sid}`, { scroll: false });
    }
    const before = new Set(itemsRef.current.map((i) => i.query_id));
    const afterCount = itemsRef.current.length;
    const startedAt = Date.now();
    setText("");
    setPending({ sessionId: sid, question: q, startedAt, events: [] });
    const body: QueryRequest = {
      question: q, session_id: sid, as_of_valid: asOfValid || null, as_of_system: asOfSystem || null,
      mode_hint: mode || null, strict: policy ? policy === "strict" : null, stream,
    };
    abort.current = new AbortController();
    let r: ApiResult<QueryResponse>;
    try {
      r = stream
        ? await streamQuery(token, body, (ev) => setPending((p) => p && { ...p, events: [...p.events, ev] }), abort.current.signal)
        : await call<QueryResponse>(token, "POST", "/v1/query", { json: body, signal: abort.current.signal });
    } catch (e) {
      if ((e as Error).name !== "AbortError") throw e;
      // 停止等待：接口那边照常执行，回答出现在会话里
      setPending(null);
      rememberSession(who, sid, q);
      setSessions(listSessions(who));
      await loadSession(sid);
      return;
    }
    setPending(null);
    if (r.ok || r.status === 502) {                     // 502 时提问也记入了会话（status=failed）
      rememberSession(who, sid, q);
      setSessions(listSessions(who));
      const s = await loadSession(sid);
      if (s.ok) {
        const fresh = s.data!.items.filter((i) => !before.has(i.query_id));
        const mine = [...fresh].reverse().find((i) => i.question === q) ?? fresh[0];
        if (mine) {
          const done = r.ok && r.data!.status === "completed";
          setExtra((x) => ({
            ...x, [mine.query_id]: {
              notices: done ? r.data!.notices : undefined,
              durationMs: done ? Date.now() - startedAt : undefined,
              error: r.ok ? undefined : r.error?.message,
            },
          }));
        }
      }
    } else {
      setErrors((es) => [...es, { question: q, result: r, afterCount }]);
    }
  };

  const cancel = async () => {
    if (!sessionId) return;
    setCancelling(true);
    const r = await call<CancelResult>(token, "POST", `/v1/sessions/${encodeURIComponent(sessionId)}/cancel`);
    setCancelErr(r.ok ? null : r);
    setCancelling(false);
    await loadSession(sessionId);
  };

  const newChat = () => {
    abort.current?.abort();
    reset(null);
    router.replace("/", { scroll: false });
  };

  // 等待中的提问只在它所属的会话里显示（等的时候切到别的会话，那边不该出现它）
  const pendingHere = pending && pending.sessionId === sessionId ? pending : null;
  const empty = !items.length && !pendingHere && !errors.length && !loadErr && !loading;

  // 会话里的记录，加上没记进会话的本地失败（按发生时的位置插入）
  const thread: React.ReactNode[] = [];
  const errsAt = (n: number) => errors.filter((e) => e.afterCount === n);
  items.forEach((it, idx) => {
    errsAt(idx).forEach((e, j) => thread.push(
      <div key={`err-${idx}-${j}`} className="grid gap-3"><UserBubble text={e.question} /><ErrorTurn result={e.result} /></div>));
    thread.push(
      <div key={it.query_id} className="grid gap-3">
        <UserBubble text={it.question} at={it.asked_at} understood={it.standalone_question} />
        <AssistantTurn it={it} extra={extra[it.query_id]} onCancel={cancel} cancelling={cancelling} onOpen={openSource} />
      </div>);
  });
  errors.filter((e) => e.afterCount >= items.length).forEach((e, j) => thread.push(
    <div key={`err-end-${j}`} className="grid gap-3"><UserBubble text={e.question} /><ErrorTurn result={e.result} /></div>));

  return (
    <div className="grid h-[calc(100dvh-6.5rem)] grid-cols-1 gap-4 md:grid-cols-[15rem_1fr]">
      {/* 会话列表 */}
      <aside className="hidden min-h-0 flex-col gap-2 rounded-xl bg-card p-2 ring-1 ring-foreground/10 md:flex">
        <Button variant="outline" className="justify-start" onClick={newChat}><MessageSquarePlusIcon />新对话</Button>
        <div className="px-1 pt-1 text-xs text-muted-foreground">{active?.label ?? "未选身份"} 的会话</div>
        <div className="min-h-0 flex-1 overflow-y-auto">
          {sessions.length === 0 && <p className="px-2 py-1 text-xs text-muted-foreground">还没有会话。</p>}
          {sessions.map((s) => (
            <div key={s.session_id} className={cn("group flex items-center rounded-md hover:bg-muted",
              s.session_id === sessionId && "bg-muted")}>
              <Link href={`/?session=${s.session_id}`} className="min-w-0 flex-1 px-2 py-1.5">
                <div className="truncate text-sm">{s.first_question}</div>
                <div className="text-[11px] text-muted-foreground">{formatTime(s.updated_at)}</div>
              </Link>
              <Button size="icon-xs" variant="ghost" className="mr-1 opacity-0 group-hover:opacity-100" aria-label="从列表移除"
                onClick={() => { forgetSession(who, s.session_id); setSessions(listSessions(who)); }}>
                <XIcon />
              </Button>
            </div>
          ))}
        </div>
        <form className="flex gap-1" onSubmit={(e) => { e.preventDefault(); if (openId.trim()) { router.push(`/?session=${openId.trim()}`); setOpenId(""); } }}>
          <Input value={openId} onChange={(e) => setOpenId(e.target.value)} placeholder="按会话 id 打开" className="h-7 font-mono text-xs" aria-label="会话 id" />
          <Button type="submit" size="sm" variant="secondary">打开</Button>
        </form>
        <p className="px-1 text-[11px] leading-snug text-muted-foreground">接口没有列出会话的操作，这里只是本浏览器的记录；别人的会话打开得到 404。</p>
      </aside>

      {/* 对话 */}
      <section className="flex min-h-0 flex-col overflow-hidden rounded-xl bg-card ring-1 ring-foreground/10">
        <div className="flex items-center gap-2 border-b px-4 py-2 text-sm">
          <span className="font-medium">{sessionId ? "会话" : "新对话"}</span>
          {sessionId && <span className="font-mono text-xs text-muted-foreground">{sessionId}</span>}
          <Button size="xs" variant="ghost" className="ml-auto md:hidden" onClick={newChat}><MessageSquarePlusIcon />新对话</Button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          <div className="mx-auto grid max-w-3xl gap-6">
            {ready && !active && (
              <p className="text-sm text-muted-foreground">还没有身份。先到 <Link href="/identities" className="text-primary underline">身份</Link> 页导入令牌表。</p>
            )}
            {empty && active && (
              <div className="grid justify-items-center gap-4 pt-16 text-center">
                <div className="font-heading text-xl font-semibold">有什么想问的？</div>
                <p className="max-w-md text-sm text-muted-foreground">
                  每条结论都标明是否已验证，并附引用：哪份文档、哪个版本、哪一处，以及在目标时间是否有效。
                </p>
                <div className="flex flex-wrap justify-center gap-2">
                  {SAMPLES.map((s) => <Button key={s} size="sm" variant="outline" onClick={() => send(s)}>{s}</Button>)}
                </div>
              </div>
            )}
            {loading && <p className="flex items-center gap-2 text-sm text-muted-foreground"><Spinner />读取会话…</p>}
            <ApiError result={loadErr} title="读取会话失败" />
            {thread}
            {pendingHere && (
              <div className="grid gap-3">
                <UserBubble text={pendingHere.question} />
                <PendingTurn startedAt={pendingHere.startedAt} events={pendingHere.events} />
              </div>
            )}
            <ApiError result={cancelErr} title="取消失败" />
            <div ref={bottom} />
          </div>
        </div>

        {/* 输入框 */}
        <div className="border-t px-4 py-3">
          <div className="mx-auto grid max-w-3xl gap-2">
            <div className="rounded-xl border bg-background focus-within:ring-3 focus-within:ring-ring/50">
              <Textarea value={text} onChange={(e) => setText(e.target.value)} maxLength={2000} aria-label="问题"
                placeholder={active ? "输入问题，Enter 发送，Shift+Enter 换行" : "先选择身份"} disabled={!active}
                className="max-h-48 min-h-14 resize-none border-0 bg-transparent shadow-none focus-visible:ring-0 dark:bg-transparent"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); send(); }
                }} />
              <div className="flex flex-wrap items-center gap-2 px-2 pb-2">
                <NativeSelect size="sm" aria-label="模式" value={mode} onChange={(e) => setMode(e.target.value as "" | Mode)}>
                  <NativeSelectOption value="">自动模式</NativeSelectOption>
                  <NativeSelectOption value="fast">Fast</NativeSelectOption>
                  <NativeSelectOption value="deep">Deep</NativeSelectOption>
                  <NativeSelectOption value="verified">Verified</NativeSelectOption>
                </NativeSelect>
                <NativeSelect size="sm" aria-label="回答方式" value={policy} onChange={(e) => setPolicy(e.target.value as "" | "lenient" | "strict")}
                  title="宽松：可以按文件里的单价算合计、按常识判断对象属于哪一类（都标未验证）；严格：只认文件原文">
                  <NativeSelectOption value="">默认（{tenantStrict === null ? "租户设置" : tenantStrict ? "严格" : "宽松"}）</NativeSelectOption>
                  <NativeSelectOption value="lenient">宽松</NativeSelectOption>
                  <NativeSelectOption value="strict">严格</NativeSelectOption>
                </NativeSelect>
                <Button type="button" size="sm" variant={showTime || asOfValid || asOfSystem ? "secondary" : "ghost"} onClick={() => setShowTime((v) => !v)}>
                  <CalendarIcon />时间点
                </Button>
                <div className="flex items-center gap-1.5">
                  <Checkbox id="stream" checked={stream} onCheckedChange={(v) => setStream(v === true)} />
                  <Label htmlFor="stream" className="text-xs font-normal">流式</Label>
                </div>
                {(asOfValid || asOfSystem) && (
                  <ToneBadge tone="info">{asOfValid && `实际生效 ${asOfValid}`}{asOfValid && asOfSystem && " · "}{asOfSystem && `截至 ${asOfSystem} 所知`}</ToneBadge>
                )}
                <div className="ml-auto flex items-center gap-2">
                  {pending && !pendingHere && (
                    <span className="text-xs text-muted-foreground">另一个会话的提问还在等待回答</span>
                  )}
                  {pending ? (
                    <Button type="button" size="icon-sm" variant="outline" aria-label="停止等待" title="停止等待（接口照常执行）"
                      onClick={() => abort.current?.abort()}>
                      <SquareIcon />
                    </Button>
                  ) : (
                    <Button type="button" size="icon-sm" aria-label="发送" disabled={!text.trim() || !active} onClick={() => send()}>
                      <ArrowUpIcon />
                    </Button>
                  )}
                </div>
              </div>
              {showTime && (
                <div className="flex flex-wrap items-center gap-3 border-t px-3 py-2 text-xs">
                  <label className="flex items-center gap-1.5 text-muted-foreground">
                    某天实际生效
                    <Input type="date" className="h-7 w-36" value={asOfValid} onChange={(e) => setAsOfValid(e.target.value)} aria-label="某天实际生效（as_of_valid）" />
                  </label>
                  <label className="flex items-center gap-1.5 text-muted-foreground">
                    截至某天系统所知（回溯）
                    <Input type="date" className="h-7 w-36" value={asOfSystem} onChange={(e) => setAsOfSystem(e.target.value)} aria-label="截至某天系统所知（as_of_system）" />
                  </label>
                  {(asOfValid || asOfSystem) && (
                    <Button size="xs" variant="ghost" onClick={() => { setAsOfValid(""); setAsOfSystem(""); }}>清除</Button>
                  )}
                  <span className="text-muted-foreground">都不填时按问题里的时间点，没有则按当前生效。</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </section>

      <SourceDrawer stack={drawer} onPush={(v) => setDrawer((s) => [...s, v])}
        onBack={() => setDrawer((s) => s.slice(0, -1))} onClose={() => setDrawer([])} />
    </div>
  );
}

export default function Page() {
  return (
    <Suspense fallback={<p className="text-sm text-muted-foreground">加载中…</p>}>
      <ChatPage />
    </Suspense>
  );
}
