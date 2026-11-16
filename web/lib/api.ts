// 调用对外接口。浏览器只访问本服务的 /eaer-api/*，由 next.config.ts 的 rewrites 代理到接口进程。
// HTTP 错误不抛异常：返回状态码与错误体（{code, message, request_id}），由页面原样展示。

import type { ApiErrorBody, QueryRequest, QueryResponse } from "./types";

export const API_BASE = "/eaer-api";

export interface ApiResult<T> {
  ok: boolean;
  status: number; // 0 表示没连上（接口进程或前端代理不可用）
  data: T | null;
  error: ApiErrorBody | null;
  requestId: string | null;
  retryAfter: string | null;
  raw?: string; // 非 JSON 响应的正文（如代理报错）
}

export interface CallOptions {
  json?: unknown;
  form?: FormData;
  signal?: AbortSignal;
}

function headers(token: string | null, extra?: Record<string, string>): Record<string, string> {
  const h: Record<string, string> = { ...(extra ?? {}) };
  if (token) h["Authorization"] = `Bearer ${token}`;
  return h;
}

export async function call<T>(token: string | null, method: string, path: string, opts: CallOptions = {}): Promise<ApiResult<T>> {
  let resp: Response;
  try {
    resp = await fetch(API_BASE + path, {
      method,
      headers: headers(token, opts.json !== undefined ? { "Content-Type": "application/json" } : undefined),
      body: opts.json !== undefined ? JSON.stringify(opts.json) : opts.form,
      signal: opts.signal,
      cache: "no-store",
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    return {
      ok: false, status: 0, data: null, requestId: null, retryAfter: null,
      error: { code: "network_error", message: `无法连接：${(e as Error).message}`, request_id: "" },
    };
  }
  return parse<T>(resp);
}

async function parse<T>(resp: Response): Promise<ApiResult<T>> {
  const requestId = resp.headers.get("X-Request-Id");
  const retryAfter = resp.headers.get("Retry-After");
  const text = await resp.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    // 非 JSON：多半是前端代理连不上接口进程
    return {
      ok: false, status: resp.status, data: null, requestId, retryAfter, raw: text.slice(0, 500),
      error: {
        code: "bad_gateway",
        message: resp.status >= 500 ? "接口进程没有响应（检查 EAER_API_URL 与接口进程是否在运行）" : "响应不是 JSON",
        request_id: requestId ?? "",
      },
    };
  }
  if (resp.ok) return { ok: true, status: resp.status, data: body as T, error: null, requestId, retryAfter };
  return { ok: false, status: resp.status, data: null, error: body as ApiErrorBody, requestId, retryAfter };
}

export interface StreamEvent {
  event: string;
  data: unknown;
}

/** 流式提问（SSE）：依次回调 mode、notice、delta、claim、done 事件；出错时（非 200）返回错误结果，不回调。 */
export async function streamQuery(
  token: string | null, body: QueryRequest, onEvent: (e: StreamEvent) => void, signal?: AbortSignal,
): Promise<ApiResult<QueryResponse>> {
  let resp: Response;
  try {
    resp = await fetch(API_BASE + "/v1/query", {
      method: "POST",
      headers: headers(token, { "Content-Type": "application/json", Accept: "text/event-stream" }),
      body: JSON.stringify({ ...body, stream: true }),
      signal,
      cache: "no-store",
    });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    return {
      ok: false, status: 0, data: null, requestId: null, retryAfter: null,
      error: { code: "network_error", message: `无法连接：${(e as Error).message}`, request_id: "" },
    };
  }
  const ctype = resp.headers.get("Content-Type") ?? "";
  if (!resp.ok || !ctype.includes("text/event-stream") || !resp.body) return parse<QueryResponse>(resp);

  const requestId = resp.headers.get("X-Request-Id");
  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let done: QueryResponse | null = null;
  for (;;) {
    const { value, done: end } = await reader.read();
    if (value) buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message";
      const dataLines: string[] = [];
      for (const line of chunk.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) dataLines.push(line.slice(5).trimStart());
      }
      if (!dataLines.length) continue;
      const data = JSON.parse(dataLines.join("\n"));
      onEvent({ event, data });
      if (event === "done") done = data as QueryResponse;
    }
    if (end) break;
  }
  if (done) return { ok: true, status: resp.status, data: done, error: null, requestId, retryAfter: null };
  return {
    ok: false, status: resp.status, data: null, requestId, retryAfter: null,
    error: { code: "stream_incomplete", message: "流式响应在 done 事件之前结束", request_id: requestId ?? "" },
  };
}
