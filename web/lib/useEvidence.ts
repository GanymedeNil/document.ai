"use client";

// 引用的证据详情（证据接口）。同一身份、同一证据只取一次；按身份分开缓存，切换身份后重新取，
// 以便看到权限差异（无权与不存在都是 404）。

import { useEffect, useState } from "react";
import { call, type ApiResult } from "./api";
import type { Evidence } from "./types";

const cache = new Map<string, Promise<ApiResult<Evidence>>>();

export function fetchEvidence(token: string | null, id: string, fresh = false): Promise<ApiResult<Evidence>> {
  const key = `${token ?? ""}|${id}`;
  if (fresh || !cache.has(key)) {
    const p = call<Evidence>(token, "GET", `/v1/evidence/${encodeURIComponent(id)}`);
    cache.set(key, p);
    // 失败的不缓存，下次重试
    p.then((r) => { if (!r.ok) cache.delete(key); });
  }
  return cache.get(key)!;
}

/** 文档归档、取消归档后，同一证据的结果会变，清掉缓存。 */
export function clearEvidenceCache(): void {
  cache.clear();
}

export function useEvidence(token: string | null, id: string | null): ApiResult<Evidence> | null {
  const [result, setResult] = useState<ApiResult<Evidence> | null>(null);
  useEffect(() => {
    let alive = true;
    setResult(null);
    if (!id) return;
    fetchEvidence(token, id).then((r) => { if (alive) setResult(r); });
    return () => { alive = false; };
  }, [token, id]);
  return result;
}
