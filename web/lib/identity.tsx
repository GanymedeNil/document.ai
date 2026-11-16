"use client";

// 演示用的身份管理：令牌保存在本浏览器的 localStorage，随时切换，便于对比不同身份看到的结果。
// 令牌表的格式与接口进程的 EAER_TOKENS_FILE 相同：{"令牌": {"user", "tenant", "roles", "principals"}}。
// 这里记下的 roles、principals 只用于显示，是否有权以接口的判断为准。

import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";
import { load, save } from "./storage";

export interface Identity {
  id: string;
  label: string;
  token: string;
  user?: string;
  tenant?: string;
  roles?: string[];
  principals?: string[];
}

interface IdentityState {
  ready: boolean;
  identities: Identity[];
  active: Identity | null;
  setActive: (id: string) => void;
  add: (i: Omit<Identity, "id">) => void;
  remove: (id: string) => void;
  importTokens: (json: string) => number;
}

const Ctx = createContext<IdentityState | null>(null);
const K_LIST = "eaer.identities";
const K_ACTIVE = "eaer.activeIdentity";

function newId(): string {
  return Math.random().toString(36).slice(2, 10);
}

/** 解析令牌表（EAER_TOKENS_FILE 的格式）；格式不对时抛出带中文说明的错误。 */
export function parseTokensFile(json: string): Omit<Identity, "id">[] {
  let raw: unknown;
  try {
    raw = JSON.parse(json);
  } catch {
    throw new Error("不是合法的 JSON");
  }
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) throw new Error("应为对象：{\"令牌\": {\"user\", \"tenant\", \"roles\"}}");
  return Object.entries(raw as Record<string, Record<string, unknown>>).map(([token, v]) => {
    const user = typeof v?.user === "string" ? v.user : undefined;
    const tenant = typeof v?.tenant === "string" ? v.tenant : undefined;
    const roles = Array.isArray(v?.roles) ? (v.roles as string[]) : [];
    const principals = Array.isArray(v?.principals) ? (v.principals as string[]) : [];
    const label = `${user ?? "未知用户"}@${tenant ?? "?"}`;
    return { label, token, user, tenant, roles, principals };
  });
}

/** 某个身份的本地记录（会话历史等）用的键：同一用户换令牌后仍能看到自己的会话。 */
export function identityKey(i: Identity | null): string {
  if (!i) return "anonymous";
  return i.user && i.tenant ? `${i.tenant}/${i.user}` : `token:${i.token.slice(0, 8)}`;
}

export function IdentityProvider({ children }: { children: ReactNode }) {
  const [ready, setReady] = useState(false);
  const [identities, setIdentities] = useState<Identity[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);

  useEffect(() => {
    setIdentities(load<Identity[]>(K_LIST, []));
    setActiveId(load<string | null>(K_ACTIVE, null));
    setReady(true);
  }, []);

  const persist = useCallback((list: Identity[], active: string | null) => {
    setIdentities(list);
    setActiveId(active);
    save(K_LIST, list);
    save(K_ACTIVE, active);
  }, []);

  const value = useMemo<IdentityState>(() => {
    const active = identities.find((i) => i.id === activeId) ?? identities[0] ?? null;
    return {
      ready,
      identities,
      active,
      setActive: (id) => persist(identities, id),
      add: (i) => {
        const item = { ...i, id: newId() };
        persist([...identities, item], active ? active.id : item.id);
      },
      remove: (id) => {
        const list = identities.filter((i) => i.id !== id);
        persist(list, activeId === id ? (list[0]?.id ?? null) : activeId);
      },
      importTokens: (json) => {
        const parsed = parseTokensFile(json);
        // 同一令牌再次导入时更新，不重复
        const byToken = new Map(identities.map((i) => [i.token, i]));
        for (const p of parsed) byToken.set(p.token, { ...p, id: byToken.get(p.token)?.id ?? newId() });
        const list = [...byToken.values()];
        persist(list, active?.id ?? list[0]?.id ?? null);
        return parsed.length;
      },
    };
  }, [ready, identities, activeId, persist]);

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useIdentity(): IdentityState {
  const v = useContext(Ctx);
  if (!v) throw new Error("useIdentity 必须在 IdentityProvider 内使用");
  return v;
}
