// 本浏览器里的记录（localStorage）。接口没有“列出会话”“列出文档”的操作，这些列表只是本浏览器用过的 id，
// 换浏览器或清除站点数据后就没有了；以接口返回的为准。读写失败（隐私模式等）时退回默认值，不影响页面。

export function load<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    return raw === null ? fallback : (JSON.parse(raw) as T);
  } catch {
    return fallback;
  }
}

export function save(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    // 忽略：只是少了一条本地记录
  }
}

export interface SessionRecord {
  session_id: string;
  first_question: string;
  last_question: string;
  updated_at: string;
}

const sessionsKey = (who: string) => `eaer.sessions.${who}`;

export function listSessions(who: string): SessionRecord[] {
  return load<SessionRecord[]>(sessionsKey(who), []);
}

export function rememberSession(who: string, sessionId: string, question: string): void {
  const list = listSessions(who);
  const now = new Date().toISOString();
  const found = list.find((s) => s.session_id === sessionId);
  const rest = list.filter((s) => s.session_id !== sessionId);
  const rec: SessionRecord = found
    ? { ...found, last_question: question, updated_at: now }
    : { session_id: sessionId, first_question: question, last_question: question, updated_at: now };
  save(sessionsKey(who), [rec, ...rest].slice(0, 200));
}

export function forgetSession(who: string, sessionId: string): void {
  save(sessionsKey(who), listSessions(who).filter((s) => s.session_id !== sessionId));
}

export interface UploadRecord {
  document_id: string;
  version_id: string;
  filename: string;
  title: string | null;
  principals: string;
  duplicate: boolean;
  uploaded_by: string;
  at: string;
}

const K_UPLOADS = "eaer.uploads";

export function listUploads(): UploadRecord[] {
  return load<UploadRecord[]>(K_UPLOADS, []);
}

export function rememberUpload(rec: UploadRecord): void {
  save(K_UPLOADS, [rec, ...listUploads().filter((u) => u.version_id !== rec.version_id)].slice(0, 500));
}
