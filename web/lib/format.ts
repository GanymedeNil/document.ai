// 显示用的中文名称与格式化。

import type { Locator, Mode, ValidGrade } from "./types";

export const MODE_LABEL: Record<Mode, string> = { fast: "Fast", deep: "Deep", verified: "Verified" };

export const GRADE_LABEL: Record<ValidGrade, string> = {
  explicit: "明确（正文写明或审核补录）",
  inferred: "推断（取审核或上传时间）",
  unknown: "未知",
};

export const GRADE_SHORT: Record<ValidGrade, string> = { explicit: "明确", inferred: "推断", unknown: "未知" };

export const TIME_SOURCE_LABEL: Record<string, string> = {
  parameter: "接口参数",
  parsed: "从问题中识别",
  model: "模型识别",
  today: "未指定时间点（按当前生效）",
};

export const QUERY_STATUS_LABEL: Record<string, string> = {
  completed: "已完成",
  running: "进行中（后台）",
  cancelled: "已取消",
  failed: "失败",
};

export const TASK_STATUS_LABEL: Record<string, string> = { open: "待审核", approved: "已通过", deleted: "已删除" };

export const EXTRACTION_LABEL: Record<string, string> = {
  queued: "排队中",
  running: "抽取中",
  done: "已完成",
  failed: "失败",
};

export const DOC_TYPE_LABEL: Record<string, string> = { normative: "规范类", record: "记录类" };

const LOCATOR_KEY: Record<string, string> = {
  para: "段落",
  sentence: "句",
  page: "页",
  table: "表",
  row: "行",
  col: "列",
  column: "列",
  cell: "单元格",
  figure: "图",
  sheet: "工作表",
  slide: "幻灯片",
  point: "数据点",
};

export function formatLocator(loc: Locator | null | undefined): string {
  if (!loc || typeof loc !== "object") return "—";
  const parts = Object.entries(loc)
    .filter(([, v]) => v !== null && v !== undefined && v !== "")
    .map(([k, v]) => `${LOCATOR_KEY[k] ?? k} ${typeof v === "object" ? JSON.stringify(v) : String(v)}`);
  return parts.length ? parts.join(" · ") : "—";
}

export function formatTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("zh-CN", { hour12: false });
}

export function shortId(id: string | null | undefined): string {
  return id ? id.slice(0, 8) : "—";
}

export function uuid(): string {
  // 非安全上下文（如用局域网地址访问）没有 randomUUID
  const ru = (crypto as Partial<Crypto>).randomUUID;
  if (ru) return ru.call(crypto);
  const b = new Uint8Array(16);
  crypto.getRandomValues(b);
  b[6] = (b[6] & 0x0f) | 0x40;
  b[8] = (b[8] & 0x3f) | 0x80;
  const h = [...b].map((x) => x.toString(16).padStart(2, "0")).join("");
  return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
}
