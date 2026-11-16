"use client";

import { CircleAlertIcon } from "lucide-react";
import type { ApiResult } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Code } from "./common";

const HINT: Record<number, string> = {
  0: "前端连不上自己的代理，检查 web 服务是否在运行。",
  401: "缺少或无效的身份令牌：在右上角选择身份，或到“身份”页导入令牌表。",
  403: "身份没有该操作需要的角色（如上传、归档要 editor，审核要 reviewer，追加规则要 admin），或无法确定授权范围。",
  404: "不存在，或无权查看——接口对两者不作区分。",
  409: "当前状态不允许该操作。",
  429: "超过频率限制。",
  502: "模型服务调用失败（已按配置重试）。",
  503: "授权服务或提问服务不可用（接口拒绝请求，不会退化为放行）。",
};

export function ApiError({ result, title }: { result: ApiResult<unknown> | null; title?: string }) {
  if (!result || result.ok) return null;
  const e = result.error;
  return (
    <Alert variant="destructive" className="my-3">
      <CircleAlertIcon />
      <AlertTitle>
        {title ?? "请求失败"}（HTTP {result.status || "—"}）
      </AlertTitle>
      <AlertDescription>
        {e && (
          <p>
            <Code>{e.code}</Code> {e.message}
          </p>
        )}
        {HINT[result.status] && <p className="text-muted-foreground">{HINT[result.status]}</p>}
        {result.retryAfter && <p>请 {result.retryAfter} 秒后重试（Retry-After）。</p>}
        {result.raw && <pre className="max-h-32 overflow-auto text-xs whitespace-pre-wrap">{result.raw}</pre>}
        {(e?.request_id || result.requestId) && (
          <p className="text-xs text-muted-foreground">request_id：{e?.request_id || result.requestId}</p>
        )}
      </AlertDescription>
    </Alert>
  );
}
