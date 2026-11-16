"use client";

// 版本全文的独立页面（/versions/{id}?ev=证据 id）：审核页与直接链接用；对话里用抽屉展示同一个组件。

import { useParams, useRouter, useSearchParams } from "next/navigation";
import { Suspense } from "react";
import { ArrowLeftIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { VersionFullText } from "@/components/VersionFullText";

function VersionPage() {
  const { id } = useParams<{ id: string }>();
  const focus = useSearchParams().get("ev");
  const router = useRouter();
  return (
    <div>
      {/* 从审核页、证据页或别处点进来，返回原处；直接打开的链接没有上一页时回到对话 */}
      <Button variant="link" size="sm" className="mb-2 px-0"
        onClick={() => (window.history.length > 1 ? router.back() : router.push("/"))}>
        <ArrowLeftIcon />返回
      </Button>
      <VersionFullText versionId={id} focus={focus} />
    </div>
  );
}

export default function Page() {
  return (
    <Suspense fallback={<p className="text-sm text-muted-foreground">加载中…</p>}>
      <VersionPage />
    </Suspense>
  );
}
