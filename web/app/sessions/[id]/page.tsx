"use client";

// 会话历史（GET /v1/sessions/{id}）：每次提问的回答与引用记录；进行中的查询显示进度、可以取消。

import Link from "next/link";
import { useParams } from "next/navigation";
import { ArrowLeftIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { PageTitle } from "@/components/common";
import { SessionView } from "@/components/SessionView";

export default function SessionPage() {
  const { id } = useParams<{ id: string }>();
  return (
    <div>
      <Button variant="link" size="sm" asChild className="mb-2 px-0">
        <Link href="/sessions"><ArrowLeftIcon />会话列表</Link>
      </Button>
      <PageTitle title="会话历史" description="引用记录在文档归档后仍保留，点开引用得到归档提示；被判为不支持的结论不返回。" />
      <SessionView sessionId={id} />
    </div>
  );
}
