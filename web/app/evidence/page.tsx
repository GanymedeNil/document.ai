"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { SearchIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { PageTitle } from "@/components/common";

export default function EvidenceLookup() {
  const router = useRouter();
  const [id, setId] = useState("");
  return (
    <div className="grid gap-4">
      <PageTitle title="证据" description="也可以在回答或会话历史里点开引用。格式不对的 id 按不存在处理（404）。" />
      <Card size="sm">
        <CardContent>
          <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); if (id.trim()) router.push(`/evidence/${id.trim()}`); }}>
            <Input value={id} onChange={(e) => setId(e.target.value)} className="font-mono" aria-label="证据 id"
              placeholder="证据 id：ev_ 加 16 位十六进制，如 ev_0123456789abcdef" />
            <Button type="submit" variant="secondary"><SearchIcon />查看</Button>
          </form>
        </CardContent>
      </Card>
    </div>
  );
}
