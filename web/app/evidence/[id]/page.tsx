"use client";

import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useState } from "react";
import { ArrowLeftIcon, RefreshCwIcon } from "lucide-react";
import type { ApiResult } from "@/lib/api";
import { useIdentity } from "@/lib/identity";
import { fetchEvidence } from "@/lib/useEvidence";
import type { Evidence } from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Spinner } from "@/components/ui/spinner";
import { EvidenceCompare, EvidenceDetail } from "@/components/EvidenceView";
import { Mono, PageTitle } from "@/components/common";

export default function EvidencePage() {
  const { id } = useParams<{ id: string }>();
  const { active, ready } = useIdentity();
  const [r, setR] = useState<ApiResult<Evidence> | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (!ready) return;
    let alive = true;
    setR(null);
    fetchEvidence(active?.token ?? null, id, true).then((x) => { if (alive) setR(x); });
    return () => { alive = false; };
  }, [ready, active?.token, id, nonce]);

  return (
    <div className="grid gap-4">
      <div>
        <Button variant="link" size="sm" asChild className="px-0"><Link href="/evidence"><ArrowLeftIcon />证据</Link></Button>
      </div>
      <PageTitle title={<>证据 <Mono className="text-base">{id}</Mono></>} description={`当前身份：${active?.label ?? "未选"}`}>
        <Button variant="outline" size="sm" onClick={() => setNonce((n) => n + 1)}><RefreshCwIcon />刷新</Button>
      </PageTitle>
      {r ? <EvidenceDetail r={r} /> : <p className="flex items-center gap-2 text-sm text-muted-foreground"><Spinner />读取中…</p>}
      <EvidenceCompare id={id} />
    </div>
  );
}
