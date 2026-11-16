"use client";

// 会话列表。接口没有“列出会话”的操作，这里列的是当前身份在本浏览器里提问过的会话；
// 也可以输入会话 id 直接查看（别人的会话与不存在的会话一样返回 404）。

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { identityKey, useIdentity } from "@/lib/identity";
import { forgetSession, listSessions, type SessionRecord } from "@/lib/storage";
import { formatTime } from "@/lib/format";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Mono, PageTitle } from "@/components/common";

export default function SessionsPage() {
  const { active, ready } = useIdentity();
  const router = useRouter();
  const [list, setList] = useState<SessionRecord[]>([]);
  const [id, setId] = useState("");
  const who = identityKey(active);

  useEffect(() => {
    if (ready) setList(listSessions(who));
  }, [ready, who]);

  return (
    <div className="grid gap-4">
      <PageTitle title="会话" description="会话只属于发起它的用户，换成别的身份打开会得到 404。" />
      <Card size="sm">
        <CardContent>
          <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); if (id.trim()) router.push(`/sessions/${id.trim()}`); }}>
            <Input value={id} onChange={(e) => setId(e.target.value)} placeholder="按会话 id（UUID）查看" className="font-mono" aria-label="会话 id" />
            <Button type="submit" variant="secondary">查看</Button>
          </form>
        </CardContent>
      </Card>
      <Card>
        <CardHeader>
          <CardTitle>{active?.label ?? "（未选身份）"} 的会话</CardTitle>
          <CardDescription>接口没有列出会话的操作，这里只记录本浏览器里提问过的会话。</CardDescription>
        </CardHeader>
        <CardContent>
          {list.length === 0 ? (
            <p className="text-sm text-muted-foreground">还没有会话。到 <Link href="/" className="text-primary underline">提问</Link> 页问一个问题。</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow><TableHead>会话</TableHead><TableHead>第一个问题</TableHead><TableHead>最近的问题</TableHead><TableHead>最近提问</TableHead><TableHead /></TableRow>
              </TableHeader>
              <TableBody>
                {list.map((s) => (
                  <TableRow key={s.session_id}>
                    <TableCell><Link href={`/sessions/${s.session_id}`} className="text-primary hover:underline"><Mono>{s.session_id.slice(0, 8)}</Mono></Link></TableCell>
                    <TableCell className="whitespace-normal">{s.first_question}</TableCell>
                    <TableCell className="whitespace-normal">{s.last_question}</TableCell>
                    <TableCell className="text-xs text-muted-foreground">{formatTime(s.updated_at)}</TableCell>
                    <TableCell className="text-right">
                      <Button size="xs" variant="ghost" onClick={() => { forgetSession(who, s.session_id); setList(listSessions(who)); }}>
                        从列表移除
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
