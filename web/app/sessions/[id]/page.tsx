import { redirect } from "next/navigation";

// 会话历史已并入对话页：/sessions/{id} 跳到 /?session={id}
export default async function SessionPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  redirect(`/?session=${encodeURIComponent(id)}`);
}
