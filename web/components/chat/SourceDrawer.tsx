"use client";

// 对话里点“证据”“全文”或结论后的编号时，在右侧抽屉里展示，不离开对话。
// 抽屉里可以从证据切到全文（定位到那一处），也可以返回上一层；需要时在新页面打开。

import Link from "next/link";
import { ArrowLeftIcon, ExternalLinkIcon, XIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Drawer, DrawerClose, DrawerContent, DrawerDescription, DrawerHeader, DrawerTitle } from "@/components/ui/drawer";
import { EvidenceLoader } from "@/components/EvidenceView";
import { VersionFullText } from "@/components/VersionFullText";

export type SourceView = { kind: "evidence"; id: string } | { kind: "version"; id: string; focus?: string | null };

export function SourceDrawer({ stack, onPush, onBack, onClose }: {
  /** 抽屉里的浏览记录，最后一项是正在看的；空表示关闭 */
  stack: SourceView[];
  onPush: (v: SourceView) => void;
  onBack: () => void;
  onClose: () => void;
}) {
  const top = stack[stack.length - 1];
  const href = top?.kind === "evidence" ? `/evidence/${top.id}`
    : top ? `/versions/${top.id}${top.focus ? `?ev=${top.focus}` : ""}` : "#";
  return (
    // handleOnly：只能用关闭按钮或点遮罩关闭，免得在抽屉里选中文字时被拖动关掉
    <Drawer direction="right" handleOnly open={stack.length > 0} onOpenChange={(o) => { if (!o) onClose(); }}>
      <DrawerContent className="data-[vaul-drawer-direction=right]:w-full data-[vaul-drawer-direction=right]:sm:max-w-2xl">
        <DrawerHeader className="flex-row items-center gap-2 border-b p-3 text-left">
          {stack.length > 1 && (
            <Button size="icon-sm" variant="ghost" aria-label="返回上一层" onClick={onBack}><ArrowLeftIcon /></Button>
          )}
          <div className="min-w-0">
            <DrawerTitle>{top?.kind === "evidence" ? "证据" : "全文"}</DrawerTitle>
            <DrawerDescription className="truncate font-mono text-xs">
              {top?.kind === "evidence" ? top.id : top ? `版本 ${top.id}` : ""}
            </DrawerDescription>
          </div>
          <div className="ml-auto flex items-center gap-1">
            {top && (
              <Button size="sm" variant="ghost" asChild>
                <Link href={href} target="_blank" rel="noopener"><ExternalLinkIcon />新页面打开</Link>
              </Button>
            )}
            <DrawerClose asChild>
              <Button size="icon-sm" variant="ghost" aria-label="关闭"><XIcon /></Button>
            </DrawerClose>
          </div>
        </DrawerHeader>
        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {top?.kind === "evidence" && (
            <EvidenceLoader key={top.id} id={top.id}
              onOpenFullText={(versionId, evidenceId) => onPush({ kind: "version", id: versionId, focus: evidenceId })} />
          )}
          {top?.kind === "version" && (
            <VersionFullText key={`${top.id}:${top.focus ?? ""}`} versionId={top.id} focus={top.focus ?? null} />
          )}
        </div>
      </DrawerContent>
    </Drawer>
  );
}
