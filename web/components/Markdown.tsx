"use client";

// 把模型给出的文字（如思考过程）按 Markdown 显示。react-markdown 默认不渲染原始 HTML：思考里可能引用证据原文，
// 原文里的 HTML 只当文字显示，不会在页面里执行；链接一律在新窗口打开、不带来源页。

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";

export function Markdown({ children, className }: { children: string; className?: string }) {
  return (
    <div className={cn("md", className)}>
      <ReactMarkdown remarkPlugins={[remarkGfm]}
        components={{ a: ({ node: _node, ...props }) => <a {...props} target="_blank" rel="noopener noreferrer" /> }}>
        {children}
      </ReactMarkdown>
    </div>
  );
}
