import type { NextConfig } from "next";

// 接口进程的地址。浏览器只访问本服务的 /eaer-api/*，由 rewrites 代理到接口进程，因此不需要改后端的 CORS。
// 注意：next build 时 rewrites 被写进构建产物，生产构建要在 build 之前设置 EAER_API_URL。
const apiUrl = (process.env.EAER_API_URL ?? "http://127.0.0.1:8000").replace(/\/+$/, "");

const nextConfig: NextConfig = {
  reactStrictMode: true,
  // 不压缩代理的响应，以免缓冲流式提问（SSE）
  compress: false,
  async rewrites() {
    return [{ source: "/eaer-api/:path*", destination: `${apiUrl}/:path*` }];
  },
  experimental: {
    // 提问的同步时限默认 30 秒，加上模型调用的余量；代理默认的超时会先于接口断开
    proxyTimeout: 120_000,
  },
};

export default nextConfig;
