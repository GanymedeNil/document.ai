"""启动本地模型服务：python -m modelserver --port 8100 [--env-file .env]

环境变量（也可写在 --env-file 指定的文件里）：
  MODELSERVER_API_KEY        访问密钥（Bearer）；不设置时不校验，只应在仅本机可达时这样用
  MODELSERVER_EMBED_MODEL    默认 Qwen/Qwen3-Embedding-0.6B；MODELSERVER_EMBED_REVISION 固定提交
  MODELSERVER_RERANK_MODEL   默认 Qwen/Qwen3-Reranker-0.6B；MODELSERVER_RERANK_REVISION 固定提交
  MODELSERVER_DEVICE         auto（cuda > mps > cpu）/ cuda / mps / cpu
  MODELSERVER_MAX_LENGTH     单条输入的最大 token 数，默认 8192
  MODELSERVER_RERANK_INSTRUCTION  重排指令，默认面向“公司制度问答”的英文指令（模型说明建议用英文写指令）
模型名对外只取仓库名的最后一段（如 Qwen3-Embedding-0.6B），EAER 侧的 EAER_EMBED_MODEL 等填它。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

DEFAULTS = {
    "MODELSERVER_EMBED_MODEL": "Qwen/Qwen3-Embedding-0.6B",
    "MODELSERVER_EMBED_REVISION": "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
    "MODELSERVER_RERANK_MODEL": "Qwen/Qwen3-Reranker-0.6B",
    "MODELSERVER_RERANK_REVISION": "e61197ed45024b0ed8a2d74b80b4d909f1255473",
}


def load_env_file(path: Path) -> None:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":     # 允许加引号（便于 shell 读取含空格的值）
                v = v[1:-1]
            os.environ.setdefault(k.strip(), v)


def main() -> None:
    import uvicorn

    from modelserver.app import create_app
    from modelserver.backends import RERANK_INSTRUCTION, QwenEmbedding, QwenReranker

    p = argparse.ArgumentParser(prog="modelserver")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--env-file")
    p.add_argument("--no-embedding", action="store_true")
    p.add_argument("--no-rerank", action="store_true")
    args = p.parse_args()
    if args.env_file:
        load_env_file(Path(args.env_file))
    env = {**DEFAULTS, **{k: v for k, v in os.environ.items() if k.startswith("MODELSERVER_")}}
    device, max_len = env.get("MODELSERVER_DEVICE", "auto"), int(env.get("MODELSERVER_MAX_LENGTH", "8192"))
    emb = None if args.no_embedding else QwenEmbedding(
        env["MODELSERVER_EMBED_MODEL"], _full(env["MODELSERVER_EMBED_MODEL"], env["MODELSERVER_EMBED_REVISION"]),
        device=device, max_length=max_len)
    rr = None if args.no_rerank else QwenReranker(
        env["MODELSERVER_RERANK_MODEL"], _full(env["MODELSERVER_RERANK_MODEL"], env["MODELSERVER_RERANK_REVISION"]),
        device=device, max_length=max_len, instruction=env.get("MODELSERVER_RERANK_INSTRUCTION", RERANK_INSTRUCTION))
    app = create_app(emb, rr, embed_name=env["MODELSERVER_EMBED_MODEL"].rsplit("/", 1)[-1],
                     rerank_name=env["MODELSERVER_RERANK_MODEL"].rsplit("/", 1)[-1],
                     api_key=env.get("MODELSERVER_API_KEY"))
    uvicorn.run(app, host=args.host, port=args.port)


def _full(repo: str, revision: str) -> str:
    """把缩写的提交号补全（只查本地缓存，离线可用）；本地没有时原样返回，交给 transformers 下载。"""
    if len(revision) >= 40:
        return revision
    try:
        from huggingface_hub import scan_cache_dir
        for r in scan_cache_dir().repos:
            if r.repo_id == repo:
                for rev in r.revisions:
                    if rev.commit_hash.startswith(revision):
                        return rev.commit_hash
    except Exception:  # noqa: BLE001
        pass
    return revision


if __name__ == "__main__":
    main()
