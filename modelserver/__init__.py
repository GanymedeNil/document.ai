"""本地模型服务（参考实现）：Qwen3-Embedding 与 Qwen3-Reranker。

为 EAER 的 ModelClient（eaer/adapters/online.py）提供 Embedding 与重排两项能力（开发文档 12.1：Fast 必需），
接口与线上服务一致：
  POST /v1/embeddings   OpenAI 兼容：{model, input} → {data: [{index, embedding}], usage}
  POST /v1/rerank       常见重排协议：{model, query, documents, top_n} → {results: [{index, relevance_score}], usage}
  GET  /v1/models、GET /health

它是独立部署的进程，不属于 EAER 应用本身（模型推理库只在这里使用，见开发文档 3.2 第 3 条）。
权重来自 Qwen 官方仓库并固定到具体提交；在 Mac 上用 MPS，在 Linux 服务器上用 CUDA，没有 GPU 时用 CPU。
吞吐要求高的私有化部署可以换成 vLLM 等推理服务，只要提供同样的两个接口，EAER 侧不需要改动。

    python -m modelserver --port 8100 [--env-file .env]
"""
