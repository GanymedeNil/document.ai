"""Qwen3-Embedding 与 Qwen3-Reranker 的推理（按两个模型官方说明的用法实现）。

- Embedding：左侧填充，取最后一个 token 的隐状态，L2 归一化。查询的指令前缀由调用方（EAER 客户端）加，
  这里对输入一视同仁，所以文档与查询走同一个接口。
- 重排：Qwen3-Reranker 是生成式重排模型，用固定的对话模板把（指令，查询，文档）拼成提示，
  取最后一个位置上 “yes” 与 “no” 两个 token 的 logits，softmax 后 “yes” 的概率即相关度（0–1）。
torch 与 transformers 只在加载模型时导入，接口层可以在没有它们的环境里测试。
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

RERANK_INSTRUCTION = "Given a question about company policies, retrieve passages that answer the question"
RERANK_PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the "
                 "Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
RERANK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def rerank_pair(instruction: str, query: str, doc: str) -> str:
    return f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}"


def load(cls, repo: str, revision: str | None, **kw):
    """先只用本地缓存加载（离线部署不访问外网）；本地没有时再按固定的提交下载。"""
    try:
        return cls.from_pretrained(repo, revision=revision, local_files_only=True, **kw)
    except OSError:
        return cls.from_pretrained(repo, revision=revision, **kw)


def pick_device(requested: str = "auto") -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Encoded:
    vectors: list[list[float]]
    tokens: int


class QwenEmbedding:
    def __init__(self, repo: str, revision: str | None = None, *, device: str = "auto", max_length: int = 8192,
                 batch_size: int = 16):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.device = torch, pick_device(device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = load(AutoTokenizer, repo, revision, padding_side="left")
        self.model = load(AutoModel, repo, revision, dtype=dtype).to(self.device).eval()
        self.max_length, self.batch_size = max_length, batch_size
        self.dims = int(self.model.config.hidden_size)
        self.lock = threading.Lock()

    def encode(self, texts: list[str]) -> Encoded:
        torch = self.torch
        out: list[list[float]] = []
        tokens = 0
        with self.lock, torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                batch = self.tok(texts[i:i + self.batch_size], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
                tokens += int(batch["attention_mask"].sum())
                hidden = self.model(**batch).last_hidden_state
                vec = torch.nn.functional.normalize(hidden[:, -1].float(), p=2, dim=1)     # 左侧填充：最后一位即最后一个 token
                out += vec.cpu().tolist()
        return Encoded(out, tokens)


class QwenReranker:
    def __init__(self, repo: str, revision: str | None = None, *, device: str = "auto", max_length: int = 8192,
                 batch_size: int = 8, instruction: str = RERANK_INSTRUCTION):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch, self.device = torch, pick_device(device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.tok = load(AutoTokenizer, repo, revision, padding_side="left")
        self.model = load(AutoModelForCausalLM, repo, revision, dtype=dtype).to(self.device).eval()
        self.yes, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")
        self.prefix = self.tok.encode(RERANK_PREFIX, add_special_tokens=False)
        self.suffix = self.tok.encode(RERANK_SUFFIX, add_special_tokens=False)
        self.max_length, self.batch_size, self.instruction = max_length, batch_size, instruction
        self.lock = threading.Lock()

    def score(self, query: str, documents: list[str], instruction: str | None = None) -> tuple[list[float], int]:
        torch = self.torch
        pairs = [rerank_pair(instruction or self.instruction, query, d) for d in documents]
        scores: list[float] = []
        tokens = 0
        with self.lock, torch.inference_mode():
            for i in range(0, len(pairs), self.batch_size):
                enc = self.tok(pairs[i:i + self.batch_size], padding=False, truncation="longest_first",
                               return_attention_mask=False,
                               max_length=self.max_length - len(self.prefix) - len(self.suffix))
                enc["input_ids"] = [self.prefix + ids + self.suffix for ids in enc["input_ids"]]
                batch = self.tok.pad(enc, padding=True, return_tensors="pt").to(self.device)
                tokens += int(batch["attention_mask"].sum())
                logits = self.model(**batch).logits[:, -1, :]
                pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
                scores += torch.nn.functional.softmax(pair, dim=1)[:, 1].cpu().tolist()
        return scores, tokens
