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


def split_on_oom(run, items: list, device: str) -> list:
    """按批推理；这一批内存不足时释放缓存、对半拆开重试，拆到单条仍不够才把内存不足抛出去（由接口返回 503）。
    长段落或大批量把显存一下子撑满时，服务降速而不是失败。"""
    from modelserver.memguard import is_out_of_memory
    try:
        return run(items)
    except Exception as e:  # noqa: BLE001 — 只处理内存不足
        if not is_out_of_memory(e) or len(items) <= 1:
            raise
        _empty_cache(device)
        half = len(items) // 2
        return split_on_oom(run, items[:half], device) + split_on_oom(run, items[half:], device)


def _empty_cache(device: str) -> None:
    import torch
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


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
        tokens = [0]

        def run(chunk: list[str]) -> list[list[float]]:
            batch = self.tok(chunk, padding=True, truncation=True, max_length=self.max_length,
                             return_tensors="pt").to(self.device)
            tokens[0] += int(batch["attention_mask"].sum())
            hidden = self.model(**batch).last_hidden_state
            return torch.nn.functional.normalize(hidden[:, -1].float(), p=2, dim=1).cpu().tolist()   # 左侧填充

        with self.lock, torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                out += split_on_oom(run, texts[i:i + self.batch_size], self.device)
        return Encoded(out, tokens[0])


class QwenReranker:
    def __init__(self, repo: str, revision: str | None = None, *, device: str = "auto", max_length: int = 2048,
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
        counted = [0]

        def run(chunk: list[str]) -> list[float]:
            enc = self.tok(chunk, padding=False, truncation="longest_first", return_attention_mask=False,
                           max_length=self.max_length - len(self.prefix) - len(self.suffix))
            enc["input_ids"] = [self.prefix + ids + self.suffix for ids in enc["input_ids"]]
            batch = self.tok.pad(enc, padding=True, return_tensors="pt").to(self.device)
            counted[0] += int(batch["attention_mask"].sum())
            logits = self.model(**batch).logits[:, -1, :]
            pair = torch.stack([logits[:, self.no], logits[:, self.yes]], dim=1).float()
            return torch.nn.functional.softmax(pair, dim=1)[:, 1].cpu().tolist()

        with self.lock, torch.inference_mode():
            for i in range(0, len(pairs), self.batch_size):
                scores += split_on_oom(run, pairs[i:i + self.batch_size], self.device)
        return scores, tokens + counted[0]


# ---------------- Qwen3-VL-Embedding 与 Qwen3-VL-Reranker（多模态，按官方仓库 QwenLM/Qwen3-VL-Embedding 的用法） ----------------

VL_DEFAULT_INSTRUCTION = "Represent the user's input."
VL_RERANK_SYSTEM = ("Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
                    "Note that the answer can only be \"yes\" or \"no\".")


def split_instruction(text: str) -> tuple[str | None, str]:
    """EAER 客户端给查询加的“Instruct: …\\nQuery:…”前缀（Qwen3-Embedding 的写法）拆成（指令，正文）；
    Qwen3-VL-Embedding 把指令放在系统消息里。没有前缀的（文档）返回 (None, 原文)。"""
    if text.startswith("Instruct: ") and "\nQuery:" in text:
        head, body = text[len("Instruct: "):].split("\nQuery:", 1)
        return head.strip(), body.strip()
    return None, text


def _punct(instruction: str) -> str:
    import unicodedata
    instruction = instruction.strip()
    return instruction if not instruction or unicodedata.category(instruction[-1]).startswith("P") else instruction + "."


def vl_embedding_messages(text: str | None, images=(), instruction: str | None = None) -> list[dict]:
    """官方 format_model_input 的对话格式：系统消息是指令，用户消息依次是图片、文字。"""
    content = [{"type": "image", "image": img} for img in images]
    if text:
        content.append({"type": "text", "text": text})
    if not content:
        content.append({"type": "text", "text": "NULL"})
    return [{"role": "system", "content": [{"type": "text", "text": _punct(instruction or VL_DEFAULT_INSTRUCTION)}]},
            {"role": "user", "content": content}]


def vl_rerank_messages(instruction: str, query: str, doc: str) -> list[dict]:
    """官方 format_mm_instruction 的格式（纯文字的查询与文档）。"""
    return [{"role": "system", "content": [{"type": "text", "text": VL_RERANK_SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": "<Instruct>: " + instruction},
                                         {"type": "text", "text": "<Query>:"}, {"type": "text", "text": query},
                                         {"type": "text", "text": "\n<Document>:"}, {"type": "text", "text": doc}]}]


def _vl_dtype(torch, device: str):
    return torch.bfloat16 if device in ("cuda", "mps") else torch.float32


class QwenVLEmbedding:
    """Qwen3-VL-Embedding：右侧填充，取最后一个有效 token 的隐状态，L2 归一化（官方 _pooling_last）。
    输入可以是字符串，或 {"text": …, "images": [PIL.Image, …]}；查询的指令由 split_instruction 从前缀中取出。"""

    def __init__(self, repo: str, revision: str | None = None, *, device: str = "auto", max_length: int = 8192,
                 batch_size: int = 8):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch, self.device = torch, pick_device(device)
        self.processor = load(AutoProcessor, repo, revision, padding_side="right")
        self.model = load(AutoModel, repo, revision, dtype=_vl_dtype(torch, self.device)).to(self.device).eval()
        cfg = self.model.config
        self.dims = int(getattr(getattr(cfg, "text_config", cfg), "hidden_size"))
        self.max_length, self.batch_size = max_length, batch_size
        self.lock = threading.Lock()

    def encode(self, texts: list) -> Encoded:
        torch = self.torch
        out: list[list[float]] = []
        tokens = 0
        with self.lock, torch.inference_mode():
            for i in range(0, len(texts), self.batch_size):
                convs, images = [], []
                for item in texts[i:i + self.batch_size]:
                    raw, imgs = (item, []) if isinstance(item, str) else (item.get("text") or "", list(item.get("images") or []))
                    instruction, text = split_instruction(raw)
                    convs.append(vl_embedding_messages(text, imgs, instruction))
                    images += imgs
                prompt = self.processor.apply_chat_template(convs, add_generation_prompt=True, tokenize=False)
                batch = self.processor(text=prompt, images=images or None, padding=True, truncation=True,
                                       max_length=self.max_length, return_tensors="pt").to(self.device)
                mask = batch["attention_mask"]
                tokens += int(mask.sum())
                hidden = self.model(**batch).last_hidden_state
                last = mask.shape[1] - mask.flip(dims=[1]).argmax(dim=1) - 1          # 右侧填充：最后一个有效位置
                vec = hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float()
                out += torch.nn.functional.normalize(vec, p=2, dim=-1).cpu().tolist()
        return Encoded(out, tokens)


class QwenVLReranker:
    """Qwen3-VL-Reranker：左侧填充，最后一个位置的隐状态乘以 lm_head 中 “yes” 与 “no” 两行之差，sigmoid 即相关度
    （官方 compute_scores；与 yes/no 两个 logits 做 softmax 取 yes 的概率相同）。"""

    def __init__(self, repo: str, revision: str | None = None, *, device: str = "auto", max_length: int = 8192,
                 batch_size: int = 8, instruction: str = RERANK_INSTRUCTION):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.torch, self.device = torch, pick_device(device)
        dtype = _vl_dtype(torch, self.device)
        self.processor = load(AutoProcessor, repo, revision, padding_side="left")
        lm = load(Qwen3VLForConditionalGeneration, repo, revision, dtype=dtype)
        vocab = self.processor.tokenizer.get_vocab()
        w = lm.lm_head.weight.data
        self.direction = (w[vocab["yes"]] - w[vocab["no"]]).to(self.device, dtype)
        self.model = lm.model.to(self.device).eval()
        del lm
        self.max_length, self.batch_size, self.instruction = max_length, batch_size, instruction
        self.lock = threading.Lock()

    def score(self, query: str, documents: list[str], instruction: str | None = None) -> tuple[list[float], int]:
        torch = self.torch
        scores: list[float] = []
        tokens = 0
        with self.lock, torch.inference_mode():
            for i in range(0, len(documents), self.batch_size):
                convs = [vl_rerank_messages(_punct(instruction or self.instruction), query, d)
                         for d in documents[i:i + self.batch_size]]
                prompt = self.processor.apply_chat_template(convs, add_generation_prompt=True, tokenize=False)
                batch = self.processor(text=prompt, padding=True, return_tensors="pt").to(self.device)
                tokens += int(batch["attention_mask"].sum())
                h = self.model(**batch).last_hidden_state[:, -1]
                scores += torch.sigmoid((h * self.direction).sum(-1).float()).cpu().tolist()
        return scores, tokens
