"""在应用侧做中文分词（分词器候选之一，开发文档 3.3“中文分词”）。

OpenSearch 没有可用于 3.x 的 jieba 插件，所以 jieba 走另一条路：写入索引前和检索前都在 Python 里切好词，
用空格连接，索引里的 text 字段改用内置的 whitespace 分词器。好处是不依赖引擎插件、可以随代码维护行业词典；
代价是写入与查询两处必须用同一个分词器和同一份词典，换词典要重建索引（与换引擎分词器相同，见 5.3）。

jieba 0.42.1 是 2020 年 1 月的版本，在 Python 3.14 上导入时会对源码里的无效转义给出 SyntaxWarning，
这里在导入时屏蔽；Python 以后的版本会把它变成错误，正式采用前需要评估替代（如基于 jieba-rs 的绑定）。
"""
from __future__ import annotations

import logging
import re
import threading
import warnings
from pathlib import Path
from typing import Protocol

_KEEP = re.compile(r"[0-9a-z一-鿿]")


class Segmenter(Protocol):
    name: str
    def segment(self, text: str) -> list[str]: ...


class JiebaSegmenter:
    """jieba 搜索引擎模式：长词再切出其中的短词（“部门经理”→ 部门、经理、部门经理），兼顾召回与精度。"""

    def __init__(self, user_dict: str | Path | None = None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            import jieba
        self._jieba = jieba.Tokenizer()
        jieba.setLogLevel(logging.WARNING)
        self._lock = threading.Lock()
        self._ready = False
        self._user_dict = Path(user_dict) if user_dict else None
        self.name = "jieba-0.42.1" + (f"+{self._user_dict.name}" if self._user_dict else "")

    def _init(self) -> None:
        with self._lock:
            if not self._ready:
                self._jieba.initialize()
                if self._user_dict:
                    self._jieba.load_userdict(str(self._user_dict))
                self._ready = True

    def segment(self, text: str) -> list[str]:
        if not self._ready:
            self._init()
        out = []
        for w in self._jieba.cut_for_search(text.lower()):
            w = w.strip()
            if w and _KEEP.search(w):
                out.append(w)
        return out
