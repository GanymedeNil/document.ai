"""公开评测集的下载（开发文档 14.1）：按固定的提交号从 Hugging Face 下载，核对 SHA-256，放在仓库之外。

    python -m eaer.evalrun.datasets fetch multihop-rag        # 12 MB，ODC-BY，可对外
    python -m eaer.evalrun.datasets fetch vidore-v3-hr        # 约 53 MB（PDF 原件与问题；不下载 440 MB 的页面图片），CC BY 4.0
    python -m eaer.evalrun.datasets fetch llm-aggrefact       # 测试集，CC BY-ND 4.0（只能原样评测）；需要先在 Hugging Face
                                                             # 上接受数据集条款，并设置 HF_TOKEN（只放在 .env，不打印）

目录：EAER_DATASETS_DIR（默认 ~/.cache/eaer/datasets）。下载的文件按 license.DATASET_FILES 登记的 SHA-256 判断来源；
新数据集或新提交号要先在那里登记，否则评测报告记为未登记来源、不能对外。
ViDoRe 的问题与文档清单经 Hugging Face 的 dataset viewer 接口取 JSON（不依赖 pyarrow），按问题编号排序后保存。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(os.environ.get("EAER_DATASETS_DIR", Path.home() / ".cache" / "eaer" / "datasets"))
SPECS = {
    "multihop-rag": {"repo": "yixuantt/MultiHopRAG", "revision": "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82",
                     "files": ["corpus.json", "MultiHopRAG.json"]},
    "vidore-v3-hr": {"repo": "vidore/vidore_v3_hr", "revision": "0cdf0979f2c5a0fd3e335e6373b9da48a9fe3bc3",
                     "pattern": "pdfs/", "rows": ["queries", "documents_metadata"]},
    # 只能原样评测（CC BY-ND 4.0）；没有 pyarrow，经 dataset viewer 接口取 JSON（需要 HF_TOKEN）
    "llm-aggrefact": {"repo": "lytang/LLM-AggreFact", "revision": "981dfd0bd8e58e7238a9ab92b2e6ea44bce918e4",
                      "files": [], "rows": ["default:test"], "gated": True},
}


def _get_json(url: str, headers: dict, attempts: int = 8) -> dict:
    """取一页；被限流（429）或服务暂时不可用时按 Retry-After 或指数退避重试。每页之间稍停，免得被限流。"""
    import time
    import urllib.error
    for k in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as r:
                page = json.load(r)
            time.sleep(0.5)
            return page
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or k == attempts - 1:
                raise
            wait = float(e.headers.get("Retry-After") or 0) or min(120.0, 5.0 * 2 ** k)
            print(f"被限流或暂时不可用（HTTP {e.code}），{wait:.0f} 秒后重试", file=sys.stderr, flush=True)
            time.sleep(wait)
    raise RuntimeError("不会到这里")


def _rows(repo: str, config: str, split: str = "test") -> list[dict]:
    """dataset viewer 接口的全部行，去掉图片列。设置了 HF_TOKEN 时带上（需要授权的数据集）。"""
    out, offset = [], 0
    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    while True:
        url = ("https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
            {"dataset": repo, "config": config, "split": split, "offset": offset, "length": 100}))
        page = _get_json(url, headers)
        rows = [x["row"] for x in page.get("rows", [])]
        out += [{k: v for k, v in row.items() if k != "image"} for row in rows]
        offset += len(rows)
        if not rows or offset >= page.get("num_rows_total", 0):
            return out


def fetch(name: str, root: Path = ROOT, log=print) -> list[Path]:
    from huggingface_hub import hf_hub_download, list_repo_files

    spec = SPECS[name]
    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    if spec.get("gated") and not os.environ.get("HF_TOKEN"):
        raise SystemExit(f"{name} 在 Hugging Face 上需要先接受数据集条款：登录后在 https://huggingface.co/datasets/{spec['repo']} "
                         "页面同意，再把访问令牌放进环境变量 HF_TOKEN")
    files = spec["files"] if "files" in spec else [
        f for f in list_repo_files(spec["repo"], repo_type="dataset", revision=spec["revision"])
        if f.startswith(spec["pattern"]) and f.endswith(".pdf")]
    saved = [Path(hf_hub_download(spec["repo"], f, repo_type="dataset", revision=spec["revision"], local_dir=target))
             for f in files]
    for item in spec.get("rows", []):
        config, _, split = item.partition(":")
        rows = sorted(_rows(spec["repo"], config, split or "test"),
                      key=lambda r: json.dumps(r, sort_keys=True, ensure_ascii=False))
        p = target / f"{split if split else config}.json"
        p.write_text(json.dumps(rows, ensure_ascii=False, sort_keys=True, indent=1), encoding="utf-8")
        saved.append(p)
    for p in saved:
        log(f"{p.relative_to(root)}  {p.stat().st_size // 1024} KB  {hashlib.sha256(p.read_bytes()).hexdigest()}")
    return saved


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eaer.evalrun.datasets", description="下载公开评测集")
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("name", choices=sorted(SPECS))
    args = p.parse_args(argv)
    fetch(args.name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
