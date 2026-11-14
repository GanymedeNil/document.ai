"""LLM-AggreFact 评测（开发文档 14.1：原样评测，CC BY-ND 4.0；AC-9-3 验证器的误通过率与误拒率）。

    HF_TOKEN=… python -m eaer.evalrun.datasets fetch llm-aggrefact      # 先在 Hugging Face 上接受数据集条款
    EAER_ENV=eval python -m eaer.evalrun.aggrefact [--per-dataset 20] [--out 报告.md]

每条数据是（文档，结论，标签 1 支持 / 0 不支持）。原样交给验证层的语义检查（eaer/verifier/semantic.py）：文档整篇作为
一条证据，结论原文作为结论，不改写、不截断、不增删数据（只能原样评测；按来源数据集各抽 --per-dataset 条是抽子集，
不算改编）。语义检查判 supported 记为“支持”，unsupported 与 unverified 记为“不支持”。报告：
- 均衡准确率（LLM-AggreFact 的通行指标，正反例召回的平均），按来源数据集与合计；
- 误通过率（标签为不支持、却判为支持）与误拒率（标签为支持、却没判为支持），与准入检查（AC-9-3）的口径一致；
- 判为 unverified 的比例（拿不准时标“未验证”是设计如此，计入误拒）。
数据来源按下载文件的 SHA-256 判断（license.DATASET_FILES）；对外报告时署名并注明原样使用。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import date
from pathlib import Path

DATASET_DIR = Path(os.environ.get("EAER_DATASETS_DIR", Path.home() / ".cache" / "eaer" / "datasets")) / "llm-aggrefact"
FIELDS = ("dataset", "doc", "claim", "label")


def load(root: Path = DATASET_DIR):
    from eaer.evalrun.license import provenance_of_files
    path = root / "test.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    missing = [f for f in FIELDS if rows and f not in rows[0]]
    if missing:
        raise SystemExit(f"数据缺少字段 {missing}（实际字段 {sorted(rows[0])}）：数据集格式变了，先核对再改代码")
    return rows, [provenance_of_files([path])]


def sample(rows: list[dict], per_dataset: int, seed: int = 7) -> list[dict]:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["dataset"], []).append(r)
    rnd = random.Random(seed)
    return [x for name in sorted(by) for x in rnd.sample(by[name], min(per_dataset, len(by[name])))]


def run(model, rows: list[dict], log=lambda s: None) -> list[dict]:
    from eaer.verifier.semantic import SemanticChecker

    checker, out = SemanticChecker(model), []
    for k, r in enumerate(rows):
        try:
            v = checker.judge(r["claim"], [("ev_doc", r["doc"])], date(2026, 1, 1))
            verdict = v.status
        except Exception as e:  # noqa: BLE001 — 记下错误继续
            verdict = f"error: {type(e).__name__}"
        out.append({"dataset": r["dataset"], "label": int(r["label"]), "verdict": verdict})
        if (k + 1) % 50 == 0:
            log(f"已判 {k + 1} / {len(rows)}")
    return out


def metrics(results: list[dict]) -> dict:
    def m(rs):
        pos = [r for r in rs if r["label"] == 1]
        neg = [r for r in rs if r["label"] == 0]
        tpr = sum(r["verdict"] == "supported" for r in pos) / max(1, len(pos))
        tnr = sum(r["verdict"] != "supported" for r in neg) / max(1, len(neg))
        return {"n": len(rs), "positives": len(pos), "negatives": len(neg),
                "balanced_accuracy": round((tpr + tnr) / 2, 4),
                "false_pass": round(1 - tnr, 4), "false_reject": round(1 - tpr, 4),
                "unverified": round(sum(r["verdict"] == "unverified" for r in rs) / max(1, len(rs)), 4),
                "errors": sum(r["verdict"].startswith("error") for r in rs)}
    by: dict[str, list[dict]] = {}
    for r in results:
        by.setdefault(r["dataset"], []).append(r)
    return {"all": m(results), **{k: m(v) for k, v in sorted(by.items())}}


def report(s: dict, *, model: str) -> str:
    lines = [f"LLM-AggreFact（原样评测）：验证层语义检查，生成 {model}；unverified 计为不支持", "",
             "| 来源 | 条数（正 / 反） | 均衡准确率 | 误通过率 | 误拒率 | 未验证 |", "| --- | --- | --- | --- | --- | --- |"]
    for k, v in s.items():
        lines.append(f"| {'合计' if k == 'all' else k} | {v['n']}（{v['positives']} / {v['negatives']}） | "
                     f"{v['balanced_accuracy']:.1%} | {v['false_pass']:.1%} | {v['false_reject']:.1%} | {v['unverified']:.1%} |")
    if s["all"]["errors"]:
        lines.append(f"\n模型调用出错 {s['all']['errors']} 次（计为不支持）。")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from eaer.adapters.online import OnlineModelClient
    from eaer.evalrun.license import add_external_flag, emit, refuse_early

    p = argparse.ArgumentParser(prog="eaer.evalrun.aggrefact", description="LLM-AggreFact：验证器原样评测")
    p.add_argument("--data", default=str(DATASET_DIR))
    p.add_argument("--per-dataset", type=int, default=20)
    p.add_argument("--out")
    add_external_flag(p)
    args = p.parse_args(argv)
    if os.environ.get("EAER_ENV") != "eval":
        print("只在评测环境运行（EAER_ENV=eval）", file=sys.stderr)
        return 2
    rows, prov = load(Path(args.data))
    if (code := refuse_early(prov, external=args.external)) is not None:
        return code
    model = OnlineModelClient.from_env()
    if model.gen is None:
        print("需要生成服务（EAER_GEN_*）", file=sys.stderr)
        return 2
    results = run(model, sample(rows, args.per_dataset), log=lambda s: print(s, file=sys.stderr, flush=True))
    if args.out:
        Path(args.out).with_suffix(".results.json").write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    return emit(report(metrics(results), model=model.gen.model), prov, external=args.external, out=args.out)


if __name__ == "__main__":
    sys.exit(main())
