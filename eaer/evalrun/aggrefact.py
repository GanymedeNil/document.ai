"""LLM-AggreFact 评测（开发文档 14.1：原样评测，CC BY-ND 4.0；AC-9-3 验证器的误通过率与误拒率）。

    HF_TOKEN=… python -m eaer.evalrun.datasets fetch llm-aggrefact      # 先在 Hugging Face 上接受数据集条款
    EAER_ENV=eval python -m eaer.evalrun.aggrefact [--per-dataset 20] [--out 报告.md]

每条数据是（文档，结论，标签 1 支持 / 0 不支持）。原样交给验证层的语义检查（eaer/verifier/semantic.py）：文档整篇作为
一条证据，结论原文作为结论，不改写、不截断、不增删数据（只能原样评测；按来源数据集各抽 --per-dataset 条、正反例各半是抽子集，
不算改编）。语义检查判 supported 记为“支持”，unsupported 与 unverified 记为“不支持”。报告：
- 均衡准确率（LLM-AggreFact 的通行指标，正反例召回的平均），按来源数据集与合计；
- 误通过率（标签为不支持、却判为支持）与误拒率（标签为支持、却没判为支持），与准入检查（AC-9-3）的口径一致；
- 判为 unverified 的比例（拿不准时标“未验证”是设计如此，计入误拒）；
- 不同置信度阈值下的合计指标（一次调用：记下模型原始判断与置信度，按语义检查的规则换算）。
数据来源按下载文件的 SHA-256 判断（license.DATASET_FILES）；对外报告时署名并注明原样使用。
"""
from __future__ import annotations

import argparse
import hashlib
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
    """每个来源数据集正反例各抽一半（标签约八成是“支持”，随机抽会让有的来源几乎没有反例，均衡准确率没有意义）。"""
    by: dict[tuple[str, int], list[dict]] = {}
    for r in rows:
        by.setdefault((r["dataset"], int(r["label"])), []).append(r)
    rnd = random.Random(seed)
    half = max(1, per_dataset // 2)
    return [x for key in sorted(by) for x in rnd.sample(by[key], min(half, len(by[key])))]


def run(model, rows: list[dict], log=lambda s: None) -> list[dict]:
    """逐条交给语义检查。检查器的阈值设为 0，记下模型原始的判断与置信度，再按产品的阈值换算（at_threshold），
    这样一次调用就能看不同阈值下的误通过率与误拒率（sweep）。"""
    from eaer.verifier.semantic import DEFAULT_MIN_CONFIDENCE, SemanticChecker

    checker, out = SemanticChecker(model, min_confidence=0.0), []
    for k, r in enumerate(rows):
        # 证据 id 用产品里的格式（ev_ 加 16 位十六进制），与产品的提示一致
        eid = "ev_" + hashlib.sha256(r["doc"].encode("utf-8")).hexdigest()[:16]
        try:
            v = checker.judge(r["claim"], [(eid, r["doc"])], date(2026, 1, 1))
            row = {"raw": v.status, "confidence": v.confidence,
                   "uncited": v.reason.startswith("模型判为支持但没有指出依据")}
            row["verdict"] = at_threshold(row, DEFAULT_MIN_CONFIDENCE)
        except Exception as e:  # noqa: BLE001 — 记下错误继续
            row = {"raw": None, "confidence": None, "uncited": False, "verdict": f"error: {type(e).__name__}"}
        out.append({"dataset": r["dataset"], "label": int(r["label"]), **row})
        if (k + 1) % 50 == 0:
            log(f"已判 {k + 1} / {len(rows)}")
    return out


def at_threshold(row: dict, t: float) -> str:
    """与 SemanticChecker 相同的规则：置信度低于阈值的 supported、unsupported 一律视为 unverified。"""
    if row["raw"] in ("supported", "unsupported") and row["confidence"] < t:
        return "unverified"
    return row["raw"]


SWEEP = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


def sweep(results: list[dict]) -> list[dict]:
    """各阈值下的合计指标（出错的条目不计）。"""
    ok = [r for r in results if r.get("raw")]
    return [{"threshold": t, **{k: v for k, v in metrics([dict(r, verdict=at_threshold(r, t)) for r in ok])["all"].items()
                                if k in ("balanced_accuracy", "false_pass", "false_reject")}} for t in SWEEP]


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
                "uncited": sum(bool(r.get("uncited")) for r in rs),
                "errors": sum(r["verdict"].startswith("error") for r in rs)}
    by: dict[str, list[dict]] = {}
    for r in results:
        by.setdefault(r["dataset"], []).append(r)
    return {"all": m(results), **{k: m(v) for k, v in sorted(by.items())}}


def report(s: dict, *, model: str, swept: list[dict] | None = None) -> str:
    lines = [f"LLM-AggreFact（原样评测）：验证层语义检查，生成 {model}；unverified 计为不支持", "",
             "| 来源 | 条数（正 / 反） | 均衡准确率 | 误通过率 | 误拒率 | 未验证 |", "| --- | --- | --- | --- | --- | --- |"]
    for k, v in s.items():
        lines.append(f"| {'合计' if k == 'all' else k} | {v['n']}（{v['positives']} / {v['negatives']}） | "
                     f"{v['balanced_accuracy']:.1%} | {v['false_pass']:.1%} | {v['false_reject']:.1%} | {v['unverified']:.1%} |")
    if s["all"]["uncited"]:
        lines.append(f"\n判为支持却没有指出有效的证据 id、因此降为“未验证”的：{s['all']['uncited']} 条。")
    if swept:
        lines += ["", "不同置信度阈值下（合计；产品默认 0.7，部署时由准入检查在标定集上选定）：", "",
                  "| 阈值 | 均衡准确率 | 误通过率 | 误拒率 |", "| --- | --- | --- | --- |"]
        lines += [f"| {w['threshold']} | {w['balanced_accuracy']:.1%} | {w['false_pass']:.1%} | {w['false_reject']:.1%} |" for w in swept]
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
    return emit(report(metrics(results), model=model.gen.model, swept=sweep(results)), prov, external=args.external, out=args.out)


if __name__ == "__main__":
    sys.exit(main())
