"""评测可复现检查（开发文档 14.5、AC-14-5）：同一模型、索引、评测集版本连续评测两次，指标差异在随机波动范围内。

    # 连续跑两次并比对（参数与 python -m eaer.evalrun.fast 相同）
    EAER_ENV=eval python -m eaer.evalrun.repro run 事实表.xlsx --admin-url postgresql://… --engine [--split 测试 --per-type 5]
    # 比对已有的两次结果（python -m eaer.evalrun.fast … --json-out a.json）
    python -m eaer.evalrun.repro compare a.json b.json

比对规则：
1. 先核对两次的运行配置完全相同——生成模型与采样参数（温度、推理开关）、向量化与重排模型、索引实现、评测集文件的
   SHA-256、集合与抽样方式、文档格式、是否经编排器。配置不同不是“复现”，直接报错，不比指标；
2. 两次回答的是同一批查询，逐条配对：一次对、一次错的为不一致对。用 McNemar 精确检验（不一致对按 1:1 的二项分布）
   判断正确率的差异是否超出随机波动：双侧 p ≥ α（默认 0.05）为在随机波动范围内；
3. 另报告逐条结果一致率、回答原文完全相同的比例、模式路径一致率，以及耗时 P50 / P95 的差异（耗时受机器负载影响，
   只列出，不作为判定依据）。
生成温度设为 0（EAER_GEN_TEMPERATURE=0）时两次通常完全相同；线上服务仍可能有少量不确定性，这正是本检查要量化的。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
from pathlib import Path

GOOD = ("correct", "refused_ok")
SCORED = ("correct", "wrong", "refused_ok", "leaked", "error")
ALPHA = 0.05


def run_config(xlsx: str | Path, prov, *, engine: bool, model: str, embedder: str, reranker: str | None, index: str,
               fmt: str, split: str | None, per_type: int | None, types: set[str] | None = None) -> dict:
    """一次评测的运行配置：复现检查要求两次完全相同。"""
    return {"engine": engine, "model": model,
            "temperature": os.environ.get("EAER_GEN_TEMPERATURE"), "thinking": os.environ.get("EAER_GEN_THINKING"),
            "embedder": embedder, "reranker": reranker, "index": index, "format": fmt,
            "dataset": {"name": prov.dataset, "modified": prov.modified, "detail": prov.detail,
                        "sha256": hashlib.sha256(Path(xlsx).read_bytes()).hexdigest()},
            "split": split, "per_type": per_type, **({"types": sorted(types)} if types else {})}


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar 精确检验的双侧 p 值：b、c 为两个方向的不一致对数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


class ConfigMismatch(Exception):
    pass


def _p(xs: list[float], q: float) -> float | None:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else None


def compare(a: dict, b: dict, *, alpha: float = ALPHA) -> dict:
    if a["config"] != b["config"]:
        diff = {k: (a["config"].get(k), b["config"].get(k)) for k in sorted(set(a["config"]) | set(b["config"]))
                if a["config"].get(k) != b["config"].get(k)}
        raise ConfigMismatch(f"两次评测的运行配置不同，不能比较复现性：{diff}")
    ra = {r["query_id"]: r for r in a["results"] if r["outcome"] in SCORED}
    rb = {r["query_id"]: r for r in b["results"] if r["outcome"] in SCORED}
    if set(ra) != set(rb):
        raise ConfigMismatch(f"两次评测的可计分查询不同：只在第一次的 {sorted(set(ra) - set(rb))[:5]}，"
                             f"只在第二次的 {sorted(set(rb) - set(ra))[:5]}")
    ids = sorted(ra)
    good_a = {i for i in ids if ra[i]["outcome"] in GOOD}
    good_b = {i for i in ids if rb[i]["outcome"] in GOOD}
    lost, gained = sorted(good_a - good_b), sorted(good_b - good_a)
    p = mcnemar_exact(len(lost), len(gained))
    secs = lambda rs: [r["seconds"] for r in rs.values() if r["outcome"] != "error"]  # noqa: E731
    return {
        "config": a["config"], "n": len(ids),
        "correct": [len(good_a), len(good_b)],
        "accuracy": [round(len(good_a) / len(ids), 4) if ids else None, round(len(good_b) / len(ids), 4) if ids else None],
        "discordant": {"first_only": lost, "second_only": gained}, "p_value": round(p, 4), "alpha": alpha,
        "within_noise": p >= alpha,
        "same_outcome": sum(ra[i]["outcome"] == rb[i]["outcome"] for i in ids),
        "same_answer": sum(ra[i]["answer"] == rb[i]["answer"] for i in ids),
        "same_mode_path": sum(ra[i].get("mode_path") == rb[i].get("mode_path") for i in ids),
        "seconds_p50": [statistics.median(secs(ra)) if secs(ra) else None, statistics.median(secs(rb)) if secs(rb) else None],
        "seconds_p95": [_p(secs(ra), 0.95), _p(secs(rb), 0.95)],
    }


def report(c: dict) -> str:
    cfg, n = c["config"], c["n"]
    lines = [f"复现检查（AC-14-5）：{'编排器' if cfg['engine'] else 'Fast'}；生成 {cfg['model']}（温度 {cfg['temperature']}，"
             f"推理 {cfg['thinking']}）；向量化 {cfg['embedder']}；重排 {cfg['reranker']}；索引 {cfg['index']}；"
             f"评测集 SHA-256 {cfg['dataset']['sha256'][:12]}…（{cfg['split'] or '全部'}，每类 {cfg['per_type'] or '全部'}）",
             f"{n} 个可计分查询：两次正确 {c['correct'][0]} / {c['correct'][1]}；不一致对 {len(c['discordant']['first_only'])} + "
             f"{len(c['discordant']['second_only'])}；McNemar 精确检验 p = {c['p_value']}（α = {c['alpha']}）",
             f"逐条结果相同 {c['same_outcome']} / {n}；回答原文相同 {c['same_answer']} / {n}；模式路径相同 {c['same_mode_path']} / {n}",
             f"耗时 P50 {_fmt(c['seconds_p50'])} 秒，P95 {_fmt(c['seconds_p95'])} 秒（只列出，不作判定）"]
    if c["discordant"]["first_only"] or c["discordant"]["second_only"]:
        lines.append(f"只在第一次答对：{'、'.join(c['discordant']['first_only']) or '无'}；"
                     f"只在第二次答对：{'、'.join(c['discordant']['second_only']) or '无'}")
    lines.append("结论：差异在随机波动范围内" if c["within_noise"] else "结论：差异超出随机波动范围，评测不可复现，需要排查")
    return "\n".join(lines)


def _fmt(pair) -> str:
    return " / ".join("—" if x is None else f"{x:.2f}" for x in pair)


def _emit(c: dict) -> int:
    from eaer.evalrun.license import Provenance, emit
    d = c["config"]["dataset"]
    code = emit(report(c), [Provenance(d["name"], d["modified"], d["detail"])], external=False)
    return code or (0 if c["within_noise"] else 1)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eaer.evalrun.repro", description="评测可复现检查（AC-14-5）")
    sub = p.add_subparsers(dest="command", required=True)
    cp = sub.add_parser("compare", help="比对两次评测的 JSON 结果")
    cp.add_argument("first")
    cp.add_argument("second")
    cp.add_argument("--alpha", type=float, default=ALPHA)
    rp = sub.add_parser("run", help="连续评测两次并比对；其余参数原样交给 python -m eaer.evalrun.fast")
    rp.add_argument("--alpha", type=float, default=ALPHA)
    rp.add_argument("--keep", help="把两次结果保存到这个目录")
    args, rest = p.parse_known_args(argv)
    try:
        if args.command == "compare":
            if rest:
                p.error(f"无法识别的参数：{rest}")
            a, b = (json.loads(Path(f).read_text(encoding="utf-8")) for f in (args.first, args.second))
            return _emit(compare(a, b, alpha=args.alpha))
        from eaer.evalrun import fast
        out = Path(args.keep) if args.keep else Path(tempfile.mkdtemp(prefix="eaer-repro-"))
        out.mkdir(parents=True, exist_ok=True)
        files = [out / "first.json", out / "second.json"]
        for k, f in enumerate(files, 1):
            print(f"—— 第 {k} 次评测 ——", file=sys.stderr)
            code = fast.main([*rest, "--json-out", str(f)])
            if code:
                return code
        a, b = (json.loads(f.read_text(encoding="utf-8")) for f in files)
        return _emit(compare(a, b, alpha=args.alpha))
    except ConfigMismatch as e:
        print(str(e), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
