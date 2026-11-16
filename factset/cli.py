"""命令行入口。

    python -m factset check  事实表.xlsx              校验并比对人工答案，有问题时退出码为 1
    python -m factset export 事实表.xlsx -o 评测集.jsonl  校验通过后导出评测集
    python -m factset render 事实表.xlsx -o 文档目录 [--format docx|html]  按事实表生成文档
    python -m factset verify-docs 事实表.xlsx 文档目录    检查生成或改写后的文档
    python -m factset synth -o 合成集.xlsx [--seed 2026] [--scale 1.0]   生成合成评测集（14.2、14.3）
    python -m factset sample 事实表.xlsx -o 抽检单.xlsx [--rate 0.05]     按类别分层抽取人工抽检单（14.2 第 6 步）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import date

from .answer import Calculator, compare
from .render import Renderer
from .verify_docs import verify
from .load import LoadError, load
from .validate import validate


def _json_default(o):
    if isinstance(o, date):
        return o.isoformat()
    raise TypeError(type(o))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="factset", description="EAER 合成评测集事实表工具")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="校验事实表，并用计算结果比对人工填写的预期答案")
    c.add_argument("xlsx")
    e = sub.add_parser("export", help="校验通过后导出评测集（JSON Lines）")
    e.add_argument("xlsx")
    e.add_argument("-o", "--output", required=True)
    r = sub.add_parser("render", help="按事实表为每个版本生成一份文档和清单 manifest.json")
    r.add_argument("xlsx")
    r.add_argument("-o", "--output", required=True)
    r.add_argument("--format", choices=["docx", "html"], default="docx")
    vd = sub.add_parser("verify-docs", help="检查文档是否包含全部事实、日期写法是否符合规则")
    vd.add_argument("xlsx")
    vd.add_argument("directory")
    sy = sub.add_parser("synth", help="生成合成评测集事实表（标准答案已计算并回填）")
    sy.add_argument("-o", "--output", required=True)
    sy.add_argument("--seed", type=int, default=2026)
    sy.add_argument("--scale", type=float, default=1.0, help="样本量比例，1.0 为开发文档 14.3 的样本量")
    sa = sub.add_parser("sample", help="按类别分层抽取人工抽检单")
    sa.add_argument("xlsx")
    sa.add_argument("-o", "--output", required=True)
    sa.add_argument("--rate", type=float, default=0.05)
    args = p.parse_args(argv)

    if args.cmd == "synth":
        from .synth import GENERATOR_VERSION, generate
        from .write import write
        write(generate(args.seed, args.scale), args.output)
        fs = load(args.output)
        issues = validate(fs)
        calc = Calculator(fs)
        bad = sum(bool(compare(q, calc.compute(q))) for q in fs.queries)
        print(f"已生成 {args.output}（生成器版本 {GENERATOR_VERSION}，种子 {args.seed}，比例 {args.scale}）："
              f"{len(fs.documents)} 份文档、{len(fs.versions)} 个版本、{len(fs.facts)} 条事实、{len(fs.events)} 个事件、"
              f"{len(fs.queries)} 个查询；校验问题 {len(issues)} 个，答案不一致 {bad} 个")
        return 1 if issues or bad else 0

    try:
        fs = load(args.xlsx)
    except LoadError as err:
        print(f"无法读取：{err}", file=sys.stderr)
        return 2
    issues = validate(fs)
    if issues:
        print(f"校验未通过，共 {len(issues)} 个问题：")
        for i in issues:
            print(f"  {i}")
        return 1
    print(f"校验通过：{len(fs.documents)} 份文档、{len(fs.versions)} 个版本、"
          f"{len(fs.facts)} 条事实、{len(fs.events)} 个事件、{len(fs.queries)} 个查询")

    if args.cmd == "render":
        m = Renderer(fs).render(Path(args.output), args.format)
        print(f"已生成 {len(m['versions'])} 份文档和 manifest.json 到 {args.output}")
        return 0
    if args.cmd == "verify-docs":
        errors, notes = verify(fs, Path(args.directory))
        for i in errors:
            print(f"  ✗ {i}")
        for i in notes:
            print(f"  · {i}")
        print(f"文档检查完成：{len(errors)} 个错误，{len(notes)} 个提示")
        return 1 if errors else 0

    calc = Calculator(fs)
    if args.cmd == "sample":
        from .spotcheck import spot_check
        n = spot_check(fs, calc, Path(args.output), args.rate)
        print(f"已抽取 {n} 条到 {args.output}，请人工逐条核对“核对结果”一列")
        return 0
    results = [(q, calc.compute(q)) for q in fs.queries]

    if args.cmd == "check":
        bad = 0
        for q, exp in results:
            diffs = compare(q, exp)
            mark = "✓" if not diffs else "✗"
            print(f"  {mark} {q.id} {q.type}：{exp.answer}")
            for d in diffs:
                print(f"      - {d}")
            bad += bool(diffs)
        print(f"比对完成：{len(results) - bad} 个一致，{bad} 个不一致")
        return 1 if bad else 0

    with open(args.output, "w", encoding="utf-8") as out:
        for q, exp in results:
            rec = {
                "query_id": q.id, "type": q.type, "question": q.question,
                "role": q.role, "asked_on": q.asked_on,
                "as_of_valid": q.as_of_valid, "as_of_system": q.as_of_system,
                "target_key": q.target_key, "expected_mode": q.expected_mode,
                "expected": exp.to_dict(),
                "manual_answer": q.expected_answer,
                "manual_matches": not compare(q, exp),
            }
            out.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
    print(f"已导出 {len(results)} 条到 {args.output}")
    return 0
