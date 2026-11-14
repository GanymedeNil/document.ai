"""评测数据集的许可与对外报告拦截（开发文档 14 数据集表、AC-14-4 硬性）。

每份评测报告都带数据集标记：用了哪个数据集、许可、使用范围；来自仅内部数据集的结果另加醒目的“仅内部”标记。
生成对外报告（评测运行器的 --external）时按许可拦截，违规数应为 0：
- 仅内部开发使用的数据集（HoH、VersionQA、CHaystack）：结果不对外；
- 只能原样评测的数据集（LLM-AggreFact，CC BY-ND）：改编过（例如植入提示注入文字、改写问题）的结果不对外；
- 只借鉴任务设计与结构的数据集（RGB、DragonBall、CRUD-RAG）：不得使用其内容，出现即拦截；
- 来源未登记的数据：一律拦截（先在 DATASETS 中登记许可）。
可对外的结果附上所需的署名，自建合成集注明为合成数据。

数据来源由数据本身决定，而不是由运行命令的人声明：
1. 事实表里有“数据来源”表（列：数据集、是否改编、说明）的，按它；导入外部数据集的工具应写这张表；
2. 否则按文件的 SHA-256 查 FROZEN：冻结的评测集逐字节登记，文件改过一个字节就不再是登记过的那一份；
3. 都没有的，记为“未登记来源”。
"""
from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

EXTERNAL, AS_IS, INTERNAL, DESIGN_ONLY = "external", "external_as_is", "internal", "design_only"
SCOPE_TEXT = {EXTERNAL: "可对外展示", AS_IS: "原样评测可对外报告分数，改编后不可", INTERNAL: "仅内部开发使用，结果不对外",
              DESIGN_ONLY: "只借鉴任务设计与结构，不得使用其内容"}
UNREGISTERED = "未登记来源"


@dataclass(frozen=True)
class Dataset:
    name: str
    scope: str
    license: str
    attribution: str | None = None     # 对外报告必须带的署名或说明


DATASETS = {d.name: d for d in [
    Dataset("FRAMES", EXTERNAL, "Apache 2.0"),
    Dataset("MultiHop-RAG", EXTERNAL, "ODC-BY", "MultiHop-RAG（Tang & Yang, 2024），ODC-BY 许可"),
    Dataset("ViDoRe V3", EXTERNAL, "CC BY 4.0（各子集逐个确认）", "ViDoRe V3（Illuin Technology），CC BY 4.0，所用子集见正文"),
    Dataset("LLM-AggreFact", AS_IS, "CC BY-ND 4.0", "LLM-AggreFact（Tang et al., 2024），CC BY-ND 4.0，原样使用"),
    Dataset("HoH", INTERNAL, "仅内部"),
    Dataset("VersionQA", INTERNAL, "仅内部"),
    Dataset("CHaystack", INTERNAL, "仅内部"),
    Dataset("RGB", DESIGN_ONLY, "只借鉴设计"),
    Dataset("DragonBall", DESIGN_ONLY, "只借鉴设计"),
    Dataset("CRUD-RAG", DESIGN_ONLY, "只借鉴设计"),
    Dataset("自建合成集", EXTERNAL, "自有", "评测数据为 EAER 自建的合成数据，不是真实业务文档"),
]}

# 冻结的评测集：文件 SHA-256 → (数据集, 说明)。冻结新的评测集时在这里登记
FROZEN = {
    "036fcb2e1c7319860a78b48f6ffd0018c6938de059fedba637881d620e761231": ("自建合成集", "合成集 v1（docs/评测集/合成集-v1.xlsx）"),
    "dd53602d436179692f849fe0438628be75bead57796d4a9c67f0fb2b18fb1091": ("自建合成集", "合成集 v2（docs/评测集/合成集-v2.xlsx）"),
    "b9831df449c79d23e75582ecadedb05ee98b4de2b5821882904b1a5c6258ef30": ("自建合成集", "事实表样例（docs/事实表样例.xlsx）"),
}


# 下载的公开评测集文件：文件 SHA-256 → (数据集, 说明)。由 python -m eaer.evalrun.datasets 按固定提交号下载；
# 只有这里登记过的文件才按该数据集的许可对待，改过一个字节就是未登记来源
DATASET_FILES = {
    "20b61b5ab84de84a927420c5d265b7ec8d859ae49980699958a787ade9e4d28f":
        ("MultiHop-RAG", "corpus.json（yixuantt/MultiHopRAG@71ac0d0b）"),
    "03cfb4926461f868684903aadc8024447bdda5bb3f6804741424cce338515bff":
        ("MultiHop-RAG", "MultiHopRAG.json（yixuantt/MultiHopRAG@71ac0d0b）"),
    # LLM-AggreFact 测试集：经 dataset viewer 接口取得、排序后保存（只能原样评测）
    "9a6f04ee9f6154fc5364fef734c16c450588063f4808d9d7035bb398610afcd4":
        ("LLM-AggreFact", "test.json（lytang/LLM-AggreFact@981dfd0b，29,320 条）"),
    # ViDoRe V3 HR 子集：PDF 原件按固定提交号下载；问题与文档清单经 dataset viewer 接口取得、排序后保存（eaer/evalrun/datasets.py）
    "685ecfda8fa0c1ab537d8120a5dd1f04c703bde671de3c7104a992fd8edf26e0":
        ("ViDoRe V3", "pdfs/a_demographic_perspective_on_the_future_of_european-KJ0125152ENN.pdf（vidore/vidore_v3_hr@0cdf0979）"),
    "139828daea6050681122a19927aa1edf63df92ccb1b6db8db23c07ba2c7a2aef":
        ("ViDoRe V3", "pdfs/access_for_domestic_workers_to_labour_and_social_protection-KE0124002ENN.pdf（vidore/vidore_v3_hr@0cdf0979）"),
    "0342cfe69b002600232528032684562d3fdbcab4a603daeb5629c4e7d1d6fdb3":
        ("ViDoRe V3", "pdfs/annual_report_on_intra-eu_labour_mobility-KE0125029ENN.pdf（vidore/vidore_v3_hr@0cdf0979）"),
    "1b347c2b832740be3abf136eeedc6f7bbacbf162c33c0227fa611cbfb9e31615":
        ("ViDoRe V3", "pdfs/employment_and_social_developments_in_europe-KE0125067ENN.pdf（vidore/vidore_v3_hr@0cdf0979）"),
    "07151bec7bcdbedcf7b7efed07953cbd05e21f8a124d365d2edac195866c5b62":
        ("ViDoRe V3", "pdfs/employment_and_social_developments_in_europe_2024-KEBD24002ENN.pdf（vidore/vidore_v3_hr@0cdf0979）"),
    "0ea898b80b584e64a0d1c19ec5dc49873281e500a53f27c6d603584ed1cb9106":
        ("ViDoRe V3", "pdfs/estimating_labour_market_transitions_and_skills_investment-KE0125066ENN.pdf（vidore/vidore_v3_hr@0cdf0979）"),
    "df8794d0bf1f00124e66a705f91915e824d4077ec0ac9edfb550608b1c08337f":
        ("ViDoRe V3", "pdfs/european_network_of_public_employment_services_annual-KE0124024ENN.pdf（vidore/vidore_v3_hr@0cdf0979）"),
    "5bf0319a97dbeb45b6941547912ceb19198841bd0f75d57a8a569ff4fe770965":
        ("ViDoRe V3", "pdfs/future-oriented_occupations_in_the_eu-KE0125043ENN.pdf（vidore/vidore_v3_hr@0cdf0979）"),
    "e88b437140a6407e03bfcbd1c01f3fbb4d1f3540c953e678c0d8ed59ca39692e":
        ("ViDoRe V3", "pdfs/joint_employment_report_2025-KE0125018ENN.pdf（vidore/vidore_v3_hr@0cdf0979）"),
    "078435e3a5d09510ea9cc9f057b2bb231107b79628dbbc754ee68af4b37459a5":
        ("ViDoRe V3", "pdfs/labour_market_and_wage_developments_in_europe-KEBN24001ENN.pdf（vidore/vidore_v3_hr@0cdf0979）"),
    "b9e64a44d1c13a38e13c2c7cb836537073353c787a83026cf822d49746e4e899":
        ("ViDoRe V3", "pdfs/posting_of_workers-KE0224850ENN.pdf（vidore/vidore_v3_hr@0cdf0979）"),
    "412ea0cd6a31a0fa3cade6952249583a3c6549fffc936b0690670994c62ccff7":
        ("ViDoRe V3", "pdfs/recruitment_working_conditions_career_development-KI0124000ENN.pdf（vidore/vidore_v3_hr@0cdf0979）"),
    "1d2cfad34e3730be47abed627d0c97c706345a44324a5baee7ce350d9ed522d3":
        ("ViDoRe V3", "pdfs/research_perspectives_on_the_future_of_work-KI0124087ENN.pdf（vidore/vidore_v3_hr@0cdf0979）"),
    "77cd118bcda7694be496ced12a80c9f456f7131ecae5a4d0dbb94f3fe8c67f46":
        ("ViDoRe V3", "pdfs/undeclared_care_work_in_the_eu-TJ0125004ENN.pdf（vidore/vidore_v3_hr@0cdf0979）"),
    "ddd1ca18f5904fcfb567bf46f9203d4f9de8475ab59acd5be3101e67a11c8e06":
        ("ViDoRe V3", "queries.json（vidore/vidore_v3_hr@0cdf0979）"),
    "53698836493ea81817a6231e95df72401b4689956b70dda4eeebd62ac5bb4280":
        ("ViDoRe V3", "documents_metadata.json（vidore/vidore_v3_hr@0cdf0979）"),
}


@dataclass(frozen=True)
class Provenance:
    dataset: str                       # DATASETS 中的名字，或 UNREGISTERED
    modified: bool = False             # 是否改编过（植入文字、改写问题等）；抽取子集不算改编
    detail: str = ""


class LicenseViolation(Exception):
    def __init__(self, problems: list[str]):
        super().__init__("；".join(problems))
        self.problems = problems


def provenance_of(path: str | Path) -> Provenance:
    """事实表的数据来源：先看“数据来源”表，再按冻结评测集的 SHA-256，都没有则为未登记来源。"""
    path = Path(path)
    declared = _declared(path)
    if declared is not None:
        return declared
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest in FROZEN:
        name, detail = FROZEN[digest]
        return Provenance(name, False, detail)
    return Provenance(UNREGISTERED, False, f"{path.name}（SHA-256 {digest[:12]}…，不是登记过的冻结评测集，也没有“数据来源”表）")


def _declared(path: Path) -> Provenance | None:
    from openpyxl import load_workbook

    try:
        wb = load_workbook(path, read_only=True)
    except Exception:  # noqa: BLE001 — 不是 Excel 文件就按 SHA-256 判断
        return None
    if "数据来源" not in wb.sheetnames:
        return None
    rows = [r for r in wb["数据来源"].iter_rows(values_only=True) if any(c is not None for c in r)]
    head = [str(c).strip() if c is not None else "" for c in rows[0]] if rows else []
    if not rows[1:] or "数据集" not in head:
        return Provenance(UNREGISTERED, False, f"{path.name} 的“数据来源”表没有填写数据集")
    vals = dict(zip(head, rows[1]))
    name = str(vals.get("数据集") or "").strip()
    modified = str(vals.get("是否改编") or "").strip() in ("是", "true", "True", "1")
    return Provenance(name if name in DATASETS else UNREGISTERED, modified,
                      str(vals.get("说明") or "") if name in DATASETS else f"“数据来源”表写的“{name}”未登记许可")


def provenance_of_files(paths: list[str | Path]) -> Provenance:
    """下载的公开评测集：全部文件都在 DATASET_FILES 里、且属于同一个数据集时按它的许可，否则为未登记来源。"""
    found = []
    for p in paths:
        digest = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        if digest not in DATASET_FILES:
            return Provenance(UNREGISTERED, False, f"{Path(p).name}（SHA-256 {digest[:12]}…，不是登记过的下载文件）")
        found.append(DATASET_FILES[digest])
    names = {n for n, _ in found}
    if len(names) != 1:
        return Provenance(UNREGISTERED, False, f"文件分属不同的数据集：{sorted(names)}")
    return Provenance(names.pop(), False, "；".join(d for _, d in found))


def external_problems(provs: list[Provenance]) -> list[str]:
    """对外报告的违规项；空表示允许对外。"""
    out = []
    for p in provs:
        d = DATASETS.get(p.dataset)
        if d is None:
            out.append(f"数据来源未登记：{p.detail or p.dataset}。先在 eaer/evalrun/license.py 登记其许可")
        elif d.scope == INTERNAL:
            out.append(f"{d.name} 仅内部开发使用，结果不对外")
        elif d.scope == DESIGN_ONLY:
            out.append(f"{d.name} 只借鉴任务设计与结构，不得使用其内容")
        elif d.scope == AS_IS and p.modified:
            out.append(f"{d.name} 只能原样评测（{d.license}），改编后的结果不对外")
    return out


def stamp(text: str, provs: list[Provenance], *, external: bool) -> str:
    """给报告加数据集标记；external 时先按许可检查，违规抛 LicenseViolation。"""
    if external:
        problems = external_problems(provs)
        if problems:
            raise LicenseViolation(problems)
    lines = []
    for p in provs:
        d = DATASETS.get(p.dataset)
        scope = SCOPE_TEXT[d.scope] if d else "许可未登记，不可对外"
        lic = f"许可 {d.license}；" if d else ""
        lines.append(f"数据集：{p.dataset}{'（改编）' if p.modified else ''}（{lic}{scope}）" + (f"——{p.detail}" if p.detail else ""))
    if not external and external_problems(provs):
        lines.insert(0, "【仅内部】以下结果不得用于对外材料：" + "；".join(external_problems(provs)))
    out = "\n".join(lines) + "\n\n" + text
    if external:
        notes = [DATASETS[p.dataset].attribution for p in provs if DATASETS[p.dataset].attribution]
        out += "\n\n数据集与许可：\n" + "\n".join(f"- {n}" for n in dict.fromkeys(notes)) if notes else ""
    return out


def refuse_early(provs: list[Provenance], *, external: bool) -> int | None:
    """对外报告注定被拦截时，在跑评测之前就停下；返回退出码 3，可以继续时返回 None。"""
    problems = external_problems(provs) if external else []
    if problems:
        print("对外报告被拦截（AC-14-4），评测未运行：\n" + "\n".join(f"- {x}" for x in problems), file=sys.stderr)
        return 3
    return None


def add_external_flag(parser) -> None:
    parser.add_argument("--external", action="store_true",
                        help="生成对外报告：按数据集许可检查，不允许对外的数据集直接拦截（AC-14-4）")


def emit(text: str, provs: list[Provenance], *, external: bool, out: str | None = None) -> int:
    """评测运行器输出报告的唯一出口：打印（并可写文件）带标记的报告；对外报告违规时不输出，返回 3。"""
    try:
        stamped = stamp(text, provs, external=external)
    except LicenseViolation as e:
        print("对外报告被拦截（AC-14-4）：\n" + "\n".join(f"- {x}" for x in e.problems), file=sys.stderr)
        return 3
    print(stamped)
    if out:
        Path(out).write_text(stamped, encoding="utf-8")
    return 0
