"""合成评测集生成器（开发文档 14.2、14.3；AC-14-1、AC-14-2 程序部分、AC-14-3）。

- 生成的事实表通过校验，程序答案与回填的预期答案逐条一致；写成 xlsx 再读回完全不变；同一种子结果相同；
- 样本量：比例 1.0 时高风险类别各 300 条、其他类别各 100 条；每类 30% 为标定集、70% 为测试集；
- 每条查询确实落在它的类别要考的情形上（冲突类有冲突、无答案类拒答、归档与权限边界类应答与拒答都有……）；
- 约四成查询用改写问法（不含事实键原词），每类都有；
- 生成的文档（Word、HTML）通过文档检查；冻结的 docs/评测集/合成集-v2.xlsx 与生成器输出一致；
- 子集与人工抽检单的抽样确定、按类别分层。
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from factset.answer import Calculator, compare
from factset.load import load
from factset.subset import subset
from factset.synth import QUOTA, Synth, generate
from factset.validate import validate
from factset.write import write

FROZEN = Path("docs/评测集/合成集-v2.xlsx")


@pytest.fixture(scope="module")
def full():
    return generate(2026, 1.0)


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    path = tmp_path_factory.mktemp("synth") / "small.xlsx"
    write(generate(7, 0.1), path)
    return load(path)


def test_small_set_is_valid_and_answers_agree(small):
    assert validate(small) == []
    calc = Calculator(small)
    assert [q.id for q in small.queries if compare(q, calc.compute(q))] == []
    assert set(Counter(q.type for q in small.queries)) == set(QUOTA)


def test_full_scale_meets_sample_sizes_and_split(full):
    counts = Counter(q.type for q in full.queries)
    assert counts == Counter(QUOTA)                                    # 高风险 300、其他 100（14.3）
    for qtype in QUOTA:
        cal = sum(q.split == "标定" for q in full.queries if q.type == qtype)
        assert cal == round(QUOTA[qtype] * 0.3)
    assert len({q.id for q in full.queries}) == len(full.queries)
    sigs = [(q.question, q.role, q.asked_on, q.as_of_valid, q.as_of_system) for q in full.queries]
    assert len(set(sigs)) == len(sigs)                                 # 没有重复的提问


def test_each_query_tests_what_its_category_is_about(full):
    calc = Calculator(full)
    exp = {q.id: calc.compute(q) for q in full.queries}
    by = lambda t: [exp[q.id] for q in full.queries if q.type == t]  # noqa: E731
    assert all(e.conflict for e in by("冲突"))
    assert all(e.refuse for e in by("无答案"))
    for t in ("当前生效", "版本差异", "图表", "验证器"):
        assert not any(e.refuse or e.conflict for e in by(t))
    for t in ("归档", "权限边界"):                                     # 应答与拒答都要有
        refusals = sum(e.refuse for e in by(t))
        assert 0.2 * QUOTA[t] < refusals < 0.8 * QUOTA[t]
    assert Counter(e.answer for e in by("验证器")).keys() == {"支持", "不支持"}
    assert sum(e.grade == "inferred" for e in by("回溯")) > 0           # 回溯能看出补录前后的差别
    assert any(e.unverified for e in by("冲突")) and any(not e.unverified for e in by("冲突"))
    assert all(f.carrier == "图表" for q in full.queries if q.type == "图表"
               for f in (full.facts[i] for i in exp[q.id].basis))
    events = Counter(e.type for e in full.events)
    assert {"归档", "取消归档", "权限变更", "有效期更正"} <= set(events)


def test_round_trip_and_determinism(full, tmp_path):
    path = tmp_path / "s.xlsx"
    write(full, path)
    back = load(path)
    calc_a, calc_b = Calculator(full), Calculator(back)
    assert [calc_a.compute(q).to_dict() for q in full.queries] == [calc_b.compute(q).to_dict() for q in back.queries]
    again = generate(2026, 1.0)
    assert [(q.question, q.expected_answer) for q in again.queries] == [(q.question, q.expected_answer) for q in full.queries]
    assert generate(2027, 1.0).queries[0].question != full.queries[0].question or \
        [q.question for q in generate(2027, 1.0).queries] != [q.question for q in full.queries]


def test_paraphrased_questions(full):
    from factset.synth import ALIASES, TOPICS
    keys = [k for _, _, ks in TOPICS for k, *_ in ks]
    assert set(ALIASES) == set(keys)
    assert not [a for k, al in ALIASES.items() for a in al if k in a]       # 改写问法不含原词
    para = [q for q in full.queries if q.note == "改写问法"]
    assert 0.3 * len(full.queries) < len(para) < 0.5 * len(full.queries)
    assert all(q.target_key not in q.question for q in para)
    assert {q.type for q in para} >= {"历史时点", "回溯", "冲突", "归档", "权限边界", "当前生效", "版本差异", "验证器"}


def test_frozen_set_matches_the_generator(full):
    """冻结的评测集版本不能被生成器的改动悄悄改掉：改了生成器就要升版本、另存新文件。"""
    frozen = load(FROZEN)
    assert validate(frozen) == []
    assert [(q.id, q.type, q.question, q.expected_answer, q.split, q.note) for q in frozen.queries] == \
        [(q.id, q.type, q.question, q.expected_answer, q.split, q.note) for q in full.queries]
    assert {f.id: (f.key, f.value) for f in frozen.facts.values()} == {f.id: (f.key, f.value) for f in full.facts.values()}


def test_rendered_documents_pass_the_checks(small, tmp_path):
    from factset.render import Renderer
    from factset.verify_docs import verify

    Renderer(small).render(tmp_path / "docs", "html")
    errors, _ = verify(small, tmp_path / "docs")
    assert errors == []


def test_subset_and_spot_check(full, tmp_path):
    test = subset(full, split="测试")
    assert len(test.queries) == 1400 and all(q.split == "测试" for q in test.queries)
    some = subset(full, split="测试", per_type=5)
    assert Counter(q.type for q in some.queries) == Counter({t: 5 for t in QUOTA})
    assert [q.id for q in some.queries] == [q.id for q in subset(full, split="测试", per_type=5).queries]
    assert some.facts is full.facts and some.events == full.events

    from openpyxl import load_workbook

    from factset.spotcheck import spot_check
    n = spot_check(full, Calculator(full), tmp_path / "s.xlsx", 0.05)
    rows = list(load_workbook(tmp_path / "s.xlsx").active.iter_rows(values_only=True))[1:]
    assert n == len(rows) == sum(max(2, -(-q // 20)) for q in QUOTA.values())
    assert Counter(r[1] for r in rows) == Counter({t: max(2, -(-q // 20)) for t, q in QUOTA.items()})


def test_scoring_handles_text_values():
    from eaer.evalrun.fast import mentions
    from factset.model import Fact

    assert mentions("国内航班乘坐经济舱。", Fact(0, "F", "V", "国内航班舱位标准", "经济舱", None, "正文", None))
    assert mentions("上限为50,000元", Fact(0, "F", "V", "k", 50000, "元", "正文", None))
    assert not mentions("上限为5000元", Fact(0, "F", "V", "k", 50000, "元", "正文", None))


def test_generator_is_seeded_not_global():
    a, b = Synth(1, 0.1), Synth(1, 0.1)
    assert [q.question for q in a.build().queries] == [q.question for q in b.build().queries]


def test_lexical_only_wrapper_survives_the_snapshot():
    """分词器对比的“仅关键词”：检索服务每次先取 snapshot()，包装必须跟着传下去，否则向量一路照样执行
    （2026-11-12 起的对比中“仅关键词”一栏实际是关键词加向量，见设计变更记录）。"""
    from datetime import date

    from eaer.evalrun.analyzers import LexicalOnly
    from eaer.retrieval.index import Filters, MemoryIndex

    class Spy(MemoryIndex):
        def snapshot(self):
            return Spy.__new__(Spy)            # 另一个实例，模拟别名解析成具体索引

        def search_vector(self, vector, f, k):
            return [("ev_x", 1.0)]

    f = Filters("t1", frozenset({"R"}), date(2026, 1, 1), None)
    snap = LexicalOnly(Spy()).snapshot()
    assert isinstance(snap, LexicalOnly) and snap.search_vector([0.0], f, 5) == []
