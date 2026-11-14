"""AC-14-4（硬性）许可合规：对外报告只含允许的数据集；内部数据集的结果带标记，生成对外报告时被拦截（违规数为 0）。

- 许可登记与开发文档 14 的数据集表一致（可对外、原样评测、仅内部、只借鉴设计、自建合成集）；
- 数据来源由数据本身决定：“数据来源”表，或冻结评测集的 SHA-256；改过的文件、来源不明的文件一律视为未登记；
- 每份报告都带数据集标记，不可对外的结果另标“仅内部”；对外报告遇到仅内部、只借鉴设计、改编过的原样评测集、
  未登记来源时拦截；
- 每个评测运行器带 --external 时，注定被拦截的数据在评测开始之前就停下（不连数据库、不调模型）。
人工复核（开发文档 AC-14-4 的“人工复核”）不在自动化范围内。
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from openpyxl import load_workbook

from eaer.evalrun import license as lic
from eaer.evalrun.license import (AS_IS, DATASETS, DESIGN_ONLY, EXTERNAL, INTERNAL, UNREGISTERED, LicenseViolation,
                                  Provenance, external_problems, provenance_of, stamp)
from fixture import sample, write

V1, V2 = Path("docs/评测集/合成集-v1.xlsx"), Path("docs/评测集/合成集-v2.xlsx")


def declared(tmp_path: Path, dataset: str, modified: str = "否") -> Path:
    path = write(sample(), tmp_path / f"{dataset}.xlsx")
    wb = load_workbook(path)
    ws = wb.create_sheet("数据来源")
    ws.append(["数据集", "是否改编", "说明"])
    ws.append([dataset, modified, "测试"])
    wb.save(path)
    return path


def test_registry_matches_the_dev_doc_table():
    scopes = {name: d.scope for name, d in DATASETS.items()}
    assert scopes == {"FRAMES": EXTERNAL, "MultiHop-RAG": EXTERNAL, "ViDoRe V3": EXTERNAL, "LLM-AggreFact": AS_IS,
                      "HoH": INTERNAL, "VersionQA": INTERNAL, "CHaystack": INTERNAL,
                      "RGB": DESIGN_ONLY, "DragonBall": DESIGN_ONLY, "CRUD-RAG": DESIGN_ONLY, "自建合成集": EXTERNAL}
    for name in ("MultiHop-RAG", "ViDoRe V3", "LLM-AggreFact", "自建合成集"):    # 需署名或注明的
        assert DATASETS[name].attribution
    assert "合成数据" in DATASETS["自建合成集"].attribution


def test_provenance_comes_from_the_data(tmp_path):
    assert provenance_of(V1).dataset == provenance_of(V2).dataset == "自建合成集"
    # 冻结评测集改过（这里用 openpyxl 重存一遍，字节就变了）就不再是登记过的那一份
    copy = tmp_path / "v2-copy.xlsx"
    shutil.copy(V2, copy)
    wb = load_workbook(copy)
    wb.save(copy)
    assert provenance_of(copy).dataset == UNREGISTERED
    assert provenance_of(write(sample(), tmp_path / "s.xlsx")).dataset == UNREGISTERED      # 没声明来源的事实表
    assert provenance_of(declared(tmp_path, "HoH")) == Provenance("HoH", False, "测试")
    assert provenance_of(declared(tmp_path, "LLM-AggreFact", "是")).modified
    assert provenance_of(declared(tmp_path, "某个新数据集")).dataset == UNREGISTERED          # 写了但没登记许可


def test_fact_table_loader_ignores_the_provenance_sheet(tmp_path):
    from factset import load
    fs = load(declared(tmp_path, "FRAMES"))
    assert fs.queries and not fs.load_issues


BLOCKED = [
    ("仅内部", Provenance("HoH"), "仅内部开发使用"),
    ("仅内部", Provenance("VersionQA"), "仅内部开发使用"),
    ("仅内部", Provenance("CHaystack"), "仅内部开发使用"),
    ("只借鉴设计", Provenance("CRUD-RAG"), "不得使用其内容"),
    ("原样评测被改编", Provenance("LLM-AggreFact", modified=True), "改编后的结果不对外"),
    ("未登记", Provenance(UNREGISTERED, detail="x.xlsx"), "数据来源未登记"),
]


@pytest.mark.parametrize("kind, prov, reason", BLOCKED, ids=[f"{b[0]}-{b[1].dataset}" for b in BLOCKED])
def test_external_report_is_blocked(kind, prov, reason):
    with pytest.raises(LicenseViolation) as e:
        stamp("召回率 91%", [prov], external=True)
    assert reason in str(e.value)
    # 同一结果在内部报告里照常输出，但带“仅内部”标记
    internal = stamp("召回率 91%", [prov], external=False)
    assert internal.startswith("【仅内部】") and "召回率 91%" in internal
    # 混在可对外的数据集里也拦截
    with pytest.raises(LicenseViolation):
        stamp("x", [Provenance("FRAMES"), prov], external=True)


@pytest.mark.parametrize("prov", [Provenance("FRAMES"), Provenance("MultiHop-RAG"), Provenance("ViDoRe V3"),
                                  Provenance("LLM-AggreFact"), Provenance("自建合成集", detail="合成集 v2")],
                         ids=lambda p: p.dataset)
def test_external_report_allowed_with_attribution(prov):
    out = stamp("召回率 91%", [prov], external=True)
    assert "【仅内部】" not in out and f"数据集：{prov.dataset}" in out and "召回率 91%" in out
    if DATASETS[prov.dataset].attribution:
        assert DATASETS[prov.dataset].attribution in out
    assert external_problems([prov]) == []


RUNNERS = [
    ("eaer.evalrun.grades", []),
    ("eaer.evalrun.fast", []),
    ("eaer.evalrun.injection", []),
    ("eaer.evalrun.admission", []),
    ("eaer.evalrun.analyzers", ["--opensearch", "http://127.0.0.1:9"]),
]


@pytest.mark.parametrize("module, extra", RUNNERS, ids=[r[0].rsplit(".", 1)[1] for r in RUNNERS])
def test_every_runner_refuses_before_running(module, extra, tmp_path, monkeypatch, capsys):
    """不可对外的数据带 --external：评测开始之前就拦截（管理连接是无效地址，真开始跑就会连接失败）。"""
    import importlib

    monkeypatch.setenv("EAER_ENV", "eval")
    main = importlib.import_module(module).main
    assert main([str(declared(tmp_path, "HoH")), "--admin-url", "postgresql://nobody@127.0.0.1:9/x", *extra,
                 "--external"]) == 3
    assert "对外报告被拦截" in capsys.readouterr().err


def test_injection_runner_counts_as_modified(tmp_path, monkeypatch, capsys):
    """注入评测往语料里植入文字：对只能原样评测的数据集是改编，对外报告被拦截。"""
    from eaer.evalrun.injection import main

    monkeypatch.setenv("EAER_ENV", "eval")
    assert main([str(declared(tmp_path, "LLM-AggreFact")), "--admin-url", "postgresql://nobody@127.0.0.1:9/x",
                 "--external"]) == 3
    assert "改编后的结果不对外" in capsys.readouterr().err


def test_runner_output_is_stamped(conn, tmp_path, monkeypatch, capsys):
    """端到端：来源未登记的事实表照常评测，报告带“仅内部”标记；声明为自建合成集的可以生成对外报告并注明合成数据。"""
    import os

    from eaer.evalrun.grades import main

    admin = os.environ.get("EAER_TEST_ADMIN_URL", "postgresql://postgres:postgres@localhost:5432/postgres")
    monkeypatch.setenv("EAER_ENV", "eval")
    assert main([str(write(sample(), tmp_path / "s.xlsx")), "--admin-url", admin]) == 0
    out = capsys.readouterr().out
    assert out.startswith("【仅内部】") and f"数据集：{UNREGISTERED}" in out and "不一致：0" in out
    assert main([str(declared(tmp_path, "自建合成集")), "--admin-url", admin, "--external"]) == 0
    out = capsys.readouterr().out
    assert "【仅内部】" not in out and "合成数据" in out and "不一致：0" in out


def test_frozen_registry_is_current():
    """docs/评测集 下每个冻结的评测集都登记了（新冻结的评测集忘了登记，对外报告会被拦截）。"""
    sets = sorted(p for p in Path("docs/评测集").glob("合成集-v*.xlsx") if "抽检单" not in p.name)
    assert sets and all(provenance_of(p).dataset == "自建合成集" for p in sets), lic.FROZEN
