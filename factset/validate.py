"""事实表约束校验（开发文档 AC-14-1）。

校验不通过时不计算答案：错误的事实表算出的标准答案没有意义。
"""
from __future__ import annotations

from collections import defaultdict

from .model import (CARRIERS, DOC_TYPES, EVENT_TYPES, GRADE_RANK, MODES, QUERY_TYPES,
                    STYLE_TO_GRADE, FactSet, Issue)


def validate(fs: FactSet) -> list[Issue]:
    issues: list[Issue] = list(fs.load_issues)
    add = lambda sheet, row, msg: issues.append(Issue(sheet, row, msg))  # noqa: E731

    # 文档
    for d in fs.documents.values():
        if d.doc_type not in DOC_TYPES:
            add("文档", d.row, f"文档类型“{d.doc_type}”无效，应为规范类或记录类")
        if not d.roles:
            add("文档", d.row, "可访问角色为空")
        for r in d.roles - fs.roles.keys():
            add("文档", d.row, f"可访问角色 {r} 不在角色表中")

    # 版本
    seen_no: dict[tuple[str, int], str] = {}
    superseded_by: dict[str, str] = {}
    for v in fs.versions.values():
        doc = fs.documents.get(v.document_id)
        if doc is None:
            add("版本", v.row, f"文档 {v.document_id} 不存在")
        k = (v.document_id, v.version_no)
        if k in seen_no:
            add("版本", v.row, f"文档 {v.document_id} 的版本号 {v.version_no} 与 {seen_no[k]} 重复")
        seen_no[k] = v.id
        if v.valid_from_style not in STYLE_TO_GRADE:
            add("版本", v.row, f"生效日写法“{v.valid_from_style}”无效")
        else:
            if v.valid_from_style == "正文写明" and v.valid_from is None:
                add("版本", v.row, "生效日写法为正文写明，但生效日为空")
            if v.valid_from_style in ("仅上传时间", "无", "不适用") and v.valid_from is not None:
                add("版本", v.row, f"生效日写法为{v.valid_from_style}，生效日应为空")
            if doc is not None:
                if doc.normative and v.valid_from_style == "不适用":
                    add("版本", v.row, "规范类文档的生效日写法不能是不适用")
                if not doc.normative and v.valid_from_style != "不适用":
                    add("版本", v.row, "记录类文档的生效日写法应为不适用")
                if not doc.normative and v.doc_date is None:
                    add("版本", v.row, "记录类文档必须填写文档日期")
        if v.uploaded_on is None or v.approved_on is None:
            add("版本", v.row, "上传日和审核通过日都必须填写")
        elif v.approved_on < v.uploaded_on:
            add("版本", v.row, "审核通过日早于上传日")
        if v.supersedes:
            prev = fs.versions.get(v.supersedes)
            if prev is None:
                add("版本", v.row, f"取代的版本 {v.supersedes} 不存在")
            else:
                if prev.document_id != v.document_id:
                    add("版本", v.row, f"取代的版本 {v.supersedes} 属于另一份文档")
                if prev.version_no >= v.version_no:
                    add("版本", v.row, f"取代的版本 {v.supersedes} 的版本号不小于本版本")
            if v.supersedes in superseded_by:
                add("版本", v.row, f"{v.supersedes} 已被 {superseded_by[v.supersedes]} 取代，不能被取代两次")
            superseded_by[v.supersedes] = v.id
    # 取代关系不能成环
    for v in fs.versions.values():
        seen, cur = {v.id}, v.supersedes
        while cur:
            if cur in seen:
                add("版本", v.row, "取代关系成环")
                break
            seen.add(cur)
            cur = fs.versions[cur].supersedes if cur in fs.versions else None

    # 事实
    keys_in_version: dict[tuple[str, str], str] = {}
    for f in fs.facts.values():
        if f.version_id not in fs.versions:
            add("事实", f.row, f"版本 {f.version_id} 不存在")
        if not f.key:
            add("事实", f.row, "事实键为空")
        if f.value is None:
            add("事实", f.row, "取值为空")
        if f.carrier not in CARRIERS:
            add("事实", f.row, f"载体“{f.carrier}”无效")
        k = (f.version_id, f.key)
        if k in keys_in_version:
            add("事实", f.row, f"同一版本内事实键“{f.key}”与 {keys_in_version[k]} 重复")
        keys_in_version[k] = f.id

    # 事件
    archived: dict[str, bool] = defaultdict(bool)
    for e in fs.events:  # 已按发生日排序
        if e.type not in EVENT_TYPES:
            add("事件", e.row, f"事件类型“{e.type}”无效")
            continue
        if e.occurred_on is None:
            add("事件", e.row, "发生日为空")
            continue
        is_doc, is_ver = e.target in fs.documents, e.target in fs.versions
        if not (is_doc or is_ver):
            add("事件", e.row, f"对象 {e.target} 既不是文档也不是版本")
            continue
        targets = ([v for v in fs.versions.values() if v.document_id == e.target] if is_doc
                   else [fs.versions[e.target]])
        if any(v.approved_on and e.occurred_on < v.approved_on for v in targets):
            add("事件", e.row, "事件发生在对象审核通过之前；发布后的变化才写在事件表")
        if e.type == "有效期更正":
            if not is_ver:
                add("事件", e.row, "有效期更正的对象必须是版本")
            if e.new_valid_from is None and e.new_grade is None:
                add("事件", e.row, "有效期更正需要填写新生效日或新来源等级")
            if e.new_grade is not None and e.new_grade not in GRADE_RANK:
                add("事件", e.row, f"新来源等级“{e.new_grade}”无效")
        if e.type == "权限变更":
            if not is_doc:
                add("事件", e.row, "权限变更的对象必须是文档")
            if not e.new_roles:
                add("事件", e.row, "权限变更需要填写新可访问角色")
            else:
                for r in e.new_roles - fs.roles.keys():
                    add("事件", e.row, f"新可访问角色 {r} 不在角色表中")
        if e.type in ("归档", "取消归档"):
            want = e.type == "归档"
            for v in targets:
                if archived[v.id] == want:
                    add("事件", e.row, f"{v.id} {'已经处于归档' if want else '并未归档'}，不能{e.type}")
                archived[v.id] = want

    # 查询
    all_keys = {f.key for f in fs.facts.values()}
    for q in fs.queries:
        if q.type not in QUERY_TYPES:
            add("查询", q.row, f"查询类型“{q.type}”无效")
        if q.role not in fs.roles:
            add("查询", q.row, f"提问角色 {q.role} 不在角色表中")
        if q.asked_on is None:
            add("查询", q.row, "提问日为空")
        if not q.target_key:
            add("查询", q.row, "目标事实键为空")
        if q.type == "历史时点" and q.as_of_valid is None:
            add("查询", q.row, "历史时点查询必须填写生效时间点")
        if q.type == "回溯" and q.as_of_system is None:
            add("查询", q.row, "回溯查询必须填写回溯时间点")
        if q.as_of_system and q.asked_on and q.as_of_system > q.asked_on:
            add("查询", q.row, "回溯时间点晚于提问日")
        for f in q.basis:
            if f not in fs.facts:
                add("查询", q.row, f"依据事实 {f} 不存在")
            elif q.target_key and fs.facts[f].key != q.target_key:
                add("查询", q.row, f"依据事实 {f} 的事实键与目标事实键不一致")
        if q.should_refuse:
            if q.basis:
                add("查询", q.row, "应拒答的查询不填依据事实")
            if q.expected_grade:
                add("查询", q.row, "应拒答的查询不填预期来源等级")
        else:
            if not q.basis:
                add("查询", q.row, "不拒答的查询必须填写依据事实")
            if q.target_key and q.target_key not in all_keys:
                add("查询", q.row, f"目标事实键“{q.target_key}”在事实表中不存在")
        if q.expected_grade and q.expected_grade not in GRADE_RANK:
            add("查询", q.row, f"预期来源等级“{q.expected_grade}”无效")
        if q.expected_mode and q.expected_mode not in MODES:
            add("查询", q.row, f"预期模式“{q.expected_mode}”无效")
    return issues
