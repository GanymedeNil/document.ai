"""合成评测集生成器（开发文档 14.2 第 1 步、14.3）。

按随机种子确定地生成一张完整的事实表（角色、文档、版本、事实、事件、查询），内容为自编的企业制度：
- 规范类：30 个主题的制度，每个 1–4 个版本，生效日写法有“正文写明”“仅上传时间”“无”三种，部分新版本晚于
  生效日才审核通过（回溯能看出差别）；部分主题只对特定角色开放；
- 补充通知：对部分事实键给出不同取值，与主制度构成冲突（来源等级相同时全部列出，不同时 explicit 优先）；
- 记录类：季度费用分析报告，表格事实可直接回答，图表事实只出现在图片里（需要视觉模型）；
- 事件：归档与取消归档、权限变更（收紧或放开）、对“仅上传时间”版本补录生效日（有效期更正）。
查询按 14.3 的样本量生成：高风险类别（历史时点、回溯、冲突、归档、权限边界）各 300 条，其他类别（当前生效、
版本差异、无答案、图表、验证器）各 100 条；scale 可按比例缩小，供测试与快速试跑。每条查询的标准答案由
answer.Calculator 计算后回填，并按类别校验它确实落在该类别要考的情形上（如冲突类必须真有冲突）。
每个类别内按编号哈希排序后取 30% 为标定集、70% 为测试集（14.3），写在“集合”列。

内容是自编的（14.2 的“其次是自编内容”）；用法律法规、国家机关正式文件作种子的版本尚未做。
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, replace
from datetime import date, timedelta

from .answer import Calculator
from .model import Document, Event, Fact, FactSet, Query, Role, Version

GENERATOR_VERSION = "1"
TODAY = date(2026, 11, 30)
FIRST = date(2023, 1, 1)
LAST_CHANGE = date(2026, 10, 15)

ROLES = [("R_STAFF", "普通员工"), ("R_FIN", "财务人员"), ("R_HR", "人力资源人员"), ("R_MGR", "管理人员"),
         ("R_LEGAL", "法务人员")]
ALL = frozenset(r for r, _ in ROLES)

# (标题, 可访问角色, [(事实键, 单位, 取值规格)])；取值规格为 (下限, 上限, 步长) 或候选文本列表
TOPICS = [
    ("差旅报销管理办法", ALL, [("一线城市住宿上限", "元/晚", (450, 900, 50)), ("二线城市住宿上限", "元/晚", (300, 600, 50)),
                               ("三线城市住宿上限", "元/晚", (200, 450, 50)), ("出差伙食补贴", "元/天", (50, 150, 10)),
                               ("市内交通补贴", "元/天", (30, 100, 10))]),
    ("采购审批权限规定", frozenset({"R_FIN", "R_MGR"}), [
        ("部门经理单笔采购审批上限", "元", (20000, 200000, 10000)), ("分管副总单笔采购审批上限", "元", (100000, 1000000, 50000)),
        ("总经理单笔采购审批上限", "元", (500000, 5000000, 100000)), ("集中采购起点金额", "元", (5000, 50000, 5000))]),
    ("员工休假管理办法", ALL, [("年休假基础天数", "天", (5, 15, 1)), ("婚假天数", "天", (3, 15, 1)),
                               ("陪产假天数", "天", (7, 30, 1)), ("病假单次最长天数", "天", (10, 60, 5))]),
    ("加班管理规定", ALL, [("工作日加班餐补", "元/次", (15, 60, 5)), ("每月加班时长上限", "小时", (20, 48, 2)),
                           ("节假日加班调休比例", "%", (100, 300, 50))]),
    ("培训管理办法", ALL, [("外部培训年度预算上限", "元/人", (2000, 20000, 1000)), ("培训服务期", "年", (1, 5, 1)),
                           ("内部讲师课时津贴", "元/课时", (100, 800, 50))]),
    ("车辆使用管理规定", ALL, [("私车公用补贴", "元/公里", (0.5, 2.0, 0.1)), ("公务用车单日里程上限", "公里", (200, 800, 50))]),
    ("通讯费补贴标准", ALL, [("部门经理通讯补贴", "元/月", (100, 500, 50)), ("普通员工通讯补贴", "元/月", (50, 300, 25)),
                             ("销售人员通讯补贴", "元/月", (100, 600, 50))]),
    ("业务招待管理办法", frozenset({"R_FIN", "R_MGR", "R_STAFF"}), [
        ("人均招待标准", "元/人", (100, 500, 20)), ("单次招待费上限", "元", (2000, 20000, 1000)),
        ("招待事前审批起点", "元", (1000, 10000, 500))]),
    ("合同管理办法", frozenset({"R_LEGAL", "R_MGR", "R_FIN"}), [
        ("合同法务审核金额起点", "元", (10000, 500000, 10000)), ("合同审批办理时限", "个工作日", (3, 15, 1))]),
    ("印章使用管理规定", ALL, [("用印申请办理时限", "个工作日", (1, 5, 1)), ("公章外借最长天数", "天", (1, 10, 1))]),
    ("安全生产管理规定", ALL, [("新员工安全培训学时", "学时", (8, 72, 4)), ("年度安全再培训学时", "学时", (4, 24, 2))]),
    ("绩效考核办法", frozenset({"R_HR", "R_MGR"}), [("年度考核优秀比例上限", "%", (10, 30, 5)),
                                                   ("绩效申诉受理时限", "个工作日", (3, 15, 1))]),
    ("薪酬管理办法", frozenset({"R_HR", "R_MGR"}), [("试用期工资比例", "%", (70, 100, 5)), ("年终奖最高月数", "个月", (2, 12, 1)),
                                                   ("高管年薪上限", "万元", (100, 500, 10))]),
    ("财务报销时限规定", ALL, [("报销单据提交时限", "天", (15, 90, 15)), ("备用金核销期限", "天", (15, 60, 5))]),
    ("信息安全管理办法", ALL, [("系统密码更换周期", "天", (30, 180, 30)), ("离职账号注销时限", "个工作日", (1, 5, 1)),
                               ("数据备份保留期限", "天", (30, 365, 15))]),
    ("办公用品管理办法", ALL, [("单次办公用品领用上限", "元", (100, 2000, 100)), ("固定资产报废年限下限", "年", (3, 10, 1))]),
    ("员工福利管理办法", ALL, [("生日福利标准", "元/人", (100, 500, 50)), ("节日慰问标准", "元/人", (200, 1000, 100)),
                               ("年度体检标准", "元/人", (300, 2000, 100))]),
    ("住房补贴管理办法", frozenset({"R_HR", "R_MGR", "R_FIN"}), [
        ("新入职住房补贴", "元/月", (500, 3000, 100)), ("住房补贴发放期限", "个月", (6, 36, 6))]),
    ("外派人员管理办法", ALL, [("境外派驻补贴", "元/天", (100, 600, 20)), ("外派探亲假天数", "天", (10, 30, 1))]),
    ("招聘管理办法", ALL, [("内部推荐奖励", "元/人", (1000, 10000, 500)), ("试用期最长月数", "个月", (1, 6, 1))]),
    ("员工离职管理办法", ALL, [("离职提前通知天数", "天", (3, 30, 1)), ("离职交接办理时限", "个工作日", (3, 10, 1))]),
    ("会议费管理办法", ALL, [("会议人均日标准", "元/人/天", (200, 800, 50)), ("会议费事前审批起点", "元", (5000, 50000, 5000))]),
    ("固定资产管理办法", frozenset({"R_FIN", "R_MGR", "R_STAFF"}), [
        ("固定资产入账起点", "元", (1000, 10000, 500)), ("资产盘点周期", "个月", (3, 12, 1))]),
    ("交通工具乘坐标准", ALL, [("国内航班舱位标准", None, ["经济舱", "超级经济舱", "公务舱"]),
                               ("高铁座席标准", None, ["二等座", "一等座", "商务座"])]),
    ("应急管理规定", ALL, [("突发事件上报时限", "小时", (1, 24, 1)), ("应急演练频次", "次/年", (1, 4, 1))]),
    ("档案管理办法", ALL, [("会计档案保管年限", "年", (10, 30, 5)), ("档案借阅最长天数", "天", (3, 30, 1))]),
    ("研发项目管理办法", frozenset({"R_MGR", "R_FIN"}), [("研发项目立项审批起点", "元", (100000, 2000000, 100000)),
                                                       ("项目结题验收时限", "个月", (1, 6, 1))]),
    ("客户投诉处理规定", ALL, [("投诉首次响应时限", "小时", (1, 48, 1)), ("投诉办结时限", "个工作日", (3, 30, 1))]),
    ("终端设备使用规定", ALL, [("外设接入审批时限", "个工作日", (1, 5, 1)), ("终端补丁更新周期", "天", (7, 60, 1))]),
    ("法律事务管理办法", frozenset({"R_LEGAL", "R_MGR"}), [("诉讼案件上报时限", "个工作日", (1, 10, 1)),
                                                       ("外聘律师费审批起点", "元", (10000, 200000, 10000))]),
]
AREAS = ["差旅", "采购", "培训", "招待", "会议", "通讯", "加班", "福利"]
QUARTERS = [(2025, 1), (2025, 2), (2025, 3), (2025, 4), (2026, 1), (2026, 2), (2026, 3)]
CN_Q = "一二三四"
# 无答案类：与已有事实键相近、但事实表里没有的说法
MISSING_PREFIX = ["海外", "博士", "退休人员", "实习生", "外包人员", "驻场", "夜班", "高温", "异地", "兼职"]
MISSING_BASE = ["住宿补贴", "交通补贴", "餐补标准", "通讯补贴", "安家费", "体检标准", "培训预算", "年休假天数",
                "加班费标准", "防暑降温费", "取暖补贴", "误餐补助"]
QUOTA = {"历史时点": 300, "回溯": 300, "冲突": 300, "归档": 300, "权限边界": 300,
         "当前生效": 100, "版本差异": 100, "无答案": 100, "图表": 100, "验证器": 100}
MODE = {"当前生效": "Verified", "历史时点": "Verified", "回溯": "Verified", "版本差异": "Deep", "冲突": "Verified",
        "归档": "Verified", "权限边界": "Verified", "图表": "Fast", "验证器": "Verified"}


def _d(day: date) -> str:
    return f"{day.year} 年 {day.month} 月 {day.day} 日"


def _d2(day: date) -> str:
    return f"{day.year}年{day.month}月{day.day}日"


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@dataclass
class _Key:
    key: str
    unit: str | None
    spec: object
    doc_id: str
    roles: frozenset[str]
    carrier: str


class Synth:
    def __init__(self, seed: int = 2026, scale: float = 1.0):
        self.rng = random.Random(seed)
        self.seed, self.scale = seed, scale
        self.fs = FactSet()
        self.keys: dict[str, _Key] = {}
        self.n = {"P": 0, "F": 0, "E": 0, "Q": 0}

    # ---------- 取值 ----------

    def draw(self, spec, avoid=()) -> int | float | str:
        for _ in range(50):
            if isinstance(spec, list):
                v = self.rng.choice(spec)
            else:
                lo, hi, step = spec
                k = self.rng.randint(0, round((hi - lo) / step))
                v = round(lo + k * step, 1) if isinstance(step, float) else lo + k * step
                v = int(v) if isinstance(v, float) and v.is_integer() else v     # 与读表时的规范化一致（2.0 → 2）
            if v not in avoid:
                return v
        return v

    def _id(self, kind: str, width: int) -> str:
        self.n[kind] += 1
        return f"{kind}{self.n[kind]:0{width}d}"

    # ---------- 文档与版本 ----------

    def _versions(self, doc_id: str, n: int, *, first_style_weights=(70, 15, 15)) -> list[Version]:
        out: list[Version] = []
        nominal = FIRST + timedelta(days=self.rng.randint(0, 540))
        for no in range(1, n + 1):
            if no > 1:
                nominal = nominal + timedelta(days=self.rng.randint(180, 420))
                if nominal > LAST_CHANGE:
                    break
            weights = first_style_weights if no == 1 else (80, 20, 0)
            style = self.rng.choices(["正文写明", "仅上传时间", "无"], weights=weights)[0]
            late = self.rng.random() < 0.2
            approved = nominal + timedelta(days=self.rng.randint(1, 15)) if late else \
                nominal - timedelta(days=self.rng.randint(2, 20))
            uploaded = approved - timedelta(days=self.rng.randint(1, 5))
            out.append(Version(0, f"{doc_id}-v{no}", doc_id, no, nominal if style == "正文写明" else None, style,
                               out[-1].id if out else None, None, uploaded, approved, None))
        return out

    def build_corpus(self) -> None:
        for rid, desc in ROLES:
            self.fs.roles[rid] = Role(0, rid, desc)
        for title, roles, keys in TOPICS:
            doc_id = self._id("P", 3)
            self.fs.documents[doc_id] = Document(0, doc_id, title, "规范类", "中文", roles)
            carrier = "表格" if self.rng.random() < 0.4 else "正文"
            for key, unit, spec in keys:
                self.keys[key] = _Key(key, unit, spec, doc_id, roles, carrier)
            vs = self._versions(doc_id, self.rng.choices([1, 2, 3, 4], weights=[2, 4, 3, 2])[0])
            prev: dict[str, object] = {}
            for v in vs:
                self.fs.versions[v.id] = v
                for key, unit, spec in keys:
                    value = prev[key] if key in prev and self.rng.random() < 0.25 else \
                        self.draw(spec, avoid=(prev.get(key),))
                    prev[key] = value
                    fid = self._id("F", 4)
                    self.fs.facts[fid] = Fact(0, fid, v.id, key, value, unit, carrier, None)
        self._notices()
        self._reports()

    def _current(self, key: str) -> list[Fact]:
        return [f for f in self.fs.facts.values() if f.key == key]

    def _notices(self) -> None:
        """补充通知：对同一主题的 2–3 个事实键给出不同取值，与主制度冲突。"""
        topics = [t for t in TOPICS if t[1] == ALL and len(t[2]) >= 2 and t[2][0][2] and not isinstance(t[2][0][2], list)]
        for title, roles, keys in self.rng.sample(topics, min(14, len(topics))):
            short = title.removesuffix("管理办法").removesuffix("管理规定").removesuffix("规定").removesuffix("标准")
            doc_id = self._id("P", 3)
            self.fs.documents[doc_id] = Document(0, doc_id, f"关于调整{short}有关标准的补充通知", "规范类", "中文", roles)
            style = self.rng.choices(["无", "仅上传时间", "正文写明"], weights=[55, 20, 25])[0]
            nominal = date(2025, 6, 1) + timedelta(days=self.rng.randint(0, 420))
            approved = nominal - timedelta(days=self.rng.randint(2, 15))
            v = Version(0, f"{doc_id}-v1", doc_id, 1, nominal if style == "正文写明" else None, style, None, None,
                        approved - timedelta(days=self.rng.randint(1, 4)), approved, None)
            self.fs.versions[v.id] = v
            for key, unit, spec in self.rng.sample(keys, min(len(keys), self.rng.choice([2, 3]))):
                taken = {f.value for f in self._current(key)}
                fid = self._id("F", 4)
                self.fs.facts[fid] = Fact(0, fid, v.id, key, self.draw(spec, avoid=taken), unit, "正文", "补充通知")

    def _reports(self) -> None:
        pairs = [(a, y, q) for a in AREAS for y, q in QUARTERS]
        for area, y, q in sorted(self.rng.sample(pairs, 24), key=lambda p: (p[1], p[2], p[0])):
            period = f"{y} 年第{CN_Q[q - 1]}季度"
            doc_id = self._id("P", 3)
            roles = frozenset({"R_FIN", "R_MGR"}) if self.rng.random() < 0.3 else ALL
            self.fs.documents[doc_id] = Document(0, doc_id, f"{period}{area}费用分析报告", "记录类", "中文", roles)
            end = date(y, 3 * q, 28)
            doc_date = end + timedelta(days=self.rng.randint(10, 25))
            approved = doc_date + timedelta(days=self.rng.randint(2, 8))
            v = Version(0, f"{doc_id}-v1", doc_id, 1, None, "不适用", None, doc_date,
                        approved - timedelta(days=1), approved, None)
            self.fs.versions[v.id] = v
            for key, unit, spec, carrier in ((f"{period}{area}报销笔数", "笔", (200, 5000, 10), "表格"),
                                             (f"{period}{area}人均费用", "元", (300, 9000, 10), "表格"),
                                             (f"{period}{area}总费用", "万元", (20, 900, 1), "图表"),
                                             (f"{period}{area}预算执行率", "%", (40, 120, 1), "图表")):
                fid = self._id("F", 4)
                self.fs.facts[fid] = Fact(0, fid, v.id, key, self.draw(spec), unit, carrier, None)
                self.keys[key] = _Key(key, unit, spec, doc_id, roles, carrier)

    # ---------- 事件 ----------

    def _last_approved(self, doc_id: str) -> date:
        return max(v.approved_on for v in self.fs.versions.values() if v.document_id == doc_id)

    def build_events(self) -> None:
        conflict_docs = {self.keys[f.key].doc_id for f in self.fs.facts.values() if f.note == "补充通知"}
        policies = [d for d in self.fs.documents.values() if d.normative and not d.title.startswith("关于")]
        plain = [d for d in policies if d.id not in conflict_docs]
        self.rng.shuffle(plain)
        events: list[Event] = []
        archive_windows: dict[str, list[tuple[date, date]]] = {}
        archived = [d for d in plain if d.roles == ALL][:6]
        for d in archived:
            lo = self._last_approved(d.id) + timedelta(days=30)
            hi = TODAY - timedelta(days=60)
            if lo >= hi:
                continue
            a = lo + timedelta(days=self.rng.randint(0, (hi - lo).days))
            events.append(Event(0, "", a, "归档", d.id, "归档", None, None, None))
            end = TODAY + timedelta(days=1)
            if self.rng.random() < 0.7:
                u = a + timedelta(days=self.rng.randint(20, 100))
                if u < TODAY - timedelta(days=5):
                    events.append(Event(0, "", u, "取消归档", d.id, "恢复为可搜索", None, None, None))
                    end = u
            archive_windows[d.id] = [(a, end)]
        for d in [d for d in plain if d.id not in archive_windows][:5]:
            lo = self._last_approved(d.id) + timedelta(days=20)
            hi = TODAY - timedelta(days=40)
            if lo >= hi:
                continue
            p = lo + timedelta(days=self.rng.randint(0, (hi - lo).days))
            new = ALL if d.roles != ALL else self.rng.choice([frozenset({"R_FIN", "R_MGR"}), frozenset({"R_HR", "R_MGR"})])
            events.append(Event(0, "", p, "权限变更", d.id, "调整可访问角色", None, None, new))
        for v in sorted(self.fs.versions.values(), key=lambda v: v.id):
            if v.valid_from_style != "仅上传时间" or not self.fs.documents[v.document_id].normative:
                continue
            if self.rng.random() < 0.35:
                continue
            c = v.approved_on + timedelta(days=self.rng.randint(20, 300))
            if c >= TODAY - timedelta(days=10):
                continue
            if any(a <= c < e for a, e in archive_windows.get(v.document_id, [])):
                continue
            prev = self.fs.versions.get(v.supersedes) if v.supersedes else None
            succ = next((s for s in self.fs.versions.values() if s.supersedes == v.id), None)
            true_from = v.approved_on - timedelta(days=self.rng.randint(0, 10))
            if prev is not None and prev.valid_from is not None and true_from <= prev.valid_from:
                continue
            if succ is not None and succ.valid_from is not None and true_from >= succ.valid_from:
                continue
            if prev is not None and true_from <= prev.approved_on:
                continue
            events.append(Event(0, "", c, "有效期更正", v.id, f"审核人补录生效日 {true_from.isoformat()}",
                                true_from, "explicit", None))
        events.sort(key=lambda e: (e.occurred_on, e.target, e.type))
        for e in events:
            self.fs.events.append(replace(e, id=self._id("E", 3)))

    # ---------- 查询 ----------

    def _visible_role(self, doc_id: str) -> str:
        roles = sorted(self.fs.documents[doc_id].roles)
        return "R_STAFF" if "R_STAFF" in roles else roles[0]

    def _query(self, qtype, question, key, role, asked_on=TODAY, as_of_valid=None, as_of_system=None) -> Query:
        return Query(0, "", qtype, question, "", False, role, asked_on, as_of_valid, as_of_system, (), None, None, key,
                     None)

    def candidates(self, calc: Calculator) -> dict[str, list[Query]]:
        r = self.rng
        out: dict[str, list[Query]] = {t: [] for t in QUOTA}
        archived_docs = {e.target for e in self.fs.events if e.type == "归档"}
        acl_docs = {e.target: e for e in self.fs.events if e.type == "权限变更"}
        unstable = archived_docs | set(acl_docs)          # 提问日可见性会变的文档：只用于归档、权限边界两类
        all_norm = [k for k in self.keys.values() if self.fs.documents[k.doc_id].normative and k.carrier != "图表"]
        norm_keys = [k for k in all_norm if k.doc_id not in unstable]
        conflict_keys = sorted({f.key for f in self.fs.facts.values() if f.note == "补充通知"})
        restricted = [k for k in all_norm if k.roles != ALL or k.doc_id in acl_docs]

        def role(k: _Key) -> str:
            return r.choice(sorted(self.fs.documents[k.doc_id].roles))

        def versions_of(key):
            return sorted((self.fs.versions[f.version_id] for f in self._current(key)
                           if self.fs.versions[f.version_id].document_id == self.keys[key].doc_id),
                          key=lambda v: v.version_no)

        def add(qtype, question, k, **kw):
            out[qtype].append(self._query(qtype, question, k.key, kw.pop("role", None) or role(k), **kw))

        cur = ["{k}是多少？", "{k}现在是多少？", "目前{k}的标准是多少？", "请问{k}是多少？", "按现行规定，{k}是多少？",
               "现行的{k}是多少？", "{k}按最新规定是多少？", "公司规定的{k}是多少？", "我想了解一下{k}是多少。",
               "{k}的现行标准是什么？"]
        for k in norm_keys:
            for t in cur:
                add("当前生效", t.format(k=k.key), k)
        hist = ["{d}，{k}是多少？", "{d}实际生效的{k}是多少？", "在{d}，{k}按当时生效的规定是多少？", "{d}那天{k}是多少？"]
        for k in norm_keys:
            for v in versions_of(k.key):
                start = v.valid_from or v.approved_on
                for _ in range(6):
                    d = start + timedelta(days=r.randint(-30, 240))
                    if FIRST <= d < TODAY:
                        add("历史时点", r.choice(hist).format(d=r.choice([_d, _d2])(d), k=k.key), k, as_of_valid=d)
        back1 = ["截至{s}，系统中记录的{k}是多少？", "截至{s}，系统里登记的{k}是多少？"]
        back2 = ["按截至{s}的系统记录，{d}的{k}是多少？", "按截至{s}系统中的记录，{d}实际生效的{k}是多少？"]
        for k in norm_keys:
            for v in versions_of(k.key):
                for _ in range(6):
                    sday = v.approved_on + timedelta(days=r.randint(-25, 150))
                    if not FIRST <= sday < TODAY:
                        continue
                    if r.random() < 0.5:
                        add("回溯", r.choice(back1).format(s=_d(sday), k=k.key), k, as_of_system=sday)
                    else:
                        d = sday - timedelta(days=r.randint(0, 200))
                        add("回溯", r.choice(back2).format(s=_d(sday), d=_d(d), k=k.key), k, as_of_valid=d,
                            as_of_system=sday)
        diff = ["{ym}修订前后，{k}有什么变化？", "《{t}》最近一次修订把{k}从多少改成了多少？",
                "《{t}》修订后，{k}和修订前相比有什么不同？"]
        for k in norm_keys:
            vs = versions_of(k.key)
            if len(vs) >= 2 and isinstance(k.spec, tuple):
                when = vs[-1].valid_from or vs[-1].approved_on
                for t in diff:
                    for _ in range(2):
                        add("版本差异", t.format(ym=f"{when.year} 年 {when.month} 月", k=k.key,
                                                  t=self.fs.documents[k.doc_id].title), k)
        conf = ["{k}是多少？", "{k}按规定是多少？", "请问{k}的标准是多少？", "{k}现在执行多少？", "目前{k}是多少？"]
        for key in conflict_keys:
            k = self.keys[key]
            for t in conf:
                for _ in range(2):
                    add("冲突", t.format(k=key), k)
                    d = TODAY - timedelta(days=r.randint(30, 500))
                    add("冲突", f"{r.choice([_d, _d2])(d)}，{key}是多少？", k, as_of_valid=d)
        arch = ["{k}是多少？", "请问{k}是多少？", "{k}按规定是多少？", "目前{k}的标准是多少？"]
        for e in [e for e in self.fs.events if e.type == "归档"]:
            unarch = next((u.occurred_on for u in self.fs.events if u.type == "取消归档" and u.target == e.target
                           and u.occurred_on > e.occurred_on), None)
            for k in [k for k in all_norm if k.doc_id == e.target]:
                days = [e.occurred_on - timedelta(days=r.randint(1, 40)) for _ in range(3)]
                days += [e.occurred_on + timedelta(days=r.randint(0, 20)) for _ in range(3)]
                if unarch:
                    days += [unarch + timedelta(days=r.randint(0, 20)) for _ in range(3)]
                for asked in days:
                    for t in arch:
                        add("归档", t.format(k=k.key), k, asked_on=min(asked, TODAY))
        perm = ["{k}是多少？", "请问{k}现在是多少？", "{k}按规定是多少？"]
        for k in restricted:
            change = acl_docs.get(k.doc_id)
            for rl in sorted(ALL):
                for t in perm:
                    if change is None:
                        add("权限边界", t.format(k=k.key), k, role=rl)
                    else:
                        for asked in (change.occurred_on - timedelta(days=r.randint(1, 30)),
                                      change.occurred_on + timedelta(days=r.randint(0, 30))):
                            add("权限边界", t.format(k=k.key), k, role=rl, asked_on=min(asked, TODAY))
        missing = [p + b for p in MISSING_PREFIX for b in MISSING_BASE if (p + b) not in self.keys]
        miss = ["{k}是多少？", "{k}的标准是多少？", "请问{k}怎么规定的？"]
        for key in missing:
            out["无答案"].append(self._query("无答案", r.choice(miss).format(k=key), key, r.choice(sorted(ALL))))
        chart = ["{k}是多少？", "《{t}》中{k}是多少？", "请问{k}是多少？"]
        for k in [k for k in self.keys.values() if k.carrier == "图表"]:
            for t in chart:
                add("图表", t.format(k=k.key, t=self.fs.documents[k.doc_id].title), k)
        for k in [k for k in norm_keys if isinstance(k.spec, tuple)]:
            top = calc.compute(self._query("当前生效", "", k.key, role(k)))
            if top.refuse or top.conflict:
                continue
            value = self.fs.facts[top.basis[0]].value
            avoid = {f.value for f in self._current(k.key)}
            claims = ["结论：{k}为 {v} {u}", "结论：按现行规定，{k}为 {v} {u}", "结论：{k}目前是 {v} {u}"]
            for claimed in (value, value, value, self.draw(k.spec, avoid=avoid), self.draw(k.spec, avoid=avoid),
                            self.draw(k.spec, avoid=avoid)):
                v = f"{claimed:,}" if isinstance(claimed, (int, float)) else claimed
                add("验证器", r.choice(claims).format(k=k.key, v=v, u=k.unit or "").rstrip() + "。", k)
        return out

    @staticmethod
    def fits(qtype: str, e) -> bool:
        """计算出的标准答案确实落在该类别要考的情形上。"""
        if qtype == "冲突":
            return e.conflict
        if qtype == "无答案":
            return e.refuse
        if qtype in ("当前生效", "版本差异", "图表", "验证器"):
            return not e.refuse and not e.conflict
        if qtype in ("历史时点", "回溯"):           # 允许“那时尚未生效、尚未审核通过”的拒答
            return not e.conflict
        if qtype in ("归档", "权限边界"):
            return not e.conflict
        return True

    def build_queries(self) -> None:
        calc = Calculator(self.fs)
        pools = self.candidates(calc)
        chosen: list[tuple[Query, object]] = []
        seen: set = set()
        for qtype, quota in QUOTA.items():
            want = max(10, round(quota * self.scale))
            picked = []
            pool = pools[qtype]
            self.rng.shuffle(pool)
            for q in pool:
                sig = (q.question, q.role, q.asked_on, q.as_of_valid, q.as_of_system)
                if sig in seen:
                    continue
                e = calc.compute(q)
                if not self.fits(qtype, e):
                    continue
                seen.add(sig)
                picked.append((q, e))
                if len(picked) >= want:
                    break
            chosen += picked
        for i, (q, e) in enumerate(chosen, start=1):
            qid = f"Q{i:04d}"
            answer = e.answer if not e.refuse else "拒答"
            chosen[i - 1] = (replace(q, id=qid, row=i + 1, expected_answer=answer, should_refuse=e.refuse,
                                     basis=e.basis, expected_grade=e.grade,
                                     expected_mode=None if e.refuse else MODE.get(q.type)), e)
        by_type: dict[str, list[Query]] = {}
        for q, _ in chosen:
            by_type.setdefault(q.type, []).append(q)
        split: dict[str, str] = {}
        for qs in by_type.values():
            ordered = sorted(qs, key=lambda q: _h(f"{self.seed}:{q.id}"))
            n_cal = round(len(ordered) * 0.3)
            for j, q in enumerate(ordered):
                split[q.id] = "标定" if j < n_cal else "测试"
        self.fs.queries = [replace(q, split=split[q.id]) for q, _ in chosen]

    def build(self) -> FactSet:
        self.build_corpus()
        self.build_events()
        self.build_queries()
        return self.fs


def generate(seed: int = 2026, scale: float = 1.0) -> FactSet:
    return Synth(seed, scale).build()
