"""确定性检查（开发文档 9.1、9.2；AC-9-1）。

每条结论依次经过：
0. 引用核对（9.1）：引用了不在本次检索结果中的证据 id，直接 unsupported；
1. authorized：调用授权接口 check 再核对一次；不通过的证据移除并记录安全告警，结论按剩余证据继续检查，
   一条都不剩时 unsupported；
2. locator_exists：证据存在；证据 id 能由版本、模态、定位和内容哈希重新算出（定位与内容未被篡改）；
   定位的形状对该模态成立（页码、表格单元格、图表区域）；原件仍在文件存储中；
3. value_match：结论中的数值、日期规范化后（全角半角、万与千、千分位、日期写法）都出现在所引证据中；
   英文证据另认：不写“年”的年份、英文日期（1 January 2024）、thousand / million / billion、per cent、
   空格千分位（3 456），以及写明了百分比单位的证据里不带 % 的数值（表头“(%)”）。不带年份的日期（“3月15日”
   “11月起”“November 6”）按月日比较，证据或标注中同月同日的任一完整日期都算找到；带年份的日期仍须年份也对上。
   宽松模式（formulas=True）另认结论里写明的算式（“每晚 680 元 × 7 晚 = 4,760 元”）：算术成立、单价在所引证据中、
   证据以外的数只是乘除里的正整数数量（晚数、天数、人数）时判 unverified 并写出算式；算术不成立判 unsupported。
   证据中没有、但等于所引证据中
   两个数值之差或之和的（“相差 10 个百分点”），只核对得了算术，判 unverified 并写明算式。所引证据的标注
   （标题、版本号、记录类的文档日期、其他文档在目标时间的生效与作废日期、出处）也算证据：这些是系统记录的元数据，
   生成时与原文一起给模型看，结论复述它们（“第 1 版（文档日期 2026-10-13）显示……”）不是编造；
4. time_valid：所引版本在目标时间有效。只有起决定作用的边界是 explicit 时才判“确定无效”（unsupported）；
   有效但仅由 inferred 或空值支撑时为 unverified。一条结论引用多个版本时取最差的结果。

结果：任一 unsupported 即 unsupported；否则任一 unverified 即 unverified；全部通过才交给语义检查（9.3）。
各检查都是纯函数，便于逐条构造正反例；DeterministicChecker 负责从数据库取数据并按顺序组装。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Literal

import psycopg

from eaer.domain.dates import EN_DM, EN_DMY, EN_MD, EN_MDY, EN_MY, MONTHS
from eaer.domain.evidence import content_hash, evidence_id
from eaer.domain.identity import Scope
from eaer.lifecycle.asof import validity_at

Outcome = Literal["pass", "unsupported", "unverified"]
_RANK = {"pass": 0, "unverified": 1, "unsupported": 2}


@dataclass(frozen=True)
class Check:
    outcome: Outcome
    reason: str = ""


PASS = Check("pass")


def worst(checks: list[Check]) -> Check:
    return max(checks, key=lambda c: _RANK[c.outcome], default=PASS)


# ---------------------------------------------------------------- value_match

_FULLWIDTH = str.maketrans("０１２３４５６７８９．，：／－％　", "0123456789.,:/-% ")
_ORDINAL = re.compile(r"第\s*[0-9一二三四五六七八九十百零〇]+\s*[条款项章节版页段期号份次届批轮名位]")
_YMD = re.compile(r"(?<!\d)(\d{4})\s*(?:年|-|/|\.)\s*(\d{1,2})\s*(?:月|-|/|\.)\s*(\d{1,2})\s*[日号]?")
_YM = re.compile(r"(?<!\d)(\d{4})\s*(?:年\s*(\d{1,2})\s*月|-(\d{1,2})(?![\d-]))")
_Y = re.compile(r"(?<!\d)(\d{4})\s*年")
# 不带年份的月日、月份（“3月15日”“11月6号”“自11月起”）：先认带年份的写法，剩下的才按这两种认。
# “3个月”是时长，“个”隔开了数字与“月”，不会被认成月份
_MD = re.compile(r"(?<![\d.])(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")
_M = re.compile(r"(?<![\d.])(\d{1,2})\s*月(?!\s*\d)")
# 英文日期：月份名首字母大写（“may”是情态动词），如 1 January 2024、January 1, 2024、Jan. 2024（eaer/domain/dates.py）
_MONTHS, _EN_DMY, _EN_MDY, _EN_MY = MONTHS, EN_DMY, EN_MDY, EN_MY
_NUM = re.compile(r"(?<![\dA-Za-z_.])(\d+(?:\.\d+)?)\s*(亿|万|千|thousand\b|million\b|billion\b|bn\b)?\s*"
                  r"(%|per\s*cent\b|percent\b)?")   # 紧跟字母的数字是编号（P002、v2）；percentage 不是百分号
_MULT = {None: Decimal(1), "千": Decimal(1000), "万": Decimal(10000), "亿": Decimal(100000000),
         "thousand": Decimal(1000), "million": Decimal(1000000), "billion": Decimal(1000000000), "bn": Decimal(1000000000)}
# 空格作千分位（欧盟出版物的写法：3 456、738 000）。与相邻的几个数无法区分（表格一行“9 700 9 930”），
# 所以只用在证据一侧，作为另一种读法补充进去，不替换原读法
_SPACE_THOUSANDS = re.compile(r"(?<![\d.,])\d{1,3}(?:[   ]\d{3})+(?![\d.,])")
# 证据里写明了百分比单位（表头“(%)”、图题“占比（%）”、per cent）时，其中不带 % 的数值也可以是百分比
_PCT_UNIT = re.compile(r"%|per\s*cent\b|percent\b|百分比")
_YEARS = range(1900, 2101)


@dataclass(frozen=True)
class Facts:
    quantities: frozenset[tuple[Decimal, str]]       # (数值, "num" 或 "pct")
    dates: frozenset[tuple[int | None, int | None, int | None]]   # (年, 月, 日)；不带年份的日期年为 None


def _norm_text(text: str) -> str:
    s = text.translate(_FULLWIDTH)
    s = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", s)            # 千分位
    return _ORDINAL.sub(" ", s)                              # 第三条、第 2 版之类的序号不是取值


def extract(text: str) -> Facts:
    s, dates = _strip_dates(text)
    return Facts(_quantities(s), frozenset(dates))


def _quantities(s: str) -> frozenset[tuple[Decimal, str]]:
    qs: set[tuple[Decimal, str]] = set()
    for m in _NUM.finditer(s):
        if (q := _num(m)) is not None:
            qs.add(q)
    return frozenset(qs)


def _num(m: re.Match) -> tuple[Decimal, str] | None:
    try:
        v = Decimal(m.group(1)) * _MULT[m.group(2)]
    except InvalidOperation:
        return None
    return v.normalize(), "pct" if m.group(3) else "num"


def _strip_dates(text: str) -> tuple[str, set[tuple[int | None, int | None, int | None]]]:
    """规范化后的文字与其中的日期；日期所在的位置换成空格（日期里的数字不再当数值）。"""
    s = _norm_text(text)
    dates: set[tuple[int | None, int | None, int | None]] = set()

    def take(pattern, build):
        nonlocal s
        out = []
        for m in pattern.finditer(s):
            d = build(m)
            if d is not None:
                dates.add(d)
                out.append(m.span())
        for a, b in reversed(out):
            s = s[:a] + " " + s[b:]

    def ymd(m):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return (y, mo, d) if 1 <= mo <= 12 and 1 <= d <= 31 else None

    def ym(m):
        mo = int(m.group(2) or m.group(3))
        return (int(m.group(1)), mo, None) if 1 <= mo <= 12 else None

    def en(y, mo, d=None):
        return (int(y) if y else None, _MONTHS[mo], int(d) if d else None) if d is None or 1 <= int(d) <= 31 else None

    def md(m):
        mo, d = int(m.group(1)), int(m.group(2))
        return (None, mo, d) if 1 <= mo <= 12 and 1 <= d <= 31 else None

    take(_YMD, ymd)
    take(_EN_DMY, lambda m: en(m.group(3), m.group(2), m.group(1)))
    take(_EN_MDY, lambda m: en(m.group(3), m.group(1), m.group(2)))
    take(_EN_MY, lambda m: en(m.group(2), m.group(1)))
    take(_YM, ym)
    take(_Y, lambda m: (int(m.group(1)), None, None))
    take(_MD, md)
    take(EN_MD, lambda m: en(None, m.group(1), m.group(2)))
    take(EN_DM, lambda m: en(None, m.group(2), m.group(1)))
    take(_M, lambda m: (None, int(m.group(1)), None) if 1 <= int(m.group(1)) <= 12 else None)
    return s, dates


def _space_grouped(text: str) -> set[tuple[Decimal, str]]:
    """空格分隔的数字串的各种千分位读法：“738 000 698 000”可能是 738000 与 698000，也可能是一个数，
    相邻的几组合起来的每一种都算（只用于证据一侧）。"""
    out: set[tuple[Decimal, str]] = set()
    for m in _SPACE_THOUSANDS.finditer(text.translate(_FULLWIDTH)):
        g = re.split(r"[   ]", m.group())[:12]
        out |= {(Decimal("".join(g[i:j + 1])).normalize(), "num") for i in range(len(g)) for j in range(i + 1, len(g))}
    return out


def _year(v: Decimal) -> int | None:
    """可能是年份的数值（英文与表头里的年份不写“年”，抽出来是数值）。"""
    return int(v) if v == v.to_integral_value() and int(v) in _YEARS else None


def _fmt(v: Decimal) -> str:
    s = format(v, "f")
    return s.rstrip("0").rstrip(".") if "." in s else s


def _date_found(d: tuple[int | None, int | None, int | None], pool) -> bool:
    """结论里的日期在 pool 中能找到：写了的部分都要对上；结论不带年份时只比月日（证据不带年份时，结论的年份对不上）。"""
    y, m, day = d
    return any((y is None or e[0] == y) and (m is None or e[1] == m) and (day is None or e[2] == day) for e in pool)


def _date_str(d: tuple[int | None, int | None, int | None]) -> str:
    y, m, day = d
    if y is None:
        return f"{m}月{day}日" if day else f"{m}月"
    return "-".join(str(x) for x in d if x is not None)


def _pool(texts) -> tuple[set, set]:
    """一组文字里的数值与日期（证据一侧的读法：另认空格千分位、写明百分比单位时不带 % 的数、年份与数值互通）。"""
    have_q: set = set()
    have_d: set = set()
    for t in texts:
        f = extract(t)
        qs = set(f.quantities) | _space_grouped(t)
        if _PCT_UNIT.search(_norm_text(t)):
            qs |= {(v, "pct") for v, k in qs if k == "num"}
        have_q |= qs
        have_d |= f.dates
    have_d |= {(y, None, None) for v, k in have_q if k == "num" and (y := _year(v)) is not None}
    have_q |= {(Decimal(d[0]), "num") for d in have_d if d[0] is not None}
    return have_q, have_d


def value_match(claim: str, evidence_texts: list[str], target_dates: tuple[date, ...] = (),
                labels: list[str] | tuple[str, ...] = (), *, formulas: bool = False,
                derived: frozenset[Decimal] = frozenset()) -> Check:
    """target_dates：本次查询的目标日期（生效时间点、回溯时间点）。结论复述这些日期是在说明时间依据，
    不需要出现在证据里——所引版本在目标日期是否有效由 time_valid 核对；其余日期仍须在证据中找到。
    labels：所引证据的标注（label_text），其中的数值与日期同样算找到；差与和只用原文的数值推算。
    formulas（宽松模式）：结论里写明的算式（“每晚 680 元 × 7 晚 = 4,760 元”）按 check_formulas 核对，
    成立的判 unverified 并写出算式；不成立的判 unsupported。derived：同一回答前面的结论已由算式算出的数，可作运算数。"""
    want = extract(claim)
    if not want.quantities and not want.dates:
        return Check("pass", "结论不含数值或日期")
    have_q, have_d = _pool(evidence_texts)
    lab_q, lab_d = _pool(labels)
    targets = {(t.year, t.month, t.day) for t in target_dates}
    lacking = [(v, k) for v, k in sorted(want.quantities) if (v, k) not in have_q | lab_q]
    missing = [f"{_fmt(v)}{'%' if k == 'pct' else ''}" for v, k in lacking]
    dates = [_date_str(d) for d in sorted(want.dates, key=str)
             if not _date_found(d, have_d | lab_d) and not _date_found(d, targets)]
    if not missing and not dates:
        return PASS
    if formulas and not dates:
        f = check_formulas(claim, {v for v, _ in have_q} | set(derived))
        if f.wrong:
            return Check("unsupported", f"算式不成立：{'；'.join(f.wrong)}")
        if f.valid and all(v in f.explained for v, _ in lacking):
            counts = f"，其中证据以外的数量（{'、'.join(_fmt(c) for c in f.counts)}）取自问题" if f.counts else ""
            return Check("unverified", f"{FORMULA_REASON}：{'；'.join(f.valid)}；代码核对了算术{counts}，含义待核实")
    sums = None if dates else _derivations([v for v, _ in lacking], have_q)
    if sums:
        return Check("unverified", f"数值由所引证据计算得出：{'；'.join(sums)}，代码只核对了算术，含义待核实")
    return Check("unsupported", f"证据中没有：{'、'.join(missing + dates)}")


# ---------------------------------------------------------------- 算式（宽松模式）

FORMULA_REASON = "数值按算式由所引证据计算得出"

# 运算符：数值之间的文字里找。“元/晚”里的斜杠是单位，不是除号：只有两数之间只隔一个“/”才算除
_OP = re.compile(r"乘以|除以|加上|减去|[×✕*÷+\-−]|(?<![A-Za-z])[xX](?![A-Za-z])")
_OP_OF = {"乘以": "*", "×": "*", "✕": "*", "*": "*", "x": "*", "X": "*", "除以": "/", "÷": "/", "加上": "+", "+": "+",
          "减去": "-", "-": "-", "−": "-"}
_OP_SHOW = {"*": "×", "/": "÷", "+": "+", "-": "−"}
_CLAUSE = re.compile(r"[，。；;,：:！？!?\n（）()【】]")
_EQ_SIGNS = str.maketrans({"＝": "=", "＋": "+", "＊": "*", "≈": "="})


@dataclass
class Formulas:
    valid: list[str] = field(default_factory=list)       # 成立且运算数都有来历的算式（写进原因）
    wrong: list[str] = field(default_factory=list)       # 算术不成立的算式
    explained: set[Decimal] = field(default_factory=set)  # 由成立的算式说明了来历的数（运算数与结果）
    counts: list[Decimal] = field(default_factory=list)   # 证据以外、作为乘除因子的数量（晚数、天数、人数）
    results: set[Decimal] = field(default_factory=set)    # 成立的算式算出的数


def _segment(seg: str) -> list | None:
    """等号之间的一段：数值与运算符交替（[数, 符, 数, …]）；数值之间的文字里必须有且只有一种运算符，否则不是算式。"""
    nums = list(_NUM.finditer(seg))
    if not nums:
        return None
    out: list = []
    for i, m in enumerate(nums):
        q = _num(m)
        if q is None:
            return None
        if i:
            between = seg[nums[i - 1].end():m.start()]
            ops = {"/"} if between.strip() == "/" else {_OP_OF[t] for t in _OP.findall(between)}
            if len(ops) != 1:
                return None
            out.append(ops.pop())
        out.append(q)
    return out


def _evaluate(items: list) -> Decimal | None:
    """按先乘除后加减求值；乘除里的百分数按小数算（680 × 80% = 544）。"""
    terms: list[list] = [[("+", items[0])]]
    for op, x in zip(items[1::2], items[2::2]):
        if op in "*/":
            terms[-1].append((op, x))
        else:
            terms.append([(op, x)])
    total = Decimal(0)
    for t in terms:
        product = len(t) > 1
        val = lambda q: q[0] / 100 if product and q[1] == "pct" else q[0]   # noqa: E731
        acc = val(t[0][1])
        for op, x in t[1:]:
            if op == "*":
                acc *= val(x)
            elif val(x) == 0:
                return None
            else:
                acc /= val(x)
        total += -acc if t[0][0] == "-" else acc
    return total


def _show(items: list) -> str:
    return " ".join(_OP_SHOW[x] if isinstance(x, str) else _fmt(x[0]) + ("%" if x[1] == "pct" else "") for x in items)


def _same(a: Decimal, b: Decimal, written: Decimal | None) -> bool:
    """按结论写出的那个数的小数位数比较（1000 ÷ 3 = 333.33）；两边都是算式时要求相等。"""
    if written is None:
        return abs(a - b) < Decimal("1e-9")
    exp = written.as_tuple().exponent
    q = Decimal(1).scaleb(exp if isinstance(exp, int) and exp < 0 else 0)
    return a.quantize(q) == b.quantize(q)


def _sources_ok(items: list, available: set[Decimal]) -> tuple[bool, list[Decimal]]:
    """运算数的来历：在所引证据中（或前面算出的）；或者是乘除式里的正整数因子（问题给的晚数、天数、人数），
    且同一个乘除式里至少有一个数在证据中。返回 (是否都有来历, 用到的证据以外的因子)。"""
    terms: list[list] = [[items[0]]]
    for op, x in zip(items[1::2], items[2::2]):
        (terms[-1].append if op in "*/" else lambda q: terms.append([q]))(x)
    counts: list[Decimal] = []
    for t in terms:
        known = [q[0] in available for q in t]
        if len(t) == 1:
            if not known[0]:
                return False, []
            continue
        if not any(known):
            return False, []
        for q, k in zip(t, known):
            if not k:
                if q[1] != "num" or q[0] <= 0 or q[0] != q[0].to_integral_value():
                    return False, []
                counts.append(q[0])
    return True, counts


def check_formulas(claim: str, available: set[Decimal]) -> Formulas:
    """结论里写明的算式（宽松模式）：逐个分句找“表达式 = 表达式 [= …]”，核对算术与运算数的来历（见 _sources_ok）。
    前面成立的算式算出的数，后面的算式可以接着用（“680 × 7 = 4,760，4,760 + 840 = 5,600”）。"""
    s, _ = _strip_dates(claim)
    out, avail = Formulas(), set(available)
    for clause in _CLAUSE.split(s.translate(_EQ_SIGNS)):
        if "=" not in clause:
            continue
        segs = [_segment(x) for x in clause.split("=")]
        if len(segs) < 2 or any(x is None for x in segs):
            continue
        vals = [_evaluate(x) for x in segs]
        if any(v is None for v in vals):
            continue
        written = [x[0][0] if len(x) == 1 else None for x in segs]
        text = " = ".join(_show(x) for x in segs)
        if not all(_same(a, b, wb if wb is not None else wa)
                   for a, b, wa, wb in zip(vals, vals[1:], written, written[1:])):
            out.wrong.append(f"{text}（左边算出来是 {_fmt(vals[0].normalize())}）")
            continue
        ok, counts = _sources_ok(segs[0], avail)
        if not ok:
            continue
        out.valid.append(text)
        out.counts += [c for c in dict.fromkeys(counts) if c not in out.counts]
        nums = {q[0] for x in segs for q in x if not isinstance(q, str)}
        out.explained |= nums
        out.results |= {w for w in written[1:] if w is not None} | {v.normalize() for v in vals}
        avail |= out.results
    return out


def _derivations(values: list[Decimal], have_q: set) -> list[str] | None:
    """结论里证据中没有的数值，是否都等于所引证据中两个数值的差或和（按结论写的小数位数比较）：
    “相差 10 个百分点”（96% 与 86%）、“上升了 2 个百分点”、“两项合计 1,610”。年份不参与（2023 − 2018 = 5 之类
    几乎什么都能凑出来）。有一个凑不出就返回 None。"""
    pool = sorted({v for v, _ in have_q if _year(v) is None})
    if len(pool) < 2 or len(pool) > 60:
        return None
    out = []
    for want in values:
        exp = want.as_tuple().exponent if want.as_tuple().exponent < 0 else 0
        q = Decimal(1).scaleb(exp)
        hit = next((f"{_fmt(want)} = {_fmt(a)} {op} {_fmt(b)}" for i, a in enumerate(pool) for b in pool[:i]
                    for op, r in (("−", a - b), ("+", a + b)) if r.quantize(q) == want.quantize(q)), None)
        if hit is None:
            return None
        out.append(hit)
    return out


def formula_results(claim: str) -> set[Decimal]:
    """结论里算术成立的算式算出的数（不论运算数的来历）。只用于已通过 value_match 的结论：同一回答后面的结论
    可以接着用这些数（“住宿 4,760 元、伙食 840 元，合计 4,760 + 840 = 5,600 元”分在两条结论里写）。"""
    s, _ = _strip_dates(claim)
    return check_formulas(claim, {q[0] for q in _quantities(s)}).results


# ---------------------------------------------------------------- locator_exists

def _nonneg_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def locator_problem(modality: str, locator: Any) -> str | None:
    """定位的形状是否成立；成立返回 None，否则返回原因。"""
    if not isinstance(locator, dict):
        return "定位不是对象"
    sec = locator.get("section", [])
    if not isinstance(sec, list) or not all(isinstance(x, str) for x in sec):
        return "章节路径格式错误"
    if "page" in locator and not (_nonneg_int(locator["page"]) and locator["page"] >= 1):
        return "页码必须是从 1 开始的整数"
    if modality == "text":
        return None if _nonneg_int(locator.get("para")) else "文本证据缺少段落位置"
    if modality == "table":
        if not _nonneg_int(locator.get("table")):
            return "表格证据缺少表格序号"
        if not (_nonneg_int(locator.get("row")) and locator["row"] >= 1):
            return "表格行号必须从 1 开始（第 0 行是表头）"
        if not (_nonneg_int(locator.get("column")) and locator["column"] >= 1):
            return "表格列号必须从 1 开始（第 0 列是行名）"
        return None
    if modality == "figure":
        if not _nonneg_int(locator.get("figure")):
            return "图表证据缺少图表序号"
        box = locator.get("bbox")
        if box is not None:
            ok = (isinstance(box, list) and len(box) == 4 and all(isinstance(x, (int, float)) for x in box)
                  and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)
            if not ok:
                return "图表区域必须是页面内的 [x0, y0, x1, y1]"
        return None
    return f"未知模态 {modality}"


def locator_exists(row: dict | None, *, raw_exists: bool | None = True) -> Check:
    """row：evidence 表的一行（含 version_id），None 表示证据不存在。"""
    if row is None:
        return Check("unsupported", "证据不存在")
    problem = locator_problem(row["modality"], row["locator"])
    if problem:
        return Check("unsupported", problem)
    if content_hash(row["content"]) != row["content_hash"]:
        return Check("unsupported", "证据内容与内容哈希不符")
    if evidence_id(str(row["version_id"]), row["modality"], row["locator"], row["content_hash"]) != row["id"]:
        return Check("unsupported", "证据 id 与定位、内容不符")
    if raw_exists is False:
        return Check("unsupported", "原件已不在文件存储中")
    return PASS


# ---------------------------------------------------------------- time_valid

def label_text(doc_type: str | None, title: str, version_no: int | None, doc_date: date | None,
               validity: dict | None, publisher: str | None) -> str:
    """一条证据的标注里给模型看的元数据（与 eaer.orchestrator.fast.evidence_block 相同的取舍）：标题与版本号；
    记录类给文档日期，其他给目标时间的生效与作废日期；有出处时给出处。日期写成 ISO 格式。"""
    parts = [f"《{title}》第 {version_no} 版"]
    if doc_type == "record":
        if doc_date is not None:
            parts.append(f"文档日期 {doc_date.isoformat()}")
    elif validity:
        if validity.get("valid_from") is not None:
            parts.append(f"生效 {validity['valid_from'].isoformat()}")
        if validity.get("valid_to") is not None:
            parts.append(f"作废 {validity['valid_to'].isoformat()}")
    if publisher:
        parts.append(f"出处 {publisher}")
    return "；".join(parts)


def time_valid(doc_type: str | None, doc_date: date | None, validity: dict | None, on: date) -> Check:
    """validity：目标时间点所用的有效期行（当前或回溯时刻记录的），None 表示那时没有记录。"""
    if doc_type == "record":
        if doc_date is None:
            return Check("unverified", "记录类文档没有文档日期")
        return PASS if doc_date <= on else Check("unsupported", f"文档日期 {doc_date} 晚于目标时间 {on}")
    if validity is None:
        return Check("unsupported", "目标时间没有该版本的有效期记录")
    vf, vt = validity["valid_from"], validity["valid_to"]
    fg, tg = validity["valid_from_grade"], validity["valid_to_grade"]
    if vt is not None and vt <= on:
        return (Check("unsupported", f"已于 {vt} 作废") if tg == "explicit"
                else Check("unverified", f"推断已于 {vt} 作废（{tg}）"))
    if vf is not None and vf > on:
        return (Check("unsupported", f"{vf} 才生效") if fg == "explicit"
                else Check("unverified", f"推断 {vf} 才生效（{fg}）"))
    if vf is None:
        return Check("unverified", "生效日为空")
    if fg != "explicit":
        return Check("unverified", f"生效日来源为 {fg}")
    if vt is not None and tg != "explicit":
        return Check("unverified", f"作废日来源为 {tg}")
    return PASS


# ---------------------------------------------------------------- 组装


@dataclass(frozen=True)
class Target:
    """目标时间：on 为判断有效的日期；system_t 不为空时是回溯查询（按当时记录的有效期）。"""
    on: date
    system_t: datetime | None = None


@dataclass
class Result:
    outcome: Outcome
    checks: dict[str, Check]
    evidence_ids: list[str]                        # 移除无权证据后剩下的
    removed: list[str] = field(default_factory=list)
    alerts: list[str] = field(default_factory=list)


class DeterministicChecker:
    def __init__(self, conn: psycopg.Connection, authorizer, scope: Scope, store=None):
        self.conn, self.authorizer, self.scope, self.store = conn, authorizer, scope, store

    def check(self, claim: str, evidence_ids: list[str], *, retrieved: set[str], target: Target | None,
              version_targets: dict[str, Target] | None = None, formulas: bool = False,
              derived: frozenset[Decimal] = frozenset()) -> Result:
        """version_targets：版本 id → 该版本的目标时间。Deep 的版本比较里，“修订前”的结论引用旧版本，
        要按旧版本有效期内的日期核对 time_valid，而不是按提问日；没有列出的版本用 target。
        formulas、derived：宽松模式核对结论里写明的算式（见 value_match）。"""
        checks: dict[str, Check] = {}
        vt = version_targets or {}
        foreign = [i for i in evidence_ids if i not in retrieved]
        if not evidence_ids or foreign:
            reason = "结论没有引用证据" if not evidence_ids else f"引用了本次检索结果以外的证据：{'、'.join(foreign)}"
            checks["locator_exists"] = Check("unsupported", reason)
            return Result("unsupported", checks, list(evidence_ids))

        allowed = self.authorizer.check(self.scope, list(evidence_ids))
        removed = [i for i, ok in zip(evidence_ids, allowed) if not ok]
        keep = [i for i, ok in zip(evidence_ids, allowed) if ok]
        alerts = [f"安全告警：授权核对未通过，已移除证据 {i}" for i in removed]
        checks["authorized"] = (Check("unsupported", "所引证据全部未通过授权核对") if not keep
                                else Check("pass", f"移除 {len(removed)} 条" if removed else ""))
        if not keep:
            return Result("unsupported", checks, keep, removed, alerts)

        rows = {r["id"]: r for r in self.conn.execute(
            "SELECT e.id, e.version_id, e.modality, e.locator, e.content, e.content_hash, v.raw_object_uri, "
            "  v.doc_date, v.approved_at, v.version_no, v.publisher, d.doc_type, d.title "
            "FROM evidence e JOIN document_version v ON v.id = e.version_id "
            "JOIN document d ON d.id = v.document_id WHERE e.id = ANY(%s)", (keep,))}
        checks["locator_exists"] = worst([
            locator_exists(rows.get(i), raw_exists=(self.store.exists(rows[i]["raw_object_uri"])
                                                     if self.store is not None and i in rows else None))
            for i in keep])
        targets: set[date] = set()
        for t in ([target] if target is not None else []) + [vt[str(rows[i]["version_id"])] for i in keep
                                                              if i in rows and str(rows[i]["version_id"]) in vt]:
            targets |= {t.on} | ({t.system_t.date()} if t.system_t is not None else set())
        versions = {rows[i]["version_id"]: rows[i] for i in keep if i in rows}
        validity = {vid: self._validity(vid, vt.get(str(vid), target)) for vid in versions}
        labels = [label_text(r["doc_type"], r["title"], r["version_no"], r["doc_date"], validity[vid], r["publisher"])
                  for vid, r in versions.items()]
        checks["value_match"] = value_match(claim, [rows[i]["content"] for i in keep if i in rows], tuple(targets),
                                            labels, formulas=formulas, derived=derived)
        checks["time_valid"] = self._time(rows, keep, target, vt, validity)
        return Result(worst(list(checks.values())).outcome, checks, keep, removed, alerts)

    def _validity(self, vid, target: Target | None) -> dict | None:
        """版本在目标时间所用的有效期行：回溯查询取当时记录的，否则取当前的；没有记录时为 None。"""
        if target is not None and target.system_t is not None:
            return validity_at(self.conn, vid, target.system_t)
        return self.conn.execute(
            "SELECT valid_from, valid_from_grade, valid_to, valid_to_grade FROM version_validity "
            "WHERE version_id = %s AND recorded_at IS NOT NULL AND replaced_at IS NULL", (vid,)).fetchone()

    def _time(self, rows: dict, keep: list[str], default: Target | None, vt: dict[str, Target],
              validity: dict) -> Check:
        if default is None and not vt:
            return Check("pass", "不涉及目标时间")
        results = []
        for vid in {rows[i]["version_id"] for i in keep if i in rows}:
            target = vt.get(str(vid), default)
            if target is None:
                results.append(Check("pass", "不涉及目标时间"))
                continue
            r = next(rows[i] for i in keep if i in rows and rows[i]["version_id"] == vid)
            if target.system_t is not None:
                if r["approved_at"] is None or r["approved_at"] > target.system_t:
                    results.append(Check("unsupported", "回溯时间点该版本尚未审核通过"))
                    continue
            results.append(time_valid(r["doc_type"], r["doc_date"], validity[vid], target.on))
        return worst(results)
