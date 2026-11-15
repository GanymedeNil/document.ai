"""英文日期的月份名（验证层的数值与日期核对、入库的文档日期抽取共用）。

月份名首字母大写才认（“may”是情态动词）；可写全称或缩写，缩写后可带点：January、Jan、Jan.、Sept.。
"""
from __future__ import annotations

import re
from datetime import date

MONTHS = {m: k + 1 for k, ms in enumerate([
    ("January", "Jan"), ("February", "Feb"), ("March", "Mar"), ("April", "Apr"), ("May",), ("June", "Jun"),
    ("July", "Jul"), ("August", "Aug"), ("September", "Sep", "Sept"), ("October", "Oct"), ("November", "Nov"),
    ("December", "Dec")]) for m in ms}
# 月份名（一个捕获组）；长的在前，Sept 先于 Sep
MON = r"(" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b\.?"
# 不带捕获组的写法，用来拼进更大的日期模式
MON_NC = r"(?:" + "|".join(sorted(MONTHS, key=len, reverse=True)) + r")\b\.?"
_ORD = r"(?:st|nd|rd|th)?"
EN_DMY = re.compile(r"(?<!\d)(\d{1,2})" + _ORD + r"\s+" + MON + r"\s+(\d{4})(?!\d)")          # 30 October 2023
EN_MDY = re.compile(r"\b" + MON + r"\s+(\d{1,2})" + _ORD + r",?\s+(\d{4})(?!\d)")             # October 30, 2023
EN_MY = re.compile(r"\b" + MON + r"\s+(\d{4})(?!\d)")                                        # October 2023
# 不带年份的月日（年份由上下文或文档日期给出）：October 30、30 October。先认完整日期，剩下的才按这两种认
EN_MD = re.compile(r"\b" + MON + r"\s+(\d{1,2})" + _ORD + r"(?![\d.,]\d)")                     # October 30
EN_DM = re.compile(r"(?<![\d.,])(\d{1,2})" + _ORD + r"\s+" + MON)                              # 30 October
# 完整的英文日期（年月日），不带捕获组
EN_DATE = (r"(?:\d{1,2}" + _ORD + r"\s+" + MON_NC + r"\s+\d{4}|" + MON_NC + r"\s+\d{1,2}" + _ORD + r",?\s+\d{4})")


def parse_en(text: str) -> date | None:
    """一个完整的英文日期（“October 30, 2023”“30 Oct. 2023”）；不是或不合法时为 None。"""
    s = " ".join(text.split())
    if m := EN_MDY.fullmatch(s):
        mo, d, y = m[1], m[2], m[3]
    elif m := EN_DMY.fullmatch(s):
        d, mo, y = m[1], m[2], m[3]
    else:
        return None
    try:
        return date(int(y), MONTHS[mo], int(d))
    except ValueError:
        return None
