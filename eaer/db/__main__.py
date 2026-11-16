"""数据库迁移入口（部署用）：

    python -m eaer.db upgrade        迁移到最新版本（可重复执行，已是最新则什么都不做）
    python -m eaer.db current        显示当前版本

连接串取环境变量 EAER_DATABASE_URL。精简版的编排文件在启动接口与后台进程之前先运行 upgrade。
"""
from __future__ import annotations

import argparse
import sys

from eaer.db import connect, database_url, migrate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="eaer.db")
    p.add_argument("command", choices=["upgrade", "current"])
    args = p.parse_args(argv)
    url = database_url()
    if args.command == "upgrade":
        migrate(url)
    with connect(url) as conn:
        row = conn.execute("SELECT version_num FROM alembic_version").fetchone()
    print(f"数据库版本：{row['version_num'] if row else '未迁移'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
