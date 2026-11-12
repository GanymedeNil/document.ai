"""数据库连接。连接串来自环境变量 EAER_DATABASE_URL，格式为 postgresql://用户@主机/库名。"""
from __future__ import annotations

import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

MIGRATIONS = Path(__file__).resolve().parent / "migrations"


def database_url() -> str:
    url = os.environ.get("EAER_DATABASE_URL")
    if not url:
        raise RuntimeError("未设置环境变量 EAER_DATABASE_URL")
    return url


def sqlalchemy_url(url: str | None = None) -> str:
    url = url or database_url()
    return url.replace("postgresql://", "postgresql+psycopg://", 1)


def connect(url: str | None = None) -> psycopg.Connection:
    return psycopg.connect(url or database_url(), row_factory=dict_row)


def migrate(url: str, revision: str = "head") -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", sqlalchemy_url(url))
    command.upgrade(cfg, revision)


def downgrade(url: str, revision: str = "base") -> None:
    from alembic import command
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(MIGRATIONS))
    cfg.set_main_option("sqlalchemy.url", sqlalchemy_url(url))
    command.downgrade(cfg, revision)
