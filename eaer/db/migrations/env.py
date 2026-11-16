"""Alembic 运行环境。迁移只执行原生 SQL，不使用 ORM 元数据。"""
from alembic import context
from sqlalchemy import create_engine

from eaer.db import sqlalchemy_url

config = context.config
url = config.get_main_option("sqlalchemy.url") or sqlalchemy_url()


def run_migrations_offline() -> None:
    context.configure(url=url, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(url)
    with engine.connect() as conn:
        context.configure(connection=conn, transaction_per_migration=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
