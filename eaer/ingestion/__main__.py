"""后台任务进程入口：python -m eaer.ingestion

环境变量同接口进程（EAER_DATABASE_URL、EAER_STORAGE_DIR）。空闲时每隔 EAER_POLL_SECONDS 秒（默认 2）查看一次任务。
"""
import os
import time

import psycopg
from psycopg.rows import dict_row

from eaer.ingestion.pipeline import Ingestion, Worker
from eaer.storage import LocalFileStore


def main() -> None:
    poll = float(os.environ.get("EAER_POLL_SECONDS", "2"))
    conn = psycopg.connect(os.environ["EAER_DATABASE_URL"], autocommit=True, row_factory=dict_row)
    worker = Worker(Ingestion(conn, LocalFileStore(os.environ["EAER_STORAGE_DIR"])))
    while True:
        if worker.run_until_idle(limit=1000) == 0:
            time.sleep(poll)


if __name__ == "__main__":
    main()
