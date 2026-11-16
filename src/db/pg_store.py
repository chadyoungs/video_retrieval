from __future__ import annotations
import logging
from typing import Any, List
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.db.relational_base import RelationalStoreBase
from app.config.settings import settings

logger = logging.getLogger(__name__)


class PostgresRelationalStore(RelationalStoreBase):
    def __init__(self, dataset_key: str | None = None):
        self.dataset_key = dataset_key
        self.engine = create_engine(settings.pg_dsn, pool_pre_ping=True)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)

    def init_schema(self):
        """
        Alembic 负责建表/迁移，这里只做校验；不要在这里写 CREATE TABLE。
        这是关键：表结构交给 alembic，代码内不维护 DDL。
        """
        with self.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("PostgreSQL store connected, dataset_key=%s", self.dataset_key)

    def drop(self):
        """谨慎：业务一般不调用；如需清数据用 TRUNCATE 按 dataset_name 过滤。"""
        logger.warning("Drop operation skipped for Postgres; use alembic downgrade or manual truncate.")

    def insert_records(self, records: List[dict[str, Any]]):
        """批量写入，records 必须包含 dataset_name"""
        if not records:
            return
        stmt = text("""
            INSERT INTO video_segments (
                video_file_name,
                video_file_path,
                segment_start,
                segment_end,
                annotation,
                metadata,
                dataset_name,
                clip_id
            )
            VALUES (
                :video_file_name,
                :video_file_path,
                :segment_start,
                :segment_end,
                :annotation,
                :metadata,
                :dataset_name,
                :clip_id
            )
        """)
        with self.engine.begin() as conn:
            conn.execute(stmt, records)
        logger.debug("Inserted %d rows to postgres, dataset=%s", len(records), self.dataset_key)

    def search_metadata(self, filter_spec: dict[str, Any]) -> List[dict[str, Any]]:
        """
        元数据过滤查询，兼容你原有规则检索接口；
        filter_spec 示例：{"dataset_name": "activitynet", "clip_id": "xxx"}
        复杂 OR/AND 规则可以在这层继续扩展。
        """
        conditions = []
        params = {}
        for idx, (k, v) in enumerate(filter_spec.items()):
            key = f"p_{idx}"
            conditions.append(f"{k} = :{key}")
            params[key] = v
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        sql = text(f"SELECT * FROM video_segments WHERE {where_clause}")
        with self.engine.connect() as conn:
            res = conn.execute(sql, params)
            return [dict(r) for r in res.fetchall()]

    def close(self):
        self.engine.dispose()
