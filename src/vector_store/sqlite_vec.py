"""
sqlite-vec vector store implementation.

This backend keeps vectors in regular SQLite tables and uses sqlite-vec scalar
distance functions for similarity search. It is slower than pgvector/HNSW but
keeps Honcho fully self-contained in a single SQLite database.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any

import sqlite_vec

from src.config import settings

from . import VectorQueryResult, VectorRecord, VectorStore, VectorUpsertResult

logger = logging.getLogger(__name__)

_VALID_IDENTIFIER_PATTERN = re.compile(r"[^a-zA-Z0-9_]+")


class SQLiteVecVectorStore(VectorStore):
    _conn: sqlite3.Connection | None
    _lock: asyncio.Lock

    def __init__(self) -> None:
        super().__init__()
        self._db_path = settings.VECTOR_STORE.SQLITE_PATH
        self._conn = None
        self._lock = asyncio.Lock()

    def _table_name(self, namespace: str) -> str:
        sanitized = _VALID_IDENTIFIER_PATTERN.sub("_", namespace)
        return f"vec_{sanitized}"

    def _connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn

        db_path = Path(self._db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        self._conn = conn
        return conn

    def _ensure_table(self, namespace: str) -> str:
        conn = self._connect()
        table_name = self._table_name(namespace)
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS "{table_name}" (
                id TEXT PRIMARY KEY,
                embedding BLOB NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{{}}'
            )
            """
        )
        return table_name

    def _upsert_many_sync(
        self,
        namespace: str,
        vectors: list[VectorRecord],
    ) -> VectorUpsertResult:
        if not vectors:
            return VectorUpsertResult(ok=True)

        conn = self._connect()
        table_name = self._ensure_table(namespace)
        rows = [
            (
                vector.id,
                json.dumps(vector.embedding),
                json.dumps(vector.metadata or {}),
            )
            for vector in vectors
        ]
        conn.executemany(
            f"""
            INSERT INTO "{table_name}" (id, embedding, metadata)
            VALUES (?, vec_f32(?), ?)
            ON CONFLICT(id) DO UPDATE SET
                embedding = excluded.embedding,
                metadata = excluded.metadata
            """,
            rows,
        )
        conn.commit()
        return VectorUpsertResult(ok=True)

    async def upsert_many(
        self,
        namespace: str,
        vectors: list[VectorRecord],
    ) -> VectorUpsertResult:
        async with self._lock:
            return await asyncio.to_thread(self._upsert_many_sync, namespace, vectors)

    def _query_sync(
        self,
        namespace: str,
        embedding: list[float],
        *,
        top_k: int,
        filters: dict[str, Any] | None,
        max_distance: float | None,
    ) -> list[VectorQueryResult]:
        conn = self._connect()
        table_name = self._ensure_table(namespace)
        params: list[Any] = [json.dumps(embedding)]
        where_clauses = ["1 = 1"]

        if filters:
            for key, value in filters.items():
                where_clauses.append(f"json_extract(metadata, '$.{key}') = ?")
                params.append(value)

        if max_distance is not None:
            where_clauses.append("vec_distance_cosine(embedding, vec_f32(?)) <= ?")
            params.extend([json.dumps(embedding), max_distance])

        params.append(top_k)
        query = f"""
            SELECT
                id,
                metadata,
                vec_distance_cosine(embedding, vec_f32(?)) AS distance
            FROM "{table_name}"
            WHERE {' AND '.join(where_clauses)}
            ORDER BY distance
            LIMIT ?
        """
        rows = conn.execute(query, params).fetchall()
        results: list[VectorQueryResult] = []
        for row in rows:
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            results.append(
                VectorQueryResult(
                    id=row["id"],
                    score=row["distance"],
                    metadata=metadata,
                )
            )
        return results

    async def query(
        self,
        namespace: str,
        embedding: list[float],
        *,
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
        max_distance: float | None = None,
    ) -> list[VectorQueryResult]:
        async with self._lock:
            return await asyncio.to_thread(
                self._query_sync,
                namespace,
                embedding,
                top_k=top_k,
                filters=filters,
                max_distance=max_distance,
            )

    def _delete_many_sync(self, namespace: str, ids: list[str]) -> None:
        if not ids:
            return
        conn = self._connect()
        table_name = self._ensure_table(namespace)
        placeholders = ",".join("?" for _ in ids)
        conn.execute(f'DELETE FROM "{table_name}" WHERE id IN ({placeholders})', ids)
        conn.commit()

    async def delete_many(self, namespace: str, ids: list[str]) -> None:
        async with self._lock:
            await asyncio.to_thread(self._delete_many_sync, namespace, ids)

    def _delete_namespace_sync(self, namespace: str) -> None:
        conn = self._connect()
        table_name = self._table_name(namespace)
        conn.execute(f'DROP TABLE IF EXISTS "{table_name}"')
        conn.commit()

    async def delete_namespace(self, namespace: str) -> None:
        async with self._lock:
            await asyncio.to_thread(self._delete_namespace_sync, namespace)

    async def close(self) -> None:
        async with self._lock:
            if self._conn is None:
                return
            conn = self._conn
            self._conn = None
            await asyncio.to_thread(conn.close)
