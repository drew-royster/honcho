import pytest

from src.config import settings
from src.vector_store import VectorRecord
from src.vector_store.sqlite_vec import SQLiteVecVectorStore


@pytest.mark.asyncio
async def test_sqlite_vec_query_supports_in_filters(tmp_path):
    original_sqlite_path = settings.VECTOR_STORE.SQLITE_PATH
    settings.VECTOR_STORE.SQLITE_PATH = str(tmp_path / "sqlite-vec.db")

    store = SQLiteVecVectorStore()
    namespace = "test.document.filters"

    try:
        await store.upsert_many(
            namespace,
            [
                VectorRecord(
                    id="explicit-doc",
                    embedding=[1.0, 0.0, 0.0],
                    metadata={"level": "explicit", "session_name": "alpha"},
                ),
                VectorRecord(
                    id="derived-doc",
                    embedding=[0.9, 0.1, 0.0],
                    metadata={"level": "deductive", "session_name": "alpha"},
                ),
                VectorRecord(
                    id="other-doc",
                    embedding=[0.0, 1.0, 0.0],
                    metadata={"level": "contradiction", "session_name": "beta"},
                ),
            ],
        )

        results = await store.query(
            namespace,
            [1.0, 0.0, 0.0],
            top_k=10,
            filters={
                "level": {"in": ["explicit", "deductive"]},
                "session_name": "alpha",
            },
        )

        assert [result.id for result in results] == ["explicit-doc", "derived-doc"]
    finally:
        await store.close()
        settings.VECTOR_STORE.SQLITE_PATH = original_sqlite_path
