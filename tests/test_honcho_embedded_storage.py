import json

import pytest

from src.honcho_embedded import (
    EmbeddedHonchoLLMConfig,
    EmbeddedHonchoRuntimeConfig,
    describe_embedding_runtime,
    embedding_storage_dir,
    honcho_storage_has_content,
    legacy_memory_seed_marker_path,
    mark_legacy_memory_seeded,
    prepare_embedding_storage,
)


@pytest.fixture(autouse=True)
def mock_tracked_db():
    yield


def test_describe_embedding_runtime_uses_provider_defaults():
    runtime = EmbeddedHonchoRuntimeConfig(
        embedding=EmbeddedHonchoLLMConfig(provider="openai")
    )

    descriptor = describe_embedding_runtime(runtime)

    assert descriptor["provider"] == "openai"
    assert descriptor["model"] == "text-embedding-3-small"
    assert descriptor["dimensions"] == 1536
    assert descriptor["storage_key"].startswith("emb-")


def test_describe_embedding_runtime_changes_when_model_changes():
    runtime_a = EmbeddedHonchoRuntimeConfig(
        embedding=EmbeddedHonchoLLMConfig(
            provider="openrouter",
            model="qwen/qwen3-embedding-4b",
            base_url="https://openrouter.ai/api/v1/",
        )
    )
    runtime_b = EmbeddedHonchoRuntimeConfig(
        embedding=EmbeddedHonchoLLMConfig(
            provider="openrouter",
            model="baai/bge-m3",
            base_url="https://openrouter.ai/api/v1",
        )
    )

    descriptor_a = describe_embedding_runtime(runtime_a)
    descriptor_b = describe_embedding_runtime(runtime_b)

    assert descriptor_a["storage_key"] != descriptor_b["storage_key"]
    assert descriptor_a["fingerprint"] != descriptor_b["fingerprint"]


def test_prepare_embedding_storage_moves_legacy_flat_db(tmp_path):
    hermes_home = tmp_path
    root = hermes_home / "honcho"
    root.mkdir()
    (root / "honcho.db").write_text("legacy-db", encoding="utf-8")
    (root / "honcho.db-wal").write_text("legacy-wal", encoding="utf-8")

    runtime = EmbeddedHonchoRuntimeConfig(
        embedding=EmbeddedHonchoLLMConfig(provider="gemini")
    )

    payload = prepare_embedding_storage(str(hermes_home), runtime)
    storage_dir = embedding_storage_dir(str(hermes_home), runtime)

    assert not (root / "honcho.db").exists()
    assert not (root / "honcho.db-wal").exists()
    assert (storage_dir / "honcho.db").read_text(encoding="utf-8") == "legacy-db"
    assert (storage_dir / "honcho.db-wal").read_text(encoding="utf-8") == "legacy-wal"

    active = json.loads((root / "active-embedding-space.json").read_text())
    per_store = json.loads((storage_dir / "embedding-space.json").read_text())
    assert active["storage_key"] == payload["storage_key"]
    assert per_store["fingerprint"] == payload["fingerprint"]


def test_honcho_storage_has_content_detects_existing_rows(tmp_path):
    db_path = tmp_path / "honcho.db"
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("INSERT INTO messages (content) VALUES ('existing-memory')")
    conn.commit()
    conn.close()

    assert honcho_storage_has_content(db_path)


def test_honcho_storage_has_content_ignores_empty_tables(tmp_path):
    db_path = tmp_path / "honcho.db"
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("CREATE TABLE documents (id INTEGER PRIMARY KEY, content TEXT)")
    conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, name TEXT)")
    conn.commit()
    conn.close()

    assert not honcho_storage_has_content(db_path)


def test_mark_legacy_memory_seeded_writes_namespaced_marker(tmp_path):
    storage_dir = tmp_path / "honcho" / "stores" / "emb-test"
    storage_dir.mkdir(parents=True)

    marker = mark_legacy_memory_seeded(storage_dir, reason="existing_store")

    assert marker == legacy_memory_seed_marker_path(storage_dir)
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["reason"] == "existing_store"
