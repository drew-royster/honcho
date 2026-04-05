"""resize pgvector embeddings to configured dimensions

Revision ID: 2f14fd0951c2
Revises: e4eba9cfaa6f
Create Date: 2026-04-05 20:30:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

from migrations.utils import column_exists, get_schema, index_exists
from src.config import settings

# revision identifiers, used by Alembic.
revision: str = "2f14fd0951c2"
down_revision: str | None = "e4eba9cfaa6f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

schema = get_schema()
LEGACY_VECTOR_DIMENSIONS = 1536
TARGET_VECTOR_DIMENSIONS = settings.VECTOR_STORE.DIMENSIONS


def _ensure_no_incompatible_embeddings(table_name: str, target_dimensions: int) -> None:
    conn = op.get_bind()
    incompatible_rows = conn.execute(
        sa.text(
            f"""
            SELECT COUNT(*)
            FROM "{schema}"."{table_name}"
            WHERE embedding IS NOT NULL
              AND vector_dims(embedding) <> :target_dimensions
            """
        ),
        {"target_dimensions": target_dimensions},
    ).scalar_one()

    if incompatible_rows:
        raise RuntimeError(
            f"Cannot change {table_name}.embedding to vector({target_dimensions}) "
            + f"because {incompatible_rows} non-null embeddings already exist with a "
            + "different dimension. Clear and regenerate embeddings before running "
            + "this migration."
        )


def _resize_vector_column(
    table_name: str,
    index_name: str,
    target_dimensions: int,
) -> None:
    if not column_exists(table_name, "embedding"):
        return

    _ensure_no_incompatible_embeddings(table_name, target_dimensions)

    inspector = sa.inspect(op.get_bind())
    if index_exists(table_name, index_name, inspector):
        op.drop_index(index_name, table_name=table_name, schema=schema)

    op.execute(
        sa.text(
            f"""
            ALTER TABLE "{schema}"."{table_name}"
            ALTER COLUMN embedding TYPE vector({target_dimensions})
            """
        )
    )

    op.create_index(
        index_name,
        table_name,
        ["embedding"],
        schema=schema,
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def upgrade() -> None:
    """Resize pgvector columns to the configured embedding dimension."""
    if TARGET_VECTOR_DIMENSIONS == LEGACY_VECTOR_DIMENSIONS:
        return

    _resize_vector_column(
        "message_embeddings",
        "ix_message_embeddings_embedding_hnsw",
        TARGET_VECTOR_DIMENSIONS,
    )
    _resize_vector_column(
        "documents",
        "ix_documents_embedding_hnsw",
        TARGET_VECTOR_DIMENSIONS,
    )


def downgrade() -> None:
    """Restore pgvector columns to the legacy 1536-dimensional shape."""
    _resize_vector_column(
        "message_embeddings",
        "ix_message_embeddings_embedding_hnsw",
        LEGACY_VECTOR_DIMENSIONS,
    )
    _resize_vector_column(
        "documents",
        "ix_documents_embedding_hnsw",
        LEGACY_VECTOR_DIMENSIONS,
    )
