from __future__ import annotations

from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import JSON, Boolean, CheckConstraint, LargeBinary, Text, cast, func, text
from sqlalchemy.sql.elements import ColumnElement

from src.config import settings


IS_SQLITE = settings.DB.CONNECTION_URI.startswith("sqlite")
TEXT_TYPE = Text
JSON_TYPE = JSON
EMBEDDING_TYPE: Any = (
    LargeBinary if IS_SQLITE else Vector(settings.VECTOR_STORE.DIMENSIONS)
)


def empty_json_object_default():
    return text("'{}'") if IS_SQLITE else text("'{}'::jsonb")


def nullable_json_default():
    return text("NULL")


def id_format_check(column_name: str, *, name: str = "id_format") -> CheckConstraint | None:
    if IS_SQLITE:
        return None
    return CheckConstraint(
        f"{column_name} ~ '^[A-Za-z0-9_-]+$'",
        name=name,
    )


def json_path_text(column: Any, field_name: str) -> ColumnElement[Any]:
    if IS_SQLITE:
        return func.json_extract(column, f"$.{field_name}")
    return column[field_name].astext


def json_path_bool(column: Any, field_name: str) -> ColumnElement[bool]:
    accessor = json_path_text(column, field_name)
    if IS_SQLITE:
        return cast(accessor, Text).in_(["1", "true", "TRUE"])
    return accessor.cast(Boolean)
