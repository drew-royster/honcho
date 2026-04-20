from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

logger_name = "honcho_embedded"
logger = logging.getLogger(logger_name)

_BOOTSTRAP_LOCK = threading.Lock()
_BOOTSTRAP_STATE: tuple[tuple[str, str], ...] | None = None
_HONCHO_MODULES: SimpleNamespace | None = None
_SANITIZE_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_-]")
_SYNC_LOOP: asyncio.AbstractEventLoop | None = None
_SYNC_LOOP_THREAD: threading.Thread | None = None
_SYNC_LOOP_LOCK = threading.Lock()
_SYNC_LOOP_READY = threading.Event()
_BOOTSTRAP_ALLOWED_PRELOADS = frozenset(
    {
        "src",
        "src.honcho_embedded",
        "src.honcho_embedded.manager",
    }
)
_DIALECTIC_LEVEL_DEFAULTS: dict[str, tuple[int, int]] = {
    "MINIMAL": (0, 1),
    "LOW": (0, 5),
    "MEDIUM": (1024, 2),
    "HIGH": (1024, 4),
    "MAX": (2048, 10),
}
_EMBEDDING_DEFAULT_MODELS: dict[str, str] = {
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
    "openrouter": "openai/text-embedding-3-small",
    "openai_compatible": "text-embedding-3-small",
    "none": "none",
}
_KNOWN_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "gemini-embedding-001": 1536,
    "text-embedding-ada-002": 1536,
    "text-embedding-3-small": 1536,
    "openai/text-embedding-ada-002": 1536,
    "openai/text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "openai/text-embedding-3-large": 3072,
}


@dataclass(frozen=True)
class EmbeddedHonchoLLMConfig:
    provider: str
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    use_codex_auth: bool = False
    codex_auth_file: str | None = None
    use_responses_api: bool = False


@dataclass(frozen=True)
class EmbeddedHonchoRuntimeConfig:
    default_llm: EmbeddedHonchoLLMConfig | None = None
    embedding: EmbeddedHonchoLLMConfig | None = None


@dataclass(frozen=True)
class EmbeddedHonchoConfig:
    hermes_home: str
    storage_key: str = "emb-default"
    workspace_id: str = "hermes"
    assistant_peer_name: str = "hermes"
    user_peer_name: str | None = None
    context_tokens: int | None = None
    user_observe_me: bool = True
    user_observe_others: bool = True
    ai_observe_me: bool = True
    ai_observe_others: bool = True
    message_max_chars: int = 25_000
    dialectic_max_input_chars: int = 10_000
    dialectic_reasoning_level: str = "low"

    @property
    def storage_root_dir(self) -> Path:
        return Path(self.hermes_home).expanduser() / "honcho"

    @property
    def storage_dir(self) -> Path:
        storage_key = _SANITIZE_ID_PATTERN.sub("-", self.storage_key).strip("-")
        storage_key = storage_key or "emb-default"
        return self.storage_root_dir / "stores" / storage_key

    @property
    def db_path(self) -> Path:
        return self.storage_dir / "honcho.db"


@dataclass
class EmbeddedHonchoSession:
    key: str
    user_peer_id: str
    assistant_peer_id: str
    honcho_session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(
        self,
        role: str,
        content: str,
        *,
        created_at: datetime | None = None,
        **kwargs: Any,
    ) -> None:
        created_at = created_at or datetime.now(UTC)
        msg = {
            "role": role,
            "content": content,
            "timestamp": created_at.isoformat(),
            "_created_at": created_at,
            **kwargs,
        }
        self.messages.append(msg)
        self.updated_at = datetime.now(UTC)

    def get_history(self, max_messages: int = 50) -> list[dict[str, Any]]:
        recent = (
            self.messages[-max_messages:]
            if len(self.messages) > max_messages
            else self.messages
        )
        return [{"role": m["role"], "content": m["content"]} for m in recent]

    def clear(self) -> None:
        self.messages = []
        self.updated_at = datetime.now(UTC)


def _ensure_sync_loop() -> asyncio.AbstractEventLoop:
    global _SYNC_LOOP, _SYNC_LOOP_THREAD

    with _SYNC_LOOP_LOCK:
        if _SYNC_LOOP is not None and _SYNC_LOOP_THREAD is not None:
            if _SYNC_LOOP_THREAD.is_alive() and not _SYNC_LOOP.is_closed():
                return _SYNC_LOOP
            _SYNC_LOOP = None
            _SYNC_LOOP_THREAD = None
            _SYNC_LOOP_READY.clear()

        def _runner() -> None:
            global _SYNC_LOOP

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            _SYNC_LOOP = loop
            _SYNC_LOOP_READY.set()
            loop.run_forever()

        _SYNC_LOOP_THREAD = threading.Thread(
            target=_runner,
            daemon=True,
            name="honcho-embedded-loop",
        )
        _SYNC_LOOP_THREAD.start()

    _SYNC_LOOP_READY.wait()
    if _SYNC_LOOP is None:
        raise RuntimeError("Embedded Honcho event loop failed to initialize.")
    return _SYNC_LOOP


def _run_coro_sync(coro: Any) -> Any:
    loop = _ensure_sync_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


def _normalize_embedding_provider(config: EmbeddedHonchoLLMConfig | None) -> str:
    if config is None:
        return "none"

    provider = (config.provider or "").strip().lower()
    if provider == "google":
        return "gemini"
    if provider in {"custom", "nous"} or config.base_url:
        return "openai_compatible"
    if provider in {"openai", "gemini", "openrouter"}:
        return provider
    return provider or "none"


def _resolve_embedding_model(config: EmbeddedHonchoLLMConfig | None) -> str:
    provider = _normalize_embedding_provider(config)
    model = str(config.model or "").strip() if config is not None else ""
    if model:
        return model
    return _EMBEDDING_DEFAULT_MODELS.get(provider, "")


def _resolve_embedding_dimensions(
    provider: str,
    model: str,
) -> int | None:
    if provider == "none":
        return None
    if provider == "gemini":
        return 1536
    return _KNOWN_EMBEDDING_DIMENSIONS.get(model.lower())


def describe_embedding_runtime(
    runtime: EmbeddedHonchoRuntimeConfig | None,
) -> dict[str, Any]:
    runtime = runtime or EmbeddedHonchoRuntimeConfig()
    embedding = runtime.embedding
    provider = _normalize_embedding_provider(embedding)
    model = _resolve_embedding_model(embedding)
    base_url = str(embedding.base_url or "").strip().rstrip("/") if embedding else ""
    dimensions = _resolve_embedding_dimensions(provider, model)

    identity = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "dimensions": dimensions,
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]

    return {
        **identity,
        "fingerprint": fingerprint,
        "storage_key": f"emb-{fingerprint}",
    }


def embedding_storage_dir(
    hermes_home: str,
    runtime: EmbeddedHonchoRuntimeConfig | None = None,
) -> Path:
    descriptor = describe_embedding_runtime(runtime)
    storage_key = str(descriptor["storage_key"])
    storage_key = _SANITIZE_ID_PATTERN.sub("-", storage_key).strip("-") or "emb-default"
    return (
        Path(hermes_home).expanduser() / "honcho" / "stores" / storage_key
    )


def prepare_embedding_storage(
    hermes_home: str,
    runtime: EmbeddedHonchoRuntimeConfig | None = None,
) -> dict[str, Any]:
    descriptor = describe_embedding_runtime(runtime)
    root = Path(hermes_home).expanduser() / "honcho"
    storage_dir = embedding_storage_dir(hermes_home, runtime)
    storage_dir.mkdir(parents=True, exist_ok=True)

    # Migrate the older flat layout on first use so existing local data stays attached
    # to the first embedding space that activates the new namespaced layout.
    for suffix in ("", "-wal", "-shm", "-journal"):
        legacy_path = root / f"honcho.db{suffix}"
        target_path = storage_dir / f"honcho.db{suffix}"
        if legacy_path.exists() and not target_path.exists():
            legacy_path.replace(target_path)

    payload = {**descriptor, "storage_dir": str(storage_dir)}
    (storage_dir / "embedding-space.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root / "active-embedding-space.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _truncate_chars(content: str, limit: int) -> list[str]:
    if len(content) <= limit:
        return [content]

    prefix = "[continued] "
    chunks: list[str] = []
    remaining = content
    first = True

    while remaining:
        effective_limit = limit if first else limit - len(prefix)
        if len(remaining) <= effective_limit:
            chunks.append(remaining if first else prefix + remaining)
            break

        segment = remaining[:effective_limit]
        cut = segment.rfind("\n\n")
        if cut < effective_limit * 0.3:
            cut = segment.rfind(". ")
            if cut >= 0:
                cut += 2
        if cut < effective_limit * 0.3:
            cut = segment.rfind(" ")
        if cut < effective_limit * 0.3:
            cut = effective_limit

        chunk = remaining[:cut].rstrip()
        remaining = remaining[cut:].lstrip()
        if not first:
            chunk = prefix + chunk
        chunks.append(chunk)
        first = False

    return chunks


def _honcho_module_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _ensure_src_package() -> None:
    """Make installed Honcho wheels importable through the existing src.* paths."""
    if "src" in sys.modules:
        return

    root = _honcho_module_root()
    pkg = ModuleType("src")
    pkg.__file__ = str(root / "__init__.py")
    pkg.__package__ = "src"
    pkg.__path__ = [str(root)]  # type: ignore[attr-defined]

    spec = ModuleSpec(name="src", loader=None, is_package=True)
    spec.submodule_search_locations = [str(root)]
    pkg.__spec__ = spec

    sys.modules["src"] = pkg


def _normalize_text_llm(config: EmbeddedHonchoLLMConfig) -> tuple[str | None, dict[str, str]]:
    provider = (config.provider or "").strip().lower()
    env: dict[str, str] = {}

    if provider in {"openai-codex", "codex"}:
        env["LLM_OPENAI_USE_CODEX_AUTH"] = "true"
        env["LLM_OPENAI_USE_RESPONSES_API"] = "true"
        if config.codex_auth_file:
            env["LLM_OPENAI_CODEX_AUTH_FILE"] = config.codex_auth_file
        if config.base_url:
            env["LLM_OPENAI_CODEX_BASE_URL"] = config.base_url
        return "openai", env

    if provider == "openai":
        if config.api_key:
            env["LLM_OPENAI_API_KEY"] = config.api_key
        if config.use_responses_api:
            env["LLM_OPENAI_USE_RESPONSES_API"] = "true"
        return "openai", env

    if provider == "anthropic":
        if config.api_key:
            env["LLM_ANTHROPIC_API_KEY"] = config.api_key
        return "anthropic", env

    if provider in {"google", "gemini"}:
        if config.api_key:
            env["LLM_GEMINI_API_KEY"] = config.api_key
        return "google", env

    if provider == "groq":
        if config.api_key:
            env["LLM_GROQ_API_KEY"] = config.api_key
        return "groq", env

    if provider == "vllm":
        if config.api_key:
            env["LLM_VLLM_API_KEY"] = config.api_key
        if config.base_url:
            env["LLM_VLLM_BASE_URL"] = config.base_url
        return "vllm", env

    if provider in {"custom", "openrouter", "nous"} or config.base_url:
        if config.api_key:
            env["LLM_OPENAI_COMPATIBLE_API_KEY"] = config.api_key
        if config.base_url:
            env["LLM_OPENAI_COMPATIBLE_BASE_URL"] = config.base_url
        return "custom", env

    return None, env


def _normalize_embedding_llm(
    config: EmbeddedHonchoLLMConfig,
) -> tuple[str | None, dict[str, str]]:
    provider = (config.provider or "").strip().lower()
    env: dict[str, str] = {}

    if provider == "openai":
        if config.api_key:
            env["LLM_EMBEDDING_API_KEY"] = config.api_key
        return "openai", env

    if provider in {"google", "gemini"}:
        if config.api_key:
            env["LLM_GEMINI_API_KEY"] = config.api_key
        return "gemini", env

    if provider == "openrouter":
        if config.api_key:
            env["LLM_EMBEDDING_API_KEY"] = config.api_key
        if config.base_url:
            env["LLM_EMBEDDING_BASE_URL"] = config.base_url
        return "openrouter", env

    if provider in {"custom", "nous"} or config.base_url:
        if config.api_key:
            env["LLM_EMBEDDING_API_KEY"] = config.api_key
        if config.base_url:
            env["LLM_EMBEDDING_BASE_URL"] = config.base_url
        return "openai_compatible", env

    return None, env


def _build_env(
    config: EmbeddedHonchoConfig,
    runtime: EmbeddedHonchoRuntimeConfig,
) -> dict[str, str]:
    config.storage_dir.mkdir(parents=True, exist_ok=True)

    env: dict[str, str] = {
        "DB_CONNECTION_URI": f"sqlite+aiosqlite:///{config.db_path.resolve()}",
        "VECTOR_STORE_TYPE": "sqlite_vec",
        "VECTOR_STORE_SQLITE_PATH": str(config.db_path.resolve()),
        "VECTOR_STORE_MIGRATED": "true",
        "CACHE_ENABLED": "false",
        "METRICS_ENABLED": "false",
        "TELEMETRY_ENABLED": "false",
        "SENTRY_ENABLED": "false",
        "DREAM_ENABLED": "false",
        "NAMESPACE": config.workspace_id,
    }

    if runtime.default_llm is not None:
        provider, provider_env = _normalize_text_llm(runtime.default_llm)
        env.update(provider_env)
        if provider is not None:
            for prefix in ("DERIVER", "SUMMARY", "DREAM"):
                env[f"{prefix}_PROVIDER"] = provider
                if runtime.default_llm.model:
                    env[f"{prefix}_MODEL"] = runtime.default_llm.model
            for level, (thinking_budget, max_tool_iterations) in (
                _DIALECTIC_LEVEL_DEFAULTS.items()
            ):
                env[f"DIALECTIC_LEVELS__{level}__PROVIDER"] = provider
                if runtime.default_llm.model:
                    env[f"DIALECTIC_LEVELS__{level}__MODEL"] = runtime.default_llm.model
                env[f"DIALECTIC_LEVELS__{level}__THINKING_BUDGET_TOKENS"] = str(
                    thinking_budget
                )
                env[f"DIALECTIC_LEVELS__{level}__MAX_TOOL_ITERATIONS"] = str(
                    max_tool_iterations
                )

    if runtime.embedding is not None:
        provider, provider_env = _normalize_embedding_llm(runtime.embedding)
        env.update(provider_env)
        if provider is not None:
            env["LLM_EMBEDDING_PROVIDER"] = provider
            if runtime.embedding.model:
                env["LLM_EMBEDDING_MODEL"] = runtime.embedding.model

    return env


def _import_honcho_modules() -> SimpleNamespace:
    from src import crud, schemas
    from src.cache.client import init_cache
    from src.db import init_db
    from src.dependencies import tracked_db
    from src.deriver.enqueue import enqueue
    from src.deriver.queue_manager import QueueManager
    from src.dialectic.chat import agentic_chat

    return SimpleNamespace(
        crud=crud,
        schemas=schemas,
        init_cache=init_cache,
        init_db=init_db,
        tracked_db=tracked_db,
        enqueue=enqueue,
        QueueManager=QueueManager,
        agentic_chat=agentic_chat,
    )


def _bootstrap_honcho(
    config: EmbeddedHonchoConfig,
    runtime: EmbeddedHonchoRuntimeConfig,
) -> SimpleNamespace:
    global _BOOTSTRAP_STATE, _HONCHO_MODULES

    env_map = _build_env(config, runtime)
    desired_state = tuple(sorted(env_map.items()))

    with _BOOTSTRAP_LOCK:
        if _BOOTSTRAP_STATE is not None:
            if desired_state != _BOOTSTRAP_STATE:
                raise RuntimeError(
                    "Embedded Honcho was already initialized with a different runtime configuration."
                )
            if _HONCHO_MODULES is None:
                raise RuntimeError("Embedded Honcho bootstrap is incomplete.")
            return _HONCHO_MODULES

        preloaded = [
            name
            for name in sys.modules
            if (name == "src" or name.startswith("src."))
            and name not in _BOOTSTRAP_ALLOWED_PRELOADS
        ]
        if preloaded:
            raise RuntimeError(
                "Honcho internals were imported before embedded configuration was applied."
            )

        for key, value in env_map.items():
            os.environ[key] = value

        _ensure_src_package()
        modules = _import_honcho_modules()
        _run_coro_sync(modules.init_cache())
        _run_coro_sync(modules.init_db())
        _BOOTSTRAP_STATE = desired_state
        _HONCHO_MODULES = modules
        return modules


class EmbeddedHonchoSessionManager:
    def __init__(
        self,
        config: EmbeddedHonchoConfig,
        runtime: EmbeddedHonchoRuntimeConfig | None = None,
    ) -> None:
        self._config = config
        self._runtime = runtime or EmbeddedHonchoRuntimeConfig()
        self._modules = _bootstrap_honcho(config, self._runtime)
        self._cache: dict[str, EmbeddedHonchoSession] = {}
        self._prefetch_cache_lock = threading.Lock()
        self._context_cache: dict[str, dict[str, str]] = {}
        self._dialectic_cache: dict[str, str] = {}
        self._queue_manager: Any | None = None

    @property
    def _qm(self) -> Any:
        if self._queue_manager is None:
            self._queue_manager = self._modules.QueueManager()
        return self._queue_manager

    def _sanitize_id(self, id_str: str) -> str:
        return _SANITIZE_ID_PATTERN.sub("-", id_str)

    def _resolve_user_peer_id(self, key: str) -> str:
        if self._config.user_peer_name:
            return self._sanitize_id(self._config.user_peer_name)
        parts = key.split(":", 1)
        channel = parts[0] if len(parts) > 1 else "default"
        chat_id = parts[1] if len(parts) > 1 else key
        return self._sanitize_id(f"user-{channel}-{chat_id}")

    def get_or_create(self, key: str) -> EmbeddedHonchoSession:
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        user_peer_id = self._resolve_user_peer_id(key)
        assistant_peer_id = self._sanitize_id(self._config.assistant_peer_name)
        honcho_session_id = self._sanitize_id(key)
        session = _run_coro_sync(
            self._load_session(key, user_peer_id, assistant_peer_id, honcho_session_id)
        )
        self._cache[key] = session
        return session

    async def _load_session(
        self,
        key: str,
        user_peer_id: str,
        assistant_peer_id: str,
        honcho_session_id: str,
    ) -> EmbeddedHonchoSession:
        schemas = self._modules.schemas
        crud = self._modules.crud

        async with self._modules.tracked_db("embedded.get_or_create") as db:
            peer_config = {
                user_peer_id: schemas.SessionPeerConfig(
                    observe_me=self._config.user_observe_me,
                    observe_others=self._config.user_observe_others,
                ),
                assistant_peer_id: schemas.SessionPeerConfig(
                    observe_me=self._config.ai_observe_me,
                    observe_others=self._config.ai_observe_others,
                ),
            }
            result = await crud.get_or_create_session(
                db,
                session=schemas.SessionCreate(name=honcho_session_id, peers=peer_config),
                workspace_name=self._config.workspace_id,
            )
            await result.post_commit()

            stmt = await crud.get_messages(
                workspace_name=self._config.workspace_id,
                session_name=honcho_session_id,
                message_count_limit=200,
            )
            records = list((await db.execute(stmt)).scalars().all())
            await db.commit()

        local_messages: list[dict[str, Any]] = []
        for record in records:
            role = "assistant" if record.peer_name == assistant_peer_id else "user"
            local_messages.append(
                {
                    "role": role,
                    "content": record.content,
                    "timestamp": record.created_at.isoformat(),
                    "_created_at": record.created_at,
                    "_synced": True,
                }
            )

        return EmbeddedHonchoSession(
            key=key,
            user_peer_id=user_peer_id,
            assistant_peer_id=assistant_peer_id,
            honcho_session_id=honcho_session_id,
            messages=local_messages,
        )

    def _flush_session(self, session: EmbeddedHonchoSession) -> bool:
        pending_messages = [msg for msg in session.messages if not msg.get("_synced")]
        if not pending_messages:
            return True

        try:
            _run_coro_sync(self._persist_messages(session, pending_messages))
        except Exception:
            logger.exception("Failed to persist embedded Honcho messages for %s", session.key)
            return False

        for msg in pending_messages:
            msg["_synced"] = True
        self._cache[session.key] = session
        return True

    async def _persist_messages(
        self,
        session: EmbeddedHonchoSession,
        pending_messages: list[dict[str, Any]],
    ) -> None:
        schemas = self._modules.schemas
        crud = self._modules.crud

        message_models = []
        for message in pending_messages:
            peer_id = (
                session.user_peer_id
                if message["role"] == "user"
                else session.assistant_peer_id
            )
            message_models.append(
                schemas.MessageCreate(
                    peer_id=peer_id,
                    content=message["content"],
                    created_at=message.get("_created_at"),
                )
            )

        async with self._modules.tracked_db("embedded.persist_messages") as db:
            created_messages = await crud.create_messages(
                db,
                messages=message_models,
                workspace_name=self._config.workspace_id,
                session_name=session.honcho_session_id,
            )

        payloads = [
            {
                "workspace_name": self._config.workspace_id,
                "session_name": session.honcho_session_id,
                "message_id": message.id,
                "content": message.content,
                "peer_name": message.peer_name,
                "created_at": message.created_at,
                "message_public_id": message.public_id,
                "seq_in_session": message.seq_in_session,
                "configuration": None,
            }
            for message in created_messages
        ]

        await self._modules.enqueue(payloads)
        await self._drain_queue()

    async def _drain_queue(self) -> None:
        while True:
            claimed = await self._qm.get_and_claim_work_units()
            if not claimed:
                return

            for work_unit_key, aqs_id in claimed.items():
                worker_id = self._qm.create_worker_id()
                self._qm.track_worker_work_unit(worker_id, work_unit_key, aqs_id)
                await self._qm.process_work_unit(work_unit_key, worker_id)

    def flush_all(self) -> None:
        for session in list(self._cache.values()):
            self._flush_session(session)

    def save(self, session: EmbeddedHonchoSession) -> None:
        self._flush_session(session)

    async def _read_peer_context(
        self,
        observer: str,
        observed: str,
        *,
        search_query: str | None = None,
        max_observations: int | None = None,
    ) -> dict[str, Any]:
        schemas = self._modules.schemas
        crud = self._modules.crud

        async with self._modules.tracked_db("embedded.peer_context") as db:
            peers_result = await crud.get_or_create_peers(
                db,
                workspace_name=self._config.workspace_id,
                peers=[
                    schemas.PeerCreate(name=observer),
                    schemas.PeerCreate(name=observed),
                ],
            )
            await db.commit()
            await peers_result.post_commit()

            representation = await crud.get_working_representation(
                self._config.workspace_id,
                db=db,
                observer=observer,
                observed=observed,
                include_semantic_query=search_query,
                include_most_derived=bool(search_query),
                semantic_search_top_k=10 if search_query else None,
                max_observations=max_observations
                or 100,
            )
            peer_card = await crud.get_peer_card(
                db,
                self._config.workspace_id,
                observer=observer,
                observed=observed,
            )
            await db.commit()

        return {
            "representation": str(representation).strip(),
            "card": peer_card or [],
        }

    def _user_observer_pair(self, session: EmbeddedHonchoSession) -> tuple[str, str]:
        if self._config.ai_observe_others:
            return session.assistant_peer_id, session.user_peer_id
        return session.user_peer_id, session.user_peer_id

    def get_prefetch_context(
        self, session_key: str, user_message: str | None = None
    ) -> dict[str, str]:
        cached = self._context_cache.get(session_key)
        if cached:
            return cached

        session = self.get_or_create(session_key)
        observer, observed = self._user_observer_pair(session)
        user_ctx = _run_coro_sync(self._read_peer_context(observer, observed))
        ai_ctx = _run_coro_sync(
            self._read_peer_context(
                session.assistant_peer_id,
                session.assistant_peer_id,
            )
        )

        result = {
            "representation": user_ctx.get("representation", ""),
            "card": "\n".join(user_ctx.get("card", [])),
            "ai_representation": ai_ctx.get("representation", ""),
            "ai_card": "\n".join(ai_ctx.get("card", [])),
        }
        with self._prefetch_cache_lock:
            self._context_cache[session_key] = result
        return result

    def prefetch_context(self, session_key: str, user_message: str | None = None) -> None:
        def _run() -> None:
            try:
                self.get_prefetch_context(session_key, user_message)
            except Exception:
                logger.exception("Embedded Honcho context prefetch failed for %s", session_key)

        threading.Thread(
            target=_run,
            daemon=True,
            name="honcho-embedded-context-prefetch",
        ).start()

    def get_peer_card(self, session_key: str) -> list[str]:
        session = self.get_or_create(session_key)
        observer, observed = self._user_observer_pair(session)
        return list(
            _run_coro_sync(self._read_peer_context(observer, observed)).get("card", [])
        )

    def _peer_observer_pair(
        self, session: EmbeddedHonchoSession, *, peer: str
    ) -> tuple[str, str]:
        if peer == "ai":
            return session.assistant_peer_id, session.assistant_peer_id
        return self._user_observer_pair(session)

    def search_context(
        self,
        session_key: str,
        query: str,
        max_tokens: int = 800,
        *,
        peer: str = "user",
    ) -> str:
        session = self.get_or_create(session_key)
        observer, observed = self._peer_observer_pair(session, peer=peer)
        context = _run_coro_sync(
            self._read_peer_context(observer, observed, search_query=query)
        )

        parts: list[str] = []
        representation = context.get("representation", "")
        if representation:
            parts.append(representation)
        card = context.get("card", [])
        if card:
            parts.append("\n".join(f"- {item}" for item in card))

        combined = "\n\n".join(parts).strip()
        max_chars = max_tokens * 4
        if max_chars > 0 and len(combined) > max_chars:
            combined = combined[:max_chars].rsplit(" ", 1)[0] + " …"
        return combined

    async def _dialectic_query_async(
        self,
        session: EmbeddedHonchoSession,
        query: str,
        *,
        peer: str,
    ) -> str:
        observer, observed = self._peer_observer_pair(session, peer=peer)

        return await self._modules.agentic_chat(
            workspace_name=self._config.workspace_id,
            session_name=None,
            query=query,
            observer=observer,
            observed=observed,
            reasoning_level=self._config.dialectic_reasoning_level,
        )

    def dialectic_query(
        self,
        session_key: str,
        query: str,
        reasoning_level: str | None = None,
        peer: str = "user",
    ) -> str:
        del reasoning_level
        session = self.get_or_create(session_key)
        if len(query) > self._config.dialectic_max_input_chars:
            query = query[: self._config.dialectic_max_input_chars].rsplit(" ", 1)[0]
        return _run_coro_sync(self._dialectic_query_async(session, query, peer=peer))

    def prefetch_dialectic(self, session_key: str, query: str) -> None:
        def _run() -> None:
            try:
                result = self.dialectic_query(session_key, query)
                if result:
                    with self._prefetch_cache_lock:
                        self._dialectic_cache[session_key] = result
            except Exception:
                logger.exception(
                    "Embedded Honcho dialectic prefetch failed for %s", session_key
                )

        threading.Thread(
            target=_run,
            daemon=True,
            name="honcho-embedded-dialectic-prefetch",
        ).start()

    def pop_dialectic_result(self, session_key: str) -> str:
        with self._prefetch_cache_lock:
            return self._dialectic_cache.pop(session_key, "")

    async def _create_conclusion_async(
        self,
        session: EmbeddedHonchoSession,
        content: str,
    ) -> bool:
        schemas = self._modules.schemas
        crud = self._modules.crud

        if self._config.ai_observe_others:
            observer = session.assistant_peer_id
            observed = session.user_peer_id
        else:
            observer = session.user_peer_id
            observed = session.user_peer_id

        async with self._modules.tracked_db("embedded.create_conclusion") as db:
            docs = await crud.create_observations(
                db,
                observations=[
                    schemas.ConclusionCreate(
                        content=content.strip(),
                        observer_id=observer,
                        observed_id=observed,
                        session_id=session.honcho_session_id,
                    )
                ],
                workspace_name=self._config.workspace_id,
            )
            return bool(docs)

    def create_conclusion(self, session_key: str, content: str) -> bool:
        if not content or not content.strip():
            return False
        session = self.get_or_create(session_key)
        return bool(_run_coro_sync(self._create_conclusion_async(session, content)))

    def migrate_memory_files(self, session_key: str, memory_dir: str) -> bool:
        session = self.get_or_create(session_key)
        memory_path = Path(memory_dir)
        if not memory_path.exists():
            return False

        uploads: list[tuple[str, str]] = []
        files = [
            ("MEMORY.md", session.user_peer_id, "Long-term agent notes and preferences"),
            ("USER.md", session.user_peer_id, "User profile and preferences"),
            ("SOUL.md", session.assistant_peer_id, "Agent persona and identity configuration"),
        ]

        for filename, peer_name, description in files:
            file_path = memory_path / filename
            if not file_path.exists():
                continue
            content = file_path.read_text(encoding="utf-8").strip()
            if not content:
                continue

            wrapped = (
                "<prior_memory_file>\n"
                "<context>\n"
                "This file was consolidated from local Hermes memory before embedded Honcho was activated.\n"
                f"{description}. Treat it as background context.\n"
                "</context>\n\n"
                f"{content}\n"
                "</prior_memory_file>"
            )
            for chunk in _truncate_chars(wrapped, self._config.message_max_chars):
                uploads.append((peer_name, chunk))

        if not uploads:
            return False

        for peer_name, content in uploads:
            role = "assistant" if peer_name == session.assistant_peer_id else "user"
            session.add_message(role, content)

        return self._flush_session(session)
