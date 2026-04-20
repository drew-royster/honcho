"""Embedded local Honcho memory provider for Hermes.

This provider ships inside the Honcho wheel as a Hermes memory plugin under
``plugins/memory/honcho_local`` so Hermes can discover it without any source
changes. It runs Honcho in-process via ``src.honcho_embedded`` and reuses
Hermes runtime provider/model configuration instead of maintaining a separate
Honcho credential stack.
"""

from __future__ import annotations

import json
import logging
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any

from agent.memory_provider import MemoryProvider

from src.honcho_embedded import (
    EmbeddedHonchoConfig,
    EmbeddedHonchoLLMConfig,
    EmbeddedHonchoRuntimeConfig,
    EmbeddedHonchoSessionManager,
    prepare_embedding_storage,
)

logger = logging.getLogger(__name__)


PROFILE_SCHEMA = {
    "name": "honcho_profile",
    "description": (
        "Retrieve the user's Honcho peer card, a compact list of durable facts and "
        "preferences learned across sessions."
    ),
    "parameters": {"type": "object", "properties": {}, "required": []},
}

SEARCH_SCHEMA = {
    "name": "honcho_search",
    "description": (
        "Semantic search over Honcho's local memory store. Returns relevant raw context "
        "without additional synthesis."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for in Honcho memory.",
            },
            "max_tokens": {
                "type": "integer",
                "description": "Approximate token budget for returned context (default 800, max 2000).",
            },
        },
        "required": ["query"],
    },
}

CONTEXT_SCHEMA = {
    "name": "honcho_context",
    "description": (
        "Ask Honcho a natural-language question and get a synthesized answer from its "
        "local memory graph."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A natural-language question about the user or assistant.",
            },
            "peer": {
                "type": "string",
                "description": "Which peer to query: 'user' (default) or 'ai'.",
            },
        },
        "required": ["query"],
    },
}

CONCLUDE_SCHEMA = {
    "name": "honcho_conclude",
    "description": (
        "Persist a conclusion about the user into Honcho's local long-term memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "conclusion": {
                "type": "string",
                "description": "A durable fact or preference to remember.",
            }
        },
        "required": ["conclusion"],
    },
}

ALL_TOOL_SCHEMAS = [PROFILE_SCHEMA, SEARCH_SCHEMA, CONTEXT_SCHEMA, CONCLUDE_SCHEMA]


def _load_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _read_main_model() -> str | None:
    cfg = _load_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, str) and model_cfg.strip():
        return model_cfg.strip()
    if isinstance(model_cfg, dict):
        default = str(model_cfg.get("default") or model_cfg.get("model") or "").strip()
        return default or None
    return None


def _write_codex_auth_file(hermes_home: str) -> str | None:
    try:
        from hermes_cli.auth import _read_codex_tokens

        payload = _read_codex_tokens()
    except Exception as exc:
        logger.debug("Hermes Codex token export failed: %s", exc)
        return None

    target = Path(hermes_home).expanduser() / "honcho" / "codex-auth.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return str(target)


def _resolve_main_text_runtime(hermes_home: str) -> EmbeddedHonchoLLMConfig | None:
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        runtime = resolve_runtime_provider()
    except Exception as exc:
        logger.warning("Failed to resolve Hermes main runtime for embedded Honcho: %s", exc)
        return None

    provider = str(runtime.get("provider") or "").strip().lower()
    model = _read_main_model()
    base_url = str(runtime.get("base_url") or "").strip() or None
    api_key = str(runtime.get("api_key") or "").strip() or None

    if provider == "google":
        provider = "gemini"
    if provider == "custom" and not base_url:
        return None
    if provider == "openai-codex":
        auth_file = _write_codex_auth_file(hermes_home)
        if not auth_file:
            logger.warning("Codex is active in Hermes, but tokens could not be exported for Honcho.")
            return None
        return EmbeddedHonchoLLMConfig(
            provider="openai-codex",
            model=model,
            base_url=base_url,
            codex_auth_file=auth_file,
            use_codex_auth=True,
            use_responses_api=True,
        )

    if provider in {"openai", "anthropic", "gemini", "groq", "vllm"}:
        return EmbeddedHonchoLLMConfig(
            provider=provider,
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

    if provider in {"openrouter", "nous", "custom"} or base_url:
        return EmbeddedHonchoLLMConfig(
            provider=provider or "custom",
            model=model,
            base_url=base_url,
            api_key=api_key,
        )

    logger.debug("Hermes main provider %r is not supported for embedded Honcho text runtime.", provider)
    return None


def _resolve_embeddings_runtime() -> EmbeddedHonchoLLMConfig | None:
    try:
        from agent.auxiliary_client import _resolve_task_provider_model
        from hermes_cli.runtime_provider import resolve_runtime_provider
    except Exception as exc:
        logger.debug("Hermes embedding runtime helpers unavailable: %s", exc)
        return None

    try:
        requested_provider, model, base_url, api_key, _api_mode = _resolve_task_provider_model(
            "embeddings"
        )
    except Exception as exc:
        logger.debug("Hermes auxiliary.embeddings resolution failed: %s", exc)
        requested_provider, model, base_url, api_key = "auto", None, None, None

    requested_provider = (requested_provider or "auto").strip().lower()
    model = str(model or "").strip() or None
    base_url = str(base_url or "").strip() or None
    api_key = str(api_key or "").strip() or None

    try:
        runtime = resolve_runtime_provider(
            requested_provider=None if requested_provider == "auto" else requested_provider,
            explicit_api_key=api_key,
            explicit_base_url=base_url,
        )
    except Exception as exc:
        logger.debug("Hermes embedding runtime provider resolution failed: %s", exc)
        runtime = {
            "provider": requested_provider if requested_provider != "auto" else "",
            "base_url": base_url,
            "api_key": api_key,
        }

    provider = str(runtime.get("provider") or requested_provider or "").strip().lower()
    runtime_base_url = str(runtime.get("base_url") or base_url or "").strip() or None
    runtime_api_key = str(runtime.get("api_key") or api_key or "").strip() or None

    if provider == "google":
        provider = "gemini"
    if provider in {
        "openai-codex",
        "codex",
        "anthropic",
        "groq",
        "vllm",
        "qwen-oauth",
        "google-gemini-cli",
        "copilot",
        "xai",
    }:
        logger.warning(
            "Embedded Honcho cannot use Hermes provider=%r for embeddings. "
            "Configure auxiliary.embeddings explicitly.",
            provider,
        )
        return None
    if provider == "openai":
        return EmbeddedHonchoLLMConfig(provider="openai", model=model, api_key=runtime_api_key)
    if provider in {"gemini", "openrouter"}:
        return EmbeddedHonchoLLMConfig(
            provider=provider,
            model=model,
            base_url=runtime_base_url,
            api_key=runtime_api_key,
        )
    if provider in {"nous", "custom"} or runtime_base_url:
        return EmbeddedHonchoLLMConfig(
            provider=provider or "custom",
            model=model,
            base_url=runtime_base_url,
            api_key=runtime_api_key,
        )

    logger.warning(
        "Embedded Honcho could not derive a compatible embeddings runtime from Hermes provider=%r. "
        "Add auxiliary.embeddings in config.yaml if your main provider does not expose embeddings.",
        provider or requested_provider or "auto",
    )
    return None


def _resolve_runtime_config(hermes_home: str) -> EmbeddedHonchoRuntimeConfig:
    return EmbeddedHonchoRuntimeConfig(
        default_llm=_resolve_main_text_runtime(hermes_home),
        embedding=_resolve_embeddings_runtime(),
    )


class HonchoLocalMemoryProvider(MemoryProvider):
    """Hermes memory provider that runs Honcho locally in-process."""

    def __init__(self) -> None:
        self._manager: EmbeddedHonchoSessionManager | None = None
        self._session_key = ""
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: threading.Thread | None = None
        self._sync_thread: threading.Thread | None = None
        self._first_turn_context: str | None = None
        self._first_turn_lock = threading.Lock()
        self._context_tokens: int | None = None
        self._message_max_chars = 25_000
        self._cron_skipped = False

    @property
    def name(self) -> str:
        return "honcho_local"

    def is_available(self) -> bool:
        return True

    def get_config_schema(self) -> list[dict[str, Any]]:
        return []

    def save_config(self, values: dict[str, Any], hermes_home: str) -> None:
        del values, hermes_home

    def initialize(self, session_id: str, **kwargs) -> None:
        agent_context = kwargs.get("agent_context", "")
        platform = kwargs.get("platform", "cli")
        if agent_context in ("cron", "flush") or platform == "cron":
            self._cron_skipped = True
            return

        hermes_home = str(kwargs.get("hermes_home") or Path("~/.hermes").expanduser())
        assistant_peer_name = str(kwargs.get("agent_identity") or "hermes").strip() or "hermes"
        workspace_id = str(kwargs.get("agent_workspace") or "hermes").strip() or "hermes"

        self._session_key = session_id or "hermes-default"
        self._context_tokens = 800

        runtime = _resolve_runtime_config(hermes_home)
        storage_info = prepare_embedding_storage(hermes_home, runtime)
        config = EmbeddedHonchoConfig(
            hermes_home=hermes_home,
            storage_key=str(storage_info["storage_key"]),
            workspace_id=workspace_id,
            assistant_peer_name=assistant_peer_name,
            context_tokens=self._context_tokens,
        )
        logger.info(
            "Embedded Honcho using embedding space %s (%s / %s)",
            storage_info["fingerprint"],
            storage_info["provider"],
            storage_info["model"] or "provider-default",
        )
        self._manager = EmbeddedHonchoSessionManager(config=config, runtime=runtime)

        session = self._manager.get_or_create(self._session_key)
        self._message_max_chars = config.message_max_chars

        try:
            if not session.messages:
                memories_dir = Path(hermes_home).expanduser() / "memories"
                self._manager.migrate_memory_files(self._session_key, str(memories_dir))
        except Exception as exc:
            logger.debug("Embedded Honcho memory-file migration skipped: %s", exc)

        try:
            self._manager.prefetch_context(self._session_key)
            self.queue_prefetch("What should I know about this user?")
        except Exception as exc:
            logger.debug("Embedded Honcho prewarm failed: %s", exc)

    def _truncate_to_budget(self, text: str) -> str:
        if not self._context_tokens:
            return text
        budget_chars = self._context_tokens * 4
        if len(text) <= budget_chars:
            return text
        truncated = text[:budget_chars]
        last_space = truncated.rfind(" ")
        if last_space > budget_chars * 0.8:
            truncated = truncated[:last_space]
        return truncated + " …"

    def _format_prefetch_context(self, ctx: dict[str, str]) -> str:
        parts: list[str] = []
        rep = ctx.get("representation", "")
        if rep:
            parts.append(f"## User Representation\n{rep}")
        card = ctx.get("card", "")
        if card:
            parts.append(f"## User Peer Card\n{card}")
        ai_rep = ctx.get("ai_representation", "")
        if ai_rep:
            parts.append(f"## AI Self-Representation\n{ai_rep}")
        ai_card = ctx.get("ai_card", "")
        if ai_card:
            parts.append(f"## AI Identity Card\n{ai_card}")
        return "\n\n".join(parts).strip()

    def system_prompt_block(self) -> str:
        if self._cron_skipped or not self._manager or not self._session_key:
            return ""

        with self._first_turn_lock:
            if self._first_turn_context is None:
                try:
                    ctx = self._manager.get_prefetch_context(self._session_key)
                    self._first_turn_context = self._format_prefetch_context(ctx)
                except Exception as exc:
                    logger.debug("Embedded Honcho first-turn context fetch failed: %s", exc)
                    self._first_turn_context = ""

        header = (
            "# Honcho Memory\n"
            "Active (local embedded mode). Relevant local memory may be auto-injected and "
            "the Honcho tools are available for explicit recall and writes."
        )
        if self._first_turn_context:
            return f"{header}\n\n{self._first_turn_context}"
        return header

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        del query, session_id
        if self._cron_skipped:
            return ""
        if self._prefetch_thread and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=3.0)
        with self._prefetch_lock:
            result = self._prefetch_result
            self._prefetch_result = ""
        if not result:
            return ""
        return f"## Honcho Context\n{self._truncate_to_budget(result)}"

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        del session_id
        if self._cron_skipped or not self._manager or not self._session_key or not query:
            return

        def _run() -> None:
            try:
                result = self._manager.dialectic_query(self._session_key, query, peer="user")
                if result and result.strip():
                    with self._prefetch_lock:
                        self._prefetch_result = result
            except Exception as exc:
                logger.debug("Embedded Honcho prefetch failed: %s", exc)

        self._prefetch_thread = threading.Thread(
            target=_run,
            daemon=True,
            name="honcho-local-prefetch",
        )
        self._prefetch_thread.start()

        try:
            self._manager.prefetch_context(self._session_key, query)
        except Exception as exc:
            logger.debug("Embedded Honcho context prefetch failed: %s", exc)

    @staticmethod
    def _chunk_message(content: str, limit: int) -> list[str]:
        if len(content) <= limit:
            return [content]

        prefix = "[continued] "
        chunks: list[str] = []
        remaining = content
        first = True
        while remaining:
            effective = limit if first else limit - len(prefix)
            if len(remaining) <= effective:
                chunks.append(remaining if first else prefix + remaining)
                break

            segment = remaining[:effective]
            cut = segment.rfind("\n\n")
            if cut < effective * 0.3:
                cut = segment.rfind(". ")
                if cut >= 0:
                    cut += 2
            if cut < effective * 0.3:
                cut = segment.rfind(" ")
            if cut < effective * 0.3:
                cut = effective

            chunk = remaining[:cut].rstrip()
            remaining = remaining[cut:].lstrip()
            if not first:
                chunk = prefix + chunk
            chunks.append(chunk)
            first = False

        return chunks

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        del session_id
        if self._cron_skipped or not self._manager or not self._session_key:
            return

        def _sync() -> None:
            try:
                session = self._manager.get_or_create(self._session_key)
                for chunk in self._chunk_message(user_content, self._message_max_chars):
                    session.add_message("user", chunk)
                for chunk in self._chunk_message(assistant_content, self._message_max_chars):
                    session.add_message("assistant", chunk)
                self._manager.save(session)
            except Exception as exc:
                logger.debug("Embedded Honcho sync_turn failed: %s", exc)

        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=5.0)
        self._sync_thread = threading.Thread(
            target=_sync,
            daemon=True,
            name="honcho-local-sync",
        )
        self._sync_thread.start()

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if action != "add" or target != "user" or not content:
            return
        if self._cron_skipped or not self._manager or not self._session_key:
            return

        def _write() -> None:
            try:
                self._manager.create_conclusion(self._session_key, content)
            except Exception as exc:
                logger.debug("Embedded Honcho memory mirror failed: %s", exc)

        threading.Thread(
            target=_write,
            daemon=True,
            name="honcho-local-memwrite",
        ).start()

    def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        del messages
        if self._cron_skipped or not self._manager:
            return
        if self._sync_thread and self._sync_thread.is_alive():
            self._sync_thread.join(timeout=10.0)
        try:
            self._manager.flush_all()
        except Exception as exc:
            logger.debug("Embedded Honcho session-end flush failed: %s", exc)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        if self._cron_skipped:
            return []
        return list(ALL_TOOL_SCHEMAS)

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        del kwargs
        if self._cron_skipped:
            return json.dumps({"error": "Honcho is not active in cron context."})
        if not self._manager or not self._session_key:
            return json.dumps({"error": "Honcho is not active for this session."})

        try:
            if tool_name == "honcho_profile":
                card = self._manager.get_peer_card(self._session_key)
                return json.dumps({"result": card or "No profile facts available yet."})

            if tool_name == "honcho_search":
                query = str(args.get("query") or "").strip()
                if not query:
                    return json.dumps({"error": "Missing required parameter: query"})
                max_tokens = min(int(args.get("max_tokens", 800)), 2000)
                result = self._manager.search_context(
                    self._session_key,
                    query,
                    max_tokens=max_tokens,
                )
                return json.dumps({"result": result or "No relevant context found."})

            if tool_name == "honcho_context":
                query = str(args.get("query") or "").strip()
                if not query:
                    return json.dumps({"error": "Missing required parameter: query"})
                peer = str(args.get("peer") or "user").strip() or "user"
                result = self._manager.dialectic_query(self._session_key, query, peer=peer)
                return json.dumps({"result": result or "No result from Honcho."})

            if tool_name == "honcho_conclude":
                conclusion = str(args.get("conclusion") or "").strip()
                if not conclusion:
                    return json.dumps({"error": "Missing required parameter: conclusion"})
                ok = self._manager.create_conclusion(self._session_key, conclusion)
                if ok:
                    return json.dumps({"result": f"Conclusion saved: {conclusion}"})
                return json.dumps({"error": "Failed to save conclusion."})

            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as exc:
            logger.error("Embedded Honcho tool %s failed: %s", tool_name, exc)
            return json.dumps({"error": f"Honcho {tool_name} failed: {exc}"})

    def shutdown(self) -> None:
        for thread in (self._prefetch_thread, self._sync_thread):
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        if self._manager:
            with suppress(Exception):
                self._manager.flush_all()


def register(ctx) -> None:
    ctx.register_memory_provider(HonchoLocalMemoryProvider())
