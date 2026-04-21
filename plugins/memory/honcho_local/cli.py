from __future__ import annotations

from hermes_constants import get_hermes_home

from plugins.memory.honcho_local import (
    _read_main_model,
    _resolve_embeddings_runtime,
    _resolve_main_text_runtime,
)
from src.honcho_embedded import (
    EmbeddedHonchoRuntimeConfig,
    describe_embedding_runtime,
    embedding_storage_dir,
    resolve_embedded_dream_settings,
)


def honcho_local_command(args) -> None:
    command = getattr(args, "honcho_local_command", None)
    if command == "status":
        _cmd_status()
        return
    print("Usage: hermes honcho_local status")


def _cmd_status() -> None:
    hermes_home = get_hermes_home()
    text_runtime = _resolve_main_text_runtime(str(hermes_home))
    embedding_runtime = _resolve_embeddings_runtime()
    runtime = (
        EmbeddedHonchoRuntimeConfig(
            default_llm=text_runtime,
            embedding=embedding_runtime,
        )
        if text_runtime or embedding_runtime
        else None
    )
    runtime_descriptor = describe_embedding_runtime(runtime)
    dream_settings = resolve_embedded_dream_settings(runtime)
    storage_dir = embedding_storage_dir(str(hermes_home), runtime)
    db_path = storage_dir / "honcho.db"

    print("\n  Honcho Local Memory")
    print(f"  HERMES_HOME: {hermes_home}")
    print(f"  Storage dir: {storage_dir}")
    print(f"  SQLite DB:   {db_path}")
    print(
        "  Embedding space: "
        f"{runtime_descriptor['fingerprint']}"
        + (
            f" ({runtime_descriptor['provider']} / {runtime_descriptor['model']})"
            if runtime_descriptor["provider"] != "none"
            else " (none)"
        )
    )
    print(f"  Main model:  {_read_main_model() or '(provider default)'}")
    if text_runtime:
        print(
            "  Text LLM:    "
            f"{text_runtime.provider}"
            + (f" / {text_runtime.model}" if text_runtime.model else " / (provider default)")
        )
    else:
        print("  Text LLM:    unresolved")
    if embedding_runtime:
        print(
            "  Embeddings:  "
            f"{embedding_runtime.provider}"
            + (f" / {embedding_runtime.model}" if embedding_runtime.model else " / (provider default)")
        )
    else:
        print("  Embeddings:  unresolved")
    print(
        "  Dreams:      "
        + (
            "enabled"
            if dream_settings["enabled"]
            else "disabled"
        )
    )
    if dream_settings["enabled"]:
        provider_label = dream_settings["provider"] or "unresolved"
        model_label = dream_settings["model"] or "(provider default)"
        print(f"  Dream LLM:   {provider_label} / {model_label}")
        print(
            "  Dreaming:    "
            f"idle {dream_settings['idle_timeout_minutes']}m, "
            f"threshold {dream_settings['document_threshold']} docs, "
            f"min gap {dream_settings['min_hours_between_dreams']}h"
        )
        print(
            "  Dream Poll:  "
            f"{dream_settings['queue_poll_seconds']}s"
        )
    print()


def register_cli(subparser) -> None:
    subs = subparser.add_subparsers(dest="honcho_local_command")
    subs.add_parser("status", help="Show local Honcho memory status")
    subparser.set_defaults(func=honcho_local_command)
